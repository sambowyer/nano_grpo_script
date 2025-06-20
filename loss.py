from typing import Any, Dict, Tuple, Union

import torch
from deepspeed import DeepSpeedEngine
from transformers import PreTrainedModel

from utils import compute_token_log_probs


def compute_pg_loss(
    policy_model: Union[DeepSpeedEngine, PreTrainedModel],
    reference_model: Union[DeepSpeedEngine, PreTrainedModel],
    batch: Dict[str, torch.Tensor],
    total_response_len: int,
    TEMPERATURE: float,
    KL_COEFFICIENT: float,
    algo_config: Dict[str, Any] = None, 
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute the policy gradient loss with KL penalty between policy and reference models.
    Supports GRPO, Dr. GRPO, and DAPO variants.

    This function:
    1. Computes log probabilities for both policy and reference models
    2. Calculates importance sampling ratio between current and old policy
    3. Implements clipping with configurable low/high bounds
    4. Optionally adds KL divergence penalty
    5. Supports various normalization schemes for advantages and length

    Args:
        policy_model: The model being trained
        reference_model: The reference model for KL penalty calculation
        batch: Dictionary containing:
            - input_ids: Tensor of shape [batch_size, seq_len]
            - attention_mask: Tensor of shape [batch_size, seq_len]
            - labels: Tensor of shape [batch_size, seq_len] with -100 for ignored positions
            - advantages: Tensor of shape [batch_size, seq_len]
            - old_logps: Optional tensor of shape [batch_size, seq_len-1] with old log probs
            - adv_den: Optional tensor with advantage denominators for Dr. GRPO/DAPO

        algo_config: Configuration for the algorithm variant:
            - eps_low: Lower clipping bound (default: 0.2)
            - eps_high: Higher clipping bound (default: eps_low or 0.28 for DAPO)
            - norm_adv: Whether to normalize advantages by std (default: "std" for GRPO, "none" for Dr. GRPO/DAPO)
            - length_norm: Whether to use response-level length normalization (default: True for GRPO, False for Dr. GRPO/DAPO)

    Returns:
        Tuple containing:
            - loss: Combined policy gradient and KL penalty loss (scalar tensor)
            - metrics: Dictionary with detailed loss components
    """
    # Set default configuration if not provided
    if algo_config is None:
        algo_config = {
            "eps_low": 0.2,
            "eps_high": 0.2,
            "norm_adv": "std",  # "std" or "none"
            "length_norm": True,  # True for GRPO, False for Dr. GRPO/DAPO
        }

    input_ids = batch["input_ids"]  # [batch_size, seq_len]
    attention_mask = batch["attention_mask"]  # [batch_size, seq_len]
    labels = batch["labels"]  # [batch_size, seq_len]
    advantages = batch["advantages"]  # [batch_size, seq_len]

    model_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

    labels_mask = (labels[..., 1:] != -100).float()  # [batch_size, seq_len-1]

    # Compute reference log probabilities for KL penalty
    with torch.no_grad():
        ref_logps = compute_token_log_probs(
            reference_model, model_inputs, TEMPERATURE
        )  # [batch_size, seq_len-1]

    # Compute current log probabilities
    logps = compute_token_log_probs(
        policy_model, model_inputs, TEMPERATURE
    )  # [batch_size, seq_len-1]

    # Compute importance sampling ratio (if old_logps are available)
    if "old_logps" in batch:
        # Use stored old log probabilities (GRPO/Dr. GRPO/DAPO)
        old_logps = batch["old_logps"][..., 1:]
        ratio = torch.exp(logps - old_logps)
    else:
        # Fallback to policy gradient (no ratio/clipping)
        ratio = torch.ones_like(logps)

    # Advantage normalization is controlled by algo_config and already done in process_training_episodes
    adv = advantages[..., 1:]
    # No secondary normalization needed here

    # Compute clipped surrogate objective
    clipped_ratio = torch.clamp(
        ratio, min=1.0 - algo_config["eps_low"], max=1.0 + algo_config["eps_high"]
    )

    # Sign-aware clipping: use min for positive advantages, max for negative advantages
    use_min = adv >= 0
    surrogate1 = ratio * adv * labels_mask
    surrogate2 = clipped_ratio * adv * labels_mask
    policy_loss_per_token = -torch.where(
        use_min, torch.min(surrogate1, surrogate2), torch.max(surrogate1, surrogate2)
    )

    # Compute KL penalty separately (not inside surrogate clipping)
    kl_penalty = torch.exp(ref_logps - logps) - (ref_logps - logps) - 1
    kl_penalty = kl_penalty * labels_mask

    # Compute entropy for monitoring
    entropy = -logps.sum() / labels_mask.sum()

    # Length normalization is controlled by algo_config
    if algo_config["length_norm"]:
        # Original GRPO with response-level length normalization
        # Properly divide each response's loss by its length before averaging
        tok_per_resp = labels_mask.sum(-1)  # [B]
        policy_loss = (
            policy_loss_per_token.sum(-1) / tok_per_resp.clamp(min=1.0)
        ).mean()
    else:
        # Dr. GRPO / DAPO with token-level normalization
        if "adv_den" in batch:
            # Use provided token budget (Dr. GRPO)
            policy_loss = policy_loss_per_token.sum() / batch["adv_den"].sum()
        else:
            # Fallback to total response length (similar to Dr. GRPO)
            policy_loss = policy_loss_per_token.sum() / total_response_len

    # Apply KL penalty (separately from surrogate clipping)
    loss = policy_loss + KL_COEFFICIENT * kl_penalty.sum() / total_response_len

    # Compute metrics for clip rates - masked to only include valid response tokens
    with torch.no_grad():
        clip_low_rate = (
            (ratio < 1.0 - algo_config["eps_low"]) & (adv < 0) & (labels_mask > 0)
        ).float().sum() / labels_mask.sum()
        clip_high_rate = (
            (ratio > 1.0 + algo_config["eps_high"]) & (adv > 0) & (labels_mask > 0)
        ).float().sum() / labels_mask.sum()
        clip_rate = (
            ((ratio < 1.0 - algo_config["eps_low"]) & (adv < 0) & (labels_mask > 0))
            | ((ratio > 1.0 + algo_config["eps_high"]) & (adv > 0) & (labels_mask > 0))
        ).float().sum() / labels_mask.sum()

    metrics = {
        "policy_loss": policy_loss.item(),
        "kl_penalty": kl_penalty.sum().item() / total_response_len,
        "entropy": entropy.item(),
        "clip_ratio/low_rate": clip_low_rate.item(),
        "clip_ratio/high_rate": clip_high_rate.item(),
        "clip_ratio/region_rate": clip_rate.item(),
    }

    return loss, metrics

