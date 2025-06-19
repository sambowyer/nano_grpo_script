import os
from pathlib import Path

## CHANGE THESE PATHS TO YOUR OWN
SCRATCH = Path("/user/work/dg22309/grpo/nano_script")
os.environ["HF_HOME"] = str("/user/work/dg22309/huggingface")

## CHANGE THESE TO YOUR OWN WANDB ENTITY AND PROJECT
WANDB_ENTITY = "sam-bowyer-bristol"
WANDB_PROJECT = "nano-grpo"

import argparse
import gc
import re
import time
from typing import Any, Dict, List, Tuple, Union

import deepspeed
import numpy as np
import torch
from datasets import load_dataset
from deepspeed import DeepSpeedEngine
from tqdm import trange
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from vllm import LLM, SamplingParams

import wandb
from utils import (
    compute_token_log_probs,
    dump_episodes,
    evaluate_on_test_set,
    find_free_port,
    find_last_checkpoint,
    prepare_model_inputs,
    load_model_into_vllm,
)


def preprocess_example(
    example: Dict[str, Any],
    tokenizer: AutoTokenizer,
    SYSTEM_MESSAGE: str,
    PROMPT_TEMPLATE: str,
) -> Dict[str, Any]:
    """
    Preprocess an example from the dataset to create a prompt/chat template for the model to follow.

    Args:
        example (Dict[str, Any]): An example from the dataset
        tokenizer (AutoTokenizer): The tokenizer to use
        SYSTEM_MESSAGE (str): The system message to use
        PROMPT_TEMPLATE (str): The prompt template to use

    Returns:
        Dict[str, Any]: A dictionary containing the prompt and input_ids
    """
    numbers: List[int] = example["nums"]
    target: int = example["target"]

    prefix = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {
            "role": "user",
            "content": PROMPT_TEMPLATE.format(numbers=numbers, target=target),
        },
        {"role": "assistant", "content": "Let me solve this step by step.\n<think>"},
    ]
    input_ids = tokenizer.apply_chat_template(
        prefix, tokenize=True, continue_final_message=True
    )
    prompt = tokenizer.decode(
        input_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    return {"prompt": prompt, "input_ids": input_ids}


def format_reward_func(completion: str, EOS_TOKEN: str) -> float:
    """
    This function is used to reward the model for following the format of the prompt.
    It checks that the model has included a <think> tag and a <answer> tag.
    It also checks that the content within <answer>...</answer> conforms to a
    specified pattern (only digits, + - * / ( ) . and whitespace).

    Format: <think>...</think><answer>...</answer>

    Also checks that the content within <answer>...</answer> conforms to a
    specified pattern (only digits, + - * / ( ) . and whitespace).

    Args:
        completion (str): Generated output
        EOS_TOKEN (str): End of sequence token

    Returns:
        float: Reward score
    """
    # Define the allowed pattern (only numbers, +, -, *, /, (, ), ., and whitespace)
    allowed_pattern = r"^[\d+\-*/().\s]+$"

    try:
        # Synthetically prepend <think> (if your pipeline relies on that to ease matching)
        completion = "<think>" + completion

        # Strip EOS token if present
        if completion.endswith(EOS_TOKEN):
            completion = completion[: -len(EOS_TOKEN)]

        # Check if the format is correct
        # Pattern means:
        # 1) <think>...contents not including other <think> tags...</think>
        # 2) \n
        # 3) <answer>...anything...</answer>
        regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\n<answer>([\s\S]*?)<\/answer>$"
        match = re.search(regex, completion, re.DOTALL)

        if match is None or len(match.groups()) != 2:
            # Format is incorrect
            return 0.0
        else:
            # Extract the content inside <answer>...</answer>
            answer_content = match.group(2).strip()

            # Check if answer content matches the allowed pattern
            if not re.match(allowed_pattern, answer_content):
                # If it doesn't match, reward is 0.5
                return 0.5
            else:
                # If both format and pattern are correct, reward is 1
                return 1.0
    except Exception:
        # Any error leads to 0 reward
        return 0.0


def equation_reward_func(completion: str, nums: List[int], target: int) -> float:
    """
    Evaluates completion based on mathematical correctness of the answer

    Args:
        completion (str): Generated output
        target (str): Expected answer
        nums (list): Available numbers to use in the equation

    Returns:
        float: Reward score
    """
    try:
        # add synthetic <think> as its already part of the prompt and prefilled for the assistant to more easily match the regex
        completion = "<think>" + completion
        # Check if the format is correct
        match = re.search(r"<answer>(.*?)<\/answer>", completion)
        if match is None:
            return 0.0
        # Extract the "answer" part from the completion
        equation = match.group(1).strip()
        # Extract all numbers from the equation
        used_numbers = [int(n) for n in re.findall(r"\d+", equation)]

        # Check if all numbers are used exactly once
        if sorted(used_numbers) != sorted(nums):
            return 0.0
        # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
        allowed_pattern = r"^[\d+\-*/().\s]+$"
        if not re.match(allowed_pattern, equation):
            return 0.0

        # Evaluate the equation with restricted globals and locals
        result = eval(equation, {"__builtins__": None}, {})
        # Check if the equation is correct and matches the ground truth
        if abs(float(result) - float(target)) < 1e-5:
            return 1.0
        else:
            return 0.0
    except Exception:
        # If evaluation fails, reward is 0
        return 0.0


def compute_reward(
    completion: str, sample: Dict[str, Any], EOS_TOKEN: str
) -> Tuple[float, Dict[str, float]]:
    """
    Compute the reward for a given completion.

    Args:
        completion (str): The completion to evaluate
        sample (Dict[str, Any]): The sample to evaluate
        EOS_TOKEN (str): The end of sequence token

    Returns:
        Tuple[float, Dict[str, float]]: A tuple containing the reward (float) and metrics (dict of partial-rewards)
    """
    nums = sample["nums"]
    target = sample["target"]

    format_reward = format_reward_func(completion, EOS_TOKEN)
    equation_reward = equation_reward_func(
        completion=completion, nums=nums, target=target
    )

    reward = format_reward + equation_reward

    metrics = {
        "format_reward": format_reward,
        "equation_reward": equation_reward,
    }

    return reward, metrics


def create_training_episodes(
    samples: List[Dict[str, Any]],
    all_generations: List[List[int]],
    all_finish_reasons: List[str],
    tokenizer: AutoTokenizer,
    EOS_TOKEN_ID: int,
    EOS_TOKEN: str,
    GENERATIONS_PER_SAMPLE: int,
    policy_model=None,  # For old_logps calculation
    temperature=1.0,  # For old_logps calculation
    dynamic_sampling=False,  # For DAPO
    algo_config=None,  # Added algo_config parameter
    token_budget=1024,  # Added token budget parameter with default
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Process model generations and calculate rewards for training episodes.
    Implements DAPO dynamic sampling by discarding groups with uniform rewards,
    but falls back to minimal noise addition when discarding would result in an empty batch.

    Args:
        samples (List[Dict[str, Any]]): List of samples (i.e. prompts/questions) from the dataset
        all_generations (List[List[int]]): List of generations for each sample (i.e. responses)
            - List of token IDs for each generation 
            - (i.e. len(all_generations) == len(samples) * GENERATIONS_PER_SAMPLE)
            - (i.e. all_generations[0] is the list of token IDs for the first sample)
            - This list is flattened across samples, but we reform it into groups of GENERATIONS_PER_SAMPLE for each sample at the start of this function
        all_finish_reasons (List[str]]): List of finish reasons for each generation in all_generations 
        tokenizer (AutoTokenizer): The tokenizer to use
        EOS_TOKEN_ID (int): The end of sequence token ID
        EOS_TOKEN (str): The end of sequence token
        GENERATIONS_PER_SAMPLE (int): The number of generations per sample
        policy_model (Optional[Union[DeepSpeedEngine, PreTrainedModel]]): The policy model to use for old_logps calculation
        temperature (float): The temperature to use for the policy model (for old_logps calculation)
        dynamic_sampling (bool): Whether to use dynamic sampling (DAPO)
        algo_config (Optional[Dict[str, Any]]): The algorithm configuration
            - eps_low: Lower clipping bound (default: 0.2)
            - eps_high: Higher clipping bound (default: eps_low or 0.28 for DAPO)
            - norm_adv: Whether to normalize advantages by std (default: "std" for GRPO, "none" for Dr. GRPO/DAPO)
            - length_norm: Whether to use response-level length normalization (default: True for GRPO, False for Dr. GRPO/DAPO)
        token_budget (int): The token budget to use

    Returns:
        Tuple[Dict[str, Any], Dict[str, Any]]:
            - episodes (Dict[str, Any]): Dictionary with processed data for training
                - "all_query_token_ids" (List[int]): List of token IDs for all queries
                - "all_response_token_ids" (List[int]): List of token IDs for all responses
                - "all_advantages" (List[List[float]]): List of advantages for all responses
                - "adv_den" (List[int]): List of advantage denominators for all responses
                - "all_old_logps" (List[List[float]]): List of old log probabilities for all responses
                - "empty_batch" (bool): Whether the batch is empty
            - stats (Dict[str, Any]): Dictionary with generation statistics
                - "response_lengths" (List[int]): List of response lengths
                - "rewards" (List[float]): List of rewards
                - "non_stop_rate" (List[bool]): List of non-stop rates
                - "uniform_groups_found" (int): Number of uniform groups found
                - "uniform_groups_discarded" (int): Number of uniform groups discarded
                - "noise_fallback_used" (int): Number of times noise fallback was used
                - "empty_batch" (bool): Whether the batch is empty
    """
    assert len(all_generations) == len(all_finish_reasons)
    assert len(all_generations) == len(samples) * GENERATIONS_PER_SAMPLE

    # Process responses and calculate rewards


    # Indices to reform all_generations into groups of GENERATIONS_PER_SAMPLE for each sample
    groups = [
        list(range(i, i + GENERATIONS_PER_SAMPLE))
        for i in range(0, len(all_generations), GENERATIONS_PER_SAMPLE)
    ]  # example: [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

    (
        all_query_token_ids,
        all_responses_token_ids,
        all_advantages,
        all_old_logps,
        adv_den,
    ) = ([], [], [], [], [])

    stats = {
        "response_lengths": [],
        "rewards": [],
        "non_stop_rate": [],
        "uniform_groups_found": 0,  # Track groups with uniform rewards
        "uniform_groups_discarded": 0,  # Track groups that were discarded
        "noise_fallback_used": 0,  # Track when noise fallback was needed
        "empty_batch": False,  # Track if we have an empty batch
    }

    # Track when we've found at least one non-uniform group
    have_non_uniform_group = False

    for sample_idx, (sample, group_indices) in enumerate(zip(samples, groups)):
        # Each iteration processes a group of GENERATIONS_PER_SAMPLE responses for a single sample

        # Get the token IDs for the responses and finish reasons in this group
        # Both of these are lists of length GENERATIONS_PER_SAMPLE
        response_token_ids = [all_generations[i] for i in group_indices]
        finish_reasons = [all_finish_reasons[i] for i in group_indices]

        assert len(response_token_ids) == len(finish_reasons) == GENERATIONS_PER_SAMPLE

        # Decode the responses to text
        responses = tokenizer.batch_decode(
            response_token_ids, skip_special_tokens=False
        )

        # Compute the rewards and metrics for each response
        rewards_and_metrics = [
            compute_reward(resp, sample, EOS_TOKEN) for resp in responses
        ]
        rewards, reward_metrics = zip(*rewards_and_metrics)
        rewards = np.array(rewards, dtype=np.float32)

        # Check that the rewards and metrics are the correct shape
        assert rewards.shape == (GENERATIONS_PER_SAMPLE,)
        assert len(reward_metrics) == GENERATIONS_PER_SAMPLE

        # Dynamic sampling with empty batch prevention
        if dynamic_sampling:
            # Check if rewards are uniform (all identical)
            reward_range = rewards.max() - rewards.min()
            is_uniform = reward_range < 1e-6

            if is_uniform:
                # First, just track that we found a uniform group
                stats["uniform_groups_found"] += 1

                # Only discard if we have at least one non-uniform group already
                # This prevents empty batches while prioritizing discarding
                if have_non_uniform_group:
                    stats["uniform_groups_discarded"] += 1
                    print(
                        f"DAPO: Discarded uniform reward group {sample_idx} with reward value {rewards[0]}"
                    )
                    continue  # Skip this group
                else:
                    # If this would create an empty batch, use minimal noise as fallback
                    stats["noise_fallback_used"] += 1
                    # Deterministic noise based on sample index for reproducibility
                    rng = np.random.RandomState(42 + sample_idx)
                    noise = 1e-6 * rng.normal(size=rewards.shape)
                    rewards = rewards + noise
                    print(
                        f"DAPO: Empty batch prevention - adding minimal noise to group {sample_idx}"
                    )
            else:
                have_non_uniform_group = (
                    True  # Mark that we've found a non-uniform group
                )

        # Group-wise normalization
        advantages = rewards - rewards.mean()

        # For GRPO mode with std normalization, also divide by std
        if algo_config is not None and algo_config.get("norm_adv") == "std":
            advantages = advantages / (rewards.std() + 1e-8)

        # Convert to native Python types for stability
        advantages = advantages.tolist()

        # Compute old log probabilities if policy model is provided
        # old_logps_group is a list of length GENERATIONS_PER_SAMPLE
        # Each element is a list of per-token log probabilities for a single response in this group
        old_logps_group = []
        if policy_model is not None:
            for i, response in enumerate(response_token_ids):
                # each iteration here is a single response 
                # (i.e. a single generation out of GENERATIONS_PER_SAMPLE many)

                # Get the query token IDs for the sample
                query = sample["input_ids"]

                # Combine the query and response token IDs
                combined_ids = torch.tensor(
                    [query + response], device=policy_model.device
                )

                # Create an attention mask and labels tensor
                attention_mask = torch.ones_like(combined_ids)
                labels = torch.tensor(
                    [[-100] * len(query) + response], device=policy_model.device
                )

                # Create a model inputs dictionary
                model_inputs = {
                    "input_ids": combined_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }

                # Compute the old log probabilities for the response
                with torch.no_grad():
                    old_logp = compute_token_log_probs(
                        policy_model, model_inputs, temperature
                    )
                    response_len = len(response)

                    # old_logp is a tensor of shape (1, seq_len)
                    # We want to get the log probabilities for the response tokens
                    # (i.e. the tokens after the query tokens)
                    old_logps_group.append(
                        old_logp[0, -response_len:].detach().cpu().tolist()
                    )

        # per_token_advantages is a list of length GENERATIONS_PER_SAMPLE
        # Each element is a list of length len(resp)
        # Each element is the advantage for the response
        # (i.e. the advantage is the same for each token in the response)
        per_token_advantages = [
            [adv] * len(resp) for adv, resp in zip(advantages, response_token_ids)
        ]

        # Extend the lists with the new episode data
        # (Note: this is a flattened list of all the data for all the responses over all groups, hence the .extend())
        all_query_token_ids.extend([sample["input_ids"]] * len(response_token_ids))
        all_responses_token_ids.extend(response_token_ids)
        all_advantages.extend(per_token_advantages)
        if policy_model is not None:
            all_old_logps.extend(old_logps_group)

        # Use configured token budget for Dr. GRPO/DAPO instead of hardcoded value
        if algo_config is not None and not algo_config.get("length_norm", True):
            # For Dr. GRPO and DAPO, use token_budget parameter
            adv_den.extend([token_budget] * len(response_token_ids))
        else:
            # For GRPO, use response length (though this isn't used in GRPO mode)
            adv_den.extend([len(resp) for resp in response_token_ids])

        # Convert stats to Python native types for JSON serialization
        stats["rewards"].extend([float(r) for r in rewards])
        stats["non_stop_rate"].extend([bool(fr != "stop") for fr in finish_reasons])
        stats["response_lengths"].extend([int(len(ids)) for ids in response_token_ids])
        for rm in reward_metrics:
            for k, v in rm.items():
                # Ensure each metric is a native Python type
                if isinstance(v, (np.integer, np.floating)):
                    v = float(v)
                stats.setdefault(f"reward_metrics/{k}", []).append(v)

    episodes = {
        "all_query_token_ids": all_query_token_ids,
        "all_response_token_ids": all_responses_token_ids,
        "all_advantages": all_advantages,
        "adv_den": adv_den,
    }

    if policy_model is not None and all_old_logps:
        episodes["all_old_logps"] = all_old_logps

    # Check if we have any valid groups after discarding
    if not all_query_token_ids:
        print("[WARNING] All groups had uniform rewards and were discarded!")
        print(
            "This batch will be skipped. This may indicate a problem with reward design."
        )
        stats["all_uniform"] = True
        stats["empty_batch"] = True

        # Set a flag to indicate empty batch but provide minimal data for error handling
        episodes = {
            "all_query_token_ids": [],
            "all_response_token_ids": [],
            "all_advantages": [],
            "adv_den": [],
            "empty_batch": True,
        }
    else:
        # Mark as a valid batch with data
        episodes["empty_batch"] = False

    # Sanity check: with our fallback, we should never have empty batches
    # But just in case, add a flag that can be checked in the main function
    if not all_query_token_ids:
        stats["empty_batch"] = True
        print(
            "[WARNING] Something went wrong - empty batch despite fallback mechanism!"
        )

    return episodes, stats


def compute_pg_loss(
    policy_model: Union[DeepSpeedEngine, PreTrainedModel],
    reference_model: Union[DeepSpeedEngine, PreTrainedModel],
    batch: Dict[str, torch.Tensor],
    total_response_len: int,
    TEMPERATURE: float,
    KL_COEFFICIENT: float,
    algo_config: Dict[str, Any] = None,  # Added algorithm configuration
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

    # Advantage normalization is controlled by algo_config and already done in create_training_episodes
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


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train R1 model with PPO")
    parser.add_argument(
        "--algo",
        type=str,
        choices=["grpo", "dr_grpo", "dapo", "optimal"],
        default="dapo",
        help="Algorithm variant: grpo (original), dr_grpo (unbiased), dapo (state-of-the-art), or optimal (best combined configuration)",
    )
    parser.add_argument(
        "--optimal",
        action="store_true",
        help="Use optimal configuration (equivalent to --algo=optimal)",
    )
    parser.add_argument(
        "--eps_low", type=float, default=0.2, help="Lower clipping epsilon"
    )
    parser.add_argument(
        "--eps_high",
        type=float,
        default=None,
        help="Higher clipping epsilon (for DAPO)",
    )
    parser.add_argument(
        "--norm_adv",
        type=str,
        choices=["std", "none"],
        default=None,
        help="Advantage normalization: std (GRPO), none (Dr. GRPO/DAPO)",
    )
    parser.add_argument(
        "--length_norm",
        action="store_true",
        default=None,
        help="Use response length normalization (GRPO)",
    )
    parser.add_argument(
        "--dyn_sample",
        action="store_true",
        default=None,
        help="Use dynamic sampling (DAPO)",
    )
    parser.add_argument(
        "--kl_coeff", type=float, default=0.001, help="KL coefficient for PPO"
    )
    parser.add_argument(
        "--group_size",
        type=int,
        default=None,
        help="Group size for sampling (number of responses per prompt)",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1024,
        help="Maximum tokens to generate in each response",
    )
    parser.add_argument(
        "--token_budget",
        type=int,
        default=1024,
        help="Token budget for normalization in Dr. GRPO/DAPO (mathematical constant)",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Temperature for sampling"
    )
    parser.add_argument(
        "--model_name", type=str, default="Qwen/Qwen2.5-3B", help="Model name/path"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-6, help="Learning rate for training"
    )
    args = parser.parse_args()

    # Apply --optimal flag if set (overrides --algo)
    if args.optimal:
        args.algo = "optimal"

    # Set default algorithm configurations based on chosen algorithm
    if args.algo == "optimal":
        # Optimal configuration combines the best elements of all methods
        args.norm_adv = "none"  # Dr. GRPO (removes std normalization bias)
        args.length_norm = False  # Dr. GRPO (removes length normalization bias)
        args.dyn_sample = False  # DAPO (ensures non-zero advantages)
        args.eps_low = 0.2  # Standard lower clip bound
        args.eps_high = 0.28  # DAPO "Clip-Higher" (prevents entropy collapse)
        args.kl_coeff = 0.0  # Remove KL divergence term for pure rule-based rewards
        if args.token_budget is None:
            args.token_budget = 1024  # Default token budget for normalization
        if args.group_size is None:
            args.group_size = 16  # Larger group for better statistics
    else:
        # Default configurations for other algorithms
        if args.norm_adv is None:
            args.norm_adv = "std" if args.algo == "grpo" else "none"
        if args.length_norm is None:
            args.length_norm = args.algo == "grpo"
        if args.dyn_sample is None:
            args.dyn_sample = args.algo == "dapo"
        if args.eps_high is None:
            args.eps_high = 0.28 if args.algo == "dapo" else args.eps_low
        if args.token_budget is None:
            args.token_budget = 1024  # Default token budget for all algorithms

    # Create algorithm configuration
    algo_config = {
        "eps_low": args.eps_low,
        "eps_high": args.eps_high,
        "norm_adv": args.norm_adv,
        "length_norm": args.length_norm,
        "token_budget": args.token_budget,
    }

    # Needed to stop DeepSpeed from complaining
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(find_free_port())
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"

    ############################################
    # Hyperparameters
    ############################################

    # Model configuration
    MODEL_NAME = args.model_name
    MODEL_CHAT_NAME = MODEL_NAME + "-Instruct"

    # RL parameters
    # Total number of training iterations
    NUM_ITERATIONS = 200  # 1000
    # Number of episodes to collect per iteration for training
    EPISODES_PER_ITERATION = 64
    # Number of responses to generate for each input prompt
    GENERATIONS_PER_SAMPLE = args.group_size if args.group_size is not None else 4
    # Controls how much the policy can deviate from the reference model
    KL_COEFFICIENT = args.kl_coeff

    # Training hyperparameters
    # Batch size for each GPU device during training
    PER_DEVICE_BATCH_SIZE = 4
    # Learning rate for model updates
    LEARNING_RATE = 1e-6

    # Sampling parameters
    # Maximum number of tokens to generate in each response
    MAX_RESPONSE_TOKENS = args.max_tokens
    # Controls randomness in generation (higher = more random)
    TEMPERATURE = args.temperature
    # Nucleus sampling parameter (1.0 = disabled)
    TOP_P = 1.0
    # Top-k sampling parameter (-1 = disabled)
    TOP_K = -1  # no top k

    # DeepSpeed configuration
    deepspeed_config = {
        "bf16": {"enabled": True},
        "zero_optimization": {"stage": 2, "overlap_comm": False},
        "train_batch_size": EPISODES_PER_ITERATION,
        "train_micro_batch_size_per_gpu": PER_DEVICE_BATCH_SIZE,
        "gradient_accumulation_steps": EPISODES_PER_ITERATION // PER_DEVICE_BATCH_SIZE,
        "gradient_clipping": 1.0,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.learning_rate,
                "betas": (0.9, 0.999),
                "eps": 1e-8,
                "weight_decay": 0.0,
                "torch_adam": True,
            },
        },
    }
    ref_deepspeed_config = {
        "bf16": {"enabled": True},
        "train_batch_size": EPISODES_PER_ITERATION,
        "train_micro_batch_size_per_gpu": PER_DEVICE_BATCH_SIZE,
        "gradient_accumulation_steps": EPISODES_PER_ITERATION // PER_DEVICE_BATCH_SIZE,
    }

    model_name_short = MODEL_NAME.split("/")[-1]

    # Get algorithm name directly from the argument
    algo_map = {
        "grpo": "GRPO",
        "dr_grpo": "Dr.GRPO",
        "dapo": "DAPO",
        "optimal": "Optimal",
    }
    algo_name = algo_map[args.algo]

    # Format run name with algorithm variant
    RUN_NAME = f"{model_name_short}_{args.algo}_el{args.eps_low}_eh{args.eps_high}_t{TEMPERATURE}_kl{KL_COEFFICIENT}_lr{LEARNING_RATE}"
    if args.algo == "optimal":
        RUN_NAME = f"{model_name_short}_optimal_g{GENERATIONS_PER_SAMPLE}_t{TEMPERATURE}_lr{LEARNING_RATE}"

    EXP_DIR = SCRATCH / "runs" / RUN_NAME
    EXP_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Using algorithm: {algo_name} with configuration:")
    print(f"  - eps_low: {algo_config['eps_low']}")
    print(f"  - eps_high: {algo_config['eps_high']}")
    print(f"  - norm_adv: {algo_config['norm_adv']}")
    print(f"  - length_norm: {algo_config['length_norm']}")
    print(f"  - dynamic sampling: {args.dyn_sample}")
    print(f"Logs and Checkpoints will be saved to: {EXP_DIR}")

    ############################################
    # Prompts and Dataset
    ############################################

    SYSTEM_MESSAGE = (
        "You are a helpful assistant. You first think about the reasoning process in the mind "
        "and then provide the user with the answer."
    )
    PROMPT_TEMPLATE = (
        "Using the numbers {numbers}, create an equation that equals {target}. "
        "You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. "
        "Show your work in <think> </think> tags. And return the final equation and answer in "
        "<answer> </answer> tags, for example <answer>(1 + 2) / (3 * 5)</answer>."
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHAT_NAME)
    EOS_TOKEN_ID = AutoTokenizer.from_pretrained(MODEL_NAME).eos_token_id
    EOS_TOKEN = tokenizer.convert_ids_to_tokens(EOS_TOKEN_ID)

    dataset = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split="train")
    dataset = dataset.map(
        preprocess_example,
        num_proc=6,
        fn_kwargs={
            "tokenizer": tokenizer,
            "SYSTEM_MESSAGE": SYSTEM_MESSAGE,
            "PROMPT_TEMPLATE": PROMPT_TEMPLATE,
        },
    )

    # Split dataset
    train_test_split = dataset.train_test_split(test_size=500, seed=42)
    train_dataset = train_test_split["train"]
    test_dataset = train_test_split["test"]

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    ############################################
    # Initialize Models
    ############################################

    policy_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        device_map=0,
    )
    reference_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        device_map=0,
    )
    policy_model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    # Initialize DeepSpeed engines
    policy_model, *_ = deepspeed.initialize(
        model=policy_model,
        config=deepspeed_config,
        model_parameters=policy_model.parameters(),
    )
    reference_model, *_ = deepspeed.initialize(
        model=reference_model,
        config=ref_deepspeed_config,
    )

    reference_model.module.cpu()

    ############################################
    # Initialize vLLM (Inference) engine
    ############################################

    inference_engine = LLM(
        model=MODEL_NAME,
        skip_tokenizer_init=False,
        gpu_memory_utilization=0.3,
        enable_prefix_caching=True,
        swap_space=1,
        scheduling_policy="fcfs",
        dtype=torch.bfloat16,
        max_model_len=2048,
        enable_sleep_mode=True,
    )

    # Wandb for logging
    wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        name=RUN_NAME,
        config={
            "model_name": MODEL_NAME,
            "learning_rate": LEARNING_RATE,
            "num_iterations": NUM_ITERATIONS,
            "episodes_per_iteration": EPISODES_PER_ITERATION,
            "group_size": GENERATIONS_PER_SAMPLE,
            "kl_coefficient": KL_COEFFICIENT,
            "temperature": TEMPERATURE,
            "algorithm": args.algo,
            "algorithm_name": algo_name,
            "eps_low": algo_config["eps_low"],
            "eps_high": algo_config["eps_high"],
            "norm_adv": algo_config["norm_adv"],
            "length_norm": algo_config["length_norm"],
            "token_budget": algo_config["token_budget"],
            "max_tokens": args.max_tokens,
            "dynamic_sampling": args.dyn_sample,
        },
    )

    # Load checkpoint if it exists
    begin_iter = 0
    ckpt_path, ckpt_iter = find_last_checkpoint(EXP_DIR)
    if ckpt_path is not None:
        print(f"Resuming from checkpoint {ckpt_path} at iteration {ckpt_iter}")
        out = policy_model.load_checkpoint(ckpt_path / "deepspeed")
        if out is None:
            raise RuntimeError(f"Failed to load checkpoint {ckpt_path}")
        begin_iter = ckpt_iter + 1
        load_model_into_vllm(policy_model, inference_engine)

    for iteration in trange(begin_iter, NUM_ITERATIONS):
        print(f"Iteration {iteration}/{NUM_ITERATIONS}")

        metrics = {}

        #########################################################
        # Evaluation
        #########################################################

        eval_stats = None
        if iteration % 25 == 0:
            print("Evaluating on eval set...")
            eval_episodes, eval_stats = evaluate_on_test_set(
                inference_engine=inference_engine,
                test_dataset=test_dataset,
                tokenizer=tokenizer,
                eos_token=EOS_TOKEN,
                eval_sampling_params=SamplingParams(
                    temperature=0.3,
                    max_tokens=args.max_tokens,  # Use the parameter from args
                    n=1,
                    detokenize=False,
                    stop_token_ids=[EOS_TOKEN_ID],
                ),
                reward_func=lambda completion, sample: compute_reward(
                    completion, sample, EOS_TOKEN
                ),
            )
            eval_episode_table = dump_episodes(
                episodes=eval_episodes,
                episodes_stats=eval_stats,
                exp_dir=EXP_DIR,
                tokenizer=tokenizer,
                iteration=iteration,
                is_eval=True,
            )
            wandb.log({"eval/episodes": eval_episode_table, "iteration": iteration})

        #########################################################
        # Generate Episodes
        #########################################################

        # Sample training batch
        num_samples = EPISODES_PER_ITERATION // GENERATIONS_PER_SAMPLE
        indices = np.random.choice(len(train_dataset), size=num_samples, replace=False)
        samples = train_dataset.select(indices)

        gen_time = time.time()

        # Sample responses
        # Convert token IDs to string prompts that vLLM can process
        prompts = [
            tokenizer.decode(ids, skip_special_tokens=False)
            for ids in samples["input_ids"]
        ]

        outputs = inference_engine.generate(
            prompts=prompts,  # Use decoded text prompts instead of token IDs
            sampling_params=SamplingParams(
                n=GENERATIONS_PER_SAMPLE,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                top_k=TOP_K,
                max_tokens=MAX_RESPONSE_TOKENS,
                detokenize=False,
                stop_token_ids=[EOS_TOKEN_ID],
            ),
        )
        all_generations = [list(g.token_ids) for out in outputs for g in out.outputs]
        all_finish_reasons = [g.finish_reason for out in outputs for g in out.outputs]
        inference_engine.sleep(1)

        print(f"Generated {len(all_generations)} responses")
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(1)

        print(
            f"Time taken to generate {len(all_generations)} responses: {time.time() - gen_time} seconds"
        )

        # Process responses and calculate rewards
        episodes, episodes_stats = create_training_episodes(
            samples,
            all_generations,
            all_finish_reasons,
            tokenizer,
            EOS_TOKEN_ID,
            EOS_TOKEN,
            GENERATIONS_PER_SAMPLE,
            policy_model=policy_model,  # Always pass policy_model for old_logps calculation
            temperature=TEMPERATURE,
            dynamic_sampling=args.dyn_sample,
            algo_config=algo_config,  # Pass the algorithm configuration
            token_budget=args.token_budget,  # Pass the token budget parameter
        )

        # Safety check for empty batches (shouldn't happen with our fallback)
        if episodes_stats.get("empty_batch", False):
            print(
                "[ERROR] Empty batch encountered despite fallback! Skipping iteration."
            )
            continue  # Skip to the next iteration of the main loop

        for k, v in episodes_stats.items():
            # Only include metric keys in metrics dictionary
            if isinstance(v, list):
                metrics.setdefault(k, []).extend(v)
            elif k not in ["empty_batch"]:  # Skip internal flags
                metrics[k] = v

        # Critical check: If we have no valid samples after dynamic sampling
        # This double-check is redundant with the above but kept for safety
        if episodes_stats.get("all_uniform", False):
            print(
                "[ERROR] Empty batch: Dynamic sampling discarded ALL groups (all had uniform rewards)"
            )
            print("Skipping this batch and continuing to the next iteration.")
            continue  # Skip to the next iteration of the main loop

        episode_table = dump_episodes(
            episodes=episodes,
            episodes_stats=episodes_stats,
            exp_dir=EXP_DIR,
            tokenizer=tokenizer,
            iteration=iteration,
        )

        #########################################################
        # Training
        #########################################################

        # Prepare training batch
        model_inputs = prepare_model_inputs(
            query_token_ids=episodes["all_query_token_ids"],
            response_token_ids=episodes["all_response_token_ids"],
            advantages=episodes["all_advantages"],
            device="cuda",
            old_logps=episodes.get("all_old_logps", None),
            adv_den=episodes.get("adv_den", None),
        )

        # Calculate losses and update model
        policy_model.train()
        reference_model.module.cuda()
        reference_model.eval()

        total_response_len = (model_inputs["labels"] != -100).sum().item()

        train_time = time.time()

        for i in trange(
            0,
            EPISODES_PER_ITERATION,
            PER_DEVICE_BATCH_SIZE,
            desc="Gradient Accumulation",
        ):
            batch = {
                k: v[i : i + PER_DEVICE_BATCH_SIZE] for k, v in model_inputs.items()
            }

            # Compute policy gradient loss
            loss, loss_metrics = compute_pg_loss(
                policy_model=policy_model,
                reference_model=reference_model,
                batch=batch,
                total_response_len=total_response_len,
                TEMPERATURE=TEMPERATURE,
                KL_COEFFICIENT=KL_COEFFICIENT,
                algo_config=algo_config,
            )

            # Track metrics
            metrics.setdefault("loss", []).append(loss.item())
            grad_norm = policy_model.get_global_grad_norm()
            if grad_norm is not None:
                grad_norm = grad_norm.item()
            metrics.setdefault("grad_norm", []).append(grad_norm)
            for k, v in loss_metrics.items():
                metrics.setdefault(k, []).append(
                    v.item() if isinstance(v, torch.Tensor) else v
                )

            # Backpropagation and optimization step
            policy_model.backward(loss, scale_wrt_gas=False)

            # Free memory
            del loss, loss_metrics
            if policy_model.is_gradient_accumulation_boundary():
                reference_model.module.cpu()

            policy_model.step()

        print(f"Time taken to train: {time.time() - train_time} seconds")

        #########################################################
        # Update inference engine weights
        #########################################################

        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(1)

        inference_engine.wake_up()
        load_model_into_vllm(policy_model, inference_engine)

        #########################################################
        # Log metrics
        #########################################################

        # FIX: Handle both list and scalar values in metrics
        train_metrics = {}
        for k, v in metrics.items():
            # All keys are valid in our new implementation
            if isinstance(v, list):
                if None not in v:  # Only include if no None values
                    train_metrics[k] = np.mean(v)
            else:  # Handle scalar values directly
                train_metrics[k] = v

        train_metrics["learning_rate"] = policy_model.get_lr()[0]
        logs = {
            "iteration": iteration,
            f"episodes/iter_{iteration:06d}": episode_table,
            **{f"train/{k}": v for k, v in train_metrics.items()},
        }
        if eval_stats is not None:
            logs.update({f"eval/{k}": np.mean(v) for k, v in eval_stats.items()})
        wandb.log(logs)

        selected_keys = [
            "train/kl_penalty",
            "train/rewards",
            "train/reward_metrics/format_reward",
            "train/reward_metrics/equation_reward",
            "eval/rewards",
            "eval/reward_metrics/format_reward",
            "eval/reward_metrics/equation_reward",
        ]
        selected_metrics = {k: logs[k] for k in selected_keys if k in logs}
        print(f"KEY METRICS: {selected_metrics}")

        # NOTE: temporary
        if iteration % 2000 == 0 and iteration != 0:
            policy_model.module.save_pretrained(
                str(EXP_DIR / "checkpoints" / f"ckpt_{iteration:06d}" / "hf_model")
            )
            policy_model.save_checkpoint(
                str(EXP_DIR / "checkpoints" / f"ckpt_{iteration:06d}" / "deepspeed")
            )


if __name__ == "__main__":
    main()
    