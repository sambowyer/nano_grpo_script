from typing import Any, Dict, List, Tuple
from pathlib import Path
import json

import numpy as np
import torch
import wandb
from transformers import AutoTokenizer, PreTrainedModel

from rewards import compute_reward
from utils import compute_token_log_probs

def process_training_episodes(
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
    algo_config=None,  
    token_budget=1024, 
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
                - "all_response_token_ids" (List[List[int]]): List of token IDs for all responses
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


def dump_episodes(
    episodes: Dict[str, Any],
    episodes_stats: Dict[str, Any],
    exp_dir: Path,
    tokenizer: AutoTokenizer,
    iteration: int,
    is_eval: bool = False,
) -> wandb.Table:
    """
    Dump episodes to a JSON file and create a wandb table.

    Args:
        episodes: Dictionary containing all query and response token IDs
        episodes_stats: Dictionary containing rewards and response lengths
        exp_dir: Path to the experiment directory
        tokenizer: Tokenizer for decoding token IDs
        iteration: Current training iteration
        is_eval: Whether the episodes are for evaluation

    Returns:
        wandb.Table: Table containing query, response, reward, and response length
    """

    query_token_ids = episodes["all_query_token_ids"]
    response_token_ids = episodes["all_response_token_ids"]
    rewards = episodes_stats["rewards"]
    response_lengths = episodes_stats["response_lengths"]

    query_texts = tokenizer.batch_decode(
        query_token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    response_texts = tokenizer.batch_decode(
        response_token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )

    if not is_eval:
        print(f"########## Example 1 (Reward: {rewards[0]}, Response Length: {response_lengths[0]})")
        print(f"#### Query:\n`{query_texts[0]}`")
        print(f"#### Response:\n`{response_texts[0]}`\n\n")

        print(f"########## Example 2 (Reward: {rewards[1]}, Response Length: {response_lengths[1]})")
        print(f"#### Query:\n`{query_texts[1]}`")
        print(f"#### Response:\n`{response_texts[1]}`\n\n")

    if is_eval:
        episodes_dir = exp_dir / "eval_episodes"
    else:
        episodes_dir = exp_dir / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)

    # Convert NumPy types to native Python types to make them JSON serializable
    def numpy_to_python(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, list):
            return [numpy_to_python(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: numpy_to_python(v) for k, v in obj.items()}
        else:
            return obj

    # Convert rewards to Python floats for JSON serialization
    python_rewards = [numpy_to_python(r) for r in rewards]
    
    with open(episodes_dir / f"eps_{iteration:06d}.json", "w") as f:
        json.dump(
            [
                {
                    "query": query_texts[i],
                    "response": response_texts[i],
                    "reward": python_rewards[i],
                }
                for i in range(len(query_texts))
            ],
            f,
        )

    # Create wandb table
    table = wandb.Table(columns=["query", "response", "reward", "response_length"])
    for i in range(len(query_texts)):
        table.add_data(query_texts[i], response_texts[i], python_rewards[i], response_lengths[i])

    return table

