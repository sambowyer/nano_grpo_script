import os
from pathlib import Path

import argparse
import gc
import re
import time
from typing import Any, Dict, List, Tuple, Union, Callable

import deepspeed
import numpy as np
import torch
from datasets import load_dataset, Dataset
from deepspeed import DeepSpeedEngine
from tqdm import trange
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from vllm import LLM, SamplingParams

from rewards import compute_reward
from episodes import process_training_episodes, dump_episodes
from loss import compute_pg_loss
from utils import (
    preprocess_example,
    find_free_port,
    find_last_checkpoint,
    prepare_model_inputs,
    load_model_into_vllm,
)

import wandb

## CHANGE THESE TO YOUR OWN WANDB ENTITY AND PROJECT
WANDB_ENTITY = "sam-bowyer-bristol"
WANDB_PROJECT = "nano-grpo"

## CHANGE THESE PATHS TO YOUR OWN
SCRATCH = Path("/user/work/dg22309/grpo/nano_script")
os.environ["HF_HOME"] = str("/user/work/dg22309/huggingface")


def evaluate_on_test_set(
    inference_engine: LLM,
    test_dataset: Dataset,
    tokenizer: AutoTokenizer,
    eos_token: str,
    eval_sampling_params: SamplingParams,
    reward_func: Callable[[str, Dict[str, Any]], Tuple[float, Dict[str, float]]],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Evaluate the model on a test dataset by generating responses and computing rewards.

    Args:
        inference_engine: The vLLM Engine instance used for text generation
        test_dataset: Dataset containing test samples
        tokenizer: Tokenizer for decoding generated token IDs
        eos_token: End of sequence token string
        eval_sampling_params: Dictionary of parameters for controlling the generation process
        reward_func: Function that computes rewards for generated responses. Takes a response
            string and sample dict as input, returns a tuple of (overall_reward, reward_components)

    Returns:
        Dictionary containing evaluation statistics and episodes.
    """
    # Convert token IDs to string prompts that vLLM can process
    prompts = [tokenizer.decode(ids, skip_special_tokens=False) for ids in test_dataset["input_ids"]]
    
    # Use text prompts instead of token IDs
    generations = inference_engine.generate(
        prompts=prompts, sampling_params=eval_sampling_params
    )

    metrics = {
        "response_lengths": [],
        "rewards": [],
        "non_stop_rate": [],
    }

    all_query_token_ids = []
    all_responses_token_ids = []

    for i, sample in enumerate(test_dataset):
        query_token_ids = sample["input_ids"]
        response_token_ids = generations[i].outputs[0].token_ids
        finish_reason = generations[i].outputs[0].finish_reason

        response = tokenizer.decode(response_token_ids, skip_special_tokens=False)
        reward, reward_components = reward_func(response, sample)

        all_query_token_ids.append(query_token_ids)
        all_responses_token_ids.append(response_token_ids)

        # Ensure reward is a native Python float, not numpy float
        if isinstance(reward, (np.integer, np.floating)):
            reward = float(reward)

        metrics["rewards"].append(reward)
        metrics["non_stop_rate"].append(finish_reason != "stop")
        metrics["response_lengths"].append(len(response_token_ids))
        for k, v in reward_components.items():
            # Convert any numpy values to Python native types
            if isinstance(v, (np.integer, np.floating)):
                v = float(v)
            metrics.setdefault(f"reward_metrics/{k}", []).append(v)

    episodes = {
        "all_query_token_ids": all_query_token_ids,
        "all_response_token_ids": all_responses_token_ids,
    }

    return episodes, metrics


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
                    max_tokens=MAX_RESPONSE_TOKENS,
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
        episodes, episodes_stats = process_training_episodes(
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
    