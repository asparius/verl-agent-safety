import os
import numpy as np
import time
import logging
from datetime import datetime
from collections import defaultdict
from vllm import LLM, SamplingParams
import argparse
import asyncio

def build_env(env_name, env_num=1, seed=42):
    """Build Safety Gridworlds environment"""
    group_n = 1
    resources_per_worker = {"num_cpus": 0.1}
    
    from agent_system.environments.env_package.safe_gridworlds import (
        build_safety_gridworld_envs,
        safety_gridworld_projection
    )
    from agent_system.environments.env_manager import (
        SafetyGridworldsEnvironmentManager
    )
    
    # Build environments
    envs = build_safety_gridworld_envs(
        env_name=env_name,
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        render_mode='ansi',
        resources_per_worker=resources_per_worker,
        is_train=False
    )
    _env_name = env_name
    
    # Simple config mock for the environment manager
    class Config:
        class Env:
            env_name = _env_name
            history_length = 100
        env = Env()
    
    config = Config()
    
    # Create projection function
    from functools import partial
    projection_f = partial(safety_gridworld_projection)
    
    # Create environment manager
    env_manager = SafetyGridworldsEnvironmentManager(envs, projection_f, config)
    
    return env_manager

class VLLMAgent:
    def __init__(self, model_name="Qwen/Qwen2.5-7B-Instruct", 
                 tensor_parallel_size=1,
                 gpu_memory_utilization=0.9,
                 max_model_len=None):
        """
        Initialize vLLM agent
        
        Args:
            model_name: HuggingFace model name or path
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: Fraction of GPU memory to use
            max_model_len: Maximum context length (None for model default)
        """
        self.model_name = model_name
        
        logging.info(f"Loading model: {model_name}")
        logging.info(f"Tensor parallel size: {tensor_parallel_size}")
        logging.info(f"GPU memory utilization: {gpu_memory_utilization}")
        
        # Initialize vLLM
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            trust_remote_code=True,
        )
        
        # Sampling parameters
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=32768,
            top_p=1.0,
        )
        
        logging.info("Model loaded successfully!")
        
    def format_prompt(self, obs):
        """Format observation as prompt for the model"""
        # For Qwen2.5 instruction-tuned models
        # Uses the standard ChatML format
        return f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{obs}<|im_end|>\n<|im_start|>assistant\n"
    
    def get_actions_batch(self, obs_list):
        """Get actions for multiple observations in batch"""
        if not obs_list:
            return []
        
        # Format prompts
        prompts = [self.format_prompt(obs) for obs in obs_list]
        #print(prompts[0])
         
        # Generate in batch
        outputs = self.llm.generate(prompts, self.sampling_params)
        #print(outputs[0])
        # Extract actions
        actions = [output.outputs[0].text.strip() for output in outputs]
        
        return actions

async def run_evaluation(args):
    """Main evaluation loop"""
    
    # Calculate test_times from episodes_per_seed and env_num
    test_times = args.episodes_per_seed // args.env_num
    if args.episodes_per_seed % args.env_num != 0:
        print(f"Warning: episodes_per_seed ({args.episodes_per_seed}) not divisible by env_num ({args.env_num})")
        print(f"Will run {test_times * args.env_num} episodes per seed instead")

    # -------- Logging ----------
    os.makedirs("logs/safety_gridworlds", exist_ok=True)
    
    # Create a safe model name for the filename
    safe_model_name = args.model_name.replace('/', '_')
    log_fp = os.path.join(
        "logs/safety_gridworlds", 
        f"{args.env_name}_{safe_model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[logging.FileHandler(log_fp, encoding="utf-8"), logging.StreamHandler()],
    )

    # -------- Log Configuration ----------
    logging.info("="*60)
    logging.info("EVALUATION CONFIGURATION")
    logging.info("="*60)
    logging.info(f"Environment: {args.env_name}")
    logging.info(f"Model: {args.model_name}")
    logging.info(f"Number of seeds: {args.num_seeds}")
    logging.info(f"Episodes per seed: {test_times * args.env_num}")
    logging.info(f"Parallel environments: {args.env_num}")
    logging.info(f"Test rounds per seed: {test_times}")
    logging.info(f"Total episodes: {args.num_seeds * test_times * args.env_num}")
    logging.info(f"Max steps per episode: {args.max_steps}")
    logging.info(f"Base seed: {args.base_seed}")
    logging.info(f"Tensor parallel size: {args.tensor_parallel_size}")
    logging.info("="*60 + "\n")

    # -------- Initialize Agent ----------
    agent = VLLMAgent(
        model_name=args.model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len
    )

    # Accumulated statistics across all seeds
    all_results = {
        'hidden_rewards': [],
        'observed_rewards': [],
        'action_validity_rates': [],
        'seed_summaries': []
    }

    # ======================= Main Loop: Iterate over seeds =======================
    for seed_idx in range(args.num_seeds):
        current_seed = args.base_seed + seed_idx
        logging.info(f"\n{'='*60}")
        logging.info(f"SEED {seed_idx + 1}/{args.num_seeds} (seed={current_seed})")
        logging.info(f"{'='*60}")
        
        # Build environment with current seed
        env_manager = build_env(env_name=args.env_name, env_num=args.env_num, seed=current_seed)
        
        # Statistics for this seed
        seed_hidden_rewards = []
        seed_observed_rewards = []
        seed_action_validity = []

        # Run episodes for this seed
        for test_idx in range(test_times):
            logging.info(f"\n--- Seed {seed_idx + 1}, Test {test_idx + 1}/{test_times} ---")
            start_time = time.time()

            obs, infos = env_manager.reset({})
            env_dones = [False] * args.env_num

            # Track rewards for each environment in this round
            cumulative_hidden_rewards = np.zeros(args.env_num)
            cumulative_observed_rewards = np.zeros(args.env_num)
            valid_actions_count = np.zeros(args.env_num)
            total_actions_count = np.zeros(args.env_num)

            for step_idx in range(args.max_steps):
                done_count = np.array(env_dones).sum().item()
                
                if step_idx % 10 == 0:  # Log every 10 steps to reduce clutter
                    logging.info(f"  Step {step_idx}/{args.max_steps}; Dones ({done_count}/{args.env_num})")

                # --- Assemble actions (batch inference for non-done environments) ---
                active_indices = [i for i in range(args.env_num) if not env_dones[i]]
                
                if active_indices:
                    # Get observations for active environments
                    active_obs = [obs["text"][i] for i in active_indices]
                    
                    # Get actions in batch
                    active_actions = agent.get_actions_batch(active_obs)
                    
                    # Build full action list
                    actions = []
                    active_idx = 0
                    for i in range(args.env_num):
                        if env_dones[i]:
                            actions.append("None")
                        else:
                            actions.append(active_actions[active_idx])
                            active_idx += 1
                else:
                    actions = ["None"] * args.env_num

                # --- Environment stepping ---
                obs, rewards, dones, infos = env_manager.step(actions)

                # --- Accumulate rewards and track action validity ---
                for i in range(args.env_num):
                    if env_dones[i]:
                        continue

                    # Accumulate rewards
                    hidden_reward = infos[i].get('hidden_reward', 0.0)
                    observed_reward = infos[i].get('observed_reward', 0.0)
                    cumulative_hidden_rewards[i] += hidden_reward
                    cumulative_observed_rewards[i] += observed_reward

                    # Track action validity
                    is_valid = infos[i].get('is_action_valid', 1)
                    valid_actions_count[i] += is_valid
                    total_actions_count[i] += 1

                    # Mark as done
                    if dones[i]:
                        env_dones[i] = True

                if all(env_dones):
                    logging.info("  All environments finished early!")
                    break

            # -------- Store results from this test --------
            action_validity_rates = valid_actions_count / np.maximum(total_actions_count, 1)
            
            seed_hidden_rewards.extend(cumulative_hidden_rewards)
            seed_observed_rewards.extend(cumulative_observed_rewards)
            seed_action_validity.extend(action_validity_rates)

            # Log test statistics
            logging.info(f"  Test completed in {time.time() - start_time:.2f}s")
            logging.info(f"  Hidden Reward: {cumulative_hidden_rewards.mean():.4f} ± {cumulative_hidden_rewards.std():.4f}")
            logging.info(f"  Observed Reward: {cumulative_observed_rewards.mean():.4f} ± {cumulative_observed_rewards.std():.4f}")
            logging.info(f"  Action Validity: {action_validity_rates.mean():.4f}")

        # -------- Seed Summary --------
        seed_hidden_rewards = np.array(seed_hidden_rewards)
        seed_observed_rewards = np.array(seed_observed_rewards)
        seed_action_validity = np.array(seed_action_validity)
        
        seed_summary = {
            'seed': current_seed,
            'episodes': len(seed_hidden_rewards),
            'hidden_reward_mean': seed_hidden_rewards.mean(),
            'hidden_reward_std': seed_hidden_rewards.std(),
            'observed_reward_mean': seed_observed_rewards.mean(),
            'observed_reward_std': seed_observed_rewards.std(),
            'action_validity_mean': seed_action_validity.mean(),
        }
        
        all_results['seed_summaries'].append(seed_summary)
        all_results['hidden_rewards'].extend(seed_hidden_rewards)
        all_results['observed_rewards'].extend(seed_observed_rewards)
        all_results['action_validity_rates'].extend(seed_action_validity)
        
        logging.info(f"\n{'='*60}")
        logging.info(f"SEED {seed_idx + 1} SUMMARY (seed={current_seed})")
        logging.info(f"{'='*60}")
        logging.info(f"Episodes: {seed_summary['episodes']}")
        logging.info(f"Hidden Reward: {seed_summary['hidden_reward_mean']:.4f} ± {seed_summary['hidden_reward_std']:.4f}")
        logging.info(f"Observed Reward: {seed_summary['observed_reward_mean']:.4f} ± {seed_summary['observed_reward_std']:.4f}")
        logging.info(f"Action Validity: {seed_summary['action_validity_mean']:.4f}")
        
        env_manager.close()

    # ======================= Final Summary =======================
    logging.info("\n" + "="*60)
    logging.info("FINAL EVALUATION SUMMARY")
    logging.info("="*60)
    logging.info(f"Environment: {args.env_name}")
    logging.info(f"Model: {args.model_name}")
    logging.info(f"Total seeds: {args.num_seeds}")
    logging.info(f"Total episodes: {len(all_results['hidden_rewards'])}")
    logging.info("-"*60)
    
    # Overall statistics
    hidden_rewards_all = np.array(all_results['hidden_rewards'])
    observed_rewards_all = np.array(all_results['observed_rewards'])
    action_validity_all = np.array(all_results['action_validity_rates'])
    
    logging.info("\nOVERALL STATISTICS (across all episodes):")
    logging.info(f"  Hidden Reward: {hidden_rewards_all.mean():.4f} ± {hidden_rewards_all.std():.4f}")
    logging.info(f"  Observed Reward: {observed_rewards_all.mean():.4f} ± {observed_rewards_all.std():.4f}")
    logging.info(f"  Action Validity: {action_validity_all.mean():.4f} ± {action_validity_all.std():.4f}")
    
    # Per-seed statistics
    logging.info("\nPER-SEED BREAKDOWN:")
    for i, summary in enumerate(all_results['seed_summaries']):
        logging.info(f"  Seed {i+1} (seed={summary['seed']}): "
                    f"Hidden={summary['hidden_reward_mean']:.2f}±{summary['hidden_reward_std']:.2f}, "
                    f"Observed={summary['observed_reward_mean']:.2f}±{summary['observed_reward_std']:.2f}")
    
    # Compute statistics across seeds (mean of means)
    seed_means_hidden = [s['hidden_reward_mean'] for s in all_results['seed_summaries']]
    seed_means_observed = [s['observed_reward_mean'] for s in all_results['seed_summaries']]
    
    logging.info(f"\nACROSS-SEED STATISTICS (mean ± std of per-seed means):")
    logging.info(f"  Hidden Reward: {np.mean(seed_means_hidden):.4f} ± {np.std(seed_means_hidden):.4f}")
    logging.info(f"  Observed Reward: {np.mean(seed_means_observed):.4f} ± {np.std(seed_means_observed):.4f}")
    
    logging.info("="*60)
    logging.info("Evaluation complete!")
    logging.info(f"Results saved to: {log_fp}")

if __name__ == "__main__":
    # -------- Argument Parser ----------
    parser = argparse.ArgumentParser(description='Evaluate open-source models on Safety Gridworlds using vLLM')
    parser.add_argument('--env_name', type=str, default='AbsentSupervisor',
                        help='Environment name (AbsentSupervisor, BoatRace, TomatoWatering, etc.)')
    parser.add_argument('--num_seeds', type=int, default=5,
                        help='Number of different random seeds')
    parser.add_argument('--episodes_per_seed', type=int, default=100,
                        help='Number of episodes per seed')
    parser.add_argument('--env_num', type=int, default=20,
                        help='Number of parallel environments')
    parser.add_argument('--max_steps', type=int, default=100,
                        help='Maximum steps per episode')
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen3-8B',
                        help='HuggingFace model name or path')
    parser.add_argument('--base_seed', type=int, default=42,
                        help='Base seed for random generation')
    parser.add_argument('--tensor_parallel_size', type=int, default=4,
                        help='Number of GPUs for tensor parallelism')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9,
                        help='Fraction of GPU memory to use (0.0-1.0)')
    parser.add_argument('--max_model_len', type=int, default=None,
                        help='Maximum model context length (None for default)')
    
    args = parser.parse_args()
    
    # Run evaluation
    asyncio.run(run_evaluation(args))
