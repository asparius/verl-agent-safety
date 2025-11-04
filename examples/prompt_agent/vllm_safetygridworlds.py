import os
import numpy as np
import time
import logging
from datetime import datetime
from collections import defaultdict
from vllm import LLM, SamplingParams
import argparse
import asyncio
import ray

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
            history_length = 10
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
        return f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{obs}<|im_end|>\n<|im_start|>assistant\n"
    
    def get_actions_batch(self, obs_list):
        """Get actions for multiple observations in batch"""
        if not obs_list:
            return []
        
        prompts = [self.format_prompt(obs) for obs in obs_list]
        outputs = self.llm.generate(prompts, self.sampling_params)
        actions = [output.outputs[0].text.strip() for output in outputs]
        
        return actions


async def run_evaluation(args):
    """Main evaluation loop"""
    
    # Calculate test_times
    test_times = args.episodes_per_seed // args.env_num
    if args.episodes_per_seed % args.env_num != 0:
        print(f"Warning: episodes_per_seed ({args.episodes_per_seed}) not divisible by env_num ({args.env_num})")
        print(f"Will run {test_times * args.env_num} episodes per seed instead")

    # -------- Logging ----------
    os.makedirs("qwen2.5-1.5b-logs/safety_gridworlds", exist_ok=True)
    safe_model_name = args.model_name.replace('/', '_')
    log_fp = os.path.join(
        "qwen2.5-1.5b-logs/safety_gridworlds", 
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

    # -------- Accumulated statistics ----------
    all_results = {
        'hidden_rewards': [],
        'observed_rewards': [],
        'action_validity_rates': [],
        'seed_summaries': []
    }

    # ======================= Main Loop =======================
    for seed_idx in range(args.num_seeds):
        current_seed = args.base_seed + seed_idx
        logging.info(f"\n{'='*60}")
        logging.info(f"SEED {seed_idx + 1}/{args.num_seeds} (seed={current_seed})")
        logging.info(f"{'='*60}")
        
        env_manager = build_env(env_name=args.env_name, env_num=args.env_num, seed=current_seed)
        
        seed_hidden_rewards = []
        seed_observed_rewards = []
        seed_action_validity = []

        for test_idx in range(test_times):
            logging.info(f"\n--- Seed {seed_idx + 1}, Test {test_idx + 1}/{test_times} ---")
            start_time = time.time()

            obs, infos = env_manager.reset({})
            env_dones = [False] * args.env_num

            cumulative_hidden_rewards = np.full(args.env_num, np.nan, dtype=np.float64)
            cumulative_observed_rewards = np.zeros(args.env_num, dtype=np.float64)
            valid_actions_count = np.zeros(args.env_num, dtype=np.float64)
            total_actions_count = np.zeros(args.env_num, dtype=np.float64)

            for step_idx in range(args.max_steps):
                done_count = np.array(env_dones).sum().item()
                if step_idx % 10 == 0:
                    logging.info(f"  Step {step_idx}/{args.max_steps}; Dones ({done_count}/{args.env_num})")

                active_indices = [i for i in range(args.env_num) if not env_dones[i]]
                if active_indices:
                    active_obs = [obs["text"][i] for i in active_indices]
                    active_actions = agent.get_actions_batch(active_obs)
                    
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

                obs, rewards, dones, infos = env_manager.step(actions)

                for i in range(args.env_num):
                    if env_dones[i]:
                        continue

                    hidden_reward = infos[i].get('hidden_reward')
                    observed_reward = infos[i].get('observed_reward', 0.0)

                    # Handle hidden_reward safely (may be None)
                    if hidden_reward is not None:
                        if np.isnan(cumulative_hidden_rewards[i]):
                            cumulative_hidden_rewards[i] = 0.0
                        cumulative_hidden_rewards[i] += float(hidden_reward)

                    cumulative_observed_rewards[i] += float(observed_reward)

                    is_valid = infos[i].get('is_action_valid', 1)
                    valid_actions_count[i] += float(is_valid)
                    total_actions_count[i] += 1.0

                    if dones[i]:
                        env_dones[i] = True

                if all(env_dones):
                    logging.info("  All environments finished early!")
                    break

            action_validity_rates = valid_actions_count / np.maximum(total_actions_count, 1.0)

            seed_hidden_rewards.extend(cumulative_hidden_rewards.tolist())
            seed_observed_rewards.extend(cumulative_observed_rewards.tolist())
            seed_action_validity.extend(action_validity_rates.tolist())

            logging.info(f"  Test completed in {time.time() - start_time:.2f}s")
            logging.info(f"  Hidden Reward: {np.nanmean(cumulative_hidden_rewards):.4f} ± {np.nanstd(cumulative_hidden_rewards):.4f}")
            logging.info(f"  Observed Reward: {cumulative_observed_rewards.mean():.4f} ± {cumulative_observed_rewards.std():.4f}")
            logging.info(f"  Action Validity: {action_validity_rates.mean():.4f}")

        seed_hidden_rewards = np.array(seed_hidden_rewards)
        seed_observed_rewards = np.array(seed_observed_rewards)
        seed_action_validity = np.array(seed_action_validity)
        
        seed_summary = {
            'seed': current_seed,
            'episodes': len(seed_hidden_rewards),
            'hidden_reward_mean': float(np.nanmean(seed_hidden_rewards)),
            'hidden_reward_std': float(np.nanstd(seed_hidden_rewards)),
            'observed_reward_mean': float(seed_observed_rewards.mean()),
            'observed_reward_std': float(seed_observed_rewards.std()),
            'action_validity_mean': float(seed_action_validity.mean()),
        }
        
        all_results['seed_summaries'].append(seed_summary)
        all_results['hidden_rewards'].extend(seed_hidden_rewards.tolist())
        all_results['observed_rewards'].extend(seed_observed_rewards.tolist())
        all_results['action_validity_rates'].extend(seed_action_validity.tolist())
        
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
    
    hidden_rewards_all = np.array(all_results['hidden_rewards'])
    observed_rewards_all = np.array(all_results['observed_rewards'])
    action_validity_all = np.array(all_results['action_validity_rates'])
    
    logging.info("\nOVERALL STATISTICS (across all episodes):")
    logging.info(f"  Hidden Reward: {np.nanmean(hidden_rewards_all):.4f} ± {np.nanstd(hidden_rewards_all):.4f}")
    logging.info(f"  Observed Reward: {observed_rewards_all.mean():.4f} ± {observed_rewards_all.std():.4f}")
    logging.info(f"  Action Validity: {action_validity_all.mean():.4f} ± {action_validity_all.std():.4f}")
    
    logging.info("\nPER-SEED BREAKDOWN:")
    for i, summary in enumerate(all_results['seed_summaries']):
        logging.info(f"  Seed {i+1} (seed={summary['seed']}): "
                     f"Hidden={summary['hidden_reward_mean']:.2f}±{summary['hidden_reward_std']:.2f}, "
                     f"Observed={summary['observed_reward_mean']:.2f}±{summary['observed_reward_std']:.2f}")
    
    seed_means_hidden = [s['hidden_reward_mean'] for s in all_results['seed_summaries']]
    seed_means_observed = [s['observed_reward_mean'] for s in all_results['seed_summaries']]
    
    logging.info(f"\nACROSS-SEED STATISTICS (mean ± std of per-seed means):")
    logging.info(f"  Hidden Reward: {np.mean(seed_means_hidden):.4f} ± {np.std(seed_means_hidden):.4f}")
    logging.info(f"  Observed Reward: {np.mean(seed_means_observed):.4f} ± {np.std(seed_means_observed):.4f}")
    
    logging.info("="*60)
    logging.info("Evaluation complete!")
    logging.info(f"Results saved to: {log_fp}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate open-source models on Safety Gridworlds using vLLM')
    parser.add_argument('--env_name', type=str, default='AbsentSupervisor')
    parser.add_argument('--num_seeds', type=int, default=5)
    parser.add_argument('--episodes_per_seed', type=int, default=20)
    parser.add_argument('--env_num', type=int, default=10)
    parser.add_argument('--max_steps', type=int, default=100)
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-3B')
    parser.add_argument('--base_seed', type=int, default=42)
    parser.add_argument('--tensor_parallel_size', type=int, default=1)
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9)
    parser.add_argument('--max_model_len', type=int, default=None)
    parser.add_argument('--num_cpus', type=int, default=None)
    parser.add_argument('--num_gpus', type=int, default=None)
    
    args = parser.parse_args()
    
    # -------- Initialize Ray ----------
    if ray.is_initialized():
        print("Shutting down existing Ray connection...")
        ray.shutdown()
    
    ray_address_env = os.environ.get('RAY_ADDRESS')
    
    print("Initializing Ray...")
    if ray_address_env:
        print(f"RAY_ADDRESS detected: {ray_address_env}")
        print("Connecting to existing Ray cluster (ignoring num_cpus/num_gpus)...")
        ray.init(address='auto', ignore_reinit_error=True, log_to_driver=True)
    else:
        try:
            print("Attempting to start new Ray head node...")
            ray_init_kwargs = {
                'ignore_reinit_error': True,
                'log_to_driver': True,
            }
            if args.num_cpus is not None:
                ray_init_kwargs['num_cpus'] = args.num_cpus
            if args.num_gpus is not None:
                ray_init_kwargs['num_gpus'] = args.num_gpus
            ray.init(**ray_init_kwargs)
        except ValueError as e:
            if "num_cpus and num_gpus must not be provided" in str(e):
                print("Detected existing cluster connection. Retrying without resource specifications...")
                ray.init(ignore_reinit_error=True, log_to_driver=True)
            else:
                raise
    
    print("Ray initialized successfully!")
        
    try:
        asyncio.run(run_evaluation(args))
    finally:
        if ray.is_initialized():
            print("\nShutting down Ray...")
            ray.shutdown()

