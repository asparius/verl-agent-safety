import os
import json
import numpy as np
import time
import logging
import random
from datetime import datetime
from collections import defaultdict, Counter
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
            history_length = 4
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
                 max_model_len=None,
                 temperature=0.0,
                 n_samples=1):
        """
        Initialize vLLM agent
        """
        self.model_name = model_name
        self.temperature = temperature
        self.n_samples = n_samples

        logging.info(f"Loading model: {model_name}")
        logging.info(f"Tensor parallel size: {tensor_parallel_size}")
        logging.info(f"GPU memory utilization: {gpu_memory_utilization}")
        logging.info(f"Temperature: {temperature}")
        logging.info(f"Majority voting samples: {n_samples}")

        # Initialize vLLM
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            trust_remote_code=True,
        )

        # Sampling parameters
        # When n_samples > 1 we ask vLLM for n completions per prompt and then
        # do majority voting on the client side (mirrors the OpenRouter version).
        self.sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=16384,
            top_p=1.0,
            n=n_samples,
        )

        logging.info("Model loaded successfully!")

    def format_prompt(self, obs):
        """Format observation as prompt for the model"""
        return f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{obs}<|im_end|>\n<|im_start|>assistant\n"

    def get_actions_batch(self, obs_list):
        """Get actions for multiple observations in batch.

        When n_samples > 1, performs majority voting across the n completions
        returned by vLLM for each prompt (matching OpenRouter version behaviour).
        """
        if not obs_list:
            return []

        prompts = [self.format_prompt(obs) for obs in obs_list]
        outputs = self.llm.generate(prompts, self.sampling_params)

        actions = []
        for output in outputs:
            candidates = [o.text.strip() for o in output.outputs]
            if len(candidates) == 1:
                actions.append(candidates[0])
            else:
                most_common, _ = Counter(candidates).most_common(1)[0]
                actions.append(most_common)

        return actions


async def run_evaluation(args):
    """Main evaluation loop"""

    # Calculate test_times
    test_times = args.episodes_per_seed // args.env_num
    if args.episodes_per_seed % args.env_num != 0:
        print(f"Warning: episodes_per_seed ({args.episodes_per_seed}) not divisible by env_num ({args.env_num})")
        print(f"Will run {test_times * args.env_num} episodes per seed instead")

    # -------- Determine which episodes to save ----------
    total_episodes = args.num_seeds * test_times * args.env_num
    num_to_save = max(1, int(total_episodes * args.save_pct / 100.0))
    # Each episode is identified by a global index: (seed_idx, test_idx, env_idx)
    # We pre-select which global episode indices to save
    all_episode_indices = list(range(total_episodes))
    random.seed(args.base_seed)
    save_indices = set(random.sample(all_episode_indices, min(num_to_save, total_episodes)))
    saved_episodes = []  # will hold the trajectory dicts

    # -------- Logging ----------
    os.makedirs("logs/safety_gridworlds", exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    safe_model_name = args.model_name.replace('/', '_')
    log_fp = os.path.join(
        "logs/safety_gridworlds",
        f"{args.env_name}_{safe_model_name}_{timestamp}.log"
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[logging.FileHandler(log_fp, encoding="utf-8"), logging.StreamHandler()],
    )

    # -------- Log Configuration ----------
    logging.info("=" * 60)
    logging.info("EVALUATION CONFIGURATION")
    logging.info("=" * 60)
    logging.info(f"Environment: {args.env_name}")
    logging.info(f"Model: {args.model_name}")
    logging.info(f"Number of seeds: {args.num_seeds}")
    logging.info(f"Episodes per seed: {test_times * args.env_num}")
    logging.info(f"Parallel environments: {args.env_num}")
    logging.info(f"Test rounds per seed: {test_times}")
    logging.info(f"Total episodes: {total_episodes}")
    logging.info(f"Max steps per episode: {args.max_steps}")
    logging.info(f"Base seed: {args.base_seed}")
    logging.info(f"Saving {num_to_save}/{total_episodes} episodes ({args.save_pct}%)")
    logging.info(f"Temperature: {args.temperature}")
    logging.info(f"Majority voting samples: {args.n_samples}")
    logging.info(f"Tensor parallel size: {args.tensor_parallel_size}")
    logging.info("=" * 60 + "\n")

    # -------- Initialize Agent ----------
    agent = VLLMAgent(
        model_name=args.model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        temperature=args.temperature,
        n_samples=args.n_samples,
    )

    # Accumulated statistics across all seeds
    all_results = {
        'hidden_rewards': [],
        'observed_rewards': [],
        'action_validity_rates': [],
        'seed_summaries': []
    }

    global_episode_counter = 0  # tracks the flat episode index across all seeds/tests/envs

    # ======================= Main Loop: Iterate over seeds =======================
    for seed_idx in range(args.num_seeds):
        current_seed = args.base_seed + seed_idx
        logging.info(f"\n{'=' * 60}")
        logging.info(f"SEED {seed_idx + 1}/{args.num_seeds} (seed={current_seed})")
        logging.info(f"{'=' * 60}")
        
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

            # Figure out which envs in this batch should be saved
            batch_start = global_episode_counter
            saving_envs = set()
            for ei in range(args.env_num):
                if (batch_start + ei) in save_indices:
                    saving_envs.add(ei)

            # Per-env trajectory buffers (only for episodes we're saving)
            trajectories = {}
            for ei in saving_envs:
                trajectories[ei] = {
                    'env_name': args.env_name,
                    'model': args.model_name,
                    'seed': current_seed,
                    'seed_idx': seed_idx,
                    'test_idx': test_idx,
                    'env_idx': ei,
                    'global_episode_idx': batch_start + ei,
                    'steps': [],
                    'cumulative_hidden_reward': None,
                    'cumulative_observed_reward': 0.0,
                    'action_validity_rate': None,
                    'num_steps': 0,
                }

            # Track rewards for each environment in this round
            cumulative_hidden_rewards = np.full(args.env_num, np.nan, dtype=np.float64)
            cumulative_observed_rewards = np.zeros(args.env_num, dtype=np.float64)
            valid_actions_count = np.zeros(args.env_num, dtype=np.float64)
            total_actions_count = np.zeros(args.env_num, dtype=np.float64)

            for step_idx in range(args.max_steps):
                done_count = np.array(env_dones).sum().item()

                if step_idx % 10 == 0:  # Log every 10 steps to reduce clutter
                    logging.info(f"  Step {step_idx}/{args.max_steps}; Dones ({done_count}/{args.env_num})")

                # --- Assemble actions for non-done environments (batched via vLLM) ---
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

                # --- Capture pre-step observations for saved episodes ---
                pre_step_obs = {}
                for ei in saving_envs:
                    if not env_dones[ei]:
                        pre_step_obs[ei] = obs["text"][ei]

                # --- Environment stepping ---
                obs, rewards, dones, infos = env_manager.step(actions)

                # --- Accumulate rewards and track action validity ---
                for i in range(args.env_num):
                    if env_dones[i]:
                        continue

                    # Get rewards from info
                    hidden_reward = infos[i].get('hidden_reward')
                    observed_reward = infos[i].get('observed_reward', 0.0)

                    # Handle hidden_reward: only accumulate if not None
                    if hidden_reward is not None:
                        # If this is the first non-None hidden reward, initialize to 0
                        if np.isnan(cumulative_hidden_rewards[i]):
                            cumulative_hidden_rewards[i] = 0.0
                        cumulative_hidden_rewards[i] += float(hidden_reward)

                    # Always accumulate observed reward
                    cumulative_observed_rewards[i] += float(observed_reward)

                    # Track action validity
                    is_valid = infos[i].get('is_action_valid', 1)
                    valid_actions_count[i] += float(is_valid)
                    total_actions_count[i] += 1.0

                    # --- Record step for saved episodes ---
                    if i in saving_envs:
                        step_record = {
                            'step': step_idx,
                            'observation': pre_step_obs.get(i, None),
                            'action': actions[i],
                            'hidden_reward': float(hidden_reward) if hidden_reward is not None else None,
                            'observed_reward': float(observed_reward),
                            'is_action_valid': bool(is_valid),
                            'done': bool(dones[i]),
                        }
                        trajectories[i]['steps'].append(step_record)

                    # Mark as done
                    if dones[i]:
                        env_dones[i] = True

                if all(env_dones):
                    logging.info("  All environments finished early!")
                    break

            # -------- Finalize saved trajectories for this batch --------
            action_validity_rates = valid_actions_count / np.maximum(total_actions_count, 1.0)

            for ei in saving_envs:
                traj = trajectories[ei]
                traj['cumulative_hidden_reward'] = (
                    float(cumulative_hidden_rewards[ei])
                    if not np.isnan(cumulative_hidden_rewards[ei])
                    else None
                )
                traj['cumulative_observed_reward'] = float(cumulative_observed_rewards[ei])
                traj['action_validity_rate'] = float(action_validity_rates[ei])
                traj['num_steps'] = len(traj['steps'])
                saved_episodes.append(traj)

            # Advance global counter
            global_episode_counter += args.env_num

            # -------- Store results from this test --------
            seed_hidden_rewards.extend(cumulative_hidden_rewards.tolist())
            seed_observed_rewards.extend(cumulative_observed_rewards.tolist())
            seed_action_validity.extend(action_validity_rates.tolist())

            # Log test statistics (use nanmean to ignore NaN values)
            logging.info(f"  Test completed in {time.time() - start_time:.2f}s")
            logging.info(f"  Hidden Reward: {np.nanmean(cumulative_hidden_rewards):.4f} ± {np.nanstd(cumulative_hidden_rewards):.4f}")
            logging.info(f"  Observed Reward: {cumulative_observed_rewards.mean():.4f} ± {cumulative_observed_rewards.std():.4f}")
            logging.info(f"  Action Validity: {action_validity_rates.mean():.4f}")

        # -------- Seed Summary --------
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
        
        logging.info(f"\n{'=' * 60}")
        logging.info(f"SEED {seed_idx + 1} SUMMARY (seed={current_seed})")
        logging.info(f"{'=' * 60}")
        logging.info(f"Episodes: {seed_summary['episodes']}")
        logging.info(f"Hidden Reward: {seed_summary['hidden_reward_mean']:.4f} ± {seed_summary['hidden_reward_std']:.4f}")
        logging.info(f"Observed Reward: {seed_summary['observed_reward_mean']:.4f} ± {seed_summary['observed_reward_std']:.4f}")
        logging.info(f"Action Validity: {seed_summary['action_validity_mean']:.4f}")
        
        env_manager.close()

    # ======================= Save Episode Trajectories =======================
    episodes_fp = os.path.join(
        "logs/safety_gridworlds",
        f"{args.env_name}_{safe_model_name}_{timestamp}_episodes.json"
    )
    with open(episodes_fp, 'w', encoding='utf-8') as f:
        json.dump({
            'metadata': {
                'env_name': args.env_name,
                'model': args.model_name,
                'total_episodes': total_episodes,
                'saved_episodes': len(saved_episodes),
                'save_pct': args.save_pct,
                'num_seeds': args.num_seeds,
                'base_seed': args.base_seed,
                'max_steps': args.max_steps,
                'timestamp': timestamp,
            },
            'episodes': saved_episodes,
        }, f, indent=2, ensure_ascii=False)

    logging.info(f"\nSaved {len(saved_episodes)}/{total_episodes} episode trajectories to: {episodes_fp}")

    # ======================= Final Summary =======================
    logging.info("\n" + "=" * 60)
    logging.info("FINAL EVALUATION SUMMARY")
    logging.info("=" * 60)
    logging.info(f"Environment: {args.env_name}")
    logging.info(f"Model: {args.model_name}")
    logging.info(f"Total seeds: {args.num_seeds}")
    logging.info(f"Total episodes: {len(all_results['hidden_rewards'])}")
    logging.info("-" * 60)
    
    # Overall statistics
    hidden_rewards_all = np.array(all_results['hidden_rewards'])
    observed_rewards_all = np.array(all_results['observed_rewards'])
    action_validity_all = np.array(all_results['action_validity_rates'])
    
    logging.info("\nOVERALL STATISTICS (across all episodes):")
    logging.info(f"  Hidden Reward: {np.nanmean(hidden_rewards_all):.4f} ± {np.nanstd(hidden_rewards_all):.4f}")
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
    
    logging.info("=" * 60)
    logging.info("Evaluation complete!")
    logging.info(f"Results saved to: {log_fp}")
    logging.info(f"Episodes saved to: {episodes_fp}")


if __name__ == "__main__":
    # -------- Argument Parser ----------
    parser = argparse.ArgumentParser(description='Evaluate open-source models on Safety Gridworlds using vLLM')
    parser.add_argument('--env_name', type=str, default='IslandNavigation',
                        help='Environment name (AbsentSupervisor, BoatRace, TomatoWatering, etc.)')
    parser.add_argument('--num_seeds', type=int, default=5,
                        help='Number of different random seeds')
    parser.add_argument('--episodes_per_seed', type=int, default=10,
                        help='Number of episodes per seed')
    parser.add_argument('--env_num', type=int, default=10,
                        help='Number of parallel environments')
    parser.add_argument('--max_steps', type=int, default=50,
                        help='Maximum steps per episode')
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-3B',
                        help='HuggingFace model name to load via vLLM')
    parser.add_argument('--base_seed', type=int, default=42,
                        help='Base seed for random generation')
    parser.add_argument('--save_pct', type=float, default=5.0,
                        help='Percentage of episodes to save as full trajectories (default: 5)')
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='Sampling temperature (0=greedy). Set to 1.0 for models that require it')
    parser.add_argument('--n_samples', type=int, default=1,
                        help='Majority voting samples per action (use 5 with temperature=1 models)')
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
