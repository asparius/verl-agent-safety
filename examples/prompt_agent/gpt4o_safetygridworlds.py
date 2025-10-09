import os
import numpy as np
import time
import logging
from datetime import datetime
from collections import defaultdict
from openai import OpenAI

def build_env(env_name, env_num=1):
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
        seed=42,
        env_num=env_num,
        group_n=group_n,
        render_mode='ansi',
        resources_per_worker=resources_per_worker,
        is_train=False
    )
    
    # Simple config mock for the environment manager
    class Config:
        class Env:
            env_name = env_name
            history_length = 0  # No history for initial eval
        env = Env()
    
    config = Config()
    
    # Create projection function
    from functools import partial
    projection_f = partial(safety_gridworld_projection, env_name=env_name)
    
    # Create environment manager
    env_manager = SafetyGridworldsEnvironmentManager(envs, projection_f, config)
    
    return env_manager

class Agent:
    def __init__(self, model_name="gpt-4o"):
        self.model_name = model_name
        self.client = OpenAI(
            api_key=os.environ['OPENAI_API_KEY'],
        )
        
    def get_action_from_gpt(self, obs):
        """Get action from GPT-4o"""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user", 
                    "content": obs
                }
            ],
            temperature=0.4,
            n=1,
            stop=None
        )
        action = response.choices[0].message.content.strip()
        return action

if __name__ == "__main__":

    # -------- Logging ----------
    os.makedirs("logs/safety_gridworlds", exist_ok=True)
    log_fp = os.path.join(
        "logs/safety_gridworlds", f"run_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[logging.FileHandler(log_fp, encoding="utf-8"), logging.StreamHandler()],
    )

    # -------- Parameters ----------
    max_steps = 100  # AI Safety Gridworlds typically use 100 steps
    env_num = 50     # Number of parallel environments
    test_times = 3   # Number of test rounds
    env_name = "BoatRace"  # Can be: "BoatRace", "TomatoWatering", "AbsentSupervisor", etc.

    # -------- Environment and agent setup ----------
    logging.info(f"Setting up environment: {env_name}")
    env_manager = build_env(env_name, env_num)
    agent = Agent()

    # Accumulated statistics
    cumulative_hidden_rewards_all = []
    cumulative_observed_rewards_all = []
    action_validity_rates_all = []

    # ======================= Main Loop =======================
    for test_idx in range(test_times):
        logging.info(f"\n========== Start test {test_idx} for {env_name} ==========")
        start_time = time.time()

        obs, infos = env_manager.reset({})
        env_dones = [False] * env_num

        # Track rewards for each environment in this round
        cumulative_hidden_rewards = np.zeros(env_num)
        cumulative_observed_rewards = np.zeros(env_num)
        valid_actions_count = np.zeros(env_num)
        total_actions_count = np.zeros(env_num)

        for step_idx in range(max_steps):
            done_count = np.array(env_dones).sum().item()
            logging.info(f"Step {step_idx}/{max_steps}; Dones ({done_count}/{env_num})")

            # --- Assemble actions ---
            actions = []
            for i in range(env_num):
                if env_dones[i]:
                    actions.append("None")  # Placeholder for done environments
                else:
                    actions.append(agent.get_action_from_gpt(obs["text"][i]))

            # --- Environment stepping ---
            obs, rewards, dones, infos = env_manager.step(actions)

            # --- Accumulate rewards and track action validity ---
            for i in range(env_num):
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
                logging.info("All environments finished early!")
                break

        # -------- Single round results --------
        # Compute action validity rate
        action_validity_rates = valid_actions_count / np.maximum(total_actions_count, 1)

        # Store results
        cumulative_hidden_rewards_all.extend(cumulative_hidden_rewards)
        cumulative_observed_rewards_all.extend(cumulative_observed_rewards)
        action_validity_rates_all.extend(action_validity_rates)

        # Log round statistics
        logging.info(f"\nTest {test_idx} Results:")
        logging.info(f"  Avg Hidden Reward: {cumulative_hidden_rewards.mean():.4f} ± {cumulative_hidden_rewards.std():.4f}")
        logging.info(f"  Avg Observed Reward: {cumulative_observed_rewards.mean():.4f} ± {cumulative_observed_rewards.std():.4f}")
        logging.info(f"  Avg Action Validity Rate: {action_validity_rates.mean():.4f} ± {action_validity_rates.std():.4f}")
        logging.info(f"Test {test_idx} time elapsed: {time.time() - start_time:.2f}s\n")

    # ======================= Final Summary =======================
    logging.info("=============== Final Summary ===============")
    logging.info(f"Environment: {env_name}")
    logging.info(f"Total tests: {test_times} | Envs / test: {env_num} | Total episodes: {env_num * test_times}")
    
    cumulative_hidden_rewards_all = np.array(cumulative_hidden_rewards_all)
    cumulative_observed_rewards_all = np.array(cumulative_observed_rewards_all)
    action_validity_rates_all = np.array(action_validity_rates_all)
    
    logging.info(f"\nOverall Statistics:")
    logging.info(f"  Hidden Reward: {cumulative_hidden_rewards_all.mean():.4f} ± {cumulative_hidden_rewards_all.std():.4f}")
    logging.info(f"  Observed Reward: {cumulative_observed_rewards_all.mean():.4f} ± {cumulative_observed_rewards_all.std():.4f}")
    logging.info(f"  Action Validity Rate: {action_validity_rates_all.mean():.4f} ± {action_validity_rates_all.std():.4f}")
    
    env_manager.close()
    logging.info("\nEvaluation complete!")