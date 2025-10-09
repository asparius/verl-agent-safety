import ray
import gymnasium
import safe_grid_gym
import numpy as np 

class SafetyGridworldWorker:

    """"
    Ray remote actor that holds an independent instance of a Safety Gridworld environment.
    Each actor manages one environment instance.
    """

    def __init__(self, env_name, render_mode='ansi', env_kwargs={}):

        """Initialize the Safety Gridworld environment in this worker
        
        Args:
            env_name: Name of the gridworld (e.g., 'AbsentSupervisor', 'BoatRace', 'TomatoWatering')
            render_mode: Rendering mode ('ansi', 'rgb_array', 'human')
            env_kwargs: Additional kwargs for environment initialization
        """
        if env_kwargs is None:
            env_kwargs = {}

        self.env = gymnasium.make(f"{env_name}-v0", render_mode=render_mode,**env_kwargs)
        self.env_name = env_name


    
    def step(self, action):

        """Execute a step in the environment
        
        Returns:
            observation, reward, done, info
        """
        obs,reward, terminated,_,info = self.env.step(action)
        return obs, reward, terminated, info

    def reset(self, seed_for_reset):
        """Reset the environment with given seed
        
        Returns:
            observation, info
        """
        obs, info = self.env.reset(seed=seed_for_reset)
        return obs,info

    def render(self):
        """Render the environment
        
        Returns:
            Rendered output (format depends on render_mode)
        """
        rendered = self.env.render()
        return rendered

    def close(self):
        """Close the environment"""
        self.env.close()



class SafetyGridworldMultiProcessEnv(gymnasium.Env):

    """
    Ray-based wrapper for AI Safety Gridworlds environments.
    Each Ray actor creates an independent environment instance.
    The main process communicates with Ray actors to collect step/reset results.
    
    This supports parallel rollouts and group-based training (for GRPO/GiGPO).
    """

    def __init__(self,
                env_name='AbsentSupervisor',
                seed=0,
                env_num=1,
                group_n=1,
                render_mode='ansi',
                resources_per_worker={'num_cpus':0.1},
                is_train=True,
                env_kwargs=None):


        """
        Args:
            env_name: Name of the Safety Gridworld (e.g., 'AbsentSupervisor', 'BoatRace', 'TomatoWatering')
            seed: Random seed for reproducibility
            env_num: Number of different environments (different initial states)
            group_n: Number of same environments in each group (for GRPO and GiGPO)
            render_mode: Rendering mode ('ansi', 'rgb_array', 'human')
            resources_per_worker: Ray resources allocation per worker
            is_train: Whether this is for training or evaluation
            env_kwargs: Additional kwargs for environment initialization
        """
        super().__init__()


        if not ray.is_initialized():
            ray.init()

        self.env_name = env_name
        self.is_train = is_train
        self.group_n = group_n
        self.env_num = env_num
        self.num_processes = env_num * group_n
        self.render_mode = render_mode
        np.random.seed(seed)

        if env_kwargs is None:
            env_kwargs = {}

        env_worker = ray.remote(**resources_per_worker)(SafetyGridworldWorker)
        self.workers = []
        for i in range(self.num_processes):
            worker = env_worker.remote(env_name, render_mode, env_kwargs)
            self.workers.append(worker)


    def step(self, actions):


        """
        Perform step in parallel across all environments.
        
        Args:
            actions: list[int], length must match self.num_processes
            
        Returns:
            obs_list: List of observations
            reward_list: List of observed rewards
            done_list: List of done flags
            info_list: List of info dicts (includes hidden_reward for safety analysis)
        """
        assert len(actions) == self.num_processes , \
                f"Expected {self.num_processes} actions , got {len(actions)}"


        futures = []

        for worker, action in zip(self.workers, actions):
            future = worker.step(action)
            futures.append(future)

        results = rat.get(futures)

        obs_list, reward_list, done_list, info_list = [], [], [], []
        for obs, reward, done, info in results:
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info) 

        if isinstance(obs_list[0], np.ndarray):
            obs_list = np.array(obs_list)

        return obs_list, reward_list, done_list, info_list



    def reset(self):

        if self.is_train:
            seeds = np.random.randint(0, 2**16- 1, size=self.env_num)

        else:
            seeds = np.random.randint(2**16, 2**32 -1, size=self.env_num)


        seeds = np.repeat(seeds, self.group_n)
        seeds = seeds.tolist()

        futures = []
        for i, worker in enumerate(self.workers):
            future = worker.reset.remote(seeds[i])
            futures.append(future)

        results = ray.get(futures)
        obs_list, info_list = [], []

        for obs, info in results:
            obs_list.append(obs)
            info_list.append(info)


        if isinstance(obs_list[0], np.ndarray):
            obs_list = np.array(obs_list)

        return obs_list, info_list             


    def render(self, env_idx=None):
        """
        Request rendering from Ray actor environments.
        
        Args:
            env_idx: If specified, render only that environment index.
                    Otherwise, render all environments.
        
        Returns:
            Single render output or list of render outputs
        """
        if env_idx is not None:
            future = self.workers[env_idx].render.remote()
            return ray.get(future)
        else:
            futures = []
            for worker in self.workers:
                future = worker.render.remote()
                futures.append(future)
            results = ray.get(futures)
            return results

    def close(self):
        """
        Close all Ray actors and clean up resources
        """
        # Kill all Ray actors
        for worker in self.workers:
            ray.kill(worker)

    def __del__(self):
        """Cleanup on deletion"""
        self.close()


def build_safety_gridworld_envs(
        env_name='AbsentSupervisor',
        seed=0,
        env_num=1,
        group_n=1,
        render_mode='ansi',
        resources_per_worker={"num_cpus": 0.1},
        is_train=True,
        env_kwargs=None):
    """
    Factory function to build Safety Gridworlds multi-process environment.
    
    Args:
        env_name: Name of the gridworld environment. Options include:
            - 'AbsentSupervisor'
            - 'BoatRace'
            - 'ConveyorBelt'
            - 'DistributionalShift'
            - 'FriendFoe'
            - 'IslandNavigation'
            - 'RocksDiamonds'
            - 'SafeInterruptibility'
            - 'SideEffectsSokoban'
            - 'TomatoWatering'
            - 'TomatoCrmdp'
            - 'WhiskyGold'
        seed: Random seed
        env_num: Number of different environment instances
        group_n: Group size for grouped algorithms (GRPO/GiGPO)
        render_mode: 'ansi', 'rgb_array', or 'human'
        resources_per_worker: Ray resource allocation
        is_train: Training vs evaluation mode
        env_kwargs: Additional environment parameters
        
    Returns:
        SafetyGridworldMultiProcessEnv instance
    """
    return SafetyGridworldMultiProcessEnv(
        env_name=env_name,
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        render_mode=render_mode,
        resources_per_worker=resources_per_worker,
        is_train=is_train,
        env_kwargs=env_kwargs
    )
