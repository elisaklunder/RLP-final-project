import gymnasium as gym
import numpy as np
from stable_baselines3.common.env_util import make_vec_env


class SB3EnvironmentHandler:
    def __init__(self, env_type: str, human_render: bool = False, num_envs: int = 1):
        self.env_type = env_type
        self.num_envs = num_envs
        self.human_render = human_render
        self.env = make_vec_env(lambda: gym.make(env_type), n_envs=num_envs)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def close(self):
        self.env.close()


class OurEnvironmentHandler:
    def __init__(
        self,
        env_id: str,
        num_envs: int = 1,
        run_name: str = "experiment",
        seed: int = 42,
        human_render: bool = False,
    ):
        """
        A handler that sets up and manages environments (including vectorized environments).

        Args:
            env_id (str): Gymnasium environment ID.
            num_envs (int): Number of parallel environments.
            capture_video (bool): Whether to record video from the first environment.
            run_name (str): Run name for video logging.
            seed (int): Environment seed.
        """
        self.env_id = env_id
        self.num_envs = num_envs
        self.run_name = run_name
        self.human_render = human_render
        self.seed = seed
        self.envs = self._make_vector_envs()

    def _make_env(self, env_id, idx):
        def thunk():
            if self.env_id == "FlappyBird-v0":
                if self.human_render:
                    env = gym.make(env_id, render_mode="human", use_lidar=False)
                else:
                    env = gym.make(env_id, use_lidar=False)
            else:
                if self.human_render:
                    env = gym.make(env_id, render_mode="human")
                else:
                    env = gym.make(env_id)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            return env

        return thunk

    def _make_vector_envs(self):
        envs = gym.vector.SyncVectorEnv(
            [self._make_env(self.env_id, i) for i in range(self.num_envs)]
        )
        envs.reset(seed=self.seed)
        return envs

    def reset(self):
        return self.envs.reset(seed=self.seed)

    def step(self, actions: np.ndarray):
        return self.envs.step(actions)

    def close(self):
        self.envs.close()

    @property
    def single_observation_space(self):
        return self.envs.single_observation_space

    @property
    def single_action_space(self):
        return self.envs.single_action_space
