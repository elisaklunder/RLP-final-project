# from trading_env import download_dataset, make_features
import flappy_bird_gymnasium  # noqa
import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

from metadrive.envs.metadrive_env import MetaDriveEnv


class SB3EnvironmentHandler:
    def __init__(self, env_id: str, human_render: str = None, num_envs: int = 1):
        self.env_id = env_id
        self.num_envs = num_envs
        self.human_render = human_render
        self.envs = self._make_vector_envs()

    def make_env_flappy_bird(self, seed):
        def _init():
            env = gym.make(
                "FlappyBird-v0", use_lidar=False, render_mode=self.human_render
            )
            env = Monitor(env)
            env.reset(seed=seed)
            return env

        return _init

    def make_env_metadrive(self, seed):
        def _init():
            env = MetaDriveEnv(
                dict(
                    map="C",
                    discrete_action=True,
                    discrete_throttle_dim=3,
                    discrete_steering_dim=3,
                    horizon=500,
                    random_spawn_lane_index=False,
                    num_scenarios=1,
                    start_seed=0,
                    traffic_density=0,
                    accident_prob=0,
                    log_level=50,
                )
            )
            env = Monitor(env)
            env.reset(seed=seed)
            return env

        return _init

    def _make_vector_envs(self):
        """
        Sets up the vectorized environments using SB3 wrappers.
        """
        if self.env_id == "FlappyBird-v0":
            # Use SubprocVecEnv (or DummyVecEnv) from stable_baselines3
            envs = SubprocVecEnv(
                [self.make_env_flappy_bird(i) for i in range(self.num_envs)]
            )
        elif self.env_id == "MetaDriveEnv":
            envs = SubprocVecEnv(
                [self.make_env_metadrive(i) for i in range(self.num_envs)]
            )
        else:
            # Easiest approach: stable-baselines3’s make_vec_env
            envs = make_vec_env(
                self.env_id,
                n_envs=self.num_envs,
                # pass extra arguments via env_kwargs if needed
                monitor_wrapper=True
            )

        return envs

    def reset(self):
        return self.envs.reset()

    def step(self, action):
        return self.envs.step(action)

    def close(self):
        self.envs.close()


# class OurEnvironmentHandler:
#     def __init__(
#         self,
#         env_id: str = "FlappyBird-v0",
#         num_envs: int = 1,
#         run_name: str = "experiment",
#         # seed: int = 42,
#         human_render: str | None = None,
#         max_steps_per_episode: int = 1000,
#     ):
#         """
#         A handler that sets up and manages environments (including vectorized environments).

#         Args:
#             env_id (str): Gymnasium environment ID.
#             num_envs (int): Number of parallel environments.
#             capture_video (bool): Whether to record video from the first environment.
#             run_name (str): Run name for video logging.
#             seed (int): Environment seed.
#         """
#         self.env_id = env_id
#         self.num_envs = num_envs
#         self.run_name = run_name
#         self.human_render = human_render
#         # self.seed = seed
#         self.max_steps_per_episode = max_steps_per_episode
#         self.envs = self._make_vector_envs()

#     def _make_env(self, seed):
#         """
#         Returns a function that, when called, creates a single Flappy Bird environment with a step limit.
#         """

#         def _init():
#             if self.env_id == "FlappyBird-v0":
#                 env = gym.make(
#                     self.env_id, use_lidar=False, render_mode=self.human_render
#                 )
#                 # env = TimeLimit(env, max_episode_steps=self.max_steps_per_episode)
#             elif self.env_id == "TradingEnv":
#                 df = make_features(download_dataset())
#                 env = gym.make(
#                     self.env_id,
#                     name="BTCUSD",
#                     df=df,
#                     positions=[-1, -0.5, 0, 0.5, 1],
#                     trading_fees=0.01 / 100,
#                     borrow_interest_rate=0.0003 / 100,
#                     render_mode=self.human_render,
#                 )
#             elif self.env_id == "MetaDriveEnv":
#                 env = MetaDriveEnv(
#                     dict(
#                         map="C",
#                         # This policy setting simplifies the task
#                         discrete_action=True,
#                         discrete_throttle_dim=3,
#                         discrete_steering_dim=3,
#                         horizon=500,
#                         # scenario setting
#                         random_spawn_lane_index=False,
#                         num_scenarios=1,
#                         start_seed=0,
#                         traffic_density=0,
#                         accident_prob=0,
#                         log_level=50,
#                     )
#                 )
#             else:
#                 env = gym.make(self.env_id)

#             self.action_dim = env.action_space.n
#             self.state_dim = env.observation_space.shape[0]

#             env.reset(seed=seed)
#             return env

#         return _init

#     def _make_vector_envs(self):
#         """Sets up the vectorized environments and initializes the agent."""

#         envs = AsyncVectorEnv([self._make_env(seed=i) for i in range(self.num_envs)])
#         self.envs = envs
#         return envs

#     def reset(self):
#         return self.envs.reset()

#     def step(self, actions: np.ndarray):
#         return self.envs.step(actions)

#     def close(self):
#         self.envs.close()

#     @property
#     def single_observation_space(self):
#         return self.envs.single_observation_space

#     @property
#     def single_action_space(self):
#         return self.envs.single_action_space


# if __name__ == "__main__":
#     env_handler = OurEnvironmentHandler(env_id="MetaDriverEnv", human_render=False)
#     obs = env_handler.reset()
#     print(f"Initial observation: {obs}")
#     action = np.array([0, 1])
#     obs, reward, done, info = env_handler.step(action)
#     print(f"Next observation: {obs}, reward: {reward}, done: {done}")
#     env_handler.close()
#     print("Environment closed.")
