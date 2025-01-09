import gymnasium
import flappy_bird_gymnasium 
from stable_baselines3.common.env_util import make_vec_env

class EnvironmentHandler:
    def __init__(self, env_type: str, human_render: bool = False, num_envs: int = 1):
        self.env_type = env_type
        self.num_envs = num_envs
        self.human_render = human_render
        self.env = make_vec_env(lambda: gymnasium.make(env_type), n_envs=num_envs)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def close(self):
        self.env.close()
