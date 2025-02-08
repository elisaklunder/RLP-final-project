import os
from functools import partial

from IPython.display import Image, clear_output
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv

from metadrive.envs.metadrive_env import MetaDriveEnv

set_random_seed(0)

# class MyMapManager(BaseManager):
#     PRIORITY = 0

#     def __init__(self):
#         super(MyMapManager, self).__init__()
#         self.current_map = None
#         self.all_maps = {idx: None for idx in range(3)}  # store the created map
#         self._map_shape = ["S", "C", "X"]  # three types of maps

#     def reset(self):
#         idx = self.engine.global_random_seed % 3
#         if self.all_maps[idx] is None:
#             # create maps on the fly
#             new_map = PGMap(
#                 map_config=dict(
#                     type=PGMap.BLOCK_SEQUENCE, config=self._map_shape[idx], lane_num=3
#                 )
#             )
#             self.all_maps[idx] = new_map

#         # attach map in the world
#         map = self.all_maps[idx]
#         map.attach_to_world()
#         self.current_map = map
#         return dict(current_map=self._map_shape[idx])

#     def before_reset(self):
#         if self.current_map is not None:
#             self.current_map.detach_from_world()
#             self.current_map = None

#     def destroy(self):
#         # clear all maps when this manager is destroyed
#         super(MyMapManager, self).destroy()
#         for map in self.all_maps.values():
#             if map is not None:
#                 map.destroy()
#         self.all_maps = None


# # Specify where to spawn the car
# MY_CONFIG = dict(
#     agent_configs={
#         "default_agent": dict(
#             spawn_lane_index=(FirstPGBlock.NODE_1, FirstPGBlock.NODE_2, 0),
#             destination=(FirstPGBlock.NODE_1, FirstPGBlock.NODE_2, 0),
#             show_navi_mark=True,
#         )
#     },
# )


# class MyEnv(BaseEnv):
#     @classmethod
#     def default_config(cls):
#         config = super(MyEnv, cls).default_config()
#         config.update(MY_CONFIG)
#         return config

#     def setup_engine(self):
#         super(MyEnv, self).setup_engine()
#         self.engine.register_manager("map_manager", MyMapManager())

#     def reward_function(self, agent):
#         return 0, {}

#     def cost_function(self, agent):
#         return 0, {}

#     def done_function(self, agent):
#         return False, {}

#     def get_single_observation(self):
#         return DummyObservation()


def create_env(shape: str = "S", need_monitor=False):
    env = MetaDriveEnv(
        dict(
            map=shape,
            # This policy setting simplifies the task
            discrete_action=True,
            discrete_throttle_dim=3,
            discrete_steering_dim=3,
            horizon=500,
            # scenario setting
            random_spawn_lane_index=False,
            num_scenarios=1,
            start_seed=0,
            traffic_density=0,
            accident_prob=0,
            log_level=50,
        )
    )
    if need_monitor:
        env = Monitor(env)
    return env


if __name__ == "__main__":
    frames_to_save = []

    # Create the environment
    train_env = SubprocVecEnv([partial(create_env, True) for _ in range(4)])
    model = PPO("MlpPolicy", train_env, n_steps=4096, verbose=1)
    model.learn(300_000, log_interval=4)

    clear_output()
    print("Training is finished! Generate gif ...")

    total_reward = 0
    env = create_env()
    obs, _ = env.reset()
    try:
        for i in range(1000):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            total_reward += reward
            ret = env.render(
                mode="topdown",
                screen_record=True,
                window=False,
                screen_size=(600, 600),
                camera_position=(50, -50),
            )
            if done:
                print("episode_reward", total_reward)
                break

        env.top_down_renderer.generate_gif()
    finally:
        env.close()
    print("gif generation is finished ...")

    Image(open("demo.gif", "rb").read())
