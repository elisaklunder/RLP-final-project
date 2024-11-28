import flappy_bird_gymnasium  # noqa: F401
import gymnasium


class EnvironmentHandler:
    def __init__(self, env_type, human_render: bool = True, seed: int = 42):
        """
        Args:
        - environment: The environment instance (FlppyBird or FinRL).
        - seed: random seed for initial resetting of the environment.
        """
        self.env = None
        self.env_type = env_type
        self.seed = seed

        if self.env_type == "FlappyBird":
            if human_render:
                self.env = gymnasium.make(
                    "FlappyBird-v0", render_mode="human", use_lidar=False
                )
            else:
                self.env = gymnasium.make("FlappyBird-v0", use_lidar=False)
            self.env.reset(seed=self.seed)

    def reset(self, seed=None):
        """
        Reset the environment and return the initial observation.
        """
        if self.env_type == "FlappyBird":
            if seed:
                self.seed = seed
            return self.env.reset(seed=self.seed)

    def step(self, action):
        """
        Perform a step in the environment with the given action.
        """
        if self.env_type == "FlappyBird":
            return self.env.step(action)

    def close(self):
        """
        Close the environment.
        """
        if hasattr(self.env, "close"):
            self.env.close()


if __name__ == "__main__":
    env_handler = EnvironmentHandler(env_type="FlappyBird")

    # Run a random action loop
    while True:
        # Sample a random action
        action = env_handler.env.action_space.sample()

        # Step through the environment
        obs, reward, terminated, _, info = env_handler.step(action)

        # Check if the game is over
        if terminated:
            break

    # Close the environment
    env_handler.close()
