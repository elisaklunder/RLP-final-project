import random
import time
from dataclasses import dataclass

# Optional: flappy bird support (if installed)
import flappy_bird_gymnasium  # noqa: F401
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard.writer import SummaryWriter


@dataclass
class PPOConfig:
    exp_name: str = "ppo_experiment"
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    human_render: bool = False

    # Algorithm specific arguments
    env_id: str = "CartPole-v1"
    total_timesteps: int = 100_000
    learning_rate: float = 0.0005
    num_envs: int = 4
    num_steps: int = 100
    gamma: float = 0.95
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 10
    norm_adv: bool = True
    clip_coef: float = 0.2
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

class EnvironmentHandler:
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
            if self.human_render:
                env = gym.make(env_id, render_mode="human", use_lidar=False)
            else:
                env = gym.make(env_id, use_lidar=False)
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


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, observation_space, action_space):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, action_space.n), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), self.critic(x)


class PPOTrainer:
    def __init__(self, config: PPOConfig):
        self.config = config

        # Seeding
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        torch.backends.cudnn.deterministic = self.config.torch_deterministic

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and self.config.cuda else "cpu"
        )

        # Create the run name
        self.run_name = f"{self.config.env_id}__{self.config.exp_name}__{self.config.seed}__{int(time.time())}"
        self.writer = SummaryWriter(f"logs/tensorboard/{self.config.env_id}/{self.run_name}")
        self._write_hyperparams()

        # Setup environment
        self.env_handler = EnvironmentHandler(
            env_id=self.config.env_id,
            num_envs=self.config.num_envs,
            run_name=self.run_name,
            seed=self.config.seed,
            human_render=self.config.human_render,
        )

        # Check discrete action space
        assert isinstance(
            self.env_handler.single_action_space, gym.spaces.Discrete
        ), "Only discrete action space is supported"

        # Setup agent and optimizer
        self.agent = Agent(
            self.env_handler.single_observation_space,
            self.env_handler.single_action_space,
        ).to(self.device)

        self.optimizer = optim.Adam(
            self.agent.parameters(), lr=self.config.learning_rate, eps=1e-5
        )

        # Prepare buffers
        self.batch_size = self.config.num_envs * self.config.num_steps
        self.minibatch_size = self.batch_size // self.config.num_minibatches
        self.num_iterations = self.config.total_timesteps // self.batch_size

        self.obs = torch.zeros(
            (self.config.num_steps, self.config.num_envs)
            + self.env_handler.single_observation_space.shape
        ).to(self.device)
        self.actions = torch.zeros(
            (self.config.num_steps, self.config.num_envs)
            + self.env_handler.single_action_space.shape
        ).to(self.device)
        self.logprobs = torch.zeros((self.config.num_steps, self.config.num_envs)).to(
            self.device
        )
        self.rewards = torch.zeros((self.config.num_steps, self.config.num_envs)).to(
            self.device
        )
        self.dones = torch.zeros((self.config.num_steps, self.config.num_envs)).to(
            self.device
        )
        self.values = torch.zeros((self.config.num_steps, self.config.num_envs)).to(
            self.device
        )

    def _write_hyperparams(self):
        params_str = "|param|value|\n|-|-|\n"
        for key, value in vars(self.config).items():
            params_str += f"|{key}|{value}|\n"
        self.writer.add_text("hyperparameters", params_str)

    def train(self):
        global_step = 0
        start_time = time.time()
        next_obs, _ = self.env_handler.reset()
        next_obs = torch.Tensor(next_obs).to(self.device)
        next_done = torch.zeros(self.config.num_envs).to(self.device)

        for iteration in range(1, self.num_iterations + 1):

            # Temporary storages to compute mean and std error for episodes ended in this iteration
            iteration_episode_returns = []
            iteration_episode_lengths = []

            for step in range(0, self.config.num_steps):
                global_step += self.config.num_envs
                self.obs[step] = next_obs
                self.dones[step] = next_done

                with torch.no_grad():
                    action, logprob, value = self.agent.get_action_and_value(
                        next_obs
                    )
                    self.values[step] = value.flatten()

                self.actions[step] = action
                self.logprobs[step] = logprob

                # Step environment
                next_obs_arr, reward_arr, terminated_arr, truncated_arr, infos = (
                    self.env_handler.step(action.cpu().numpy())
                )
                next_done = np.logical_or(terminated_arr, truncated_arr)
                self.rewards[step] = torch.tensor(reward_arr).to(self.device).view(-1)
                next_obs = torch.Tensor(next_obs_arr).to(self.device)
                next_done = torch.Tensor(next_done).to(self.device)

                # Logging episode returns

                if "episode" in infos:
                    # Get completed episodes
                    completed_episodes = infos["episode"]["_r"]
                    if completed_episodes.any():
                        # Get returns and lengths for completed episodes
                        returns = infos["episode"]["r"][completed_episodes]
                        lengths = infos["episode"]["l"][completed_episodes]
                        
                        # Add to iteration storage
                        iteration_episode_returns.extend(returns.tolist())
                        iteration_episode_lengths.extend(lengths.tolist())

            # Perform PPO update
            self._update(global_step, start_time, next_obs, next_done)

            # After updating, log mean and standard error of episodes ended in this iteration
            if len(iteration_episode_returns) > 0:
                mean_return = np.mean(iteration_episode_returns)
                mean_length = np.mean(iteration_episode_lengths)
                self.writer.add_scalar("rollout/ep_rew_mean", mean_return, global_step)
                self.writer.add_scalar("rollout/ep_len_mean", mean_length, global_step)

    def _log_episode_return(self, global_step, info):
        self.writer.add_scalar(
            "rollout/ep_rew_mean", info["episode"]["r"], global_step
        )
        self.writer.add_scalar(
            "rollout/ep_len_mean", info["episode"]["l"], global_step
        )

    def _update(self, global_step, start_time, next_obs, next_done):
        # Bootstrap value
        with torch.no_grad():
            next_value = self.agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(self.rewards).to(self.device)
            lastgaelam = 0
            for t in reversed(range(self.config.num_steps)):
                if t == self.config.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - self.dones[t + 1]
                    nextvalues = self.values[t + 1]
                delta = (
                    self.rewards[t]
                    + self.config.gamma * nextvalues * nextnonterminal
                    - self.values[t]
                )
                advantages[t] = lastgaelam = (
                    delta
                    + self.config.gamma
                    * self.config.gae_lambda
                    * nextnonterminal
                    * lastgaelam
                )
            returns = advantages + self.values

        # Flatten the batch
        b_obs = self.obs.reshape(
            (-1,) + self.env_handler.single_observation_space.shape
        )
        b_logprobs = self.logprobs.reshape(-1)
        b_actions = self.actions.reshape(
            (-1,) + self.env_handler.single_action_space.shape
        )
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = self.values.reshape(-1)

        # Optimize policy and value network
        b_inds = np.arange(self.batch_size)
        clipfracs = []
        for epoch in range(self.config.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, self.batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, newvalue = self.agent.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > self.config.clip_coef)
                        .float()
                        .mean()
                        .item()
                    ]

                mb_advantages = b_advantages[mb_inds]
                if self.config.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - self.config.clip_coef, 1 + self.config.clip_coef
                ) # OUR MODIFICATION GOES HERE
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                loss = (pg_loss + v_loss * self.config.vf_coef)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.agent.parameters(), self.config.max_grad_norm
                )
                self.optimizer.step()

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # Logging
        self.writer.add_scalar("train/value_loss", v_loss.item(), global_step)
        self.writer.add_scalar("train/policy_gradient_loss", pg_loss.item(), global_step)
        self.writer.add_scalar("train/clip_fraction", np.mean(clipfracs), global_step)
        self.writer.add_scalar("train/explained_variance", explained_var, global_step)

    def save_model(self, path: str):
        torch.save({
            'agent_state_dict': self.agent.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
        }, path)
        print(f"Model saved to {path}")

    def load_model(self, path: str):
        checkpoint = torch.load(path)
        self.agent.load_state_dict(checkpoint['agent_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.config = checkpoint['config']
        print(f"Model loaded from {path}")

    def evaluate(self, num_episodes: int = 50):
        # Create a single environment instead of a vectorized one
        eval_env = gym.make(self.config.env_id, render_mode="human", use_lidar=False)
        eval_env = gym.wrappers.RecordEpisodeStatistics(eval_env)
        
        for ep in range(num_episodes):
            obs, _ = eval_env.reset()
            terminated = truncated = False

            while not (terminated or truncated):
                obs_tensor = torch.Tensor(obs)
                action, _, _ = self.agent.get_action_and_value(obs_tensor)  
                action_value = action.numpy().item()
                obs, reward, terminated, truncated, _ = eval_env.step(action_value)
                    
        eval_env.close()

if __name__ == "__main__":
    # config = PPOConfig(total_timesteps=100_000, 
    #                    env_id="FlappyBird-v0", 
    #                    learning_rate=0.0005, 
    #                    num_steps=100, 
    #                    update_epochs=10, 
    #                    gamma=0.95)
                       
    # trainer = PPOTrainer(config)
    # trainer.train()
    # trainer.save_model("logs/checkpoints/clean_ppo_model_flappybird.pth")
    
    config = PPOConfig(env_id="FlappyBird-v0", human_render=True)
    trainer = PPOTrainer(config)   
    trainer.load_model("logs/checkpoints/clean_ppo_model_flappybird.pth")
    trainer.evaluate()