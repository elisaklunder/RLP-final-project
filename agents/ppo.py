import flappy_bird_gymnasium  # noqa: F401
import gymnasium
import torch
import torch.nn as nn
import torch.optim as optim
from envs.environment_handler import EnvironmentHandler


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=64):
        super(ActorCritic, self).__init__()
        # Shared base layers
        self.base = self._init_base(state_dim, hidden_size)
        # Actor head
        self.actor = self._init_actor(hidden_size, action_dim)
        # Critic head
        self.critic = self._init_critic(hidden_size)

    def _init_base(self, state_dim, hidden_size):
        """Initialize the shared base layers."""
        return nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh(),
        )

    def _init_actor(self, hidden_size, action_dim):
        """Initialize the actor head."""
        return nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_dim),
            nn.Softmax(dim=-1),  # For discrete actions
        )

    def _init_critic(self, hidden_size):
        """Initialize the critic head."""
        return nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        base_out = self.base(x)
        action_probs = self.actor(base_out)
        state_value = self.critic(base_out)
        return action_probs, state_value

def collect_trajectories(env_handler, policy_net, t_max, device):
    states = []
    actions = []
    rewards = []
    dones = []
    log_probs = []
    state_values = []

    state, _ = env_handler.reset()
    for t in range(t_max):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        action_probs, state_value = policy_net(state_tensor)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        next_state, reward, done, _, _ = env_handler.step(action.item())

        states.append(state)
        actions.append(action.item())
        rewards.append(reward)
        dones.append(done)
        log_probs.append(log_prob)
        state_values.append(state_value)

        if done:
            state, _ = env_handler.reset()
        else:
            state = next_state

    return {
        "states": states,
        "actions": actions,
        "rewards": rewards,
        "dones": dones,
        "log_probs": log_probs,
        "state_values": state_values,
    }


def ppo_loss(old_log_probs, new_log_probs, advantages, clip_epsilon):
    ratio = torch.exp(new_log_probs - old_log_probs)
    clipped_ratio = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) # Our change goes here
    policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
    return policy_loss


def value_loss(returns, state_values):
    return nn.MSELoss()(state_values.squeeze(-1), returns)


def compute_gae(rewards, dones, state_values, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0
    returns = []
    state_values = state_values + [0]  # Append 0 for the last state value
    for t in reversed(range(len(rewards))):
        delta = (
            rewards[t] + gamma * state_values[t + 1] * (1 - dones[t]) - state_values[t]
        )
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)
        returns.insert(0, gae + state_values[t])
    return advantages, returns


class PPOAgent:
    def __init__(self, env_handler, total_timesteps=100000, device="cpu"):
        self.env_handler = env_handler
        self.total_timesteps = total_timesteps
        self.device = device

        # Get state and action dimensions
        obs_space = self.env_handler.env.observation_space
        act_space = self.env_handler.env.action_space

        self.state_dim = obs_space.shape[0]
        if isinstance(act_space, gymnasium.spaces.Discrete):
            self.action_dim = act_space.n
        else:
            raise NotImplementedError(
                "Only discrete action spaces are supported."
            )

        self.policy_net = ActorCritic(self.state_dim, self.action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=3e-4)

    def train(self, num_epochs=50, t_max=2048, batch_size=64, clip_epsilon=0.2):
        for epoch in range(num_epochs):
            trajectories = collect_trajectories(
                env_handler, self.policy_net, t_max, self.device
            )

            # Convert lists to tensors
            states = torch.FloatTensor(trajectories["states"]).to(self.device)
            actions = torch.LongTensor(trajectories["actions"]).to(self.device)
            rewards = trajectories["rewards"]
            dones = trajectories["dones"]
            old_log_probs = (
                torch.stack(trajectories["log_probs"]).detach().to(self.device)
            )
            state_values = [
                sv.detach().cpu().item() for sv in trajectories["state_values"]
            ]

            # Compute advantages and returns
            advantages, returns = compute_gae(rewards, dones, state_values)
            advantages = torch.FloatTensor(advantages).to(self.device)
            returns = torch.FloatTensor(returns).to(self.device)

            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Update policy and value network
            dataset = torch.utils.data.TensorDataset(
                states, actions, old_log_probs, returns, advantages
            )
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=True
            )

            for _ in range(4):  # Number of epochs per batch
                for batch in dataloader:
                    b_states, b_actions, b_old_log_probs, b_returns, b_advantages = (
                        batch
                    )

                    action_probs, state_values = self.policy_net(b_states)
                    dist = torch.distributions.Categorical(action_probs)
                    new_log_probs = dist.log_prob(b_actions)

                    # Compute losses
                    policy_loss = ppo_loss(
                        b_old_log_probs, new_log_probs, b_advantages, clip_epsilon
                    )
                    v_loss = value_loss(b_returns, state_values)

                    loss = policy_loss + 0.5 * v_loss

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            print(f"Epoch {epoch + 1}/{num_epochs} completed.")

    def evaluate(self, episodes=10):
        rewards = []
        for episode in range(episodes):
            obs, _ = self.env_handler.reset()
            episode_reward = 0
            done = False
            while not done:
                state_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    action_probs, _ = self.policy_net(state_tensor)
                dist = torch.distributions.Categorical(action_probs)
                action = dist.sample().item()
                obs, reward, done, _, _ = self.env_handler.step(action)
                episode_reward += reward
            rewards.append(episode_reward)
            print(f"Episode {episode + 1}: Reward = {episode_reward}")
        return rewards

    def save(self, path="ppo_model.pth"):
        torch.save(self.policy_net.state_dict(), path)
        print(f"Model saved to {path}.")

    def load(self, path="ppo_model.pth"):
        self.policy_net.load_state_dict(torch.load(path))
        print(f"Model loaded from {path}.")


if __name__ == "__main__":
    import flappy_bird_gymnasium  # noqa: F401
    import gymnasium

    env_handler = EnvironmentHandler(env_type="FlappyBird", human_render=False)
    agent = PPOAgent(env_handler, device="cpu")  # Use 'cuda' if you have a GPU

    agent.train(num_epochs=10000, t_max=2048, batch_size=64, clip_epsilon=0.2)

    # Evaluate the trained agent
    agent.evaluate(episodes=5)

    # Save the model
    agent.save("ppo_flappy_bird.pth")
