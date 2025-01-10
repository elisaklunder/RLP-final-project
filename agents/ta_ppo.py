import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import flappy_bird_gymnasium
import gymnasium as gym
import numpy as np
from collections import deque
from typing import Tuple, List
import csv
torch.manual_seed(0)

class ActorCritic(nn.Module):
    """
    Define the actor-critic architecture
    Actor: Neural network -> Maps a state to a probability distribution over the action space (defines the policy)
    Critic: Neural network -> Maps a state to a single scalar (value function)
    """

    def __init__(self, state_space: int, action_space: int) -> None:
        super().__init__()

        self.actor = nn.Sequential(
            nn.Linear(state_space, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_space),
            nn.Softmax(dim=-1)
        )

        self.critic = nn.Sequential(
            nn.Linear(state_space, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def act(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select an action in state s by sampling from the policy conditioned on s.
        This function will be used to interact with the environment and gather experiences
        Input: 
            state -> tensor of shape [batch_size, state_dim]
        Output:
            action -> tensor of shape [batch_size], type long, the index of the chosen action
            log_prob -> tensor of shape [batch_size], the log probability of the chosen action
            state_value -> tensor of shape [batch_size], the critic's prediction for the value function at that state
        """

        with torch.inference_mode():
            action_probs: torch.Tensor = self.actor(state)
            state_value: torch.Tensor = self.critic(state).squeeze(1)
        dist = Categorical(probs=action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, state_value

    def act_deterministically(self, state: torch.Tensor) -> torch.Tensor:
        """
        Select the optimal action in state s defined by `a = argmax pi(a|s)`
        This function will be used to evaluate the agent during training.
        Input:
            state -> tensor of shape [batch_size, state_dim]
        Output:
            action -> tensor of shape [batch_size], type long, the index of the best action in that state
        """

        with torch.inference_mode():
            action_probs: torch.Tensor = self.actor(state)
        return action_probs.argmax(dim=-1)

    def evaluate(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Given a state and the action taken by the 'old' policy, evaluate the action using the current policy.
        This function will be used to train the agent.
        Input:
            state -> tensor of shape [batch_size, state_dim]
            action -> tensor of shape [batch_size], type long, the actions taken in the given states
        Output:
            log_prob -> tensor of shape [batch_size], the log probability of the old action for that state
            according to the new policy
            state_values -> tensor of shape [batch_size], the estimate for the value function in the state
            according to the critic
            entropy -> tensor of shape [batch_size], the entropy of the policy for that state (used for regularization)
        """

        action_probs = self.actor(state)
        state_value = self.critic(state).squeeze(1)
        dist = Categorical(probs=action_probs)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, state_value, entropy

class RolloutBuffer:
    """
    Memory buffer, will be used to store the agent's experiences from the interaction stage.
    We will be learning from them in the training stage.
    It's tuples (s, a, r, log(pi(a|s)), V(s)), and an indicator for whether `s` is terminal
    """

    def __init__(self) -> None:
        self.states: List[torch.Tensor] = []
        self.actions: List[torch.Tensor] = []
        self.log_probs: List[torch.Tensor] = []
        self.rewards: List[float] = []
        self.state_values: List[torch.Tensor] = []
        self.dones: List[bool] = []

    def clear(self) -> None:
        self.__init__()

class Agent:
    """The agent that we are going to train with PPO"""

    def __init__(self, state_space: int, action_space: int) -> None:
        self.buffer = RolloutBuffer()
        self.policy = ActorCritic(state_space, action_space)
        self.optimizer = torch.optim.Adam(params=self.policy.parameters(), lr=1e-4)

    def choose_action(self, state: np.ndarray) -> np.ndarray:
        """
        Interact with the environment. Store the experience
        Input:
            state: given from env.step()
        Return:
            action: to be passed into env.step() again
        """

        state: torch.Tensor = torch.from_numpy(state).float().unsqueeze(0)
        action, log_prob, state_value = self.policy.act(state=state)

        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.log_probs.append(log_prob)
        self.buffer.state_values.append(state_value)

        return action.numpy()[0]

    def choose_action_deterministic(self, state: np.ndarray) -> np.ndarray:
        """
        Interact with the environment deterministically. Same structure as `choose_action`
        """

        state: torch.Tensor = torch.from_numpy(state).float().unsqueeze(0)
        action = self.policy.act_deterministically(state=state)
        return action.numpy()[0]

    def store_outcome(self, reward: float, done: bool) -> None:
        """
        After interacting with the environment, store the reward and the terminal indicator to the experiences.
        Call this after each env.step() if `choose_action` is used.
        """

        self.buffer.rewards.append(reward)
        self.buffer.dones.append(done)

    def learn(self) -> None:
        """
        Train the agent!!
        """

        # Get the experiences. Let n be the number of interactions with the environment
        old_states = torch.cat(self.buffer.states) # [n, state_dim]
        old_actions = torch.cat(self.buffer.actions) # [n]
        old_log_probs = torch.cat(self.buffer.log_probs) # [n]
        old_state_values = torch.cat(self.buffer.state_values) # [n]

        # Callculate the return for each state in linear time
        returns = deque(maxlen=len(old_states))
        discounted_return = 0
        for reward, done in zip(reversed(self.buffer.rewards), reversed(self.buffer.dones)):
            if done: discounted_return = reward
            discounted_return = reward + .99 * discounted_return
            returns.appendleft(discounted_return)

        returns = torch.tensor(returns).float() # [n]
        # Standardize returns by taking the z-score
        # returns = (returns - returns.mean()) / (returns.std() + torch.finfo().eps)

        # Compute advantages (return - V(s)) using the critic
        advantages = returns - old_state_values # [n]
        # Actually standardizing the advantages instead seems to be better, the policy is more stable
        # When standardizing the returns it happens that the agent learns a bit (~return of 12) 
        # and then collapses to baseline
        advantages = (advantages - advantages.mean()) / (advantages.std() + torch.finfo().eps)

        # Update the policy a few times
        for epoch in range(10):

            # Get the new policy's predictions
            #  [n]          [n]         [n]
            log_probs, state_values, entropy = self.policy.evaluate(state=old_states, action=old_actions)

            # Compute the ratio pi(s) / pi_old(s)
            # Since we are working with log probabilities, use the fact that
            # pi(s) / pi_old(s) = e^(log(pi(s)) - log(pi_old(s)))
            # Here we are using that log(a/b) = log(a) - log(b)
            # and that x = e^(log(x))
            ratios = (log_probs - old_log_probs).exp() # [n]

            # Compute the 2 components of the surrogate loss. r*A and clip(r*A)
            surr1 = ratios * advantages # [n]
            surr2 = ratios.clip(.8, 1.2) * advantages # [n]

            # PPO's loss function
            loss = -(
                torch.min(surr1, surr2) +
                .5 * F.mse_loss(state_values, returns) -
                .01 * entropy
            )

            # Update
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Wipe the buffer
        self.buffer.clear()

def evaluate_agent(agent: Agent, env: gym.Env) -> Tuple[float, float]:
    """
    Evaluate the agent by running it for a few episodes with full exploitation
    Store the average return and average episode length
    """

    returns = []
    lengths = []
    for i in range(10):
        state, _ = env.reset()
        done = False
        current_return = 0
        episode_length = 0
        while not done:
            episode_length += 1
            action = agent.choose_action_deterministic(state=state)
            state, reward, terminated, truncated, _ = env.step(action)
            current_return += reward
            done = terminated or truncated
        returns.append(current_return)
        lengths.append(episode_length)
    return np.mean(returns), np.mean(episode_length)

def train() -> None:
    """
    Train an agent on Flappy Bird!
    """

    env = gym.make('FlappyBird-v0', use_lidar=False)
    file = open('returns.csv', 'w')
    writer = csv.writer(file)
    writer.writerow(['PPO_Returns'])
    agent = Agent(env.observation_space.shape[0], env.action_space.n)

    # Perform that many updates
    for update in range(2000):

        # For each update, gather experiences from that many full episodes
        # TODO: It would be way faster to not have 200 episodes of a single environment
        # but instead 1 episode in 200 environments ;))
        # The agent can act in all of them with 1 forward pass. Be really careful how the experiences
        # are stored and how the returns are calculated
        for episode in range(200):
            state, _ = env.reset()
            done = False
            while not done:
                action = agent.choose_action(state=state)
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                agent.store_outcome(reward=reward, done=done)

        # Do the update
        agent.learn()

        # Perform the evaluation
        r, l = evaluate_agent(agent=agent, env=env)
        print(f'Update: {update} | Average Return: {r:.4f} | Average Episode Length: {l:.4f}', flush=True)
        writer.writerow([r])
    file.close()

if __name__ == '__main__':
    train()