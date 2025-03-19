import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# Define a simple 1D environment
class Simple1DEnv:
    def __init__(self, size=10):
        self.size = size
        self.goal = size - 1
        self.state = 0

    def reset(self):
        self.state = 0
        return np.array([self.state], dtype=np.float32)

    def step(self, action):
        # Action: 0 for left, 1 for right
        if action == 0:
            self.state = max(0, self.state - 1)
        elif action == 1:
            self.state = min(self.goal, self.state + 1)
        # Reward: 1 if the goal is reached, otherwise a small negative penalty.
        reward = 1.0 if self.state == self.goal else -0.1
        done = self.state == self.goal
        return np.array([self.state], dtype=np.float32), reward, done, {}

# Hyperparameters
learning_rate = 3e-4
gamma = 0.99           # Discount factor
lmbda = 0.95           # GAE parameter
eps_clip = 0.2         # PPO clipping epsilon
K_epochs = 4           # Number of PPO epochs per update
max_timesteps = 100    # Maximum timesteps per episode
total_timesteps = 5000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Actor-Critic Network with a shared feature layer
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(ActorCritic, self).__init__()
        # Shared network layers
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh()
        )
        # Actor head for action probabilities
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        # Critic head for state value estimation
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        shared_out = self.shared(x)
        action_probs = self.actor(shared_out)
        state_value = self.critic(shared_out)
        return action_probs, state_value

    def act(self, state):
        state = torch.FloatTensor(state).to(device)
        action_probs, state_value = self.forward(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), state_value

    def evaluate(self, state, action):
        action_probs, state_value = self.forward(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        return action_logprobs, torch.squeeze(state_value), dist_entropy

# PPO Agent that uses the ActorCritic network
class PPO:
    def __init__(self, state_dim, action_dim):
        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        # Create a separate copy for the old policy
        self.policy_old = ActorCritic(state_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()

    def update(self, memory):
        # Convert lists to tensors
        states = torch.FloatTensor(np.array(memory.states)).to(device)
        actions = torch.LongTensor(memory.actions).to(device)
        old_logprobs = torch.FloatTensor(memory.logprobs).to(device)
        returns = torch.FloatTensor(memory.returns).to(device)
        advantages = torch.FloatTensor(memory.advantages).to(device)

        for _ in range(K_epochs):
            # Evaluate current policy for given states and actions
            logprobs, state_values, dist_entropy = self.policy.evaluate(states, actions)
            # Calculate the ratio (pi_theta / pi_theta_old)
            ratios = torch.exp(logprobs - old_logprobs)
            # Compute surrogate loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * advantages
            loss_actor = -torch.min(surr1, surr2).mean()
            # Compute value function loss
            loss_critic = self.MseLoss(state_values, returns)
            # Total loss (with an entropy bonus for exploration)
            loss = loss_actor + 0.5 * loss_critic - 0.01 * dist_entropy.mean()

            # Take gradient descent step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Update the old policy with current policy weights
        self.policy_old.load_state_dict(self.policy.state_dict())

# Memory to store trajectories
class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.state_values = []
        self.returns = []
        self.advantages = []

    def clear(self):
        self.actions.clear()
        self.states.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.is_terminals.clear()
        self.state_values.clear()
        self.returns.clear()
        self.advantages.clear()

# Compute returns and advantages using Generalized Advantage Estimation (GAE)
def compute_gae(memory, last_value, gamma=0.99, lmbda=0.95):
    rewards = memory.rewards
    is_terminals = memory.is_terminals
    state_values = memory.state_values + [last_value]
    gae = 0
    returns = []
    advantages = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * state_values[step + 1] * (1 - is_terminals[step]) - state_values[step]
        gae = delta + gamma * lmbda * (1 - is_terminals[step]) * gae
        advantages.insert(0, gae)
        returns.insert(0, gae + state_values[step])
    memory.returns = returns
    memory.advantages = advantages

def main():
    # Create an instance of our custom environment
    env = Simple1DEnv(size=10)
    state_dim = 1   # The state is a single number (position)
    action_dim = 2  # Two possible actions: left or right

    ppo_agent = PPO(state_dim, action_dim)
    memory = Memory()

    timestep = 0
    episode = 0

    while timestep < total_timesteps:
        state = env.reset()
        for t in range(max_timesteps):
            timestep += 1
            # Use the old policy to decide an action
            action, logprob, state_value = ppo_agent.policy_old.act(state)
            next_state, reward, done, _ = env.step(action)

            # Store the transition in memory
            memory.states.append(state)
            memory.actions.append(action)
            memory.logprobs.append(logprob.item())
            memory.rewards.append(reward)
            memory.is_terminals.append(done)
            memory.state_values.append(state_value.item())

            state = next_state

            if done:
                break

        # If the episode didn't end, get the value for the last state; otherwise, it's 0.
        if done:
            last_value = 0
        else:
            _, _, last_value = ppo_agent.policy_old.act(state)
            last_value = last_value.item()

        # Compute returns and advantages using GAE
        compute_gae(memory, last_value, gamma, lmbda)

        # Update the PPO agent using the collected trajectory
        ppo_agent.update(memory)
        memory.clear()
        episode += 1

        if episode % 10 == 0:
            print(f"Episode {episode} \t Timestep {timestep}")

if __name__ == "__main__":
    main()
