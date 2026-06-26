import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# Define a simple DQN trading agent
class DQNTradingAgent(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQNTradingAgent, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

    def act(self, state):
        q_values = self.forward(state)
        action = torch.argmax(q_values).item()
        return action

# Adversarial attack: FGSM (Fast Gradient Sign Method)
def fgsm_attack(agent, state, epsilon):
    state.requires_grad = True
    q_values = agent(state)
    target_action = torch.argmax(q_values).item()
    loss = -q_values[0, target_action]
    loss.backward()
    perturbed_state = state + epsilon * state.grad.sign()
    return torch.clamp(perturbed_state, 0, 1)  # Ensure state remains valid

# Simulate a trading environment
class TradingEnvironment:
    def __init__(self, price_series):
        self.price_series = price_series
        self.current_step = 0

    def reset(self):
        self.current_step = 0
        return torch.tensor([self.price_series[self.current_step]], dtype=torch.float32)

    def step(self, action):
        self.current_step += 1
        if self.current_step >= len(self.price_series):
            done = True
            reward = 0
        else:
            done = False
            reward = self.price_series[self.current_step] if action == 1 else -self.price_series[self.current_step]
        next_state = torch.tensor([self.price_series[self.current_step]], dtype=torch.float32) if not done else None
        return next_state, reward, done

if __name__ == '__main__':
    # Dummy price series for testing
    price_series = np.sin(np.linspace(0, 10, 100))  # Simulated price data
    env = TradingEnvironment(price_series)

    # Initialize DQN trading agent
    state_dim = 1
    action_dim = 2  # Buy or Sell
    agent = DQNTradingAgent(state_dim, action_dim)
    optimizer = optim.Adam(agent.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Train the agent on the environment
    num_episodes = 10
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.act(state.unsqueeze(0))
            next_state, reward, done = env.step(action)
            if next_state is not None:
                target = reward + 0.99 * torch.max(agent(next_state.unsqueeze(0)))
            else:
                target = torch.tensor(reward, dtype=torch.float32)
            q_values = agent(state.unsqueeze(0))
            loss = criterion(q_values[0, action], target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            state = next_state

    # Test the agent with adversarial attack
    state = env.reset()
    done = False
    total_reward = 0
    epsilon = 0.1  # Perturbation magnitude
    while not done:
        # Apply adversarial attack
        perturbed_state = fgsm_attack(agent, state.unsqueeze(0), epsilon).squeeze(0)
        action = agent.act(perturbed_state.unsqueeze(0))
        next_state, reward, done = env.step(action)
        total_reward += reward
        state = next_state

    print(f"Total reward under adversarial attack: {total_reward}")