import torch
import torch.nn as nn
import torch.optim as optim

# Define the game parameters
num_players = 2
num_actions = 2
num_states = num_actions ** num_players

# Define the payoff matrix for the game (Prisoner's Dilemma)
payoff_matrix = torch.tensor([[3, 0], [5, 1]], dtype=torch.float32)

# Define the utility functions for the agent and adversary
def agent_utility(state, action):
    return torch.sum(payoff_matrix[:, action] * state)

def adversary_utility(state, action):
    return -torch.sum(payoff_matrix[:, action] * state)

# Define the agent and adversary networks
class Agent(nn.Module):
    def __init__(self):
        super(Agent, self).__init__()
        self.fc = nn.Linear(num_states, num_actions)

    def forward(self, state):
        return self.fc(state)

class Adversary(nn.Module):
    def __init__(self):
        super(Adversary, self).__init__()
        self.fc = nn.Linear(num_states, num_actions)

    def forward(self, state):
        return self.fc(state)

# Initialize the agent, adversary, and optimizers
agent = Agent()
adversary = Adversary()
agent_optimizer = optim.Adam(agent.parameters(), lr=0.01)
adversary_optimizer = optim.Adam(adversary.parameters(), lr=0.01)

# Training loop
num_episodes = 1000

for episode in range(num_episodes):
    # Generate a random state for each episode
    state = torch.rand(num_states)

    # Agent's turn: maximize utility
    agent_optimizer.zero_grad()
    agent_action_probs = torch.softmax(agent(state), dim=-1)
    agent_action = torch.multinomial(agent_action_probs, 1)
    agent_util = agent_utility(state, agent_action)
    agent_loss = -agent_util  # Negative utility for maximization
    agent_loss.backward()
    agent_optimizer.step()

    # Adversary's turn: minimize agent's utility
    adversary_optimizer.zero_grad()
    adversary_action_probs = torch.softmax(adversary(state), dim=-1)
    adversary_action = torch.multinomial(adversary_action_probs, 1)
    adversary_util = adversary_utility(state, adversary_action)
    adversary_loss = -adversary_util  # Negative utility for minimization
    adversary_loss.backward()
    adversary_optimizer.step()

# Evaluate the learned strategies to determine Nash equilibria
agent_strategy = torch.softmax(agent(torch.ones(num_states)), dim=-1)
adversary_strategy = torch.softmax(adversary(torch.ones(num_states)), dim=-1)

print("Agent's Strategy:", agent_strategy)
print("Adversary's Strategy:", adversary_strategy)
