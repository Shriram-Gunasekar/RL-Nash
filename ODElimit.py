import torch
import torch.nn as nn
import torch.optim as optim
import torchdiffeq

# Define the game parameters
num_players = 2
num_actions = 2
num_states = num_actions ** num_players

# Define the reward matrix for the Prisoner's Dilemma game
reward_matrix = torch.tensor([[3, 0], [5, 1]], dtype=torch.float32)

# Define the dynamic system (ODE) for the game
class GameODE(nn.Module):
    def __init__(self):
        super(GameODE, self).__init__()

    def forward(self, t, y):
        # y represents the state of the game
        # t is the time (not used in this example)
        # We use a simple linear dynamics for illustration
        return torch.matmul(y, reward_matrix.T)

# Define the Neural ODE model for solving the dynamic system
game_model = GameODE()
neural_ode = torchdiffeq.odeint_adjoint.odeint_adjoint

# Define the loss function (negative sum of rewards for optimization)
def loss_fn(state):
    return -torch.sum(torch.matmul(state, reward_matrix.T))

# Initialize the initial state of the game
initial_state = torch.ones(num_states, dtype=torch.float32) / num_states

# Use an optimizer to find the optimal control policy (strategy)
optimizer = optim.Adam(game_model.parameters(), lr=0.01)

# Training loop to find the Nash equilibrium
num_epochs = 1000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    state_trajectory = neural_ode(game_model, initial_state, torch.linspace(0, 1, 10))
    loss = loss_fn(state_trajectory[-1])
    loss.backward()
    optimizer.step()

# Get the optimal control policy (Nash equilibrium strategy)
optimal_strategy = state_trajectory[-1].detach().numpy()
nash_equilibria = [tuple(optimal_strategy.argmax(axis=0))]

print("Discovered Nash Equilibria:", nash_equilibria)
