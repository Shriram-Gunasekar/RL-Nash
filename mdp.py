import numpy as np
import mdp
from mdp import GridWorld, PolicyIteration

# Define the game parameters
num_players = 2
num_actions = 2
num_states = num_actions ** num_players

# Define the reward matrix for the Prisoner's Dilemma game
reward_matrix = np.array([[3, 0], [5, 1]])

# Define the transition function for the game
def transition(state, action):
    next_state = [np.random.randint(num_actions) for _ in range(num_players)]
    return next_state

# Define the reward function for the game
def reward(state, action, next_state):
    return reward_matrix[state[0], action] + reward_matrix[state[1], action]

# Define the game environment as a GridWorld MDP
env = GridWorld(num_states=num_states, num_actions=num_actions, transition=transition, reward=reward)

# Use Policy Iteration to find the optimal policy (strategy)
policy_iteration = PolicyIteration(env)
optimal_policy = policy_iteration.run()

# Detect Nash equilibria based on the optimal policy
nash_equilibria = []
for state in range(num_states):
    state_actions = optimal_policy[state]
    nash_condition = all(
        optimal_policy[state, action] >= optimal_policy[state, state_actions]
        for action in range(num_actions) if action != state_actions
    )
    if nash_condition:
        nash_equilibria.append(state)

print("Discovered Nash Equilibria:", nash_equilibria)
