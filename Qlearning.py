import numpy as np

# Define the game parameters
num_players = 2
num_actions = 2
num_states = num_actions ** num_players

# Define the Q-table to store Q-values for state-action pairs
Q = np.zeros((num_states, num_actions))

# Define the reward matrix for the Prisoner's Dilemma game
reward_matrix = np.array([[3, 0], [5, 1]])

# Define the learning parameters
learning_rate = 0.1
discount_factor = 0.9
num_episodes = 1000
epsilon = 0.1  # Exploration vs. exploitation trade-off

# Helper function to convert state to index
def state_to_index(state):
    return state[0] * num_actions + state[1]

# Q-learning algorithm
for episode in range(num_episodes):
    # Initialize the game state
    state = [np.random.randint(num_actions) for _ in range(num_players)]
    state_index = state_to_index(state)

    done = False
    while not done:
        # Exploration vs. exploitation
        if np.random.rand() < epsilon:
            action = np.random.randint(num_actions)
        else:
            action = np.argmax(Q[state_index])

        # Simulate the game and get rewards
        next_state = [np.random.randint(num_actions) for _ in range(num_players)]
        next_state_index = state_to_index(next_state)
        rewards = reward_matrix[state[0], action], reward_matrix[state[1], action]

        # Update Q-value using Q-learning formula
        Q[state_index, action] += learning_rate * (
            rewards[state[0]] + discount_factor * np.max(Q[next_state_index]) - Q[state_index, action]
        )

        # Transition to the next state
        state = next_state
        state_index = next_state_index

        # Check if the game is over
        done = True  # Assuming a single-step game for simplicity

# Detect Nash equilibria based on Q-values
nash_equilibria = []
for state in range(num_states):
    state_actions = np.argmax(Q[state])
    nash_condition = all(
        Q[state, action] >= Q[state, state_actions] for action in range(num_actions) if action != state_actions
    )
    if nash_condition:
        nash_equilibria.append(state)

print("Discovered Nash Equilibria:", nash_equilibria)
