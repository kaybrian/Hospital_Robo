import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
import random

# Custom environment import
from hospital_robot_env import HospitalRobotEnv


def state_to_discrete(state, grid_size):
    """
    Convert continuous state to a discrete representation

    Args:
        state (np.ndarray): Continuous state representation
        grid_size (int): Size of the grid

    Returns:
        int: Discrete state representation
    """
    # Flatten the state and create a unique index
    flattened_state = state.flatten()

    # Create a unique index based on the flattened state
    # Ensure the state is within grid bounds
    discrete_state = tuple(
        min(max(0, int(coord)), grid_size - 1)
        for coord in flattened_state
    )

    # Use a hash or unique mapping technique
    return hash(discrete_state) % (grid_size ** 6)


def run_training(episodes=10000, grid_size=10, is_training=True, render=False):
    """
    Train the Hospital Robot Navigation environment using Q-learning

    Args:
        episodes (int): Number of training episodes
        grid_size (int): Size of the grid environment
        is_training (bool): Whether to train or test
        render (bool): Whether to render the environment
    """
    # Create the environment
    env = HospitalRobotEnv()

    # Q-learning hyperparameters
    learning_rate = 1.0
    discount_factor = 0.9
    epsilon = 1.0  # Exploration rate
    epsilon_decay_rate = 0.0005
    min_epsilon = 0.01

    # Initialize Q-table (use a large number of potential states)
    num_states = grid_size ** 6  # Increased state space to accommodate more combinations
    num_actions = env.action_space.n

    if is_training:
        # Initialize Q-table with zeros
        q_table = np.zeros((num_states, num_actions))
    else:
        # Load pre-trained Q-table
        try:
            with open("hospital_robot_q_table.pkl", "rb") as f:
                q_table = pickle.load(f)
        except FileNotFoundError:
            print("No pre-trained Q-table found. Initializing a new one.")
            q_table = np.zeros((num_states, num_actions))

    # Tracking rewards
    rewards_per_episode = []

    # Training loop
    for episode in range(episodes):
        # Reset the environment
        observation, _ = env.reset()

        # Convert initial state to discrete
        state = state_to_discrete(observation, grid_size)

        total_reward = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            # Epsilon-greedy action selection
            if is_training and random.random() < epsilon:
                # Explore: choose a random action
                action = env.action_space.sample()
            else:
                # Exploit: choose the best action from Q-table
                action = np.argmax(q_table[state, :])

            # Take action
            next_observation, reward, terminated, truncated, _ = env.step(action)

            # Convert next state to discrete
            next_state = state_to_discrete(next_observation, grid_size)

            # Q-learning update
            if is_training:
                # Q-value update formula
                best_next_action = np.argmax(q_table[next_state, :])
                q_table[state, action] = q_table[state, action] + learning_rate * (
                        reward + discount_factor * q_table[next_state, best_next_action] - q_table[state, action]
                )

            # Update state and total reward
            state = next_state
            total_reward += reward

        # Store rewards
        rewards_per_episode.append(total_reward)

        # Decay epsilon
        epsilon = max(min_epsilon, epsilon - epsilon_decay_rate)

        # Reduce learning rate over time
        if epsilon < 0.1:
            learning_rate = 0.01

        # Print progress
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards_per_episode[-100:])
            print(f"Episode {episode + 1}/{episodes}, Avg Reward: {avg_reward:.2f}, Epsilon: {epsilon:.2f}")

    # Visualization of rewards
    if is_training:
        plt.figure(figsize=(10, 5))
        plt.plot(range(len(rewards_per_episode)), rewards_per_episode)
        plt.title('Rewards per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.savefig('hospital_robot_rewards.png')
        plt.close()

        # Save Q-table
        with open("hospital_robot_q_table.pkl", "wb") as f:
            pickle.dump(q_table, f)
        print("Q-table saved successfully!")

    # Close the environment
    env.close()

    return q_table

if __name__ == "__main__":
    # Train the agent
    run_training(episodes=100000, grid_size=10, is_training=True, render=False)
