import gymnasium as gym
import numpy as np
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


def test_trained_agent(render=True):
    """
    Test the trained hospital robot navigation agent

    Args:
        render (bool): Whether to render the environment
    """
    # Load the trained Q-table
    with open("hospital_robot_q_table.pkl", "rb") as f:
        q_table = pickle.load(f)

    # Create environment for testing
    render_mode = "human" if render else None
    env = HospitalRobotEnv(render_mode=render_mode)

    # Run a few test episodes
    for _ in range(30):
        observation, _ = env.reset()
        state = state_to_discrete(observation, env.grid_size)

        terminated = False
        truncated = False
        total_reward = 0

        while not (terminated or truncated):
            # Choose best action
            action = np.argmax(q_table[state, :])

            # Take action
            next_observation, reward, terminated, truncated, _ = env.step(action)

            # Update state
            state = state_to_discrete(next_observation, env.grid_size)
            total_reward += reward

            # Render the environment if required
            if render:
                env.render()

        print(f"Test Episode Total Reward: {total_reward}")

    env.close()


if __name__ == "__main__":
    # Test the trained agent
    test_trained_agent()