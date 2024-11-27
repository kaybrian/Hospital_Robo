import gymnasium as gym
import ale_py
import matplotlib.pyplot as plt
import numpy as np
import pickle
from mpl_toolkits.axisartist.floating_axes import FixedAxisArtistHelper

# entry_point follows the format "module_name:class_name"
gym.register(
    id="HospitalRobot-v0",
    entry_point="hospital_robot_env.hospital_robot_env:HospitalRobotEnv",
    max_episode_steps=200,
)

# Register ALE environments
gym.register_envs(ale_py)


def discretize_state(state):
    # Convert the entire grid state to a hashable tuple
    return tuple(state.flatten())


def run(episodes, is_training=True, render=False):
    env = gym.make("HospitalRobot-v0", render_mode="human" if render else None)

    # Use a dictionary for Q-table instead of numpy array
    if is_training:
        q = {}
    else:
        with open("HospitalRobot.pkl", "rb") as f:
            q = pickle.load(f)

    learning_rate = 0.1  # Adjusted learning rate
    discount_factor = 0.99  # Slightly higher discount factor
    epsilon = 1  # Exploration rate
    epsilon_decay_rate = 0.0005  # Slightly slower decay
    rng = np.random.default_rng()

    # Track more detailed rewards
    total_rewards_per_episode = np.zeros(episodes)
    success_episodes = 0
    total_steps_to_goal = []

    for i in range(episodes):
        # Progress and statistics print
        if i % 100 == 0:
            print(f"Episode {i}/{episodes}")
            if i > 0:
                print(f"Success rate: {success_episodes / 100 * 100:.2f}%")
                print(f"Avg steps to goal: {np.mean(total_steps_to_goal) if total_steps_to_goal else 'N/A'}")
                success_episodes = 0
                total_steps_to_goal = []

        state = env.reset()[0]  # Get the initial state
        state_key = discretize_state(state)
        terminated = False
        truncated = False
        episode_reward = 0
        episode_steps = 0

        while not (terminated or truncated):
            # Initialize Q-value for state if not exists
            if state_key not in q:
                q[state_key] = np.zeros(env.action_space.n)

            # Epsilon-greedy action selection
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample()  # Random action for exploration
            else:
                action = np.argmax(q[state_key])  # Best action according to Q-table

            new_state, reward, terminated, truncated, _ = env.step(action)
            new_state_key = discretize_state(new_state)

            # Initialize Q-value for new state if not exists
            if new_state_key not in q:
                q[new_state_key] = np.zeros(env.action_space.n)

            if is_training:
                # Q-learning update
                q[state_key][action] = q[state_key][action] + learning_rate * (
                        reward + discount_factor * np.max(q[new_state_key]) - q[state_key][action]
                )

            state = new_state
            state_key = new_state_key
            episode_reward += reward
            episode_steps += 1

        # Track episode statistics
        total_rewards_per_episode[i] = episode_reward

        # Check for successful episode (reached goal)
        if reward > 100:  # Assuming reward > 100 indicates reaching the goal
            success_episodes += 1
            total_steps_to_goal.append(episode_steps)

        # Decay epsilon
        epsilon = max(epsilon - epsilon_decay_rate, 0.01)

        # Optionally adjust learning rate
        if epsilon < 0.1:
            learning_rate = 0.01

    # Close the environment
    env.close()

    # Compute moving average of rewards
    window_size = 100
    smoothed_rewards = np.convolve(total_rewards_per_episode, np.ones(window_size) / window_size, mode='valid')

    # Plot the rewards if training
    if is_training:
        plt.figure(figsize=(10, 5))
        plt.plot(smoothed_rewards)
        plt.title('Smoothed Rewards over Episodes')
        plt.xlabel('Episodes')
        plt.ylabel('Reward')
        plt.savefig('HospitalRobot_Rewards.png')
        plt.close()

    # Save the Q-table after training
    if is_training:
        with open("HospitalRobot.pkl", "wb") as f:
            pickle.dump(q, f)

    return total_rewards_per_episode


if __name__ == "__main__":
    # Train the agent
    rewards = run(1000)

    # Test the trained agent
    run(10, is_training=False, render=True)