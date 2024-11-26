import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import matplotlib.pyplot as plt
import pickle
from stable_baselines3.common.vec_env import SubprocVecEnv
import ale_py
# Import custom environment
from hospital_robot_env import HospitalRobotEnv
import gymnasium as gym


gym.register_envs(ale_py)

def run(episodes, is_training=True, render=False):
    # Create the custom environment using SubprocVecEnv
    env = SubprocVecEnv([lambda: HospitalRobotEnv()])

    # Initialize the DQN model
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=0.0005,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=64,
        gamma=0.99,
        exploration_fraction=0.1,
        exploration_final_eps=0.05,
        target_update_interval=1000,
        verbose=1,
        policy_kwargs=dict(net_arch=[256, 256]),
        tensorboard_log="./tensorboard_logs/"
    )

    rewards_per_episode = np.zeros(episodes)

    # Training loop
    for i in range(episodes):
        obs = env.reset()
        done = [False]  # done should be a list or array
        episode_reward = 0

        while not done[0]:
            # Predict the action based on the current policy
            action, _ = model.predict(obs, deterministic=not is_training)

            # Take the action and get the new state and reward
            new_obs, reward, done, info = env.step(action)

            episode_reward += reward[0]  # reward is returned as an array

            if is_training:
                # Update the model after each step
                model.learn(total_timesteps=1, reset_num_timesteps=False)

            # Update the observation for the next step
            obs = new_obs

            # Check if episode is done
            if done[0]:  # done is returned as an array
                break

        rewards_per_episode[i] = episode_reward

        # Print the reward for the current episode
        if i % 100 == 0:
            print(f"Episode {i} Reward: {episode_reward}")

    # Post-training processing
    env.close()

    # Save training rewards
    if is_training:
        plt.plot(np.convolve(rewards_per_episode, np.ones((100,))/100, mode='valid'))
        plt.title("Hospital Robot Rewards")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.savefig('hospital_robot_rewards.png')

        # Save the trained model
        model.save("models/hospital_robot_dqn")

        # Save the rewards for analysis later
        with open("hospital_rewards.pkl", "wb") as f:
            pickle.dump(rewards_per_episode, f)

    # Evaluate the policy
    if not is_training:
        print("Evaluating trained model:")
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
        print(f"Mean reward: {mean_reward} ± {std_reward}")

if __name__ == "__main__":
    # Train the model
    run(10000, is_training=True)  # Train for 10,000 episodes

    # Evaluate the trained model
    run(10, is_training=False)  # Evaluate for 10 episodes
