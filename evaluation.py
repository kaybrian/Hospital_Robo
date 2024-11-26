import gymnasium as gym
from stable_baselines3 import DQN
import numpy as np
import time
from hospital_robot_env import HospitalRobotEnv

def evaluate_episode(env, model):
    obs, _ = env.reset()
    total_reward = 0
    steps = 0
    collisions = 0
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        prev_obs = obs
        obs, reward, terminated, truncated, info = env.step(action)

        # Count collisions
        if reward <= -5:  # Collision detected
            collisions += 1

        total_reward += reward
        steps += 1
        done = terminated or truncated

        # Add delay for visualization
        time.sleep(0.1)

    return total_reward, steps, collisions


def main():
    # Create environment with rendering enabled
    env = HospitalRobotEnv(render_mode="human")

    # Load the best model instead of the final one
    model = DQN.load("models/best_model/best_model")

    # Run several episodes with detailed metrics
    n_episodes = 5
    rewards = []
    steps_list = []
    collisions_list = []

    for episode in range(n_episodes):
        print(f"\nStarting Episode {episode + 1}")
        reward, steps, collisions = evaluate_episode(env, model)
        rewards.append(reward)
        steps_list.append(steps)
        collisions_list.append(collisions)

        print(f"Episode {episode + 1} Results:")
        print(f"Total Reward: {reward:.2f}")
        print(f"Steps Taken: {steps}")
        print(f"Collisions: {collisions}")

    print("\nOverall Statistics:")
    print(f"Average Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Average Steps: {np.mean(steps_list):.2f} ± {np.std(steps_list):.2f}")
    print(f"Average Collisions: {np.mean(collisions_list):.2f} ± {np.std(collisions_list):.2f}")

    env.close()


if __name__ == "__main__":
    main()