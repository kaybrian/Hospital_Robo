import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
# Add callback for saving best model
from stable_baselines3.common.callbacks import EvalCallback

# Import custom environment
from hospital_robot_env import HospitalRobotEnv

import ale_py


def make_env():
    """Create and wrap the hospital robot environment"""
    gym.register_envs(ale_py)
    env = HospitalRobotEnv(render_mode=None)
    return env


def main():
    # Create directories for saving models and logs
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # Create and wrap the environment
    env = DummyVecEnv([make_env])

    # Initialize the DQN agent with improved parameters
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=3e-4,  # Slightly increased learning rate
        buffer_size=100000,  # Increased buffer size
        learning_starts=5000,  # More initial random actions
        batch_size=128,  # Larger batch size
        gamma=0.99,
        exploration_fraction=0.2,  # Longer exploration
        exploration_initial_eps=1.0,
        exploration_final_eps=0.02,  # Lower final exploration
        train_freq=4,
        gradient_steps=2,  # More gradient steps
        target_update_interval=500,  # More frequent target updates
        verbose=1,
        tensorboard_log="logs/",
        policy_kwargs=dict(
            net_arch=[256, 256, 256]  # Deeper network
        )
    )

    # Train for longer
    total_timesteps = 500000  # Increased from 100000

    # Add evaluation callback
    eval_env = DummyVecEnv([make_env])
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/best_model",
        log_path="./logs/",
        eval_freq=5000,
        deterministic=True,
        render=False
    )

    # Train the agent
    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=True,
        callback=eval_callback
    )

    # Evaluate the model
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    # Save the final model
    model.save("models/hospital_robot_dqn")

    # Close the environment
    env.close()


if __name__ == "__main__":
    main()