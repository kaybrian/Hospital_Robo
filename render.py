import gymnasium as gym
import pygame
import time
import numpy as np
from hospital_robot_env import HospitalRobotEnv


def render_environment():
    # Create an instance of the environment
    env = HospitalRobotEnv(render_mode="human", grid_size=(50, 50))

    # Reset the environment
    observation, _ = env.reset()

    # Render loop
    done = False
    while not done:
        # Get action from the environment (for now, just moving randomly)
        action = env.action_space.sample()

        # Step the environment with the action
        observation, reward, terminated, truncated, info = env.step(action)

        # Render the current state of the environment
        env.render()

        # Handle pygame events like closing the window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        # Add a small delay to control the frame rate
        time.sleep(0.1)  # Adjust for a smooth visual experience, can be tweaked for speed

    # Close the environment when done
    env.close()


if __name__ == "__main__":
    render_environment()
