import gymnasium as gym
import numpy as np
import pygame
import random


class HospitalRobotEnv(gym.Env):
    """
    Custom Hospital Robot Navigation Environment

    Objective:
    - Navigate a robot carrying a patient from a starting room to the operation room
    - Avoid walls and obstacles
    - Take the shortest path
    - Minimize steps and collisions
    """

    def __init__(self, render_mode=None, grid_size=(50, 50)):
        super().__init__()

        # Environment configuration
        self.grid_size = grid_size
        self.cell_size = 20  # Pixel size of each grid cell
        self.screen_width = grid_size[0] * self.cell_size
        self.screen_height = grid_size[1] * self.cell_size

        # Action space: 4 directions (up, right, down, left)
        self.action_space = gym.spaces.Discrete(4)

        # Observation space: 2D grid representing the environment
        self.observation_space = gym.spaces.Box(
            low=0, high=5,
            shape=grid_size,
            dtype=np.int32
        )

        # Rendering
        self.render_mode = render_mode
        pygame.init()
        self.screen = None
        self.clock = None

        # Environment elements
        self.robot_pos = None
        self.patient_pos = None
        self.operation_room_pos = None
        self.walls = []

        # Game state
        self.steps = 0
        self.max_steps = 200

        # Color constants
        self.COLORS = {
            0: (255, 255, 255),  # White - empty space
            1: (0, 0, 0),  # Black - walls
            2: (0, 255, 0),  # Green - robot
            3: (255, 0, 0),  # Red - patient
            4: (0, 0, 255),  # Blue - operation room
            5: (255, 165, 0)  # Orange - obstacles
        }

    def _create_hospital_layout(self):
        """
        Create a hospital layout with:
        - Patient's room
        - Operation room
        - Walls
        - Corridors
        """
        # Initialize grid
        self.grid = np.zeros(self.grid_size, dtype=np.int32)

        # Add walls (border)
        for x in range(self.grid_size[0]):
            self.grid[x, 0] = 1
            self.grid[x, self.grid_size[1] - 1] = 1
        for y in range(self.grid_size[1]):
            self.grid[0, y] = 1
            self.grid[self.grid_size[0] - 1, y] = 1

        # Add internal walls to create corridors and rooms
        # Vertical walls
        for y in range(10, self.grid_size[1] - 10):
            if 15 < y < 35:
                self.grid[15, y] = 1
                self.grid[35, y] = 1

        # Horizontal walls
        for x in range(10, self.grid_size[0] - 10):
            if 15 < x < 35:
                self.grid[x, 15] = 1
                self.grid[x, 35] = 1

        # Add some random obstacles
        for _ in range(20):
            x = random.randint(1, self.grid_size[0] - 2)
            y = random.randint(1, self.grid_size[1] - 2)
            if self.grid[x, y] == 0:
                self.grid[x, y] = 5

    def reset(self, seed=None, options=None):
        """
        Reset the environment to initial state
        """
        super().reset(seed=seed)

        # Create hospital layout
        self._create_hospital_layout()

        # Reset game state
        self.steps = 0

        # Set patient's room (bottom left quadrant)
        while True:
            x = random.randint(1, 14)
            y = random.randint(1, 14)
            if self.grid[x, y] == 0:
                self.patient_pos = (x, y)
                self.grid[x, y] = 3
                break

        # Set operation room (top right quadrant)
        while True:
            x = random.randint(self.grid_size[0] - 15, self.grid_size[0] - 2)
            y = random.randint(self.grid_size[1] - 15, self.grid_size[1] - 2)
            if self.grid[x, y] == 0:
                self.operation_room_pos = (x, y)
                self.grid[x, y] = 4
                break

        # Set robot initial position near patient
        while True:
            x = random.randint(max(0, self.patient_pos[0] - 2),
                               min(self.grid_size[0] - 1, self.patient_pos[0] + 2))
            y = random.randint(max(0, self.patient_pos[1] - 2),
                               min(self.grid_size[1] - 1, self.patient_pos[1] + 2))
            if self.grid[x, y] == 0:
                self.robot_pos = (x, y)
                self.grid[x, y] = 2
                break

        # Prepare rendering if in human mode
        if self.render_mode == "human":
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Hospital Robot Navigation")
            self.clock = pygame.time.Clock()

        return self._get_observation(), {}

    def step(self, action):
        """
        Execute one timestep in the environment

        Actions:
        0: Up
        1: Right
        2: Down
        3: Left
        """
        # Initialize terminated and truncated flags
        terminated = False
        truncated = False

        # Movement deltas
        deltas = [
            (0, -1),  # Up
            (1, 0),  # Right
            (0, 1),  # Down
            (-1, 0)  # Left
        ]

        # Current position
        x, y = self.robot_pos

        # Calculate new position
        dx, dy = deltas[action]
        new_x, new_y = x + dx, y + dy

        # Initialize reward
        reward = 0

        # Base step penalty (smaller to encourage exploration)
        reward -= 0.1

        # Distance-based reward
        current_distance = np.sqrt((new_x - self.operation_room_pos[0]) ** 2 +
                                   (new_y - self.operation_room_pos[1]) ** 2)
        previous_distance = np.sqrt((x - self.operation_room_pos[0]) ** 2 +
                                    (y - self.operation_room_pos[1]) ** 2)

        # Reward for moving closer to the target
        if current_distance < previous_distance:
            reward += 1
        elif current_distance > previous_distance:
            reward -= 0.5

        # Collision penalties
        if self.grid[new_x, new_y] in [1, 5]:
            reward -= 5  # Collision with wall or obstacle
            new_x, new_y = x, y
        else:
            self.grid[x, y] = 0
            self.robot_pos = (new_x, new_y)
            self.grid[new_x, new_y] = 2

        # Success reward
        if self.robot_pos == self.operation_room_pos:
            reward += 200
            terminated = True

        # Increment steps
        self.steps += 1

        # Check max steps
        if self.steps >= self.max_steps:
            truncated = True

        # Optional rendering
        if self.render_mode == "human":
            self._render_frame()

        return self._get_observation(), reward, terminated, truncated, {}

    def _get_observation(self):
        """
        Convert game state to numpy array observation
        """
        return self.grid.copy()

    def _render_frame(self):
        """
        Render the current game state using Pygame
        """
        if self.render_mode != "human":
            return

        self.screen.fill(self.COLORS[0])

        # Render grid
        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                cell_value = self.grid[x, y]
                color = self.COLORS.get(cell_value, self.COLORS[0])

                # Draw cell
                rect = pygame.Rect(
                    x * self.cell_size,
                    y * self.cell_size,
                    self.cell_size,
                    self.cell_size
                )
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, (100, 100, 100), rect, 1)

        pygame.display.flip()
        self.clock.tick(10)  # Limit to 10 FPS

    def render(self):
        """
        Standard render method for Gymnasium environments
        """
        self._render_frame()

    def close(self):
        """
        Close the environment and pygame
        """
        if self.render_mode == "human":
            pygame.quit()


# Shortcut function to create vectorized environment
def make_hospital_env():
    return HospitalRobotEnv()