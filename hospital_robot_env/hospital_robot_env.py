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
       - Maximize rewards
       - Reach the operation room within a limited number of steps

       Reward:
           - +1 for moving closer to the target,
           - -0.5 for moving away,
           - -5 for collision,
           - +200 for successful delivery of the patient,
           - +10 for picking up the patient

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
        self.has_patient = False  # New state variable

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
        Create a realistic hospital layout with:
        - Multiple rooms with doorways
        - Main corridors
        - Emergency exits
        - Nursing stations
        - Waiting areas
        """
        # Initialize grid
        self.grid = np.zeros(self.grid_size, dtype=np.int32)

        # Add outer walls
        for x in range(self.grid_size[0]):
            self.grid[x, 0] = 1
            self.grid[x, self.grid_size[1] - 1] = 1
        for y in range(self.grid_size[1]):
            self.grid[0, y] = 1
            self.grid[self.grid_size[0] - 1, y] = 1

        # Create main horizontal corridor
        corridor_y = self.grid_size[1] // 2
        for x in range(1, self.grid_size[0] - 1):
            self.grid[x, corridor_y] = 0
            self.grid[x, corridor_y + 1] = 0

        # Create vertical corridors
        for x in [self.grid_size[0] // 4, self.grid_size[0] // 2, 3 * self.grid_size[0] // 4]:
            for y in range(1, self.grid_size[1] - 1):
                self.grid[x, y] = 0
                self.grid[x + 1, y] = 0

        # Create rooms
        rooms = [
            # Format: (start_x, start_y, width, height)
            (2, 2, 10, 10),  # Top-left room
            (15, 2, 10, 10),  # Top room
            (28, 2, 10, 10),  # Top-right room
            (2, 38, 10, 10),  # Bottom-left room
            (15, 38, 10, 10),  # Bottom room
            (28, 38, 10, 10),  # Bottom-right room
        ]

        # rooms with doorways
        for room in rooms:
            start_x, start_y, width, height = room

            # Room walls
            for x in range(start_x, start_x + width):
                self.grid[x, start_y] = 1  # Top wall
                self.grid[x, start_y + height] = 1  # Bottom wall
            for y in range(start_y, start_y + height):
                self.grid[start_x, y] = 1  # Left wall
                self.grid[start_x + width, y] = 1  # Right wall

            # doorway (gap in wall)
            door_pos = random.randint(2, width - 2)

            # Decide if door should be on horizontal or vertical wall
            if random.random() < 0.5:
                # Horizontal door
                if start_y < self.grid_size[1] // 2:  # Top rooms
                    self.grid[start_x + door_pos, start_y + height] = 0  # Door at bottom
                else:  # Bottom rooms
                    self.grid[start_x + door_pos, start_y] = 0  # Door at top
            else:
                # Vertical door
                if start_x < self.grid_size[0] // 2:  # Left rooms
                    self.grid[start_x + width, start_y + door_pos] = 0  # Door on right
                else:  # Right rooms
                    self.grid[start_x, start_y + door_pos] = 0  # Door on left

        # nursing stations (represented as obstacles)
        nursing_stations = [
            (self.grid_size[0] // 4, corridor_y - 3),
            (3 * self.grid_size[0] // 4, corridor_y + 3)
        ]
        for station in nursing_stations:
            x, y = station
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if self.grid[x + dx, y + dy] == 0:
                        self.grid[x + dx, y + dy] = 5

        # some random obstacles in corridors (equipment, chairs, etc.)
        for _ in range(20):
            x = random.randint(1, self.grid_size[0] - 2)
            y = random.randint(1, self.grid_size[1] - 2)
            if self.grid[x, y] == 0 and \
                    not (x in [self.grid_size[0] // 4, self.grid_size[0] // 2, 3 * self.grid_size[0] // 4] and \
                         y in range(corridor_y - 1, corridor_y + 2)):
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
                self.grid[x, y] = 3  # Patient's position
                break

        # Set operation room (top right quadrant)
        while True:
            x = random.randint(self.grid_size[0] - 15, self.grid_size[0] - 2)
            y = random.randint(self.grid_size[1] - 15, self.grid_size[1] - 2)
            if self.grid[x, y] == 0:
                self.operation_room_pos = (x, y)
                self.grid[x, y] = 4  # Operation room
                break

        # Set robot initial position near patient
        while True:
            x = random.randint(max(0, self.patient_pos[0] - 2),
                               min(self.grid_size[0] - 1, self.patient_pos[0] + 2))
            y = random.randint(max(0, self.patient_pos[1] - 2),
                               min(self.grid_size[1] - 1, self.patient_pos[1] + 2))
            if self.grid[x, y] == 0:
                self.robot_pos = (x, y)
                self.grid[x, y] = 2  # Robot's initial position
                break

        # Prepare rendering if in human mode
        if self.render_mode == "human":
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Hospital Robot Navigation")
            self.clock = pygame.time.Clock()

        self.has_patient = False  # Initialize patient status

        return self._get_observation(), {}

    def step(self, action):
        """
        Execute one timestep in the environment.
        """
        terminated = False
        truncated = False

        # Movement deltas (based on action)
        deltas = [
            (0, -1),  # Up
            (1, 0),  # Right
            (0, 1),  # Down
            (-1, 0)  # Left
        ]

        x, y = self.robot_pos
        dx, dy = deltas[action]
        new_x, new_y = x + dx, y + dy
        reward = 0

        # Check for collision or invalid move
        if self.grid[new_x, new_y] in [1, 5]:  # 1 for obstacle, 5 for invalid position
            reward -= 5  # Collision penalty
            new_x, new_y = x, y  # Stay in current position
        else:
            # Update the position
            self.grid[x, y] = 0
            self.robot_pos = (new_x, new_y)
            self.grid[new_x, new_y] = 2

            # Calculate reward based on the distance to the goal
            current_distance = np.sqrt((new_x - self.operation_room_pos[0]) ** 2 +
                                       (new_y - self.operation_room_pos[1]) ** 2)
            previous_distance = np.sqrt((x - self.operation_room_pos[0]) ** 2 +
                                        (y - self.operation_room_pos[1]) ** 2)

            if current_distance < previous_distance:
                reward += 1  # Moving closer
            elif current_distance > previous_distance:
                reward -= 0.5  # Moving away

        # Check if the robot is at the patient's position and hasn't picked up the patient yet
        if self.robot_pos == self.patient_pos and not self.has_patient:
            self.has_patient = True
            reward += 10  # Reward for picking up the patient

        # Update the observation to reflect the patient being carried
        if self.has_patient:
            self.grid[self.robot_pos[0], self.robot_pos[1]] = 4  # Robot carrying patient

        # Check if the robot reached the destination (operation room)
        if self.robot_pos == self.operation_room_pos:
            reward += 200  # Success reward
            terminated = True  # Episode ends when the goal is reached

        # Increment steps
        self.steps += 1


        # Check if we've reached the maximum allowed steps
        if self.steps >= self.max_steps:
            truncated = True



        # Prepare the observation
        obs = self._get_observation()

        # Return the new state (observation), reward, terminated flag, truncated flag, and additional info
        info = {"truncated": truncated}  # Additional information

        return obs, reward, terminated, truncated, info


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
        self.clock.tick(30)  # Limit to 30 FPS

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
