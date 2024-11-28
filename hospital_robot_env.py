import gymnasium as gym
import numpy as np
import pygame
import random
from gymnasium import spaces


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
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, grid_size=10, max_steps=100, render_mode=None):
        """
        Initialize the Hospital Robot Environment

        Args:
            grid_size (int): Size of the square grid environment
            max_steps (int): Maximum number of steps allowed in an episode
        """
        super().__init__()

        # Environment configuration
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.render_mode = render_mode

        # Action space (4 directions: Up, Right, Down, Left)
        self.action_space = spaces.Discrete(4)

        # Observation space: Robot position, patient position, operation room position
        self.observation_space = spaces.Box(
            low=0, high=grid_size - 1,
            shape=(3, 2),
            dtype=np.int32
        )

        # Rendering
        self.window = None
        self.clock = None

        # Episode tracking
        self.current_step = 0
        self.robot_pos = None
        self.patient_pos = None
        self.operation_room_pos = None
        self.has_patient = False

        # Action mapping
        self.action_to_direction = {
            0: np.array([-1, 0]),  # Up
            1: np.array([0, 1]),  # Right
            2: np.array([1, 0]),  # Down
            3: np.array([0, -1])  # Left
        }

        # Obstacles
        self.obstacles = self._generate_obstacles()

    def _generate_obstacles(self):
        """
        Generate structured obstacles in the environment based on grid size

        Returns:
            List of obstacle positions
        """
        obstacles = []

        # Different obstacle patterns based on grid size
        if self.grid_size == 10:
            # Maze-like configuration
            obstacles = [
                # Horizontal walls
                (2, 3), (2, 4), (2, 5), (2, 6),
                (5, 3), (5, 4), (5, 5), (5, 6),
                (7, 2), (7, 3), (7, 4),

                # Vertical walls
                (3, 2), (4, 2),
                (6, 5), (6, 6), (6, 7),

                # Random obstacles
                (1, 7), (3, 8), (8, 4), (9, 2)
            ]
        elif self.grid_size == 15:
            # More complex maze
            obstacles = [
                # Horizontal barriers
                (3, 4), (3, 5), (3, 6), (3, 7),
                (7, 8), (7, 9), (7, 10), (7, 11),
                (11, 3), (11, 4), (11, 5),

                # Vertical barriers
                (4, 3), (5, 3),
                (8, 6), (9, 6), (10, 6),

                # Diagonal obstacles
                (2, 9), (3, 8), (4, 7),
                (12, 5), (11, 6), (10, 7),

                # Scattered obstacles
                (5, 11), (6, 12), (1, 13), (14, 2)
            ]
        elif self.grid_size == 20:
            # Large, more intricate maze
            obstacles = [
                # Main horizontal barriers
                (5, 6), (5, 7), (5, 8), (5, 9), (5, 10),
                (10, 11), (10, 12), (10, 13), (10, 14), (10, 15),
                (15, 5), (15, 6), (15, 7),

                # Vertical barriers
                (6, 5), (7, 5),
                (12, 10), (13, 10), (14, 10),

                # Zigzag patterns
                (3, 8), (4, 9), (5, 10),
                (16, 12), (17, 13), (18, 14),

                # Scattered obstacles
                (2, 15), (7, 17), (12, 3), (18, 6),
                (9, 8), (14, 16), (3, 12), (17, 9)
            ]
        else:
            # Default random generation for other grid sizes
            num_obstacles = self.grid_size // 2
            obstacles = set()
            while len(obstacles) < num_obstacles:
                obstacle = (random.randint(0, self.grid_size - 1),
                            random.randint(0, self.grid_size - 1))
                obstacles.add(obstacle)
            obstacles = list(obstacles)

        return obstacles

    def _validate_obstacle_placement(self):
        """
        Ensure obstacles do not block the entire path

        Returns:
            bool: True if obstacle placement is valid
        """
        def is_valid_path(start, end):
            # Simple Manhattan distance check
            return (abs(start[0] - end[0]) + abs(start[1] - end[1])) < self.grid_size * 2

        # Check path from start to patient
        if not is_valid_path(self.robot_pos, self.patient_pos):
            return False

        # Check path from patient to operation room
        if not is_valid_path(self.patient_pos, self.operation_room_pos):
            return False

        return True

    def reset(self, seed=None, options=None):
        """
        Reset the environment to initial state

        Returns:
            Initial observation and info dictionary
        """
        super().reset(seed=seed)

        # Reset tracking variables
        self.current_step = 0
        self.has_patient = False

        # Set initial positions
        self.robot_pos = np.array([0, 0])
        self.patient_pos = np.array([
            random.randint(1, self.grid_size - 2),
            random.randint(1, self.grid_size - 2)
        ])
        self.operation_room_pos = np.array([
            self.grid_size - 1,
            self.grid_size - 1
        ])

        # Ensure no position conflicts
        while (tuple(self.robot_pos) in self.obstacles or
               tuple(self.patient_pos) in self.obstacles or
               tuple(self.operation_room_pos) in self.obstacles or
               np.array_equal(self.robot_pos, self.patient_pos) or
               np.array_equal(self.robot_pos, self.operation_room_pos) or
               np.array_equal(self.patient_pos, self.operation_room_pos)):
            self.robot_pos = np.array([0, 0])
            self.patient_pos = np.array([
                random.randint(1, self.grid_size - 2),
                random.randint(1, self.grid_size - 2)
            ])
            self.operation_room_pos = np.array([
                self.grid_size - 1,
                self.grid_size - 1
            ])

        # Prepare observation and info
        observation = np.array([
            self.robot_pos,
            self.patient_pos,
            self.operation_room_pos
        ])
        info = {}

        return observation, info

    def step(self, action):
        """
        Execute one time step in the environment

        Args:
            action (int): Action to take (0-3)

        Returns:
            observation, reward, terminated, truncated, info
        """
        self.current_step += 1

        # Calculate new robot position
        direction = self.action_to_direction[action]
        new_robot_pos = self.robot_pos + direction

        # Check for collisions or out of bounds
        reward = 0
        terminated = False
        truncated = self.current_step >= self.max_steps

        # Collision with obstacles or walls
        if (tuple(new_robot_pos) in self.obstacles or
                not (0 <= new_robot_pos[0] < self.grid_size and
                     0 <= new_robot_pos[1] < self.grid_size)):
            reward -= 5  # Collision penalty
            new_robot_pos = self.robot_pos  # Stay in place

        # Distance calculation for reward
        old_dist_to_patient = np.linalg.norm(self.robot_pos - self.patient_pos)
        old_dist_to_operation = np.linalg.norm(self.robot_pos - self.operation_room_pos)

        # Update robot position
        self.robot_pos = new_robot_pos

        # New distances after movement
        new_dist_to_patient = np.linalg.norm(self.robot_pos - self.patient_pos)
        new_dist_to_operation = np.linalg.norm(self.robot_pos - self.operation_room_pos)

        # Patient pickup
        if not self.has_patient and np.array_equal(self.robot_pos, self.patient_pos):
            self.has_patient = True
            reward += 10

        # Movement rewards and penalties
        if new_dist_to_patient < old_dist_to_patient:
            reward += 3  # Moving closer to patient
        elif new_dist_to_patient > old_dist_to_patient:
            reward -= 9  # Moving away from patient

        # Operation room delivery
        if (self.has_patient and
                np.array_equal(self.robot_pos, self.operation_room_pos)):
            reward += 200  # Successfully delivered patient
            terminated = True

        # Prepare observation
        observation = np.array([
            self.robot_pos,
            self.patient_pos if not self.has_patient else self.robot_pos,
            self.operation_room_pos
        ])

        # Additional info
        info = {
            "current_step": self.current_step,
            "has_patient": self.has_patient
        }

        return observation, reward, terminated, truncated, info

    def render(self):
        """
        Render the environment based on the selected render mode
        """
        if self.render_mode == 'human':
            if self.window is None:
                pygame.init()
                self.window = pygame.display.set_mode((400, 400))
                pygame.display.set_caption('Hospital Robot Navigation')
                self.clock = pygame.time.Clock()

            # Clear the screen
            self.window.fill((255, 255, 255))

            # Cell size
            cell_size = 400 // self.grid_size

            # Draw grid and obstacles
            for x in range(self.grid_size):
                for y in range(self.grid_size):
                    rect = pygame.Rect(x * cell_size, y * cell_size, cell_size, cell_size)
                    pygame.draw.rect(self.window, (200, 200, 200), rect, 1)

            for obs in self.obstacles:
                pygame.draw.rect(self.window, (100, 100, 100), pygame.Rect(obs[1] * cell_size, obs[0] * cell_size, cell_size, cell_size))

            # Draw the robot, patient, and operation room
            pygame.draw.rect(self.window, (0, 255, 0), pygame.Rect(self.robot_pos[1] * cell_size, self.robot_pos[0] * cell_size, cell_size, cell_size))
            if not self.has_patient:
                pygame.draw.rect(self.window, (255, 0, 0), pygame.Rect(self.patient_pos[1] * cell_size, self.patient_pos[0] * cell_size, cell_size, cell_size))
            pygame.draw.rect(self.window, (0, 0, 255), pygame.Rect(self.operation_room_pos[1] * cell_size, self.operation_room_pos[0] * cell_size, cell_size, cell_size))

            pygame.display.flip()
            self.clock.tick(self.metadata['render_fps'])


    def close(self):
        """
        Close the environment and clean up resources
        """
        if self.window is not None:
            pygame.quit()