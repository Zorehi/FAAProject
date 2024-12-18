import gym
from gym import spaces
import numpy as np
import pygame
import random

class MazeEnv(gym.Env):
    def __init__(self):
        super(MazeEnv, self).__init__()

        # Dimensions du labyrinthe
        self.WIDTH, self.HEIGHT = 800, 600
        self.CELL_SIZE = 40
        self.ROWS, self.COLS = self.HEIGHT // self.CELL_SIZE, self.WIDTH // self.CELL_SIZE

        # Espaces d'actions et d'observations
        self.action_space = spaces.Discrete(4)  # 0: Up, 1: Down, 2: Left, 3: Right
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.ROWS, self.COLS, 1), dtype=np.float32
        )

        # Initialisation de l'état
        self.maze = self.generate_maze()
        self.player_pos = [0, 0]  # Position initiale
        self.target_pos = [self.COLS - 1, self.ROWS - 1]  # Cible

        # PyGame pour le rendu
        self.screen = None

    def generate_maze(self):
        visited = [[False for _ in range(self.COLS)] for _ in range(self.ROWS)]
        stack = []
        current_cell = (0, 0)
        visited[0][0] = True
        maze = [["1111" for _ in range(self.COLS)] for _ in range(self.ROWS)]

        while True:
            x, y = current_cell
            neighbors = []
            for direction, (dx, dy) in {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}.items():
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.COLS and 0 <= ny < self.ROWS and not visited[ny][nx]:
                    neighbors.append((nx, ny, direction))

            if neighbors:
                nx, ny, direction = random.choice(neighbors)
                visited[ny][nx] = True
                stack.append(current_cell)

                if direction == "UP":
                    maze[y][x] = maze[y][x][:0] + "0" + maze[y][x][1:]
                    maze[ny][nx] = maze[ny][nx][:1] + "0" + maze[ny][nx][2:]
                elif direction == "DOWN":
                    maze[y][x] = maze[y][x][:1] + "0" + maze[y][x][2:]
                    maze[ny][nx] = maze[ny][nx][:0] + "0" + maze[ny][nx][1:]
                elif direction == "LEFT":
                    maze[y][x] = maze[y][x][:2] + "0" + maze[y][x][3:]
                    maze[ny][nx] = maze[ny][nx][:3] + "0" + maze[ny][nx][4:]
                elif direction == "RIGHT":
                    maze[y][x] = maze[y][x][:3] + "0" + maze[y][x][4:]
                    maze[ny][nx] = maze[ny][nx][:2] + "0" + maze[ny][nx][3:]

                current_cell = (nx, ny)
            elif stack:
                current_cell = stack.pop()
            else:
                break

        return maze

    def reset(self):
        self.maze = self.generate_maze()
        self.player_pos = [0, 0]
        return self._get_observation()

    def step(self, action):
        x, y = self.player_pos

        if action == 0 and self.maze[y][x][0] == "0":  # Up
            y -= 1
        elif action == 1 and self.maze[y][x][1] == "0":  # Down
            y += 1
        elif action == 2 and self.maze[y][x][2] == "0":  # Left
            x -= 1
        elif action == 3 and self.maze[y][x][3] == "0":  # Right
            x += 1

        self.player_pos = [x, y]

        done = self.player_pos == self.target_pos
        reward = 1 if done else -0.01  # Récompense positive pour atteindre la cible, légère pénalité sinon

        return self._get_observation(), reward, done, {}

    def render(self, mode='human'):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
            pygame.display.set_caption("Labyrinthe Aléatoire")

        self.screen.fill((255, 255, 255))

        for y in range(self.ROWS):
            for x in range(self.COLS):
                self._draw_cell(x, y, (255, 255, 255))
                self._draw_walls(x, y, self.maze[y][x])

        self._draw_cell(self.target_pos[0], self.target_pos[1], (255, 0, 0))
        self._draw_cell(self.player_pos[0], self.player_pos[1], (0, 0, 255))

        pygame.display.flip()

    def close(self):
        if self.screen:
            pygame.quit()
            self.screen = None

    def _get_observation(self):
        obs = np.zeros((self.ROWS, self.COLS, 1), dtype=np.float32)
        obs[self.player_pos[1]][self.player_pos[0]][0] = 1  # Marquer la position du joueur
        return obs

    def _draw_cell(self, x, y, color):
        pygame.draw.rect(
            self.screen, color, (x * self.CELL_SIZE, y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        )

    def _draw_walls(self, x, y, walls):
        if walls[0] == "1":
            pygame.draw.line(
                self.screen, (0, 0, 0), (x * self.CELL_SIZE, y * self.CELL_SIZE),
                ((x + 1) * self.CELL_SIZE, y * self.CELL_SIZE), 2
            )
        if walls[1] == "1":
            pygame.draw.line(
                self.screen, (0, 0, 0), (x * self.CELL_SIZE, (y + 1) * self.CELL_SIZE),
                ((x + 1) * self.CELL_SIZE, (y + 1) * self.CELL_SIZE), 2
            )
        if walls[2] == "1":
            pygame.draw.line(
                self.screen, (0, 0, 0), (x * self.CELL_SIZE, y * self.CELL_SIZE),
                (x * self.CELL_SIZE, (y + 1) * self.CELL_SIZE), 2
            )
        if walls[3] == "1":
            pygame.draw.line(
                self.screen, (0, 0, 0), ((x + 1) * self.CELL_SIZE, y * self.CELL_SIZE),
                ((x + 1) * self.CELL_SIZE, (y + 1) * self.CELL_SIZE), 2
            )

# Pour tester l'environnement
if __name__ == "__main__":
    env = MazeEnv()
    obs = env.reset()
    done = False

    while not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        env.render()

    env.close()