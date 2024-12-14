import gym
from PIL.FontFile import WIDTH
from gym import spaces
import numpy as np
import pygame
import random

WIDTH, HEIGHT = 800, 600
CELL_SIZE = 20
START_POS = (0, 0)

class MazeEnv(gym.Env):
    def __init__(self):
        super(MazeEnv, self).__init__()

        # Dimensions du labyrinthe
        self.width, self.height = WIDTH, HEIGHT
        self.cell_size = CELL_SIZE
        self.rows, self.cols = self.height // self.cell_size, self.width // self.cell_size
        self.start_pos = START_POS

        # Espaces d'actions et d'observations
        self.action_space = spaces.Discrete(4)  # 0: Up, 1: Down, 2: Left, 3: Right
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.rows, self.cols, 1), dtype=np.float32
        )

        # Initialisation de l'état
        self.maze = self.generate_maze()
        self.agent_pos = [0, 0]  # Position initiale
        self.target_pos = [self.cols - 1, self.rows - 1]  # Cible
        self.path = [self.agent_pos]

        # PyGame pour le rendu
        self.screen = None

    def generate_maze(self):
        visited = [[False for _ in range(self.cols)] for _ in range(self.rows)]
        stack = []
        current_cell = (0, 0)
        visited[0][0] = True
        maze = [["1111" for _ in range(self.cols)] for _ in range(self.rows)]

        while True:
            x, y = current_cell
            neighbors = []
            for direction, (dx, dy) in {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}.items():
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.cols and 0 <= ny < self.rows and not visited[ny][nx]:
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
        self.agent_pos = self.start_pos
        self.path = [self.agent_pos]
        return self._get_observation()

    def step(self, action):
        x, y = self.agent_pos
        previous_pos = self.agent_pos

        # Mise à jour de la position de l'agent en fonction de l'action et des murs
        if action == 0 and self.maze[y][x][0] == "0":  # Up
            y -= 1
        elif action == 1 and self.maze[y][x][1] == "0":  # Down
            y += 1
        elif action == 2 and self.maze[y][x][2] == "0":  # Left
            x -= 1
        elif action == 3 and self.maze[y][x][3] == "0":  # Right
            x += 1

        self.agent_pos = [x, y]

        # Vérification de l'état final
        done = self.agent_pos == self.target_pos

        # Gestion des boucles et calcul de la récompense
        # min_go_back = self.rows // 2
        if done:
            reward = 1.0 # Récompense forte pour atteindre l'objectif
        elif self.agent_pos in self.path:
            reward = -0.2  # Pénalité pour revenir sur ses pas
        elif previous_pos == self.agent_pos:
            reward = -1.0  # Pénalité pour taper un mur
        else:
            reward = self.compute_reward(previous_pos, self.agent_pos)

        # Mise à jour du chemin parcouru
        if self.agent_pos not in self.path:
            self.path.append(self.agent_pos)

        return self._get_observation(), reward, done, {}

    def compute_reward(self, previous_pos, new_pos):
        """
        Calcul de la récompense en fonction de la distance à l'objectif.
        """
        dist_new = np.linalg.norm(np.array(new_pos) - np.array(self.target_pos))
        dist_old = np.linalg.norm(np.array(previous_pos) - np.array(self.target_pos))

        # Calcul de la différence de distance, normalisée par la diagonale maximale
        return (dist_old - dist_new) * 2 / (self.rows * self.cols)

    def render(self, mode='human'):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Labyrinthe Aléatoire")

        self.screen.fill((255, 255, 255))

        for y in range(self.rows):
            for x in range(self.cols):
                if [x, y] in self.path:
                    self._draw_cell(x, y, (255, 255, 0))
                else:
                    self._draw_cell(x, y, (255, 255, 255))
                self._draw_walls(x, y, self.maze[y][x])

        self._draw_cell(self.start_pos[0], self.start_pos[1], (0, 255, 0))
        self._draw_cell(self.target_pos[0], self.target_pos[1], (255, 0, 0))
        self._draw_cell(self.agent_pos[0], self.agent_pos[1], (0, 0, 255))

        pygame.display.flip()

    def close(self):
        if self.screen:
            pygame.quit()
            self.screen = None

    def _get_observation(self):
        obs = np.zeros((self.rows, self.cols, 1), dtype=np.float32)
        obs[self.agent_pos[1]][self.agent_pos[0]][0] = 1  # Marquer la position du joueur
        return obs

    def _draw_cell(self, x, y, color):
        pygame.draw.rect(
            self.screen, color, (x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
        )

    def _draw_walls(self, x, y, walls):
        if walls[0] == "1":
            pygame.draw.line(
                self.screen, (0, 0, 0), (x * self.cell_size, y * self.cell_size),
                ((x + 1) * self.cell_size, y * self.cell_size), 2
            )
        if walls[1] == "1":
            pygame.draw.line(
                self.screen, (0, 0, 0), (x * self.cell_size, (y + 1) * self.cell_size),
                ((x + 1) * self.cell_size, (y + 1) * self.cell_size), 2
            )
        if walls[2] == "1":
            pygame.draw.line(
                self.screen, (0, 0, 0), (x * self.cell_size, y * self.cell_size),
                (x * self.cell_size, (y + 1) * self.cell_size), 2
            )
        if walls[3] == "1":
            pygame.draw.line(
                self.screen, (0, 0, 0), ((x + 1) * self.cell_size, y * self.cell_size),
                ((x + 1) * self.cell_size, (y + 1) * self.cell_size), 2
            )

    def save_screenshot(self, filename="final_solution.png"):
        #Fonction pour sauvegarder une capture d'écran de l'environnement
        pygame.image.save(self.screen, filename)

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