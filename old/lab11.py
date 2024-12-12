import gym
from gym import spaces
import numpy as np
import pygame
import sys
import random

grid_s = (7, 7)
start_p = (0, 0)
goal_p = (6, 4)

class LabyrinthEnv(gym.Env): #Classe qui hérite de la classe gym.Env de la librairie gym
    def __init__(self, grid_size=grid_s, start_pos=start_p, goal_pos=goal_p, obstacles=[]):
        super(LabyrinthEnv, self).__init__() #Appel du constructeur de la classe mère
        self.grid_size = grid_size #Initialisation des attributs de la classe
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        self.obstacles = obstacles

        self.visited = set()

        self.action_space = spaces.Discrete(4)  # 4 actions: haut, bas, gauche, droite
        self.observation_space = spaces.Box(low=0, high=1, shape=(grid_size[0], grid_size[1]), dtype=np.float32) #Espace d'observation

        self.state = np.zeros(grid_size) #Initialisation de l'état
        self.state[self.start_pos] = 1 #Initialisation de la position de départ
        self.agent_pos = self.start_pos #Initialisation de la position de l'agent

        # Pygame setup
        self.screen_size = 600
        self.cell_size = self.screen_size // max(self.grid_size)
        self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
        pygame.display.set_caption('Labyrinth Environment')

    def reset(self): #Fonction de réinitialisation de l'environnement
        self.state = np.zeros(self.grid_size)
        self.state[self.start_pos] = 1
        self.agent_pos = self.start_pos
        return self.state

    def step(self, action): #Fonction qui permet de faire avancer l'agent
        new_pos = list(self.agent_pos)

        if action == 0:  # haut
            new_pos[0] = max(self.agent_pos[0] - 1, 0)
        elif action == 1:  # bas
            new_pos[0] = min(self.agent_pos[0] + 1, self.grid_size[0] - 1)
        elif action == 2:  # gauche
            new_pos[1] = max(self.agent_pos[1] - 1, 0)
        elif action == 3:  # droite
            new_pos[1] = min(self.agent_pos[1] + 1, self.grid_size[1] - 1)

        new_pos = tuple(new_pos)

        if new_pos in self.obstacles or new_pos in self.visited:
            # Stay in place if trying to move into an obstacle or visited position
            new_pos = self.agent_pos
            reward = -1
            done = False
            self.visited.add(new_pos)
        elif new_pos == self.goal_pos:
            reward = 100000
            done = True
        else:
            #Calcul de la distance euclidienne entre la position de l'agent et la position de l'objectif
            reward = - (np.linalg.norm(np.array(new_pos) - np.array(self.goal_pos)))
            print(reward)
            done = False


        self.agent_pos = new_pos
        self.state[self.agent_pos] = 1 #Mise à jour de la position de l'agent dans l'état

        return self.state, reward, done, {}

    def render(self, mode='human'): #Fonction qui permet de visualiser l'environnement avec Pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        self.screen.fill((255, 255, 255))  # White background

        # Draw grid
        for row in range(self.grid_size[0]):
            for col in range(self.grid_size[1]):
                rect = pygame.Rect(col * self.cell_size, row * self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, (200, 200, 200), rect, 1)

        # Draw obstacles
        for obstacle in self.obstacles:
            rect = pygame.Rect(obstacle[1] * self.cell_size, obstacle[0] * self.cell_size, self.cell_size,
                               self.cell_size)
            pygame.draw.rect(self.screen, (0, 0, 0), rect)

        # Draw goal
        goal_rect = pygame.Rect(self.goal_pos[1] * self.cell_size, self.goal_pos[0] * self.cell_size, self.cell_size,
                                self.cell_size)
        pygame.draw.rect(self.screen, (0, 255, 0), goal_rect)

        # Draw agent
        agent_rect = pygame.Rect(self.agent_pos[1] * self.cell_size, self.agent_pos[0] * self.cell_size, self.cell_size,
                                 self.cell_size)
        pygame.draw.rect(self.screen, (55, 0, 255), agent_rect)

        pygame.display.flip()


def path_to_goal(start_pos, goal_pos, grid_size=grid_s):
    path = []
    path.append(start_pos)
    while path[-1] != goal_pos:
        move = random.randint(0, 5)
        current_pos = path[-1]
        if (move == 0 or move == 4) and current_pos[0] > 0:  # haut
            path.append((current_pos[0] - 1, current_pos[1]))
        elif move == 1 and current_pos[0] < grid_size[0] - 1:  # bas
            path.append((current_pos[0] + 1, current_pos[1]))
        elif move == 2 and current_pos[1] > 0:  # gauche
            path.append((current_pos[0], current_pos[1] - 1))
        elif (move == 3 or move == 5) and current_pos[1] < grid_size[1] - 1:  # droite
            path.append((current_pos[0], current_pos[1] + 1))

    return path



def generate_obstacles(path, num_obstacles, grid_size=grid_s):
    obstacles = set()
    attempts = 0
    max_attempts = 1000  # Maximum attempts to avoid infinite loop
    while len(obstacles) < num_obstacles and attempts < max_attempts:
        possible_obstacle = (random.randint(0, grid_size[0] - 1), random.randint(0, grid_size[1] - 1))
        if possible_obstacle not in path:
            obstacles.add(possible_obstacle)
        attempts += 1
    return list(obstacles)

if __name__ == "__main__":
    path = path_to_goal(start_p, goal_p)
    obs = generate_obstacles(path, num_obstacles=1000)
    env = LabyrinthEnv(obstacles=obs)

    done = False
    while not done:
        env.render()
        action = env.action_space.sample()  # random action
        obs, reward, done, info = env.step(action)
        pygame.time.wait(500)