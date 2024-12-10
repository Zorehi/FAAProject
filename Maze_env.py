import gym
from gym import spaces
import numpy as np
import pygame
import random 

GRID_SIZE = (5, 5)
START_POS = (0, 0)
GOAL_POS = (4, 4)

class Maze_env(gym.Env):
    def __init__(self, grid_size=GRID_SIZE, start_pos=START_POS, goal_pos=GOAL_POS):
        #Environnement héritant de la classe gym.Env
        super(Maze_env, self).__init__()       
        self.grid_size = grid_size              #Initialisation des attributs de la classe
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        

        self.action_space = spaces.Discrete(4)  # 4 actions: haut, bas, gauche, droite

        # Espace d'observation : grille de taille grid_size avec des valeurs entre -1 (obstacle) et 1 (position de l'agent) avec 0 : cases vides
        self.observation_space = spaces.Box(low=-1, high=1, shape=(grid_size[0], grid_size[1]), dtype=np.float32) 

        # Initialisation de l'état : 0 partout sauf à la position de départ --> accord avec l'espace d'observation
        self.state = np.zeros(grid_size)
        self.state[self.start_pos] = 1

        # Position de l'agent : choisie dans les constantes
        self.agent_pos = self.start_pos

        # Pygame setup
        pygame.init()
        self.cell_size = 100
        self.screen_size = (self.grid_size[1] * self.cell_size, self.grid_size[0] * self.cell_size)
        self.screen = pygame.display.set_mode(self.screen_size)
        pygame.display.set_caption('Basic Environment')

    def reset(self):
        #Réinitialisation de l'environnement au début de chaque épisode
        self.state = np.zeros(self.grid_size)
        self.state[self.start_pos] = 1
        self.agent_pos = self.start_pos

        self.obstacles = generate_obstacles(self.grid_size, self.start_pos, self.goal_pos, 5)

        for obstacle in self.obstacles:
            self.state[obstacle] = -1

        return self.state
    

    def step(self, action):
        #Fonction qui permet de faire avancer l'agent

        #Calcul de la nouvelle position de l'agent en fonction de l'action choisie
        previous_pos = self.agent_pos
        new_pos = list(self.agent_pos)

        if action == 0:  # haut
            new_pos[0] = max(self.agent_pos[0] - 1, 0)
        elif action == 1:  # bas
            new_pos[0] = min(self.agent_pos[0] + 1, self.grid_size[0] - 1)
        elif action == 2:  # gauche
            new_pos[1] = max(self.agent_pos[1] - 1, 0)
        elif action == 3:  # droite
            new_pos[1] = min(self.agent_pos[1] + 1, self.grid_size[1] - 1)


        # Vérifier si la nouvelle position est un obstacle
        if self.state[tuple(new_pos)] != -1.0:  # Si ce n'est pas un obstacle
            self.state[self.agent_pos] = 0  # Effacer l'ancienne position
            self.agent_pos = tuple(new_pos)  # Mettre à jour la position
            self.state[self.agent_pos] = 1  # Mettre à jour la nouvelle position


        #Calcul de la récompense et de la fin de l'épisode
        done = self.agent_pos == self.goal_pos  #L'épisode est terminé si l'agent atteint l'objectif

        #Récompense : 1 si l'épisode est terminé, sinon appel de la fonction compute_reward
        if done:
            reward = 1
        else:
            reward = self.compute_reward(previous_pos, self.agent_pos)

        print(reward)

        return self.state, reward, done
    
    def compute_reward(self, previous_pos, new_pos):
        #Calcul de la récompense en fonction de la distance euclidienne entre la nouvelle position et la position de l'objectif

        dist_new = np.linalg.norm(np.array(new_pos) - np.array(self.goal_pos))
        dist_old = np.linalg.norm(np.array(previous_pos) - np.array(self.goal_pos))

        return dist_old - dist_new

    def render(self, mode='human'):
        #Fonction pour afficher l'environnement avec Pygame
        self.screen.fill((255, 255, 255))  # Fond blanc

        for y in range(self.grid_size[0]):
            for x in range(self.grid_size[1]):
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                if self.state[y, x] == -1:  # Obstacle
                    pygame.draw.rect(self.screen, (0, 0, 0), rect)  # Noir pour les obstacles
                elif (y, x) == self.agent_pos:
                    pygame.draw.rect(self.screen, (0, 0, 255), rect)  # Bleu pour l'agent
                elif (y, x) == self.goal_pos:
                    pygame.draw.rect(self.screen, (0, 255, 0), rect)  # Vert pour l'objectif
                else:
                    pygame.draw.rect(self.screen, (200, 200, 200), rect, 1)  # Gris pour la grille
        pygame.display.flip()

    def close(self):
        pygame.quit()


def generate_obstacles(grid_size, start_pos, goal_pos, num_obstacles):
        """
        Génère des obstacles dans une grille.
        
        Paramètres :
            grid_size (tuple) : Taille de la grille (rows, cols).
            start_pos (tuple) : Position de départ (y, x).
            goal_pos (tuple) : Position de l'objectif (y, x).
            num_obstacles (int) : Nombre d'obstacles à générer.
        
        Retourne :
            List[tuple] : Liste des positions des obstacles.
        """
        obstacles = set()
        
        while len(obstacles) < num_obstacles:
            # Générer une position aléatoire dans la grille
            pos = (random.randint(0, grid_size[0] - 1), random.randint(0, grid_size[1] - 1))
            
            # Vérifier que la position n'est ni le départ, ni l'objectif, ni déjà un obstacle
            if pos != start_pos and pos != goal_pos:
                obstacles.add(pos)
        
        return list(obstacles)
