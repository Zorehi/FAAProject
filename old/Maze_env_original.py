import gym
from gym import spaces
import numpy as np
import pygame
import random 

GRID_SIZE = (10,10)
START_POS = (0, 0)
GOAL_POS = (8,5)
OBSTACLES = 50

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
        self.path = [self.agent_pos]

        self.obstacles = generate_obstacles(self.grid_size, self.start_pos, self.goal_pos, OBSTACLES)
        #self.obstacles = testing_obstacles()



    def setup_pygame(self):
        #Initialisation de Pygame pour afficher l'environnement
        pygame.init()
        self.cell_size = 100
        self.screen_size = (self.grid_size[1] * self.cell_size, self.grid_size[0] * self.cell_size)
        self.screen = pygame.display.set_mode(self.screen_size)
        pygame.display.set_caption('Maze Environment')
        self.use_pygame = True

    def reset(self):
        #Réinitialisation de l'environnement au début de chaque épisode

        self.state = np.zeros(self.grid_size)
        self.state[self.start_pos] = 1
        self.agent_pos = self.start_pos
        self.path = [self.agent_pos]

        for obstacle in self.obstacles:
            self.state[obstacle] = -1

        return self.state
    

    def step(self, action):
        #Fonction qui permet de faire avancer l'agent

        #Calcul de la nouvelle position de l'agent en fonction de l'action choisie
        previous_pos = self.agent_pos
        new_pos = list(self.agent_pos)
        reward = 0

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
            self.path.append(self.agent_pos)  # Ajouter la nouvelle position au chemin
        else :
            reward = -1

        #Calcul de la récompense et de la fin de l'épisode
        done = self.agent_pos == self.goal_pos  #L'épisode est terminé si l'agent atteint l'objectif

        min_go_back = self.grid_size[0] // 2
        #Récompense : 1 si l'épisode est terminé, sinon appel de la fonction compute_reward
        if done:
            reward = 1
        elif len(self.path) > min_go_back and self.agent_pos in self.path[:-min_go_back] : #Si l'agent revient sur ses pas, récompense négative : évite les boucles ou le blocage
            reward = -0.2
        else:
            reward += self.compute_reward(previous_pos, self.agent_pos)

        return self.state, reward, done
    
    def compute_reward(self, previous_pos, new_pos):
        #Calcul de la récompense en fonction de la distance euclidienne entre la nouvelle position et la position de l'objectif

        dist_new = np.linalg.norm(np.array(new_pos) - np.array(self.goal_pos))
        dist_old = np.linalg.norm(np.array(previous_pos) - np.array(self.goal_pos))

        return (dist_old - dist_new)*0.5

    def render(self, mode='human'):
        #Fonction pour afficher l'environnement avec Pygame
        self.screen.fill((255, 255, 255))  # Fond blanc

        for y in range(self.grid_size[0]):
            for x in range(self.grid_size[1]):
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)

                # Dessiner les cases visitées en jaune
                if (y, x) in self.path:
                    pygame.draw.rect(self.screen, (255, 255, 0), rect)  # Jaune pour le chemin

                if self.state[y, x] == -1:  # Obstacle
                    pygame.draw.rect(self.screen, (0, 0, 0), rect)  # Noir pour les obstacles
                elif (y, x) == self.agent_pos:
                    pygame.draw.rect(self.screen, (0, 0, 255), rect)  # Bleu pour l'agent
                elif (y, x) == self.goal_pos:
                    pygame.draw.rect(self.screen, (0, 255, 0), rect)  # Vert pour l'objectif
                else:
                    pygame.draw.rect(self.screen, (200, 200, 200), rect, 1)  # Gris pour la grille
        pygame.display.flip()

    def save_screenshot(self, filename="final_solution.png"):
        #Fonction pour sauvegarder une capture d'écran de l'environnement
        pygame.image.save(self.screen, filename)

    def close(self):
        if self.use_pygame:
            pygame.display.quit()
            pygame.quit()

def generate_obstacles(grid_size, start_pos, goal_pos, num_obstacles):
    """
    Génère des obstacles cohérents en garantissant un chemin du départ à l'arrivée.
    Utilise un algorithme DFS pour éviter les blocages.
    """
    obstacles = set()
    path = []
    visited = set()
    stack = [start_pos]  # Pile pour suivre le chemin

    # Étape 1 : Générer un chemin valide avec DFS
    while stack:
        current_pos = stack[-1]
        visited.add(current_pos)
        path.append(current_pos)

        if current_pos == goal_pos:
            break  # Chemin complet atteint l'objectif

        y, x = current_pos

        # Trouver les mouvements valides non visités
        possible_moves = []
        if y > 0 and (y - 1, x) not in visited:  # Haut
            possible_moves.append((y - 1, x))
        if y < grid_size[0] - 1 and (y + 1, x) not in visited:  # Bas
            possible_moves.append((y + 1, x))
        if x > 0 and (y, x - 1) not in visited:  # Gauche
            possible_moves.append((y, x - 1))
        if x < grid_size[1] - 1 and (y, x + 1) not in visited:  # Droite
            possible_moves.append((y, x + 1))

        # Si aucun mouvement valide, revenir en arrière
        if not possible_moves:
            stack.pop()
            path.pop()
        else:
            next_pos = random.choice(possible_moves)
            stack.append(next_pos)

    # Étape 2 : Générer des obstacles en dehors du chemin garanti
    escape = 0
    while len(obstacles) < num_obstacles and escape < 1000:
        pos = (random.randint(0, grid_size[0] - 1), random.randint(0, grid_size[1] - 1))
        if pos not in path and pos != start_pos and pos != goal_pos:
            obstacles.add(pos)
        escape += 1

    return obstacles

def testing_obstacles():
    obstacles = set()

    #obstacles : 10,11,12,22,32,42,52,53,54,44,34,24,14 :
    obstacles.add((1,0))
    obstacles.add((1,1))
    obstacles.add((1,2))
    obstacles.add((2,2))
    obstacles.add((3,2))
    obstacles.add((4,2))
    obstacles.add((5,2))
    obstacles.add((5,3))
    obstacles.add((5,4))
    obstacles.add((4,4))
    obstacles.add((3,4))
    obstacles.add((2,4))
    obstacles.add((1,4))

    return obstacles
