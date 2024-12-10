import numpy as np
import random
import matplotlib.pyplot as plt
from Maze_env import Maze_env  

# Paramètres Q-Learning
alpha = 0.1  # Taux d'apprentissage
gamma = 0.9  # Facteur de discount
epsilon = 0.1  # Exploration (epsilon-greedy)
episodes = 100  # Nombre d'épisodes d'entraînement

# Initialiser l'environnement
env = Maze_env(grid_size=(5, 5), start_pos=(0, 0), goal_pos=(4, 4))
q_table = np.zeros((env.grid_size[0], env.grid_size[1], env.action_space.n))  # Table Q
rewards_per_episode = []  # Récompenses par épisode

def choose_action(state, epsilon):
    """Choisir une action selon la stratégie epsilon-greedy."""
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()  # Exploration
    else:
        y, x = state
        return np.argmax(q_table[y, x])  # Exploitation

def train_q_learning():
    """Entraîner l'agent avec l'algorithme Q-Learning."""
    for episode in range(episodes):
        state = env.reset()
        state_pos = env.agent_pos
        done = False
        total_reward = 0  # Récompense cumulée pour cet épisode

        while not done:
            action = choose_action(state_pos, epsilon)
            next_state, reward, done = env.step(action)
            next_pos = env.agent_pos

            # Mise à jour Q-Table
            y, x = state_pos
            next_y, next_x = next_pos
            q_table[y, x, action] += alpha * (
                reward + gamma * np.max(q_table[next_y, next_x]) - q_table[y, x, action]
            )
            
            total_reward += reward  # Accumuler les récompenses
            state_pos = next_pos
            
        rewards_per_episode.append(total_reward)  # Enregistrer la récompense totale
        print(f"Épisode {episode + 1}/{episodes} - Récompense : {total_reward}")

    print("Entraînement terminé.")

    return q_table

