import numpy as np
import random
import matplotlib.pyplot as plt
import time
from Maze_env import Maze_env  

# Paramètres Q-Learning
alpha = 0.1  # Taux d'apprentissage
gamma = 0.9  # Facteur de discount
epsilon = 0.1  # Exploration (epsilon-greedy)
episodes = 100  # Nombre d'épisodes d'entraînement

rewards_per_episode = []  # Récompenses par épisode

def choose_action(state, epsilon, env, q_table):
    """Choisir une action selon la stratégie epsilon-greedy."""
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()  # Exploration
    else:
        y, x = state
        return np.argmax(q_table[y, x])  # Exploitation

def train_q_learning(env, plot_rewards=False, show_path=False):
    q_table = np.zeros((env.grid_size[0], env.grid_size[1], env.action_space.n))  # Table Q
    """Entraîner l'agent avec l'algorithme Q-Learning."""
    for episode in range(episodes):
        state = env.reset()
        state_pos = env.agent_pos
        done = False
        total_reward = 0  # Récompense cumulée pour cet épisode

        while not done:
            action = choose_action(state_pos, epsilon, env, q_table)
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

            # Affichage temps réel tous les N épisodes
            if (episode + 1) % 10 == 0:  # Afficher tous les 10 épisodes
                env.render()
                time.sleep(0.1)  # Pause pour observer (ajustable)

        rewards_per_episode.append(total_reward)  # Enregistrer la récompense totale
        #print(f"Épisode {episode + 1}/{episodes} - Récompense : {total_reward}")
        
    print("Entraînement terminé.")

    if plot_rewards:
        plotting_rewards()
    if show_path:
        showing_path()

    return q_table

def plotting_rewards():
    """Afficher les récompenses par épisode."""

    plt.plot(range(episodes), rewards_per_episode)
    plt.xlabel("Épisodes")
    plt.ylabel("Récompense cumulée")
    plt.title("Évolution de la récompense cumulée pendant l'apprentissage")
    plt.show()

def showing_path():
    action_symbols = ['↑', '↓', '←', '→']  # Symboles pour les actions (haut, bas, gauche, droite)

    print("Politique optimale apprise :")
    for y in range(env.grid_size[0]):
        row = ""
        for x in range(env.grid_size[1]):
            if (y, x) == env.start_pos:
                row += " S "  # Position de départ
            elif (y, x) == env.goal_pos:
                row += " G "  # Objectif
            elif env.state[y, x] == -1:  # Obstacles
                row += " X "
            else:
                action = np.argmax(q_table[y, x])  # Meilleure action selon Q-Table
                row += f" {action_symbols[action]} "
    print(row)