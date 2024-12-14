import numpy as np
import random
import time
import matplotlib.pyplot as plt
import pygame
import sys
import matplotlib.image as mpimg

# Paramètres Q-Learning
ALPHA = 0.1  # Taux d'apprentissage
GAMMA = 0.9  # Facteur de discount
EPSILON = 1  # Exploration (epsilon-greedy)
EPSILON_DECAY = 0.9 # Taux de décroissance de l'exploration
EPSILON_MIN = 0.01 # Valeur minimale de l'exploration
EPISODES = 1000  # Nombre d'épisodes d'entraînement
MAX_STEPS = 5000 # Nombre maximal d'étapes par épisode

def choose_action(pos, epsilon, env, q_table):
    """Choisir une action selon la stratégie epsilon-greedy."""
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()  # Exploration
    else:
        x, y = pos
        return np.argmax(q_table[y, x, :])  # Exploitation

def train_q_learning(env, alpha=ALPHA, gamma=GAMMA, epsilon=EPSILON, episodes=EPISODES, show_training=False):
    q_table = np.zeros((env.rows, env.cols, env.action_space.n))  # Table Q
    rewards_per_episode = []  # Récompenses par épisode

    print("Entraînement Q-learning en cours...")

    """Entraîner l'agent avec l'algorithme Q-Learning."""
    for episode in range(episodes):
        env.reset()
        state_pos = env.agent_pos
        total_reward = 0  # Récompense cumulée pour cet épisode

        for step in range(MAX_STEPS):
            action = choose_action(state_pos, epsilon, env, q_table)
            _, reward, done, _ = env.step(action)
            next_pos = env.agent_pos

            # Mise à jour Q-Table
            x, y = state_pos
            next_x, next_y = next_pos
            q_table[y, x, action] += alpha * (
                reward + gamma * np.max(q_table[next_y, next_x, :]) - q_table[y, x, action]
            )
            
            total_reward += reward  # Accumuler les récompenses
            state_pos = next_pos

            # Affichage temps réel tous les N épisodes
            if show_training:
                if (episode + 1) % 100 == 0:  # Afficher tous les 10 épisodes
                    env.render()
                    time.sleep(0.005)  # Pause pour observer (ajustable)

            if done:
                break

        epsilon = max(epsilon * EPSILON_DECAY, EPSILON_MIN)

        rewards_per_episode.append(total_reward)  # Enregistrer la récompense totale
        #print(f"Épisode {episode + 1}/{episodes} - Récompense : {total_reward}")
        
    #print("Entraînement Q-learning terminé.")

    return q_table, rewards_per_episode


def train_sarsa(env, alpha=ALPHA, gamma=GAMMA, epsilon=EPSILON, episodes=EPISODES, show_training=False):
    """
    Entraîne l'agent avec l'algorithme SARSA.
    Retourne la Q-Table entraînée et les récompenses cumulées.
    """
    q_table = np.zeros((env.rows, env.cols, env.action_space.n))  # Initialisation de la Q-Table
    rewards_per_episode = []  # Stockage des récompenses

    print("Entraînement SARSA en cours...")

    for episode in range(episodes):
        env.reset()
        state_pos = env.agent_pos
        action = choose_action(state_pos, epsilon, env, q_table)
        total_reward = 0

        for step in range(MAX_STEPS):
            _, reward, done, _ = env.step(action)
            next_pos = env.agent_pos
            next_action = choose_action(next_pos, epsilon, env, q_table)  # Choisir a' selon epsilon-greedy

            # Mettre à jour la Q-Table selon SARSA
            x, y = state_pos
            next_x, next_y = next_pos
            q_table[y, x, action] += alpha * (
                reward + gamma * q_table[next_y, next_x, next_action] - q_table[y, x, action]
            )

            state_pos = next_pos  # Mettre à jour l'état
            action = next_action  # Mettre à jour l'action
            total_reward += reward

            if show_training:
                #env.setup_pygame()
                if (episode + 1) % 10 == 0:  # Afficher tous les 10 épisodes
                    env.render()
                    time.sleep(0.005)  # Pause pour observer (ajustable)

            if done:
                break

        epsilon = max(epsilon * EPSILON_DECAY, EPSILON_MIN)

        rewards_per_episode.append(total_reward)  # Enregistrer la récompense totale de l'épisode
        #print(f"Épisode {episode + 1}/{episodes} - Récompense : {total_reward}")

    #print("Entraînement SARSA terminé.")

    return q_table, rewards_per_episode


def show_final_solution(env, q_table, learning_type):
    """
    Affiche la solution finale en utilisant la Q-Table entraînée.
    """
    
    env.reset()
    state_pos = env.agent_pos
    done = False
    env.render()  # Première affichage

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        x, y = state_pos
        action = np.argmax(q_table[y, x, :])  # Exploitation totale : choix de l'action optimale
        _, reward, done, _ = env.step(action)
        state_pos = env.agent_pos
        env.render()  # Affiche l'environnement après chaque étape
        pygame.time.wait(100)  # Pause pour observer l'exécution (100 ms)

    screen_path = f"final_solution_{learning_type}.png"
    env.save_screenshot(filename=screen_path)  # Sauvegarder la solution finale
    #pygame.display.quit()  # Quitter proprement Pygame après affichage


def show_final_state(learning_type):
    screen_path = f"final_solution_{learning_type}.png"
    img = mpimg.imread(screen_path)
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.axis('off')  # Masquer les axes
    plt.title("Situation finale de l'agent")
    plt.show()


def plotting_rewards(rewards):
    """Afficher l'évolution des récompenses par épisode."""

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(rewards) + 1), rewards, label="Récompense par épisode")
    plt.xlabel("Épisodes")
    plt.ylabel("Récompenses cumulées")
    plt.title("Évolution des récompenses par épisode")
    plt.grid(True)
    plt.legend()
    plt.show()


def best_worst_rewards(rewards):
    return max(rewards), min(rewards)