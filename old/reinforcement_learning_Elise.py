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
EPSILON = 0.1  # Exploration (epsilon-greedy)
EPISODES = 100  # Nombre d'épisodes d'entraînement

def choose_action(state, epsilon, env, q_table):
    """Choisir une action selon la stratégie epsilon-greedy."""
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()  # Exploration
    else:
        y, x = state
        return np.argmax(q_table[y, x])  # Exploitation

def train_q_learning(env, alpha=ALPHA, gamma=GAMMA, epsilon=EPSILON, episodes=EPISODES, show_training=False):
    q_table = np.zeros((env.grid_size[0], env.grid_size[1], env.action_space.n))  # Table Q
    rewards_per_episode = []  # Récompenses par épisode

    if show_training:
        env.setup_pygame()

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
            if show_training:
                if (episode + 1) % 10 == 0:  # Afficher tous les 10 épisodes
                    env.render()
                    time.sleep(0.1)  # Pause pour observer (ajustable)

        rewards_per_episode.append(total_reward)  # Enregistrer la récompense totale
        #print(f"Épisode {episode + 1}/{episodes} - Récompense : {total_reward}")
        
    #print("Entraînement Q-learning terminé.")

    return q_table, rewards_per_episode


def train_sarsa(env, alpha=ALPHA, gamma=GAMMA, epsilon=EPSILON, episodes=EPISODES, show_training=False):
    """
    Entraîne l'agent avec l'algorithme SARSA.
    Retourne la Q-Table entraînée et les récompenses cumulées.
    """
    q_table = np.zeros((env.grid_size[0], env.grid_size[1], env.action_space.n))  # Initialisation de la Q-Table
    rewards_per_episode = []  # Stockage des récompenses

    if show_training:
        env.setup_pygame()

    for episode in range(episodes):
        state = env.reset()
        state_pos = env.agent_pos
        action = choose_action(state_pos, epsilon, env, q_table)
        total_reward = 0
        done = False

        while not done:
            next_state, reward, done = env.step(action)
            next_pos = env.agent_pos
            next_action = choose_action(next_pos, epsilon, env, q_table)  # Choisir a' selon epsilon-greedy

            # Mettre à jour la Q-Table selon SARSA
            y, x = state_pos
            next_y, next_x = next_pos
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
                    time.sleep(0.1)  # Pause pour observer (ajustable)

        rewards_per_episode.append(total_reward)  # Enregistrer la récompense totale de l'épisode
        #print(f"Épisode {episode + 1}/{episodes} - Récompense : {total_reward}")

    #print("Entraînement SARSA terminé.")

    return q_table, rewards_per_episode


def show_final_solution(env, q_table, learnin_type):
    """
    Affiche la solution finale en utilisant la Q-Table entraînée.
    """
    # Réinitialiser l'environnement Pygame
    env.setup_pygame()
    
    env.reset()
    state_pos = env.agent_pos
    done = False
    env.render()  # Première affichage

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        y, x = state_pos
        action = np.argmax(q_table[y, x])  # Exploitation totale : choix de l'action optimale
        state, reward, done = env.step(action)
        state_pos = env.agent_pos
        env.render()  # Affiche l'environnement après chaque étape
        pygame.time.wait(500)  # Pause pour observer l'exécution (500 ms)

    screen_path = f"final_solution_{learnin_type}.png"
    env.save_screenshot(filename=screen_path)  # Sauvegarder la solution finale
    pygame.display.quit()  # Quitter proprement Pygame après affichage


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