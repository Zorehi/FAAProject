from Maze_env import Maze_env
from reinforcement_learning import train_q_learning, train_sarsa
import pygame
import numpy as np

# Un doit être True et l'autre False, si les deux sont True, Q-learning sera executé
Q = False
SARSA = True

if __name__ == "__main__":
    env = Maze_env()
    state_pos = env.agent_pos
    done = False

    if Q:
        q_table_q, _ = train_q_learning(env, show_training=False)
    elif SARSA:
        q_table_sarsa, _ = train_sarsa(env, show_training=False)

    env.setup_pygame()
    state = env.reset()
    env.render()

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        if Q :
            if not done:
                y, x = state_pos
                action = np.argmax(q_table_q[y, x])  # Exploitation totale
                state, reward, done = env.step(action)
                env.render()
                state_pos = env.agent_pos
                pygame.time.wait(500)  # Wait for 100 milliseconds
        elif SARSA:
            if not done:
                y, x = state_pos
                action = np.argmax(q_table_sarsa[y, x])
                state, reward, done = env.step(action)
                env.render()
                state_pos = env.agent_pos
                pygame.time.wait(500)
        else :
            if not done:
                action = env.action_space.sample()  # Random action
                state, reward, done = env.step(action)
                env.render()
                pygame.time.wait(500)

    env.close()