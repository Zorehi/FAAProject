from MazeEnv_Jeremy import MazeEnv as Maze_env
from ReinforcementLearning_Jeremy import train_q_learning, train_sarsa
import pygame
import numpy as np

# Un doit être True et l'autre False, si les deux sont True, Q-learning sera executé
Q = False
SARSA = True
SHOW_TRAINING = False
WAITING_TIME = 100

if __name__ == "__main__":
    env = Maze_env()
    state_pos = env.agent_pos
    done = False

    if Q:
        q_table_q, _ = train_q_learning(env, show_training=SHOW_TRAINING)
    elif SARSA:
        q_table_sarsa, _ = train_sarsa(env, show_training=SHOW_TRAINING)

    env.reset()
    env.render()

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        if Q :
            if not done:
                x, y = env.agent_pos
                action = np.argmax(q_table_q[y, x, :])  # Exploitation totale
                _, reward, done, _ = env.step(action)
                env.render()
                pygame.time.wait(WAITING_TIME)  # Wait for 100 milliseconds
        elif SARSA:
            if not done:
                x, y = env.agent_pos
                action = np.argmax(q_table_sarsa[y, x, :])
                _, reward, done, _ = env.step(action)
                env.render()
                pygame.time.wait(WAITING_TIME)
        else :
            if not done:
                action = env.action_space.sample()  # Random action
                _, reward, done, _ = env.step(action)
                env.render()
                pygame.time.wait(WAITING_TIME)

    env.close()