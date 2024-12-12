from Maze_env import Maze_env
from Q_learning import train_q_learning
import pygame
import numpy as np

Q = True

if __name__ == "__main__":
    env = Maze_env()
    state_pos = env.agent_pos
    done = False

    if Q:
        q_table = train_q_learning(env, plot_rewards=False, show_training=False)

    state = env.reset()


    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        if Q :
            if not done:
                y, x = state_pos
                action = np.argmax(q_table[y, x])  # Exploitation totale
                state, reward, done = env.step(action)
                env.render()
                state_pos = env.agent_pos
                pygame.time.wait(500)  # Wait for 100 milliseconds
        else :
            if not done:
                action = env.action_space.sample()  # Random action
                state, reward, done = env.step(action)
                env.render()
                pygame.time.wait(500)


    env.close()