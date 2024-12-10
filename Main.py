from Maze_env import Maze_env
from training_RL import train_q_learning
import pygame
import numpy as np

if __name__ == "__main__":
    env = Maze_env()
    state = env.reset()
    state_pos = env.agent_pos
    done = False

    q_table = train_q_learning()


    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        if not done:
            y, x = state_pos
            action = np.argmax(q_table[y, x])  # Exploitation totale
            state, reward, done = env.step(action)
            env.render()
            state_pos = env.agent_pos
            pygame.time.wait(1000)  # Wait for 100 milliseconds
        
        '''if not done:
            action = env.action_space.sample()  # Random action
            state, reward, done = env.step(action)
            env.render()
            pygame.time.wait(1000)'''


    env.close()