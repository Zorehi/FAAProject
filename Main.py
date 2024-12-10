from Maze_env import Maze_env
import pygame

if __name__ == "__main__":
    env = Maze_env()
    env.reset()
    env.render()

    done = False

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        if not done:
            action = env.action_space.sample()  # Random action
            state, reward, done = env.step(action)
            env.render()
            pygame.time.wait(1000)  # Wait for 100 milliseconds


    env.close()