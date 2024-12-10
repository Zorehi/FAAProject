import gym
from gym import spaces
import numpy as np

class BasicEnv(gym.Env):
    def __init__(self, grid_size=(5, 5), start_pos=(0, 0), goal_pos=(4, 4)):
        super(BasicEnv, self).__init__()
        self.grid_size = grid_size
        self.start_pos = start_pos
        self.goal_pos = goal_pos

        self.action_space = spaces.Discrete(4)  # 4 actions: up, down, left, right
        self.observation_space = spaces.Box(low=0, high=1, shape=(grid_size[0], grid_size[1]), dtype=np.float32)

        self.state = np.zeros(grid_size)
        self.state[self.start_pos] = 1
        self.agent_pos = self.start_pos

    def reset(self):
        self.state = np.zeros(self.grid_size)
        self.state[self.start_pos] = 1
        self.agent_pos = self.start_pos
        return self.state

    def step(self, action):
        new_pos = list(self.agent_pos)

        if action == 0:  # up
            new_pos[0] = max(self.agent_pos[0] - 1, 0)
        elif action == 1:  # down
            new_pos[0] = min(self.agent_pos[0] + 1, self.grid_size[0] - 1)
        elif action == 2:  # left
            new_pos[1] = max(self.agent_pos[1] - 1, 0)
        elif action == 3:  # right
            new_pos[1] = min(self.agent_pos[1] + 1, self.grid_size[1] - 1)

        self.state[self.agent_pos] = 0
        self.agent_pos = tuple(new_pos)
        self.state[self.agent_pos] = 1

        done = self.agent_pos == self.goal_pos
        reward = 1 if done else 0

        return self.state, reward, done, {}

    def render(self, mode='human'):
        for row in self.state:
            print(' '.join(['A' if cell == 1 else '.' for cell in row]))
        print()

# Example usage
if __name__ == "__main__":
    env = BasicEnv()
    env.reset()
    env.render()

    done = False
    while not done:
        action = env.action_space.sample()  # Random action
        state, reward, done, _ = env.step(action)
        env.render()
        if done:
            print("Goal reached!")