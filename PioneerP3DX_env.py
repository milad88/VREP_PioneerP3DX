import gym
from gym import spaces, logger
from Pioneer_interface import PioneerP3DX_interface
import numpy as np

from utility import Dist


class PioneerP3DX(gym.Env):
    def __init__(self):
        self.robot_interface = PioneerP3DX_interface()
        self.action_space = spaces.Box(-2., 2., dtype=np.float32, shape=[2])
        self.observation_space = spaces.Box(0, 255, shape=[64, 64, 3])
        self.target_pos_1 = [-0.05, -0.25]
        self.target_pos_2 = [-0.05, -1.25]

    def get_action_space(self):
        return self.action_space

    def step(self, action):

        self.robot_interface.move(action[0], action[1])
        obs = self.getObservation()
        pos = self.robot_interface.getPosition()
        reward = self.reward(pos)
        done = False
        if reward < 0.4:
            done = True
        return obs, reward, done, None

    def reset(self):
        return self.robot_interface.getCameraImage()[2]

    def getObservation(self):
        return self.robot_interface.getCameraImage()[2]

    def render(self, mode='human'):
        pass

    def reward(self, pos):
        base = 1
        if pos[1] > -1.25 and pos[1] < -0.25 and pos[0] > -0.35 and pos[0] < 0.35:
            base = 0
        if pos[1] > -0.775:
            return 1 / (Dist(pos[0], self.target_pos_1[0], pos[1], self.target_pos_1[1]) + base)
        return 1 / (Dist(pos[0], self.target_pos_2[0], pos[1], self.target_pos_2[1]) + base)
