from abc import ABC, abstractmethod
from typing import List

import gym
import matplotlib.pyplot as plt


class AbstractRenderer(ABC):
    @abstractmethod
    def initialize(self, env):
        pass

    @abstractmethod
    def render(self, env):
        pass


class IpynbRenderer(AbstractRenderer):
    def __init__(self):
        self.img = None

    def initialize(self, env: gym.Env):
        self.img = plt.imshow(env.render())

    def render(self, env):
        try:
            from IPython import display

            self.img.set_data(env.render())
            display.display(plt.gcf())
            display.clear_output(wait=True)

        except Exception as e:
            print(e)
