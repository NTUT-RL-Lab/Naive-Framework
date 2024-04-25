from typing import List
from gymnasium import Env, logger, Wrapper
from matplotlib.pylab import f
from director import Director
from guise import Guise


class Facade(Wrapper):
    """Facade class to wrap the environment and provide a single interface to the agent
    """

    def __init__(self, envs: List[Guise], director: Director) -> None:
        """Constructor for the Facade class
        Args:
            envs (List[Env]): List of environments to be used
        """
        self.index = 0
        self.envs = envs
        if (len(envs) == 0):
            raise ValueError("No envs provided 😫")
        for env in envs:
            env.reset()
        self.env = envs[0]
        self.director = director
        super().__init__(envs[0])

    def switch_env(self, index: int) -> None:
        """Switches the environment to the one at the index
        Args:
            index (int): Index of the environment to switch to
        """
        if (index < 0 or index >= len(self.envs)):
            raise ValueError("Invalid index provided")
        if (index == self.index):
            return
        # logger.info(f"Switching to env {index}")
        self.index = index
        self.env = self.envs[index]

    def step(self, action):
        """Step function to step the environment
        """
        observation, reward, terminated, truncated, info = super().step(
            self.env.map_action(action))
        # apply reward weights
        reward = self.env.reward(reward)

        index,  = self.director.update(
            observation, reward, terminated, truncated, info)
        self.switch_env(index)
        return observation, reward, terminated, truncated, info
