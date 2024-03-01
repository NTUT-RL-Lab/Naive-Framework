from gymnasium import Env, logger, Wrapper
from typing import List
class Facade(Wrapper):
    """Facade class to wrap the environment and provide a single interface to the agent
    """
    def __init__(self, envs: List[Env]) -> None:
        """Constructor for the Facade class
        Args:
            envs (List[Env]): List of environments to be used
        """
        self.index = 0
        self.envs = envs
        if (len(envs) == 0):
            raise ValueError("No envs provided 😫")
        self.env = envs[0]
        super().__init__(envs[0])

    def SwitchEnv(self, index: int) -> None:
        """Switches the environment to the one at the index
        Args:
            index (int): Index of the environment to switch to
        """
        self.index = index
        self.env = self.envs[index]