from gymnasium import Env, logger, Wrapper
from typing import List, override
from director import Director
class Facade(Wrapper):
    """Facade class to wrap the environment and provide a single interface to the agent
    """
    def __init__(self, envs: List[Env], director: Director) -> None:
        """Constructor for the Facade class
        Args:
            envs (List[Env]): List of environments to be used
        """
        self.index = 0
        self.envs = envs
        if (len(envs) == 0):
            raise ValueError("No envs provided 😫")
        self.env = envs[0]
        self.director = director
        super().__init__(envs[0])

    def SwitchEnv(self, index: int) -> None:
        """Switches the environment to the one at the index
        Args:
            index (int): Index of the environment to switch to
        """
        self.index = index
        self.env = self.envs[index]
    @override
    def step(self, action):
        """Step function to step the environment
        """
        result = super().step(action)
        self.director.Update(self)
        return result