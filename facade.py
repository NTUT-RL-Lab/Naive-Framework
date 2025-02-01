from typing import Any
import numpy as np
import gymnasium as gym
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
        self.blend = director.blend
        self._reward_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(1, ), dtype=np.float32)
        self._observation_space = gym.spaces.Box(
            low=0, high=255, shape=(4, 84, 84), dtype=np.float32)
        self._frame_stack = 4
        self._frames = np.zeros(
            self._observation_space.shape, dtype=np.float32)
        super().__init__(envs[0])

    @property
    def reward_space(self):
        return self._reward_space

    @property
    def observation_space(self):
        return self._observation_space

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

        if self.blend:
            observations, rewards, terminateds, truncateds, infos = [], [], [], [], []
            for index in range(len(self.envs)):
                self.switch_env(index)
                for _ in range(self._frame_stack):
                    observation, reward, terminated, truncated, info = super().step(
                        self.env.map_action(action))
                    self._frames = np.roll(self._frames, shift=-1, axis=0)
                    self._frames[-1, :, :] = observation
                observations.append(self._frames)
                rewards.append(self.env.reward(reward))
                terminateds.append(terminated)
                truncateds.append(truncated)
                infos.append(info)
            observation = np.mean(observations, axis=0)
            reward = np.max(rewards)
            terminated = np.min(terminateds) != 0
            truncated = np.min(truncateds) != 0
            info = {k: np.mean([i[k] for i in infos]) for k in infos[0]}
            return observation, reward, terminated, truncated, info
        for _ in range(self._frame_stack):
            observation, reward, terminated, truncated, info = super().step(
                self.env.map_action(action))
            self._frames = np.roll(self._frames, shift=-1, axis=0)
            self._frames[-1, :, :] = observation
        observation = self._frames
        reward = self.env.reward(reward)
        index,  = self.director.update(
            observation, reward, terminated, truncated, info)
        self.switch_env(index)
        return observation, reward, terminated, truncated, info

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if self.blend:
            observations = []
            infos = []
            for index in range(len(self.envs)):
                self.switch_env(index)
                obs, info = super().reset(seed=seed, options=options)
                self._frames = np.zeros(
                    self._observation_space.shape, dtype=np.float32)
                for _ in range(self._frame_stack):
                    observation, _ = super().step(self.env.map_action(0))
                    self._frames = np.roll(self._frames, shift=-1, axis=0)
                    self._frames[-1, :, :] = observation
                observations.append(self._frames)
                infos.append(info)
            observation = np.mean(observations, axis=0)
            return observation, {k: np.mean([i[k] for i in infos]) for k in infos[0]}
        observation, infos = super().reset(seed=seed, options=options)
        self._frames = np.zeros(
            self._observation_space.shape, dtype=np.float32)
        for _ in range(self._frame_stack):
            observation, _ = super().step(self.env.map_action(0))
            self._frames = np.roll(self._frames, shift=-1, axis=0)
            self._frames[-1, :, :] = observation
        observation = self._frames
        return observation, infos

    def close(self):
        if self.blend:
            for env in self.envs:
                env.close()
            return
        return super().close()
