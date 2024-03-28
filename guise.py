import gymnasium as gym
from gymnasium import Wrapper, Env, logger, spaces
from gymnasium.wrappers.pixel_observation import PixelObservationWrapper
import numpy as np
from gymnasium.error import DependencyNotInstalled
from ops import OPS_N, Ops
from typing import Callable


class Guise(PixelObservationWrapper):
    def __init__(self, env: Env) -> None:
        super().__init__(
            env
        )
        self.observation_space = self.observation_space['pixels']
        self.shape = (0, 0)
        # do we really need a pad? :thinking:
        self.ops_n = OPS_N
        self.mapping = {}
        self.origin_space = -1

    def rescale_observation(self, shape: tuple[int, int] | int):
        if isinstance(shape, int):
            shape = (shape, shape)
        assert len(shape) == 2 and all(
            x > 0 for x in shape
        ), f"Expected shape to be a 2-tuple of positive integers, got: {shape}"
        self.shape = tuple(shape)
        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def init_action_mapping(self, mapping: dict[int, str] | Callable[[np.ndarray], np.ndarray | int], origin_space):
        if callable(mapping):
            self.map_action = mapping
        else:
            if len(mapping) != self.action_space.n:
                raise ValueError(
                    f"Expected mapping to have length {self.action_space.n}, got {len(mapping)}")
            for key, value in mapping.items():
                try:
                    self.mapping[Ops.__members__[value]] = key
                except KeyError:
                    raise ValueError(f"Invalid operation {value}")
        self.origin_space = origin_space
        # hardcoded to discrete for now
        self.action_space = spaces.Discrete(self.ops_n)

    def map_action(self, action: np.ndarray | int) -> np.ndarray | int:
        if isinstance(action, (np.int64, int)):
            if action in self.mapping:
                return self.mapping[action]
            try:
                return self.mapping[Ops.__members__["NOOP"]]
            except KeyError:
                raise ValueError(f"No NOOP mapping defined 🤷")
        actions = np.zeros(self.origin_space)
        for id, value in enumerate(action):
            if id in self.mapping:
                actions[self.mapping[id]] = value
        return actions

    def observation(self, observation):
        obs = super().observation(observation)['pixels']
        try:
            import cv2
        except ImportError as e:
            raise DependencyNotInstalled(
                "opencv (cv2) is not installed, run `pip install gymnasium[other]`"
            ) from e
        observation = cv2.resize(
            obs, self.shape[::-1], interpolation=cv2.INTER_AREA
        )
        return observation.reshape(self.observation_space.shape)
