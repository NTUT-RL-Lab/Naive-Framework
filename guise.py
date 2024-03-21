import gymnasium as gym
from gymnasium import Wrapper, Env, logger, spaces
from gymnasium.wrappers.pixel_observation import PixelObservationWrapper
import numpy as np
from gymnasium.error import DependencyNotInstalled
from pad import Pad


class Guise(PixelObservationWrapper):
    def __init__(self, env: Env) -> None:
        super().__init__(
            env
        )
        self.observation_space = self.observation_space['pixels']
        self.shape = (0, 0)
        # do we really need a pad? :thinking:
        self.pad = Pad()
        self.action_space = spaces.Discrete(self.pad.ops_n)

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

    def init_action_mapping(self, mapping: dict[int, str]):
        if len(mapping) != self.action_space.n:
            raise ValueError(
                f"Expected mapping to have length {self.action_space.n}, got {len(mapping)}")
        self.pad.define_mapping(mapping)

    def map_action(self, action: np.ndarray) -> np.ndarray:
        return self.pad.mapped_ops(action)

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
