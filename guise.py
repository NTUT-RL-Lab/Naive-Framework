import gymnasium as gym
from gymnasium import Wrapper, Env, logger, spaces
from gymnasium.wrappers.pixel_observation import PixelObservationWrapper
import numpy as np
from gymnasium.error import DependencyNotInstalled
from ops import OPS_N, Ops
from typing import Callable


class Guise(PixelObservationWrapper):
    """Guise class to wrap the environment and provide a single interface to the agent
    """

    def __init__(self, env: Env) -> None:
        super().__init__(
            env,
            pixels_only=False,  # Hardcoded for now
        )
        self.observation_space = self.observation_space['pixels']
        # self.observation_space = self.observation_space['state']
        self.shape = (0, 0)
        self.ops_n = OPS_N
        self.mapping = {}
        self.origin_space = -1

    def rescale_observation(self, shape: tuple[int, int] | int):
        """Rescale the observation space
        """
        if isinstance(shape, int):
            shape = (shape, shape, 1)
        else:
            shape = (shape[0], shape[1], 1)
        assert len(shape) == 3 and all(
            x > 0 for x in shape
        ), f"Expected shape to be a 2-tuple of positive integers, got: {shape}"
        obs_shape = tuple(shape)
        self.shape = obs_shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def init_action_mapping(self, mapping: dict[int, str] | Callable[[np.ndarray], np.ndarray | int], origin_space):
        """Initialize the action mapping
        """
        if callable(mapping):
            pass
            # self.map_action = mapping
        else:
            self.mapping = mapping
        self.origin_space = origin_space
        # hardcoded to discrete for now
        self.action_space = spaces.Discrete(len(self.mapping))

    def map_action(self, action: np.ndarray | int) -> np.ndarray | int:
        if isinstance(action, (np.int64, int)):
            if action in self.mapping:
                return self.mapping[action]
            try:
                return self.mapping[Ops.__members__["NOOP"]]
            except KeyError:
                raise ValueError(f"No NOOP mapping defined 🤷")
        actions = np.zeros(self.origin_space)
        for eid, value in enumerate(action):
            if eid in self.mapping:
                actions[self.mapping[eid]] = value
        return actions

    def observation(self, observation):
        # obs = super().observation(observation)['pixels']
        obs = super().observation(observation)
        # obs_img = obs['pixels']
        # import cv2
        # cv2.imwrite("logs/image.png", obs_img)
        # return obs['state']
        try:
            import cv2
        except ImportError as e:
            raise DependencyNotInstalled(
                "opencv (cv2) is not installed, run `pip install gymnasium[other]`"
            ) from e
        # resize and grayscale
        observation = cv2.resize(
            cv2.cvtColor(obs['pixels'], cv2.COLOR_RGB2GRAY), self.shape[1::-1])
        # save the image
        # cv2.imwrite("logs/image.png", observation)
        return observation.reshape(self.observation_space.shape)
