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
        self.reward_coef = 1.0
        self.frames = []
        self.steps = 0
        self.record = False
        self.exp_name = "🤡"
        self.env_id = -1

    def set_info(self, exp_name: str, env_id: int):
        self.exp_name = exp_name
        self.env_id = env_id

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

    def init_reward_coef(self, reward_coef: float):
        """Initialize the reward coefficient
        """
        self.reward_coef = reward_coef

    def map_action(self, action: np.ndarray | int) -> np.ndarray | int:
        action = int(action)
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
        if self.steps % 100_0000 == 0:
            self.record = True
        if (self.record and len(self.frames) == 10000):
            import os
            if not os.path.exists("logs/videos/train/" + self.exp_name):
                os.makedirs("logs/videos/train/" + self.exp_name)
            height, width, _ = self.frames[0].shape
            out = cv2.VideoWriter(
                f"logs/videos/train/{self.exp_name}/env{self.env_id}_{self.steps//1000_000}M.avi", cv2.VideoWriter_fourcc(*'DIVX'), 30, (width, height))
            for frame in self.frames:
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            out.release()
            self.frames = []
            self.record = False
        if self.record:
            self.frames.append(obs['pixels'])
        # resize and grayscale
        observation = cv2.resize(
            cv2.cvtColor(obs['pixels'], cv2.COLOR_RGB2GRAY), self.shape[1::-1], interpolation=cv2.INTER_AREA)
        # save the image
        # cv2.imwrite("logs/image.png", observation)
        self.steps += 1
        return observation.reshape(self.observation_space.shape)

    def reward(self, reward):
        return reward * self.reward_coef
