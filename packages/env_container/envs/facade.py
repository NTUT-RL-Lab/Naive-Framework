from gymnasium import Wrapper, spaces
import numpy as np
from sympy import false


class Facade(Wrapper):
    """Facade class to wrap the environment and provide a single interface to the agent
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, envs, director) -> None:
        print("im here")
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
        self.states = ["running", "running"]
        self._reward_space = spaces.Box(
            low=0.0, high=1.0, shape=(1, ), dtype=np.float32)
        super().__init__(envs[0])
        print("🙏")
        self.frames = []
        self.frame_a = []
        self.frame_b = []

        self.frame_count = 0
        self.episode = 0

    @property
    def reward_space(self):
        return self._reward_space

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
            self.frame_count += 1
            if (self.frame_count == 1000000):
                self.frame_count = 0
            observations, rewards, terminateds, truncateds, infos = [], [], [], [], []
            for index in range(len(self.envs)):
                if self.states[index] == "ended":
                    continue
                self.switch_env(index)
                observation, reward, terminated, truncated, info = super().step(
                    self.env.map_action(action))
                observations.append(observation)
                rewards.append(self.env.reward(reward))
                terminateds.append(terminated)
                truncateds.append(truncated)
                infos.append(info)
                if (terminated | truncated):
                    self.states[index] = "ended"
            observation = np.mean(observations, axis=0)
            reward = np.sum(rewards)
            # if all states is ended, it is terminated
            flag = False
            for state in self.states:
                if state == "running":
                    flag = True
                    break
            terminated = not flag
            truncated = False
            info = {k: np.mean([i[k] for i in infos]) for k in infos[0]}
            if self.frame_count < 10000:
                self.frames.append((observation.astype(np.uint8), reward))
                self.frame_a.append(
                    (observations[0].astype(np.uint8), rewards[0]))
                if len(observations) > 1:
                    self.frame_b.append(
                        (observations[1].astype(np.uint8), rewards[1]))
            if self.frame_count == 10000:
                # save the video as a mp4 file
                import cv2
                import os
                _, height, width = self.frames[0][0].shape
                size = (width, height)
                out = cv2.VideoWriter(
                    f'not_die_logs/output_{self.episode}M.avi', cv2.VideoWriter_fourcc(*'DIVX'), 30, size, isColor=False)
                for i in range(len(self.frames)):
                    # write the reward on the top left corner
                    cv2.putText(self.frames[i][0][0], str(
                        self.frames[i][1]), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                    out.write(
                        self.frames[i][0][0])
                out.release()

                # save frame a
                out = cv2.VideoWriter(
                    f'not_die_logs/output_a_{self.episode}M.avi', cv2.VideoWriter_fourcc(*'DIVX'), 30, size, isColor=False)
                for i in range(len(self.frame_a)):
                    # write the reward on the top left corner
                    cv2.putText(self.frame_a[i][0][0], str(
                        self.frame_a[i][1]), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                    out.write(
                        self.frame_a[i][0][0])
                out.release()

                # save frame b
                out = cv2.VideoWriter(
                    f'not_die_logs/output_b_{self.episode}M.avi', cv2.VideoWriter_fourcc(*'DIVX'), 30, size, isColor=False)
                for i in range(len(self.frame_b)):
                    # write the reward on the top left corner
                    cv2.putText(self.frame_b[i][0][0], str(
                        self.frame_b[i][1]), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                    out.write(
                        self.frame_b[i][0][0])
                out.release()
                self.episode += 1
                self.frame_a = []
                self.frame_b = []
                self.frames = []

            return observation, reward, terminated, truncated, info
        observation, reward, terminated, truncated, info = super().step(
            self.env.map_action(action))
        # apply reward weights
        reward = self.env.reward(reward)

        index,  = self.director.update(
            observation, reward, terminated, truncated, info)
        self.switch_env(index)
        return observation, reward, terminated, truncated, info

    def reset(self, *, seed: int | None = None, options=None):
        if self.blend:
            observations = []
            infos = []
            for index in range(len(self.envs)):
                self.states[index] = "running"
                self.switch_env(index)
                obs, info = super().reset(seed=seed, options=options)
                observations.append(obs)
                infos.append(info)
            return np.mean(observations, axis=0), {k: np.mean([i[k] for i in infos]) for k in infos[0]}

        observation, infos = super().reset(seed=seed, options=options)
        return observation, infos

    def close(self):
        if self.blend:
            for env in self.envs:
                env.close()
            return
        return super().close()
