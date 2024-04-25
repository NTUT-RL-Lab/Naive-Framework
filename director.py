from guise import Guise
from typing import Any
import gymnasium as gym
from gymnasium import logger
from stable_baselines3 import *
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.evaluation import evaluate_policy
from coef import Coef
from typing import Any
import numpy as np
import re


class Director():
    def __init__(self, coef: Coef) -> None:
        self.coef = coef
        self.n_timestep = coef.n_timestep
        self.c_lr = coef.c_lr
        self.cap = coef.cap
        self.env_weights = coef.env_weights
        self.n_envs = coef.n_envs
        self.env_ids = coef.env_ids
        self.exp_envs = [gym.make(env_id) for env_id in self.env_ids]
        self.env_id = 0
        self.c_transition_loss = coef.c_transition_loss
        self.policy = coef.policy
        self.env_steps = [0] * self.n_envs
        self.action_mappings = coef.act_mapping
        self.rnd_score = coef.rnd_score  # random score for each env
        self.model: BaseAlgorithm = None
        self.model_class: BaseAlgorithm = None
        self.exp_name = ""
        for env_id in self.env_ids:
            self.exp_name += re.sub('[^0-9a-zA-Z]+', '_', env_id) + "_"
        self.exp_name += f"{self.n_timestep//1_000_000}M_{self.c_lr}_{self.cap}"

    def set_model(self, model: BaseAlgorithm) -> None:
        """Sets the model to be used for learning"""
        self.model = model
        self.model_class = model.__class__

    def learn(self) -> None:
        self.model.learn(total_timesteps=self.n_timestep,
                         progress_bar=True, tb_log_name=self.exp_name)

    # env_id
    def update(self, observation, reward, terminated, truncated, info) -> tuple[int, ...]:
        self.env_steps[self.env_id] += 1
        if "👻" == "🎃":
            mean, std = self.eval(env_id=self.env_id, episodes=10)
            if (mean > 10):  # arbitrary value
                self.env_id = (self.env_id + 1) % self.n_envs
        if (self.env_steps[self.env_id] > self.cap):
            self.env_steps[self.env_id] = 0
            self.env_id = (self.env_id + 1) % self.n_envs
        # logger.info(f"env_id: {self.env_id}")
        return (self.env_id,)

    def eval(self, env_id: int, episodes: int = 10) -> tuple[int, int]:
        """Evaluates the environment
        """
        self.model.save("EvalModel")
        evalModel = self.model_class.load(
            "EvalModel", env=self.exp_envs[env_id])
        mean_reward, std_reward = evaluate_policy(
            evalModel, self.exp_envs[env_id], n_eval_episodes=episodes)
        return mean_reward, std_reward

    def save(self, path: str) -> None:
        """Saves the model
        """
        self.model.save(path)

    def birth_envs(self) -> list[Guise]:
        """Births the environments
        """
        max_w, max_h = 0, 0
        envs = [gym.make(env_id, render_mode="rgb_array")
                for env_id in self.env_ids]
        disguises: list[Guise] = []
        for i in range(envs.__len__()):
            disguises.append(Guise(envs[i]))
        #     obs = disguises[i].observation_space
        #     max_w = max(max_w, obs.shape[0])
        #     max_h = max(max_h, obs.shape[1])
        all_actions = {}
        cid = 0
        # hardcode shape for now
        max_w, max_h = 84, 84
        for i in range(disguises.__len__()):
            # magic(observation normalize) happen here 🪄🕊️
            disguises[i].rescale_observation((max_w, max_h))
            for action in self.action_mappings[i].values():
                if action not in all_actions:
                    all_actions[action] = cid
                    cid += 1
            logger.info(
                f"disguises[{i}].observation_space.shape: {disguises[i].observation_space.shape}")
        logger.info(f"all_actions: {all_actions}")
        for i in range(disguises.__len__()):
            mapping = {}
            for key, value in all_actions.items():
                for k, v in self.action_mappings[i].items():
                    if v == key:
                        mapping[value] = k
                        break
                else:
                    mapping[value] = all_actions["NOOP"]
            logger.info(f"mapping for disguises[{i}]: {mapping}")
            disguises[i].init_action_mapping(
                mapping, origin_space=disguises[i].action_space.n)
            disguises[i].init_reward_coef(
                np.sqrt(np.sum(self.rnd_score))/self.rnd_score[i])
        # calculte the reward coefficient, based on the random score square of each env

        return disguises
