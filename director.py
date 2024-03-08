from typing import Any
from coef import Coef
import gymnasium as gym
from gymnasium import logger
from stable_baselines3 import *
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.evaluation import evaluate_policy
class Director():
    def __init__(self, coef:Coef) -> None:
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
    def SetModel (self, model: BaseAlgorithm) -> None: 
        self.model = model
        self.modelClass = model.__class__
    def Learn(self) -> None:
        self.model.learn(total_timesteps=self.n_timestep, progress_bar=True)
    
    def Update(self, result: tuple[Any, ...]) -> tuple[int,...]: # env_id
        self.env_steps[self.env_id] += 1
        if "👻" == "🎃":
            mean, std = self.Eval(env_id = self.env_id, episodes = 1000)
            if (mean > 10): # arbitrary value
                self.env_id = ( self.env_id + 1 ) % self.n_envs
        if (self.env_steps[self.env_id] > self.cap):
            self.env_steps[self.env_id] = 0
            self.env_id = (self.env_id + 1) % self.n_envs
        logger.info(f"env_id: {self.env_id}")
        return ( self.env_id ,)    
        
    
    def Eval(self, env_id: int , episodes : int = 1000) -> tuple[int, int]:
        """Evaluates the environment
        """
        self.model.save("EvalModel")
        evalModel = self.modelClass.load("EvalModel", env = self.exp_envs[env_id])
        mean_reward, std_reward  = evaluate_policy(evalModel, self.exp_envs[env_id], n_eval_episodes=episodes)
        return mean_reward, std_reward

        
