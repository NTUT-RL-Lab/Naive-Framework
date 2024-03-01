from facade import Facade
from coef import Coef
import gymnasium as gym
from stable_baselines3 import *
class Director():
    def __init__(self, coef:Coef) -> None:
        self.coef = coef
        self.n_timestep = coef.n_timestep
        self.c_lr = coef.c_lr
        self.cap = coef.cap
        self.env_weights = coef.env_weights
        self.n_envs = coef.n_envs
        self.env_ids = coef.env_ids
        self.envs = [gym.make(env_id) for env_id in self.env_ids]
        self.c_transition_loss = coef.c_transition_loss
        self.policy = coef.policy
        # self.model = 
        # TODO