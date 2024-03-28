import stat
from typing import List, Callable
import numpy as np
import tomllib
# c_ means a coeffiecent
# Director's Coef
'''
 Routine:
 1. Load Coef to Manager
 2. Pass the parameters to algorithms
 3. Wrap the environment (2000 things todo)
 4. Start training
    i.    Train on A
    ii.   Till A's score reach the cap
          📰 Evaluate agent on both
          🦾 Tune coef in manager
    iii.  Swtich env to B
    iv.   Train on B
    v.    Till B's score reach the cap
          📰 Evaluate agent on both
          🦾 Tune coef in manager
    vi.   Switch env to A
    vii.  Back to i. if not reach timestep n
    viii. :)
'''


class Coef:
    """Coefficient class to store hyperparameters
    """
    @staticmethod
    def from_data(
        n_timestep: int,  # total training timestemp needed
        c_lr: float,  # learning rate
        cap: float,  # everytime reach this cap, switch env
        env_weights: List[float],  # each env's importance weight
        n_envs: int,  # number of envs
        env_ids: List[str],
        # action mapping for each env
        act_mapping: List[dict[int, str]] | Callable[[np.ndarray], np.ndarray | int],
        c_transition_loss,
        policy: str,
        eval_freq: int,
        eval_episodes: int,
        seed: int,
        device: str
    ):
        """
        Args:
            n_timestep (int): total training timestemp needed
            c_lr (float): learning rate
            cap (float): everytime reach this cap, switch env
            env_weights (List[float]): each env's importance weight
            n_envs (int): number of envs
            env_ids (List[str]): envs' ids
            act_mapping (List[dict[int, str]] | Callable[[np.ndarray], np.ndarray | int]): action mapping for each env
            c_transition_loss: 
            policy (str): 
            eval_freq (int): 
            eval_episodes (int): 
            seed (int): 
            device (str):
        """
        coef = Coef()
        coef.n_timestep = n_timestep
        coef.c_lr = c_lr
        coef.cap = cap
        coef.env_weights = env_weights
        coef.n_envs = n_envs
        coef.env_ids = env_ids
        coef.c_transition_loss = c_transition_loss
        coef.act_mapping = act_mapping
        coef.policy = policy
        coef.eval_freq = eval_freq
        coef.eval_episodes = eval_episodes
        coef.seed = seed
        coef.device = device
        return coef

    def __init__(self, config_file: str = None):
        if config_file is None:
            return
        # load from file
        with open(config_file, "rb") as f:
            config = tomllib.load(f)

        self.n_timestep = config["n_timestep"]
        self.c_lr = config["c_lr"]
        self.cap = config["cap"]
        self.env_weights = config["env_weights"]
        self.env_ids = config["env_ids"]
        self.c_transition_loss = config["c_transition_loss"]
        self.policy = config["policy"]
        self.eval_freq = config["eval_freq"]
        self.eval_episodes = config["eval_episodes"]
        self.seed = config["seed"]
        self.device = config["device"]
        self.env_weights = config["env_weights"]
        self.n_envs = len(self.env_ids)

        mappings_path = "config/mappings.toml"

        self.act_mapping = []
        with open(mappings_path, "rb") as f:
            mappings = tomllib.load(f)
            for env_id in self.env_ids:
                # change key type to int
                self.act_mapping.append(
                    {int(k): v for k, v in mappings[env_id].items()})
