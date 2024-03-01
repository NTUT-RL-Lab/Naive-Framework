from typing import List
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
    def __init__(
        self,
        n_timestep : int, # total training timestemp needed
        c_lr : float, # learning rate
        cap : float,  # everytime reach this cap, switch env
        env_weights: List[float], # each env's importance weight
        n_envs : int, # number of envs
        env_ids : List[str],
        c_transition_loss, 
        policy : str,
        eval_freq: int,
        eval_episodes: int,
        seed: int ,
        device: str
    ):
        self.n_timestep = n_timestep
        self.c_lr = c_lr
        self.cap = cap
        self.env_weights = env_weights
        self.n_envs = n_envs
        self.env_ids = env_ids
        self.c_transition_loss = c_transition_loss
        self.policy = policy
        self.eval_freq = eval_freq
        self.eval_episodes = eval_episodes
        self.seed = seed
        self.device = device

