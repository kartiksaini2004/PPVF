import numpy as np
import torch
import random
from utils.env import Env
from utils.config import Config
from utils.trace import Trace

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


if __name__ == "__main__":

    cfg = (
        Config()
    )  # initializes a Config object, which holds all system parameters (e.g., privacy budgets, caching capacities).

    trace = Trace(cfg=cfg)  # processes the video request dataset based on cfg.
    env = Env(trace=trace, cfg=cfg)  #sets up the simulation environment with the trace and config.

    env.main_Test() #begins the main simulation, testing the PPVF framework.
