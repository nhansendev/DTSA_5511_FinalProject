import os
import numpy as np

# Directory the notebook is running from
SCRIPT_DIR = os.getcwd()

# For reproducibility
RANDOM_SEED = 0
RNG = np.random.default_rng(RANDOM_SEED)
