"""
@created by: heyao
@created at: 2023-07-25 21:05:29
"""
import os
import random

import numpy as np
import torch


def seed_everything(seed=42, benchmark=False):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = benchmark
