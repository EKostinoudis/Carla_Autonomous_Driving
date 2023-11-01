import os
import random
import torch
import glob
from torch.nn import DataParallel
import numpy as np

from .utils import sort_nicely
from configs import g_conf

class DataParallelWrapper(DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

    def __len__(self):
        return len(self.module)

def seed_everything(seed=0):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True


def check_saved_checkpoints(checkpoints_path):
    if not os.path.exists(checkpoints_path):
        return None
    else:
        checkpoints = glob.glob(os.path.join(checkpoints_path, '*.pth'))
        if checkpoints:
            sort_nicely(checkpoints)
            return checkpoints[-1]
        else:
            return None

def check_saved_checkpoints_in_total(checkpoints_path):
    if not os.path.exists(checkpoints_path):
        return None
    else:
        checkpoints = glob.glob(os.path.join(checkpoints_path, '*.pth'))
        if checkpoints:
            sort_nicely(checkpoints)
            return checkpoints
        else:
            return None


def update_learning_rate(optimizer, minimumlr = 0.00001):
    """
        Adjusts the learning rate based on the schedule
        """

    minlr = minimumlr
    if g_conf.LEARNING_RATE_POLICY['name'] == 'normal':
        for param_group in optimizer.param_groups:
            cur_lr = param_group['lr']
            print('Previous lr:', cur_lr)
            new_lr = cur_lr * g_conf.LEARNING_RATE_POLICY['level']
            param_group['lr'] = max(new_lr, minlr)
            print('New lr:', param_group['lr'])

    else:
        raise NotImplementedError('Not found learning rate policy !')

