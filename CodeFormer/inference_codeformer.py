import os
import random
import time
import cv2
import numpy as np
import torch
from torchvision.utils import make_grid

# 強制設為 True，避免新版 PyTorch 版本字串引發 IndexError
IS_HIGH_VERSION = True

def set_random_seed(seed):
    """Set random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_time_str():
    return time.strftime('%Y%m%d_%H%M%S', time.localtime())

def mkdir_and_rename(path):
    """mkdirs. If path exists, rename it with timestamp and create a new one."""
    if os.path.exists(path):
        new_name = path + '_archived_' + get_time_str()
        print(f'Path already exists. Rename it to {new_name}', flush=True)
        os.rename(path, new_name)
    os.makedirs(path, exist_ok=True)

def make_exp_dirs(opt):
    """Make dirs for experiments."""
    path_opt = opt['path']
    if opt['is_train']:
        mkdir_and_rename(path_opt['experiments_root'])
        mkdir_and_rename(path_opt['models'])
        mkdir_and_rename(path_opt['training_states'])
        mkdir_and_rename(path_opt['log'])
        mkdir_and_rename(path_opt['visualization'])
    else:
        mkdir_and_rename(path_opt['results_root'])
        mkdir_and_rename(path_opt['log'])
        mkdir_and_rename(path_opt['visualization'])

def check_resume(opt, resume_iter):
    """Check resume states and update opt and logger."""
    if opt['path'].get('resume_state'):
        # get all the checkpoint models
        checkpoints = [
            f for f in os.listdir(opt['path']['models']) if f.endswith('.pth')
        ]
        if checkpoints:
            checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
            opt['path']['resume_state'] = os.path.join(
                opt['path']['models'], f'net_g_{resume_iter}.pth')
            opt['logger']['print_freq'] = opt['logger']['print_freq']
            print(f"Resuming from {opt['path']['resume_state']}")

def sizeof_fmt(num, suffix='B'):
    """Format file size."""
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return f'{num:3.1f} {unit}{suffix}'
        num /= 1024.0
    return f'{num:.1f} Yi{suffix}'

def scandir(dir_path, suffix=None, recursive=False, full_path=False):
    """Scan a directory to find the interested files."""
    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    root = dir_path

    def _scandir(dir_path, suffix, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                if full_path:
                    return_path = entry.path
                else:
                    return_path = os.path.relpath(entry.path, root)

                if suffix is None:
                    yield return_path
                elif return_path.endswith(suffix):
                    yield return_path
            else:
                if recursive:
                    yield from _scandir(entry.path, suffix=suffix, recursive=recursive)
                else:
                    continue

    return _scandir(dir_path, suffix=suffix, recursive=recursive)

def gpu_is_available():
    return torch.cuda.is_available()

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')