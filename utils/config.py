"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import os
import yaml
from easydict import EasyDict
from utils.utils import mkdir_if_missing

import torch
import numpy as np
import random
import sys
from datetime import datetime
import shutil

def seed_fix(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(mode=True)

def get_random_state(p):
    state={}
    state['torch_state'] = torch.get_rng_state()
    state['numpy_state'] = np.random.get_state()
    state['random_state'] = random.getstate()
    state['cuda_state_all'] = torch.cuda.get_rng_state_all()
    state['trainloader_generator_state'] = p.trainloader_generator.get_state()
    return state

def set_random_state(state,p):
    torch.set_rng_state(state['torch_state'])
    np.random.set_state(state['numpy_state'])
    random.setstate(state['random_state'])
    torch.cuda.set_rng_state_all(state['cuda_state_all'])
    p.trainloader_generator.set_state(state['trainloader_generator_state'])

def save_code(output_dir):
    code_dir = os.path.join(output_dir, 'code')
    mkdir_if_missing(code_dir)
    dirs = ["models", "utils", "losses"]
    # exclude non-python files
    for d in dirs:
        shutil.copytree(d, os.path.join(code_dir, d),dirs_exist_ok=True, ignore=shutil.ignore_patterns('*.pyc', '__pycache__', '*.pth', '*.pkl', '*.txt'))
    # copy main file
    shutil.copyfile('train.py', os.path.join(code_dir, 'train.py'))


def create_config(args):
    # Config for environment path
    if "seed" in args.__dict__ and args.seed is not None:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        seed_fix(args.seed)

    cfg = EasyDict()
    cfg.update(args.__dict__)
    output_dir = args.output_dir
    mkdir_if_missing(output_dir)
    save_code(output_dir)

    cfg['train_db_name']=args.train_db_name
    if cfg['setup'] in ['cluster']:
        if "cluster_dir" in args.__dict__ and args.cluster_dir is not None:
            cluster_dir = args.cluster_dir
        else:
            cluster_dir = os.path.join(output_dir, 'cluster')
        mkdir_if_missing(cluster_dir)
        cfg['cluster_dir'] = cluster_dir
        cfg['cluster_checkpoint'] = os.path.join(cluster_dir, 'checkpoint.pth.tar')
        cfg['cluster_model'] = os.path.join(cluster_dir, 'model.pth.tar')
    if cfg['to_log']:
        log_name = datetime.now().strftime("%b%d_%H-%M-%S_log")
        log_output_dir = os.path.join(os.path.dirname(output_dir), f'{log_name}.out')
        log_err_dir = os.path.join(os.path.dirname(output_dir), f'{log_name}.err')
        print(f"Starting writing log to file: {log_output_dir}")
        sys.stdout = open(log_output_dir, 'w')
        sys.stderr = open(log_err_dir, 'w')

    return cfg
