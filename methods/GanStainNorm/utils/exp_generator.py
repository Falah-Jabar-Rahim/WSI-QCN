"""Experiment container generation related implementation"""

from datetime import datetime
from os import makedirs, rename
from os.path import basename, isdir, join
from shutil import copy2
from typing import Dict


def generate_experiment_container(exp_root: str, exp_type:str, config_path:str) -> Dict[str, str]:
    """Generate experiment container"""
    exp_date = str(datetime.now().date().strftime("%d%m%Y"))
    exp_time = str(datetime.now().time().strftime("%H%M%S"))
    exp_name = f'Exp-{exp_type}-{exp_date}-{exp_time}'
    print(exp_name)
    if not isdir(exp_root):
        raise RuntimeError('Experiment root folder does not exist')
    #Generate experiment directory
    experiment_directory = join(exp_root, exp_name)
    makedirs(experiment_directory)
    experiment_directories = get_experiment_directories(experiment_directory)
    #Generate experiment specific directories
    makedirs(experiment_directories['checkpoint'])
    makedirs(experiment_directories['output'])
    makedirs(experiment_directories['inference'])
    makedirs(experiment_directories['wsi'])
    makedirs(experiment_directories['ds_wsi'])
    makedirs(experiment_directories['evaluation'])
    makedirs(experiment_directories['logs'])
    copy2(config_path, experiment_directory)
    rename(
        join(experiment_directory, basename(config_path)),
        join(experiment_directory, 'config.json'))

    return experiment_directories

def get_experiment_directories(experiment_directory: str):
    """Return a dictionary of experiment directories"""
    return {
        'exp': experiment_directory,
        'checkpoint': join(experiment_directory, 'checkpoint'),
        'output': join(experiment_directory, 'output'),
        'inference': join(experiment_directory, 'inference'),
        'wsi': join(experiment_directory, 'wsi'),
        'ds_wsi': join(experiment_directory, 'ds_wsi'),
        'evaluation': join(experiment_directory, 'evaluation'),
        'logs': join(experiment_directory, 'logs')
    }
