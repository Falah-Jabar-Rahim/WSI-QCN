"""Utility functions for exection"""

import subprocess
from datetime import datetime
from os.path import join
from typing import Any, Mapping, Tuple

from data_wrangling.load_data import DataLoader, PairedDataLoader
from models.cycle_gan import build_cycle_gan
from models.pix2pix import build_pix2pix, build_uthgan
from models.adv_pix2pix import build_dense_pix2pix


def get_model(
        model_name: str,
        input_img_dim: Tuple[int, int, int],
        **kwargs):
    """Return Keras model based on model name string"""
    if model_name == 'Pix2Pix':
        vs_model = build_pix2pix(input_img_dim=input_img_dim, **kwargs)
    elif model_name == 'CycleGan':
        vs_model = build_cycle_gan(input_img_dim=input_img_dim, **kwargs)
    elif model_name == 'UthGan':
        vs_model = build_uthgan(input_img_dim=input_img_dim, **kwargs)
    elif model_name == 'DensePix2Pix':
        vs_model = build_dense_pix2pix(input_img_dim=input_img_dim, **kwargs)
    else:
        raise ValueError(f'{model_name} not found!')
    return vs_model

def get_data_loader(config: Mapping[str, Any], dataset_type: str='train'):
    """Return data loader based on the experiment type"""
    if config.get('data_loader_type', None) == 'unpaired':
        data_loader = DataLoader(config=config)
    else:
        data_loader = PairedDataLoader(
            config=config, dataset_type=dataset_type)
    return data_loader

def add_commit_hash_to_readme(readme_path: str, msg: str = 'logging') -> None:
    """Appends current commit hash to the readme file with timestamp"""
    date = str(datetime.now().date().strftime("%d-%m-%Y"))
    time = str(datetime.now().time().strftime("%H:%M:%S"))
    #commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    try:

        commit_hash = subprocess.check_output(

            ["git", "rev-parse", "HEAD"],

            stderr=subprocess.DEVNULL,

        ).decode("ascii").strip()

    except subprocess.CalledProcessError:

        commit_hash = "not-available"

    with open(join(readme_path, 'readme.txt'), 'a') as readme_file:
        readme_file.write("\n")
        readme_file.write(f'{date} {time}: {msg} | commit hash: {commit_hash}')

def copy_pretrained_cyclegan_weights_to_pix2pix(input_img_dim, pix2pix_model, **kwargs):
    """Copy CycleGAN generator discriminator weight to Pix2Pix"""
    epoch = kwargs['load_pretrained_weights']['epoch']
    checkpoint_dir = kwargs['load_pretrained_weights']['checkpoint_dir']
    weight_file = f'epoch.{epoch:03d}'
    weight_file_path = join(checkpoint_dir, weight_file)
    cyclegan_model = build_cycle_gan(input_img_dim=input_img_dim, **kwargs)
    cyclegan_model.load_weights(weight_file_path).expect_partial()
    pix2pix_model.generator.set_weights(cyclegan_model.gen_g.get_weights())
    #pix2pix_model.discriminator.set_weights(cyclegan_model.disc_x.get_weights())
    #for i, layer in enumerate(cyclegan_model.disc_x.layers[1:]):
    #    # Plus 3 for skipping the target and concatenate layers as well in Pix2Pix
    #    # CycleGAN doesn't have these layers because of unpaired data
    #    pix2pix_model.discriminator.layers[i+3].set_weights(layer.get_weights())
    print("CycleGAN weights successfully transferred to Pix2Pix")
    return pix2pix_model
