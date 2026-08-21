"""Execution point of the project for trainings"""

import json
from argparse import ArgumentParser, Namespace
from os import listdir, makedirs
from os.path import join, isdir
from typing import Any, Mapping

import tensorflow as tf
from tqdm import tqdm
#from tensorflow.keras import mixed_precision
#mixed_precision.set_global_policy("mixed_float16")
from data_wrangling.merge_tile import MergeTiles
from data_wrangling.tiling import WsiTileGenerator
from data_wrangling.utils import downsample
from models.inference import Inference
from models.utils import build_callbacks
from utils.execution_utils import (add_commit_hash_to_readme,
                                   copy_pretrained_cyclegan_weights_to_pix2pix,
                                   get_data_loader, get_model)
from utils.exp_generator import (generate_experiment_container,
                                 get_experiment_directories)
from utils.untar import untar_samples


def train(args: Namespace):
    """Initiate model training"""
    config_path = args.config_path
    config = get_config(args.config_path)
    exp_root = args.exp_root
    experiment_directories = generate_experiment_container(
        exp_root=exp_root, exp_type=config['exp_type'], config_path=config_path)
    add_commit_hash_to_readme(
        readme_path=experiment_directories['exp'], msg='train')

    data_root = args.data_root if args.data_root else config['data_root']
    if config.get('val', False):
        data_loader = get_data_loader(config=config)
        # for separate validation WSI
        val_data_loader = get_data_loader(config=config, dataset_type='val')
        train_dataset, _ = data_loader.get_train_val_dataset(data_root=data_root)
        val_dataset, _ = val_data_loader.get_train_val_dataset(data_root=data_root)
    else:
        data_loader = get_data_loader(config=config)
        train_dataset, val_dataset = data_loader.get_train_val_dataset(data_root=data_root)

    if config.get('multi_gpu', False):
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            vs_model = get_model(
                model_name=config['exp_type'],
                input_img_dim=config['input_img_dim'],
                multi_gpu=True,
                batch_size=config['batch_size'],
                normalization=config.get('normalization', 'instancenorm'),
                double_conv=config.get('double_conv', False),
                use_pix2pix_components=config.get('use_pix2pix_components', True)
            )
            if config.get('load_pretrained_weights', False):
                copy_pretrained_cyclegan_weights_to_pix2pix(
                    input_img_dim=config['input_img_dim'],
                    pix2pix_model=vs_model,
                    multi_gpu=True,
                    batch_size=config['batch_size'],
                    normalization=config.get('normalization', 'instancenorm'),
                    use_pix2pix_components=config.get('use_pix2pix_components', True),
                    load_pretrained_weights=config['load_pretrained_weights']
                )
    else:
        vs_model = get_model(
            model_name=config['exp_type'],
            input_img_dim=config['input_img_dim'],
            multi_gpu=False,
            batch_size=config['batch_size'],
            normalization=config.get('normalization', 'instancenorm'),
            double_conv=config.get('double_conv', False),
            use_pix2pix_components=config.get('use_pix2pix_components', True)
        )
        if config.get('load_pretrained_weights', False):
            copy_pretrained_cyclegan_weights_to_pix2pix(
                input_img_dim=config['input_img_dim'],
                pix2pix_model=vs_model,
                multi_gpu=False,
                batch_size=config['batch_size'],
                normalization=config.get('normalization', 'instancenorm'),
                use_pix2pix_components=config.get('use_pix2pix_components', True),
                load_pretrained_weights=config['load_pretrained_weights']
            )


    callback_list = build_callbacks(
        model_type=config['exp_type'],
        val_dataset=val_dataset,
        val_plot_count=config['val_plot_count'],
        output_path=experiment_directories['output'],
        checkpoint_path=experiment_directories['checkpoint'],
        logs_path=experiment_directories['logs']
    )

    vs_model.fit(
        x=train_dataset,
        epochs=config['epochs'],
        callbacks=callback_list
    )

def inference(args: Namespace):
    """Initiate model inference"""
    Inference(
        exp_path=args.exp_path,
        data_root=args.data_root
    ).run_inference(config=get_config(args.config_path))

def merge_tile_to_generate_wsi(args: Namespace):
    """Merge tile to generate whole slide images"""
    merge_tiles = MergeTiles(
        config=get_config(args.config_path),
        exp_path=args.exp_path,
        multiprocess= args.multiprocess,
        inference_root=args.inference_root
    )
    merge_tiles()

def tile_wsi(args: Namespace):
    """Initiate whole slide images tiling"""
    config = get_config(args.config_path)
    WsiTileGenerator(
        data_source=config['data_source'],
        output_root=config['output_root'],
        tar_wrapper=config['tar_wrapper'],
        masked_tiles=config['masked_tiles'],
        acceptable_tile_percentage=config['acceptable_tile_percentage'],
        paired_wsi=config.get('paired_wsi', False)
    ).generate_wsi_tiles(
        img_list=config['image_list'],
        tile_size=config['tile_size'],
        stride=config['stride'],
        tile_format=config['tile_format'],
        multiprocess=args.multiprocess
    )

def untar(args: Namespace):
    """Initiate sample WSIs untarring"""
    untar_samples(
        src_path=args.src_path,
        dst_path=args.dst_path,
        skip_stained=args.inference,
        num_cores=args.num_cores
    )

def downsample_wsi(args: Namespace):
    """Downsample tif WSI (by typically a factor of 10) & save as jpg"""
    exp_path = args.exp_path
    wsi_epoch_path = join(exp_path, 'wsi')
    epoch_list = [epoch for epoch in listdir(wsi_epoch_path)\
                   if isdir(join(wsi_epoch_path, epoch))]

    for epoch in epoch_list:
        dst_path = join(exp_path, 'ds_wsi', epoch)
        makedirs(dst_path, exist_ok=True)
        wsi_path = join(wsi_epoch_path, epoch)
        tif_list = [tif for tif in listdir(wsi_path) if '.tif' in tif]
        
        for tif in tqdm(tif_list):
            tif_name = tif.split('.')[0]
            input_tif = join(wsi_path, tif)
            output_tif = join(dst_path, f'ds_{tif_name}.jpg')
            downsample(file_path=input_tif, downsampled_path=output_tif, factor=5)

def add_config_path_to_subparser(subparser: ArgumentParser):
    """Add config argument to the subparser"""
    subparser.add_argument(
        '--config_path', help='Operation specific config', type=str, required=True)

def get_config(config_path: str) -> Mapping[str, Any]:
    """Return config json"""
    with open(config_path) as json_file:
        config = json.load(json_file)
    return config

def continue_train(args: Namespace):
    """Continue model training for a specific experiment"""
    config_path = join(args.exp_path, 'config.json')
    config = get_config(config_path)
    experiment_directories = get_experiment_directories(args.exp_path)
    add_commit_hash_to_readme(
        readme_path=experiment_directories['exp'], msg='continue-train')

    data_root = args.data_root if args.data_root else config['data_root']
    if config.get('val', None):
        data_loader = get_data_loader(config=config)
        # for separate validation WSI
        val_data_loader = get_data_loader(config=config, dataset_type='val')
        train_dataset, _ = data_loader.get_train_val_dataset(data_root=data_root)
        val_dataset, _ = val_data_loader.get_train_val_dataset(data_root=data_root)
    else:
        data_loader = get_data_loader(config=config)
        train_dataset, val_dataset = data_loader.get_train_val_dataset(data_root=data_root)

    print("Train element spec:", train_dataset.element_spec)

    print("Validation element spec:", val_dataset.element_spec)

    train_cardinality = tf.data.experimental.cardinality(train_dataset)

    val_cardinality = tf.data.experimental.cardinality(val_dataset)

    print("Train cardinality:", train_cardinality.numpy())

    print("Validation cardinality:", val_cardinality.numpy())

    if train_cardinality.numpy() == 0:
        raise RuntimeError(

            "The training dataset is empty. "

            "Check paired-tile selection in data_wrangling/load_data.py."

        )



    weight_file = f'epoch.{args.epoch:03d}'
    weight_file_path = join(experiment_directories['checkpoint'], weight_file)

    if config.get('multi_gpu', False):
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            vs_model = get_model(
                model_name=config['exp_type'],
                input_img_dim=config['input_img_dim'],
                multi_gpu=True,
                batch_size=config['batch_size'],
                normalization=config.get('normalization', 'instancenorm'),
                double_conv=config.get('double_conv', False),
                use_pix2pix_components=True
            )
            vs_model.load_weights(weight_file_path).expect_partial()
    else:
        vs_model = get_model(
            model_name=config['exp_type'],
            input_img_dim=config['input_img_dim'],
            multi_gpu=False,
            batch_size=config['batch_size'],
            normalization=config.get('normalization', 'instancenorm'),
            double_conv=config.get('double_conv', False),
            use_pix2pix_components=True
        )
        vs_model.load_weights(weight_file_path).expect_partial()

    callback_list = build_callbacks(
        model_type=config['exp_type'],
        val_dataset=val_dataset,
        val_plot_count=config['val_plot_count'],
        output_path=experiment_directories['output'],
        checkpoint_path=experiment_directories['checkpoint'],
        logs_path=experiment_directories['logs']
    )
    vs_model.fit(
        x=train_dataset,
        epochs=config['epochs'],
        callbacks=callback_list,
        initial_epoch=int(args.epoch)
    )

def main():
    """The main function"""

    parser = ArgumentParser()
    subparsers = parser.add_subparsers()
    train_parser = subparsers.add_parser('train', help='Train a model')
    train_parser.add_argument(
        '--exp_root', help='Root path to experiments', type=str, required=True)
    train_parser.add_argument(
        '--data_root', help='Root path to data, usually on temporary disk', type=str)
    train_parser.add_argument(
        '--multi_gpu', help='Enable training with multiple GPUs', action='store_true')
    add_config_path_to_subparser(train_parser)
    train_parser.set_defaults(func=train)

    continue_train_parser = subparsers.add_parser('continue_train', help='Continue training')
    continue_train_parser.add_argument(
        '--exp_path', help='Path to specific experiment', type=str, required=True)
    continue_train_parser.add_argument(
        '--data_root', help='Root path to data, usually on temporary disk', type=str)
    continue_train_parser.add_argument(
        '--epoch', help='Epoch to continue training from', type=int)
    continue_train_parser.set_defaults(func=continue_train)

    inference_parser = subparsers.add_parser('inference', help='Run inference')
    inference_parser.add_argument(
        '--exp_path', help='Path to specific experiment', type=str, required=True)
    inference_parser.add_argument(
        '--data_root', help='Root path to data, usually on temporary disk', type=str)
    add_config_path_to_subparser(inference_parser)
    inference_parser.set_defaults(func=inference)

    merge_tiles_parser = subparsers.add_parser('merge_tiles', help='Merge tiles after inference')
    merge_tiles_parser.add_argument(
        '--exp_path', help='Path to specific experiment', type=str, required=True)
    merge_tiles_parser.add_argument(
        '--inference_root',
        help='Root path to inference generated data, usually on temporary disk', type=str)
    merge_tiles_parser.add_argument(
        '--multiprocess', help='For WSI wise tile merging in parallel', action='store_true')
    add_config_path_to_subparser(merge_tiles_parser)
    merge_tiles_parser.set_defaults(func=merge_tile_to_generate_wsi)

    tile_images_parser = subparsers.add_parser('tile', help='Tile WSIs')
    tile_images_parser.add_argument(
        '--multiprocess', help='For parallelizing tiling', action='store_true')
    add_config_path_to_subparser(tile_images_parser)
    tile_images_parser.set_defaults(func=tile_wsi)

    untar_tiles_parser = subparsers.add_parser('untar_tiles', help='Untar tiles to temporary loc')
    untar_tiles_parser.add_argument(
        '--src_path', help='Path to source tar files', type=str, required=True)
    untar_tiles_parser.add_argument(
        '--dst_path', help='Path to untar tiles', type=str, required=True)
    untar_tiles_parser.add_argument(
        '--inference', help='If set skips stained untarring', action='store_true')
    untar_tiles_parser.add_argument(
        '--num_cores', help='Number of cores for parallel processing', type=int, required=True)
    untar_tiles_parser.set_defaults(func=untar)

    downsample_parser = subparsers.add_parser('downsample', help='Run tiff downsampling')
    downsample_parser.add_argument(
        '--exp_path', help='Path to specific experiment', type=str, required=True)
    downsample_parser.set_defaults(func=downsample_wsi)

    param = parser.parse_args()
    param.func(param)

if __name__ == '__main__':
    main()
