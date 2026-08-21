"""
Train a conditioned diffusion model.
"""

import argparse
import torch

from guided_diffusion_.guided_diffusion import dist_util, logger
from guided_diffusion_.guided_diffusion.image_datasets import load_data_stain_augmentation
from guided_diffusion_.guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion_.guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion_.guided_diffusion.train_util import TrainLoop


def main():    
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure(dir=args.checkpoint_path)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    
    
    model.to(dist_util.dev())#dist_util.dev() device
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    data = load_data_stain_augmentation(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        random_crop=args.random_crop,
        random_flip=args.random_flip,
        stain_database_path=args.stain_database_path,
        nearest_neighbours=args.nearest_neighbours,
        sigma_perturb=args.sigma_perturb,
        sigma1=args.sigma1,
        sigma2=args.sigma2,
        shift_value=args.shift_value,
        color_threshold=args.color_threshold,
        stain_threshold=args.stain_threshold,
        gaussian_blur=args.gaussian_blur, 
    )

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()


def create_argparser():
    defaults = dict(
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=4,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        
        image_size=1024,
    )
    defaults.update(model_and_diffusion_defaults())
    
    '''
    DATA_FLAGS="--data_dir ./data/Aperio/ --stain_database_path ./stain_augmentation/stain_database.pickle --random_crop True --random_flip True --nearest_neighbours 5 --sigma_perturb 0.1 --sigma1 0.7 --sigma2 0.7 --shift_value 25 --color_threshold 1000 --stain_threshold 1000000 --gaussian_blur True"
    MODEL_FLAGS="--attention_resolutions 32,16,8 --image_size 512 --num_channels 128 --num_res_blocks 3 --use_scale_shift_norm True"
    DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear" 
    TRAIN_FLAGS="--lr 1e-4 --batch_size 2"
    '''
    defaults_ = dict(
        # sava checkpoint
        checkpoint_path="./checkpoint/",
        
        #DATA_FLAGS
        data_dir="./data/Aperio/",
        random_crop=True,
        random_flip=True,
        stain_database_path='./stain_augmentation/stain_database.pickle',
        nearest_neighbours=5,
        sigma_perturb=0.1,
        sigma1=0.7,
        sigma2=0.7,
        shift_value=25,
        color_threshold=1000,
        stain_threshold=1000000,
        gaussian_blur=True, 
        
        #offical resume checkpoint
        # resume_checkpoint = './checkpoint/ema_0.9999_010000.pt',
        
        # MODEL_FLAGS
        attention_resolutions = '32,16,8',
        image_size = 256,
        num_channels = 128,
        num_res_blocks = 3,
        use_fp16 = False,
        use_scale_shift_norm = True,
        # DIFFUSION_FLAGS
        diffusion_steps = 1000,
        noise_schedule = 'linear',
        # TRAIN_FLAGS
        lr = 1e-4,
        batch_size = 16,
        is_train = True,
    )
    defaults.update(defaults_)
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
