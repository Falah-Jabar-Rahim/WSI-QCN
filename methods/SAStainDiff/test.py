import argparse
import os
import numpy as np
import torch as th
import torch.distributed as dist
from SAStainDiff.guided_diffusion_.guided_diffusion.image_datasets import _list_image_files_recursively
from SAStainDiff.guided_diffusion_.guided_diffusion import dist_util
from SAStainDiff.guided_diffusion_.guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
import time
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm
import glob


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)
    
def preprocess(image):
    transform = transforms.Compose( [transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                    ]
                                )
    image = transform(image)
    image = th.unsqueeze(image, dim=0)
    return image
    
def postprocess(image):
    image = ((image + 1) * 127.5).clamp(0, 255).to(th.uint8)
    image = image.permute(0, 2, 3, 1)
    image = image.contiguous().cpu().numpy()
    return image

def create_argparser():
    defaults = dict(
        clip_denoised=True,
    )
    defaults.update(model_and_diffusion_defaults())
    
    defaults_ = dict(
        #MODEL_FLAGS
        attention_resolutions = '32,16,8',
        image_size = 256,
        num_channels = 128,
        num_res_blocks = 3,
        use_fp16 = True,
        use_scale_shift_norm = True,
        #DIFFUSION_FLAGS
        diffusion_steps = 500,
        noise_schedule  = 'linear',
        #SAMPLE_FLAGS
        batch_size=64,
        is_train=False,
        use_ddim = True,
        timestep_respacing = 'ddim50',
        timestep_step= 1000,
        input_path='./data/test/',
        save_path='./data/sample/',
        model_path = './checkpoint/other_to_Aperio.pt',
        use_anysize=False,
    )
    defaults.update(defaults_)
    
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def any_size(all_image_files, args):   
    dist_util.setup_dist()
    print(args_to_dict(args, model_and_diffusion_defaults().keys()))

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    def model_fn(x, HE, t, y=None, ref_img=None):
        # assert y is not None
        return model(x, HE, t, y if args.class_cond else None)
    
    sample_fn = ( diffusion.p_sample_loop_any_size if not args.use_ddim else diffusion.ddim_sample_loop_any_size )
    def sample_patch(img):
        batch_size = img.shape[0]
        HE_input = img.to(dist_util.dev())
        noise = diffusion.q_sample(HE_input, th.tensor(len(diffusion.betas)-1, dtype=th.long).to(dist_util.dev()))
        sample = sample_fn(
                            model_fn,
                            img.shape,
                            HE = HE_input,
                            clip_denoised=args.clip_denoised,
                            device = dist_util.dev(),
                            progress=False,
                            noise=noise,
                        )
        return sample

    start = time.time()
    for path_i in tqdm(all_image_files): 
        name = path_i.split('/')[-1]
        
        img = np.array(Image.open(path_i))
        
        HE_input = preprocess(img).to(dist_util.dev())
        sample_img = sample_patch(HE_input)
        sample_img = postprocess(sample_img)
        
        sample_img = Image.fromarray(sample_img[0])
        sample_img.save(args.save_path + name)
        
    end = time.time()
    running_time = end-start
    print('time cost : %.5f sec' %running_time)
        
        
def main(all_image_files, args):
    dist_util.setup_dist()    
    print(args_to_dict(args, model_and_diffusion_defaults().keys()))

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    def model_fn(x, HE, t, y=None, ref_img=None):
        # assert y is not None
        return model(x, HE, t, y if args.class_cond else None)
    
    sample_fn = ( diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop )
    def sample_patch(img):
        HE_input = img.to(dist_util.dev())
        noise = diffusion.q_sample(HE_input, th.tensor(len(diffusion.betas)-1, dtype=th.long).to(dist_util.dev()))
        sample = sample_fn(
                            model_fn,
                            img.shape,
                            HE = HE_input,
                            clip_denoised=args.clip_denoised,
                            device = dist_util.dev(),
                            progress=False,
                            noise = noise,
                        )
        return sample

    start = time.time()
    for i in tqdm(range(0, len(all_image_files), args.batch_size)):
        batch_end = min(i + args.batch_size, len(all_image_files))
        actual_batch_size = batch_end - i
        
        # read image
        for j in range(actual_batch_size):
            img_array_pre = np.array(Image.open(all_image_files[i+j]))
            img_array_pre = preprocess(img_array_pre).to(dist_util.dev())
            if j == 0:
                HE_input = img_array_pre
            else:
                HE_input = th.cat((HE_input, img_array_pre), axis=0)
                
        # run sample
        sample_img = sample_patch(HE_input)
        sample_img = postprocess(sample_img)
        
        # save sample
        for j in range(actual_batch_size):
            name = all_image_files[i+j].split('/')[-1]
            save_img = Image.fromarray(sample_img[j])
            save_img.save(args.save_path + name)
            
    end = time.time()
    running_time = end-start
    print('time cost : %.5f sec' %running_time)
        
if __name__ == '__main__':
    args = create_argparser().parse_args()
           
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    all_image_files = sorted(_list_image_files_recursively(args.input_path))  
    main(all_image_files, args) if not args.use_anysize else any_size(all_image_files, args)




