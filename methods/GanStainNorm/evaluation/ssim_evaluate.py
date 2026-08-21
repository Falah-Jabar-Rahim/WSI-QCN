'''Compute SSIM score between normalized and unnormalized images'''

"""Tile wise evaluation of WSI"""

import gc
from argparse import ArgumentParser
from os import listdir, makedirs
from os.path import join
from time import perf_counter
from typing import List

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numpy import dot
from numpy import max as np_max
from numpy import min as np_min
from numpy import ndarray
from skimage.metrics import structural_similarity as simm
from tqdm import tqdm

from .dst_evaluate import load_tif


def compute_wsi_ssim(
        root_img_dir: str,
        tgt_path: str,
        result_path: str,
        tile_size: int,
        stride: int,
        cores: int
    ):
    """Read wsi from disk and compute mean SSIM score tile by tile"""
    start_time = perf_counter()
    img_list = [tif for tif in listdir(tgt_path)\
                if '.tif' in tif and 'mask_' not in tif]
    print(f'Img list length {len(img_list)}')
    results = []
    if cores > 1:
        results = Parallel(n_jobs=cores)(
            delayed(compute_tiles_ssim)(
                img_name,
                tgt_path,
                root_img_dir,
                tile_size,
                stride
            )
        for img_name in img_list)
    else:
        for img_name in tqdm(img_list):
            img_mssim_pair = compute_tiles_ssim(
                img_name=img_name,
                tgt_path=tgt_path,
                root_path=root_img_dir,
                tile_size=tile_size,
                stride=stride
            )
            results.append(img_mssim_pair)

        gc.collect()
    
    write_results(results=results, result_path=result_path)
    elapsed = (perf_counter() - start_time)//60
    print(f'Evaluation took {elapsed} minutes')

def write_results(results: List, result_path: str):
    """Write evaluation result to csv and a pickle file"""
    scores_df = pd.DataFrame(columns=['sample_name', 'ssim'])

    for score in results:
        scores_df = scores_df.append(
            {
                'sample_name': score[0],
                'ssim': score[1]
            },
            ignore_index=True
        )
    makedirs(result_path, exist_ok=True)
    scores_df.to_pickle(join(result_path, 'ssim.pkl'))
    scores_df.to_csv(join(result_path, 'ssim.csv'))


def compute_tiles_ssim(
        img_name: str,
        tgt_path: str,
        root_path: str,
        tile_size: np.ndarray,
        stride: np.ndarray
    ):
    norm_wsi = load_tif(join(tgt_path, img_name))
    org_wsi = load_tif(join(root_path, img_name))
    mask = (load_tif(join(root_path, f'mask_{img_name}'))/255).astype('int')

    img_h, img_w = org_wsi.shape[0:2]
    rows = np.floor((img_h - tile_size + stride) / stride).astype("int")
    cols = np.floor((img_w - tile_size + stride) / stride).astype("int")
    simm_list = []
    print(f'{img_name} Row: {rows} Col: {cols}')
    for row_ind in range(rows):
        for col_ind in range(cols):
            # Top-left coordinates of tile at full-res.
            h_start = row_ind * stride
            w_start = col_ind * stride
            # Bottom-right coordinates of tile at full-res.
            if mask is not None:
                mask_tile = \
                    mask[h_start : h_start + tile_size, w_start : w_start + tile_size]
                if np.sum(mask_tile) < 1: #tile_size * tile_size:
                    continue

            org_tile = org_wsi[h_start : h_start + tile_size, w_start : w_start + tile_size]
            norm_tile = norm_wsi[h_start : h_start + tile_size, w_start : w_start + tile_size]
            simm_list.append(
                compute_ssim(org_tile, norm_tile)
            )
    
    mn = np.mean(simm_list)
    print(f'{img_name} done!', mn)
    return img_name, mn

def rgb2gray(rgb: ndarray):
    """RGB to grayscale"""
    return dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def compute_ssim(ground_truth_tile: ndarray, generated_tile: ndarray) -> float:
    """Compute structural similarity index measure between ground truth and generated tile"""
    gt_tile = rgb2gray(ground_truth_tile)
    gen_tile = rgb2gray(generated_tile)
    data_range = np_max(generated_tile) - np_min(generated_tile)
    msimm, _, _ = simm(gt_tile, gen_tile, data_range=data_range, gradient=True, full=True)
    return msimm


def main():
    """"The main function"""

    parser = ArgumentParser()
    parser.add_argument(
        '--root_img_dir', help='Root path of original images', type=str, required=True)
    parser.add_argument(
        '--tgt_path', help='Target path of images to be evaluated', type=str, required=True)
    parser.add_argument(
        '--result_path', help='Pickle file path', type=str, required=True)
    parser.add_argument(
        '--tile_size', help='Pickle file path', type=int, required=False, default=256)
    parser.add_argument(
        '--stride', help='rgb or lab for CIELab', type=int, required=False, default=256)
    parser.add_argument(
        '--cores', help='Number of cores', type=int, required=False, default=1)
    param = parser.parse_args()

    compute_wsi_ssim(
        root_img_dir=param.root_img_dir,
        tgt_path=param.tgt_path,
        result_path=param.result_path,
        tile_size=param.tile_size,
        stride=param.stride,
        cores=param.cores
    )

if __name__ == '__main__':
    main()
