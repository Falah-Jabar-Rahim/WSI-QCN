"""Implement Frechet Inception Distance (FID)
Code mostly borrowed from:  https://machinelearningmastery.com/
how-to-implement-the-frechet-inception-distance-fid-from-scratch/"""

import gc
from argparse import ArgumentParser
from os import listdir, makedirs
from os.path import join
from time import perf_counter
from typing import List, Union

import attr
import numpy as np
import pandas as pd
from numpy import cov, iscomplexobj, ndarray, trace
from scipy.linalg import sqrtm
from tensorflow.keras.applications.inception_v3 import (InceptionV3,
                                                        preprocess_input)
from tqdm import tqdm

from .dst_evaluate import load_tif


@attr.s(auto_attribs=True)
class TileStats:
    """Tile statistics for reference and normalized images

    Arguments:
        mu: mean of the features
        sigma: covariance matrix of the features
    """
    mu: ndarray
    sigma: ndarray

def compute_wsi_fid(
        root_img_dir: str,
        ref_img_name: str,
        tgt_path: str,
        result_path,
        tile_size: int=256,
        stride: int=256
    ):
    """Read tiles from disk and compute FID score"""
    start_time = perf_counter()
    img_list = [tif for tif in listdir(tgt_path)\
                if '.tif' in tif and 'mask_' not in tif]
    try:
        img_list.remove(ref_img_name)
    except ValueError:
        print('Reference image not in image list')

    ref_img = load_tif(join(root_img_dir, ref_img_name))
    ref_mask = (load_tif(join(root_img_dir, f'mask_{ref_img_name}'))/255).astype('int')

    ref_bottleneck_features_arr = get_bottleneck_features(ref_img, ref_mask, tile_size, stride)
    ref_tiles_stats = TileStats(
        mu = ref_bottleneck_features_arr.mean(axis=0),
        sigma = cov(ref_bottleneck_features_arr, rowvar=False)
    )
    results = []
    for img_name in tqdm(img_list):
        norm_img = load_tif(join(tgt_path, img_name))
        norm_mask = (load_tif(join(root_img_dir, f'mask_{img_name}'))/255).astype('int')

        norm_bottleneck_features_arr = get_bottleneck_features(norm_img, norm_mask, tile_size, stride)
        norm_tiles_stats = TileStats(
            mu = norm_bottleneck_features_arr.mean(axis=0),
            sigma = cov(norm_bottleneck_features_arr, rowvar=False)
        )
        fid = compute_fid(
                reference=ref_tiles_stats,
                normalized=norm_tiles_stats
            )
        results.append((img_name, fid))
        gc.collect()

    write_results(results=results, result_path=result_path)
    elapsed = (perf_counter() - start_time)//60
    print(f'Evaluation took {elapsed} minutes')

def get_bottleneck_features(
        img: ndarray,
        mask: Union[ndarray, None],
        tile_size=256,
        stride=256
    ):
    """A generator that returns tiles from tiff in a grid-like way"""
    bottleneck_features_list = []
    model = InceptionV3(
        include_top=False, pooling='avg', input_shape=(256 ,256, 3))
        
    img_h, img_w = img.shape[0:2]
    rows = np.floor((img_h - tile_size + stride) / stride).astype("int")
    cols = np.floor((img_w - tile_size + stride) / stride).astype("int")
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

            tile = preprocess_input(
                img[h_start : h_start + tile_size, w_start : w_start + tile_size])
            bottleneck_features_list.append(
                model.predict(np.expand_dims(tile, axis=0))[0])
    return np.array(bottleneck_features_list)

def write_results(results: List, result_path: str):
    """Write evaluation result to csv and a pickle file"""
    scores_df = pd.DataFrame(columns=['sample_name', 'fid'])

    for score in results:
        scores_df = scores_df.append(
            {
                'sample_name': score[0],
                'fid': score[1]
            },
            ignore_index=True
        )
    makedirs(result_path, exist_ok=True)
    scores_df.to_pickle(join(result_path, 'fid.pkl'))
    scores_df.to_csv(join(result_path, 'fid.csv'))

def get_tile_name_list(tiles_path: str, tile_format: str):
    """Return list of tile names with the tile_format"""
    return [join(tiles_path, tile_name)
            for tile_name in listdir(tiles_path) if tile_format in tile_name]
    
# calculate frechet inception distance
def compute_fid(reference: TileStats, normalized: TileStats):
    """Method to compute FID using means and variance of reference and normalized tiles"""

    #calculate sum squared difference between means
    ssdiff = np.sum((reference.mu - normalized.mu)**2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(reference.sigma.dot(normalized.sigma))
    # check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + trace(reference.sigma + normalized.sigma - 2.0 * covmean)
    return fid

def main():
    """"The main function"""

    parser = ArgumentParser()
    parser.add_argument(
        '--root_img_dir', help='Root path of original images', type=str, required=True)
    parser.add_argument(
        '--ref_img_name', help='Reference image name', type=str, required=True)
    parser.add_argument(
        '--tgt_path', help='Target path of images to be evaluated', type=str, required=True)
    parser.add_argument(
        '--result_path', help='Path where results will be saved', type=str, required=True)
    parser.add_argument(
        '--tile_size', help='Tile size for inference', type=int, default=256)
    parser.add_argument(
        '--stride', help='Tile stride for inference', type=int, default=256)
    param = parser.parse_args()

    compute_wsi_fid(
        root_img_dir=param.root_img_dir,
        ref_img_name=param.ref_img_name,
        tgt_path=param.tgt_path,
        result_path=param.result_path,
        tile_size=param.tile_size,
        stride=param.stride
    )

if __name__ == '__main__':
    main()
