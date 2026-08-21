"""Implements Jensen Shannon divergenece based evaluation"""

from argparse import ArgumentParser
from os import listdir, makedirs
from os.path import basename, join
from time import perf_counter
from typing import List

import attr
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.spatial.distance import chebyshev, cityblock, euclidean
from scipy.spatial.distance import jensenshannon as jensh
from scipy.stats import chisquare, pearsonr
from skimage.color import rgb2lab
from tifffile import TiffFile


@attr.s(auto_attribs=True)
class EvalScore:
    """Contains different distance metrics"""
    sample_name: str
    chebyshev: float
    cityblock: float
    euclidean: float
    intersection: float
    jensen_shannon: float
    pearson_r: float

def evaluate(
        root_img_dir: str,
        ref_img_name: str,
        tgt_path: str,
        result_path,
        cores: int=1,
        colorspace: str='lab'
    ):
    """Evaluate images at root_path against ref_img"""
    img_list = [tif for tif in listdir(tgt_path)\
                if '.tif' in tif and 'mask_' not in tif]
    try:
        img_list.remove(ref_img_name)
    except ValueError:
        print('Reference image not in image list')

    eval_scores: List[EvalScore] = []
    ref_img = load_tif(join(root_img_dir, ref_img_name))

    if colorspace == 'lab':
        ref_img = rgb2lab(ref_img)
        ref_img = np.around(ref_img)
    elif colorspace == 'rgb':
        ref_img = ref_img
    else:
        raise ValueError(f'Unrecognized colorspace {colorspace}')

    ref_mask = (load_tif(join(root_img_dir, f'mask_{ref_img_name}'))/255).astype('int')
    ref_mask = ref_mask

    if cores > 1:
        eval_scores = Parallel(n_jobs=cores)(
            delayed(compute_distances)(
                ref_img,
                ref_mask,
                join(tgt_path, img_name),
                join(root_img_dir, f'mask_{img_name}')
            )
        for img_name in img_list)
    else:
        for img_name in img_list:
            try:
                eval_scores.append(
                    compute_distances(
                        ref_img=ref_img,
                        ref_mask=ref_mask,
                        tgt_img_path=join(tgt_path, img_name),
                        tgt_mask_path=join(root_img_dir, f'mask_{img_name}')
                    )
                )
                print(f'{img_name} done!')
            #pylint: disable=broad-except
            except Exception as exp:
                print(f'!!!!! Problem with sample {img_name}!!!!')
                print(str(exp))
                eval_scores.append(
                    EvalScore(
                        sample_name=img_name,
                        chebyshev=1000,
                        cityblock=1000,
                        euclidean=1000,
                        intersection=1000,
                        jensen_shannon=1000,
                        pearson_r=1000,
                    )
                )
    pickle_scores(eval_scores, result_path)

def pickle_scores(scores: List[EvalScore], result_path: str):
    """Save scores in a dataframe and pickle it"""
    scores_df = pd.DataFrame(
        columns=['sample_name', 'chebyshev', 'cityblock', 'euclidean',
                 'intersection', 'jensen_shannon', 'pearson_r']
    )

    for score in scores:
        scores_df = scores_df.append(
            {
                'sample_name': score.sample_name,
                'chebyshev': score.chebyshev,
                'cityblock': score.cityblock,
                'euclidean': score.euclidean,
                'intersection': score.intersection,
                'jensen_shannon': score.jensen_shannon,
                'pearson_r': score.pearson_r
            },
            ignore_index=True
        )
    makedirs(result_path, exist_ok=True)
    scores_df.to_pickle(join(result_path, 'results.pkl'))
    scores_df.to_csv(join(result_path, 'results.csv'))

def compute_distances(
        ref_img: np.ndarray,
        ref_mask: np.ndarray,
        tgt_img_path: str,
        tgt_mask_path: str,
        colorspace: str='lab'
    ):
    """Compute distance between color histograms of image pair"""
    tgt_img = load_tif(tgt_img_path)
    tgt_mask = (load_tif(tgt_mask_path)/255).astype('int')

    ref_mask_fltn = ref_mask.flatten()
    # if basename(tgt_img_path) == 'c_3146_HE_2019-12-04_10_58_01.tif':
    #     tgt_mask_fltn = tgt_mask[:-100, :-1][::2, ::2].flatten()
    # elif basename(tgt_img_path) == 'c_3431_HE_2019-12-04_10_42_17.tif':
    #     tgt_mask_fltn = tgt_mask[:-1, :-100][::2, ::2].flatten()
    # else:
    tgt_mask_fltn = tgt_mask.flatten()

    if colorspace == 'lab':
        ch_a, ch_b, ch_c = compute_lab_histograms(
            ref_img,
            ref_mask_fltn,
            tgt_img,
            tgt_mask_fltn
        )
    elif colorspace == 'rgb':
        ch_a, ch_b, ch_c = compute_rgb_histograms(
            ref_img,
            ref_mask_fltn,
            tgt_img,
            tgt_mask_fltn
        )
    else:
        raise ValueError(f'Unrecognized colorspace {colorspace}')

    return EvalScore(
        sample_name=basename(tgt_img_path),
        chebyshev=get_mean_chebyshev(ch_a, ch_b, ch_c),
        cityblock=get_mean_cityblock(ch_a, ch_b, ch_c),
        euclidean=get_mean_euclidean(ch_a, ch_b, ch_c),
        intersection=get_mean_intersection(ch_a, ch_b, ch_c),
        jensen_shannon=get_mean_jensen_shannon(ch_a, ch_b, ch_c),
        pearson_r=get_mean_pearson_r(ch_a, ch_b, ch_c)
    )


def compute_rgb_histograms(
        ref_img: np.ndarray,
        ref_mask_fltn: np.ndarray,
        tgt_img: np.ndarray,
        tgt_mask_fltn: np.ndarray
    ):
    """Compute histograms in red blue green color space"""
    tgt_img = tgt_img
    bin_ref_a, _ = np.histogram(
        ref_img[..., 0].flatten(), bins=256, weights=ref_mask_fltn)
    bin_tgt_a, _ = np.histogram(
        tgt_img[..., 0].flatten(), bins=256, weights=tgt_mask_fltn)

    bin_ref_b, _ = np.histogram(
        ref_img[..., 1].flatten(), bins=256, weights=ref_mask_fltn)
    bin_tgt_b, _ = np.histogram(
        tgt_img[..., 1].flatten(), bins=256, weights=tgt_mask_fltn)

    bin_ref_c, _ = np.histogram(
        ref_img[..., 2].flatten(), bins=256, weights=ref_mask_fltn)
    bin_tgt_c, _ = np.histogram(
        tgt_img[..., 2].flatten(), bins=256, weights=tgt_mask_fltn)

    return (bin_ref_a, bin_tgt_a), (bin_ref_b, bin_tgt_b), (bin_ref_c, bin_tgt_c)

def compute_lab_histograms(
        ref_img: np.ndarray,
        ref_mask_fltn: np.ndarray,
        tgt_img: np.ndarray,
        tgt_mask_fltn: np.ndarray
    ):
    """Compute distances in CIELab color spaces"""
    tgt_img = rgb2lab(tgt_img)
    tgt_img = np.around(tgt_img)

    bin_ref_a, _ = np.histogram(
        ref_img[..., 0].flatten(), bins=98, range=(1, 99), weights=ref_mask_fltn)
    bin_tgt_a, _ = np.histogram(
        tgt_img[..., 0].flatten(), bins=98, range=(1, 99), weights=tgt_mask_fltn)

    bin_ref_b, _ = np.histogram(
        ref_img[..., 1].flatten(), bins=256, range=(-128, 127), weights=ref_mask_fltn)
    bin_tgt_b, _ = np.histogram(
        tgt_img[..., 1].flatten(), bins=256, range=(-128, 127), weights=tgt_mask_fltn)

    bin_ref_c, _ = np.histogram(
        ref_img[..., 2].flatten(), bins=256, range=(-128, 127), weights=ref_mask_fltn)
    bin_tgt_c, _ = np.histogram(
        tgt_img[..., 2].flatten(), bins=256, range=(-128, 127), weights=tgt_mask_fltn)

    bin_ref_a = bin_ref_a/np.linalg.norm(bin_ref_a, 2)
    bin_tgt_a = bin_tgt_a/np.linalg.norm(bin_tgt_a, 2)
    bin_ref_b = bin_ref_b/np.linalg.norm(bin_ref_b, 2)
    bin_tgt_b = bin_tgt_b/np.linalg.norm(bin_tgt_b, 2)
    bin_ref_c = bin_ref_c/np.linalg.norm(bin_ref_c, 2)
    bin_tgt_c = bin_tgt_c/np.linalg.norm(bin_tgt_c, 2)

    return (bin_ref_a, bin_tgt_a), (bin_ref_b, bin_tgt_b), (bin_ref_c, bin_tgt_c)

def get_mean_chebyshev(ch_a, ch_b, ch_c):
    """Return mean chebyshev score"""
    return np.mean(
        [chebyshev(ch_a[0], ch_a[1]),
        chebyshev(ch_b[0], ch_b[1]),
        chebyshev(ch_c[0], ch_c[1])]
    )

def get_mean_chisquare(ch_a, ch_b, ch_c):
    """Return mean p-value from chisquare test"""
    return np.mean(
        [chisquare(ch_a[1], ch_a[0])[1],
        chisquare(ch_b[1], ch_b[0])[1],
        chisquare(ch_c[1], ch_c[0])[1]]
    )

def get_mean_cityblock(ch_a, ch_b, ch_c):
    """Return mean cityblock score"""
    return np.mean(
        [cityblock(ch_a[0], ch_a[1]),
        cityblock(ch_b[0], ch_b[1]),
        cityblock(ch_c[0], ch_c[1])]
    )

def get_mean_euclidean(ch_a, ch_b, ch_c):
    """Return mean euclidean score"""
    return np.mean(
        [euclidean(ch_a[0], ch_a[1]),
        euclidean(ch_b[0], ch_b[1]),
        euclidean(ch_c[0], ch_c[1])]
    )

def get_mean_jensen_shannon(ch_a, ch_b, ch_c):
    """Return mean jensen shannon score"""
    return np.mean(
        [jensh(ch_a[0], ch_a[1]),
        jensh(ch_b[0], ch_b[1]),
        jensh(ch_c[0], ch_c[1])]
    )

def get_mean_intersection(ch_a, ch_b, ch_c):
    """Return mean intersection of the histograms"""
    return np.mean(
        [hist_intersection(ch_a[0], ch_a[1]),
        hist_intersection(ch_b[0], ch_b[1]),
        hist_intersection(ch_c[0], ch_c[1])]
    )

def get_mean_pearson_r(ch_a, ch_b, ch_c):
    """Return mean Pearson's Correlation R value"""
    return np.mean(
        [pearsonr(ch_a[0], ch_a[1])[0],
        pearsonr(ch_b[0], ch_b[1])[0],
        pearsonr(ch_c[0], ch_c[1])[0]]
    )

def hist_intersection(hist_1, hist_2):
    """Compute histogram intersection"""
    minima = np.minimum(hist_1, hist_2)
    intersection = np.true_divide(np.sum(minima), np.sum(hist_2))
    return intersection

def load_tif(tif_path:str):
    """Return tiff img as numpy array"""
    with TiffFile(tif_path) as img:
        img_arr = img.pages[0].asarray()
    return img_arr.astype(np.uint8)

def main():
    """Main function for temporary random sample tile untarring"""

    start_time = perf_counter()
    parser = ArgumentParser()
    parser.add_argument(
        '--root_img_dir', help='Root path of original images', type=str, required=True)
    parser.add_argument(
        '--ref_img_name', help='Reference image name', type=str, required=True)
    parser.add_argument(
        '--tgt_path', help='Target path of images to be evaluated', type=str, required=True)
    parser.add_argument(
        '--result_path', help='Pickle file path', type=str, required=True)
    parser.add_argument(
        '--colorspace', help='rgb or lab for CIELab', type=str, required=False, default='lab')
    parser.add_argument(
        '--cores', help='Number of cores', type=int, required=False, default=1)

    param = parser.parse_args()

    evaluate(
        root_img_dir=param.root_img_dir,
        ref_img_name=param.ref_img_name,
        tgt_path=param.tgt_path,
        result_path=param.result_path,
        colorspace=param.colorspace,
        cores=param.cores
    )

    elapsed = (perf_counter() - start_time) // 60
    print(f'Evaluation took {elapsed} minutes')

if __name__ == '__main__':
    main()
