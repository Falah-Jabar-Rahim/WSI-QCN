"""One time preprocessing"""

import json
from argparse import ArgumentParser
from functools import partial
from multiprocessing import Pool
from os import listdir
from os.path import join
from typing import Any, Mapping, Optional, Tuple, Union

from tqdm import tqdm

from .utils import (downsample, generate_mask, recreate, rename_wsi,
                    resize_and_save_tiff_mask)


def main():
    """The main function"""

    parser = ArgumentParser()
    parser.add_argument(
        '--config_path', help='One time preprocessing config'
    )
    parser.add_argument(
        '--multiprocess', action='store_true')
    param = parser.parse_args()
    with open(param.config_path) as json_file:
        config = json.load(json_file)
    wsi_src = config['wsi_src']
    wsi_dst = config['wsi_dst']
    ds_output = config['ds_output']
    thresholds = config['thresholds']
    multiprocess = param.multiprocess
    if config['original_list']:
        wsi_pair_list = list(zip(config['original_list'], config['renamed_list']))
    else:
        wsi_pair_list = [(wsi, None) for wsi in listdir(wsi_src) if '.jpg' in wsi]
    if not multiprocess:
        for wsi_name_pair in tqdm(wsi_pair_list):
            otp_image(wsi_name_pair, wsi_src, wsi_dst, thresholds, ds_output)
    else:
        pool = Pool(6)
        pool.map(
            partial(
                otp_image,
                wsi_src=wsi_src,
                wsi_dst=wsi_dst,
                thresholds=thresholds,
                ds_output=ds_output),
            wsi_pair_list
        )

def otp_image(
        wsi_name_pair: Tuple[str, Union[str, None]],
        wsi_src: str,
        wsi_dst: str,
        thresholds: Mapping[str, Any],
        ds_output: Optional[str]
    ) -> None:
    """One time preprocess tiff images for uint conversion and mask generation"""
    wsi, renamed_wsi = wsi_name_pair
    if renamed_wsi:
        rename_wsi(
           original_file_path=join(wsi_src, wsi),
           renamed_file_path=join(wsi_src, renamed_wsi)
        )
        wsi = renamed_wsi
    recreate(
       file_path=join(wsi_src, wsi),
       recreated_path=join(wsi_dst, wsi))
    downsampled = downsample(
        file_path=join(wsi_dst, wsi),
        downsampled_path=join(ds_output, f'ds_{wsi}') if ds_output else None)
    img_type = 'unstained' if 'unstained' in wsi else 'stained'
    mask = generate_mask(
        img_file=downsampled,
        mask_path=join(ds_output, f'ds_mask_{wsi}') if ds_output else None,
        masked_img_path=join(ds_output, f'ds_masked_{wsi}') if ds_output else None,
        lower_threshold=thresholds[img_type][0],
        upper_threshold=thresholds[img_type][1])
    resize_and_save_tiff_mask(
        mask_file=mask,
        original_tiff_path=join(wsi_dst, wsi),
        tiff_mask_path=join(wsi_dst, f'mask_{wsi}')
    )

if __name__ == '__main__':
    main()
