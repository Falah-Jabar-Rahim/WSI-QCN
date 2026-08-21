"""Various util functions related to data processing"""


import json
from argparse import ArgumentParser
from os import rename
from os.path import join
from typing import Any, Mapping, Optional, Union

import numpy as np
from tifffile import TiffWriter, TiffFile
import tensorflow as tf
from PIL import Image
from scipy.ndimage.morphology import binary_fill_holes
from skimage import io, measure, morphology, transform


def rename_wsi(original_file_path: str, renamed_file_path: str):
    """Rename WSI to have shorter descriptive names"""
    rename(original_file_path, renamed_file_path)

def recreate(file_path: str, recreated_path: str):
    """Recreate tiff with uint8 dtype to reduce the size"""
    with TiffFile(file_path) as data:
        data_arr = data.pages[0].asarray()

    image_size_in_gb = (np.prod(data_arr.shape)) / (1024**3)
    bigtiff = image_size_in_gb > 4
    with TiffFile(recreated_path, 'w', bigtiff=bigtiff) as handle:
        handle.write(data_arr)

def downsample(file_path: str, downsampled_path: Optional[str]=None, factor: int=10):
    """Downsample tiffs"""
    with TiffFile(file_path) as data:
        data_arr = data.pages[0].asarray()
    downsampled = data_arr[::factor, ::factor]
    if downsampled_path:
        pil_img = Image.fromarray(downsampled)
        pil_img = pil_img.convert('RGB')
        pil_img.save(downsampled_path.replace('tif', 'jpg'), format="jpeg", quality=80)
    return downsampled

def generate_mask(
        img_file: Union[str, np.ndarray],
        mask_path: Optional[str] = None,
        masked_img_path: Optional[str] = None,
        lower_threshold: int = 20,
        upper_threshold: int = 120,
        wsi_format: str = 'tif'
    ):
    """Generate Mask"""
    if isinstance(img_file, str):
        img = Image.open(img_file)
    else:
        img = Image.fromarray(img_file)
    grayscale = np.array(img.convert('L'))
    mask = np.ones_like(grayscale, dtype='bool')
    mask[grayscale < lower_threshold] = 0
    mask[grayscale > upper_threshold] = 0
    # Tophat filter works well for the satellite tissue
    # sections that contains beehive/honeycomb like pattern
    #b_mask = morphology.black_tophat(mask, morphology.disk(20))
    #w_mask = morphology.white_tophat(mask, morphology.disk(20))
    #mask = b_mask + w_mask
    mask = morphology.dilation(mask, selem=morphology.disk(5))
    mask = binary_fill_holes(mask)
    mask = morphology.binary_erosion(mask, selem=morphology.disk(8))
    mask = morphology.dilation(mask, selem=morphology.disk(3))
    if mask_path:
        pil_img = Image.fromarray(mask.astype('uint8')*255)
        pil_img.save(
            mask_path.replace(wsi_format, 'jpg'), format="jpeg", quality=80)
    if masked_img_path:
        img = np.array(img)
        img[mask < 1] = 0
        pil_img = Image.fromarray(img)
        pil_img = pil_img.convert('RGB')
        pil_img.save(
            masked_img_path.replace(wsi_format, 'jpg'), format="jpeg", quality=80)
    return mask

def resize_and_save_tiff_mask(
        mask_file: Union[str, np.ndarray],
        original_tiff_path: str,
        tiff_mask_path: str
    ):
    """Resize and save tiff mask"""
    with TiffFile(original_tiff_path) as original_tiff:
        output_shape = original_tiff.pages[0].shape
    height, width = output_shape[:2]
    if isinstance(mask_file, str):
        img = io.imread(mask_file, as_gray=True)
        mask = img > 0
    else:
        mask = mask_file

    upsampled_mask = transform.resize(mask, (height, width)).astype('uint8')
    upsampled_mask[upsampled_mask > 0] = 255
    image_size_in_gb = (height * width) / (1024**3)
    bigtiff = image_size_in_gb > 4
    with TiffWriter(tiff_mask_path, 'w', bigtiff=bigtiff) as tiff_mask:
        tiff_mask.write(
            upsampled_mask.astype('uint8'),
            photometric='minisblack',
            planarconfig='contig',
            tile=(512, 512),
            compression='lzw'
        )

def load_jpeg(file_path, channels=3):
    """Load jpeg image from the path"""
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=channels)
    return img

def load_tiff(file_path):
    """Load tiff images from the path"""
    img = io.imread(file_path, plugin="pil", as_gray=False)
    return tf.convert_to_tensor(img)

def normalize(img):
    """Normalize data between (-1,1)"""
    #pylint: disable=unexpected-keyword-arg, no-value-for-parameter
    img = tf.cast(img, dtype=tf.float32)
    return (img / 127.5) - 1.0

def generate_intersection_mask(
        ds_img_a_path: str,
        ds_img_b_path: str,
        original_tiff_path: str,
        tiff_mask_path: str,
        ds_mask_path: Optional[str] = None,
        biggest_component: bool = False,
        ds_output: bool = False):
    """Generate intersection mask of two registered images"""
    mask_a = generate_mask(img_file=ds_img_a_path, upper_threshold=120)
    mask_b = generate_mask(img_file=ds_img_b_path, upper_threshold=180)
    intersection_mask = np.logical_and(mask_a, mask_b)
    if biggest_component:
        intersection_mask = get_biggest_component(mask=intersection_mask)
    intersection_mask = remove_small_component(mask=intersection_mask)
    if ds_mask_path:
        mask = intersection_mask.astype('uint8')
        mask[mask > 0] = 255
        pil_mask = Image.fromarray(mask)
        pil_mask = pil_mask.convert('RGB')
        pil_mask.save(ds_mask_path, format="png")
    if ds_output:
        img_a = np.array(Image.open(ds_img_a_path))
        img_b = np.array(Image.open(ds_img_b_path))
        img_a[intersection_mask < 1] = 0
        img_b[intersection_mask < 1] = 0
    resize_and_save_tiff_mask(
        mask_file=intersection_mask,
        original_tiff_path=original_tiff_path,
        tiff_mask_path=tiff_mask_path
    )

def get_biggest_component(mask: np.ndarray) -> np.ndarray:
    """Return three biggest components with respect to their centroid's y-axis"""
    new_mask = mask
    labels, label_count = measure.label(mask, return_num=True)
    if label_count > 1:
        label_size_list = [(ind, np.sum(labels == ind)) for ind in range(label_count+1) if ind > 0]
        label_size_list = sorted(label_size_list, key=lambda x:x[1], reverse=True)
        new_mask = labels == label_size_list[0][0]
    return new_mask

def remove_small_component(mask: np.ndarray, minimum_size: int=5000) -> np.ndarray:
    """Return three biggest components with respect to their centroid's y-axis"""
    new_mask = mask
    labels, label_count = measure.label(mask, return_num=True)
    if label_count > 1:
        for ind in range(1, label_count+1):
            if np.sum(labels == ind) < minimum_size:
                new_mask[labels == ind] = 0
    return new_mask

def png_to_tiff(file_path: str):
    """Convert a tiff file and write to the same location with same name"""
    img_arr = np.array(Image.open(file_path))
    image_size_in_gb = (np.prod(img_arr.shape[:3])) / (1024**3)
    bigtiff = image_size_in_gb > 4
    tiff_path = file_path.replace('.png','.tif')
    with TiffWriter(tiff_path, bigtiff=bigtiff) as tiff_file:
       tiff_file.write(
            img_arr.astype('uint8'),
            photometric='rgb',
            planarconfig='contig',
            tile=(512, 512),
            compression='lzw'
        )

def get_config(config_path: str) -> Mapping[str, Any]:
    """Return config json"""
    with open(config_path) as json_file:
        config = json.load(json_file)
    return config

def main():
    """The main function to start the flow of intersection mask script"""

    parser = ArgumentParser()
    parser.add_argument(
        '--config_path', help='Intersection mask config', type=str, required=True)

    param = parser.parse_args()
    config = get_config(config_path=param.config_path)
    if not config['sample_range']:
        sample_list = config['sample_list']
    else:
        sample_list = [f'sample_{x}' for x in range(config['sample_range'][0],
                                                    config['sample_range'][1])]
    for sample in sample_list:
        print(f'Generating intersection mask for {sample}')
        ds_unstained_path = join(
            config['data_root'], sample, f'ds_{sample}_unstained.png')
        ds_stained_path = join(
            config['data_root'], sample, f'ds_{sample}_stained.png')
        original_tiff_path = join(config['data_root'], sample, f'{sample}_unstained.tif')
        tiff_mask_path = join(config['data_root'], sample, f'{sample}_mask.tif')
        ds_mask_path = join(config['data_root'], sample, f'ds_{sample}_mask.png')\
             if config.get('downsample_mask', False) else None
        generate_intersection_mask(
            ds_img_a_path=ds_unstained_path,
            ds_img_b_path=ds_stained_path,
            original_tiff_path=original_tiff_path,
            tiff_mask_path=tiff_mask_path,
            ds_mask_path=ds_mask_path,
            biggest_component=config.get('biggest_component', False)
        )
        print(f'Intersection mask generated for {sample}')

if __name__ == '__main__':
    main()
