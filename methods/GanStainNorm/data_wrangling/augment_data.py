"""Implement different data augmentation functions"""


import numpy as np
from tensorflow import Tensor, cond, float32, int32
#pylint: disable=import-error
from tensorflow.image import (crop_and_resize, flip_left_right, flip_up_down,
                              random_brightness, random_contrast,
                              random_flip_left_right, random_flip_up_down,
                              random_hue, rot90)
#pylint: disable=import-error
from tensorflow.random import uniform

#from data_wrangling.load_data import configure_for_performance, get_datasets

TILE_SIZE = 512 #1024 #256

def color(img: Tensor) -> Tensor:
    """Apply random hue, saturation, brightness and contrast changes"""
    img = random_hue(img, 0.03)
    #img = random_saturation(img, 5, 6)
    img = random_brightness(img, 0.2)
    img = random_contrast(img, 0.7, 1.3)
    return img

def flip(img: Tensor) -> Tensor:
    """Randomly flip the input left-right and up-down"""
    img = random_flip_left_right(img)
    img = random_flip_up_down(img)
    return img

def flip_pair(img_a: Tensor, img_b: Tensor) -> Tensor:
    """Flip the image pair left-right & up-down"""
    img_a, img_b = flip_left_right(img_a), flip_left_right(img_b)
    img_a, img_b = flip_up_down(img_a), flip_up_down(img_b)
    return img_a, img_b

def rotate(img: Tensor) -> Tensor:
    """Randomly rotate the input by 90, 180 or 270"""
    return rot90(img, uniform(shape=[], minval=0, maxval=4, dtype=int32))

def rotate_pair(img_a: Tensor, img_b: Tensor) -> Tensor:
    """Randomly rotate the input image pair by 90, 180 or 270"""
    rotation_angle = uniform(shape=[], minval=0, maxval=4, dtype=int32)
    return rot90(img_a, rotation_angle), rot90(img_b, rotation_angle)

def zoom(img: Tensor) -> Tensor:
    """Randomly magnify image from 1% to 3%"""
    # Generate 20 crop settings, ranging from a 1% to 3% crop.
    scales = list(np.arange(0.85, 1.0, 0.01))
    boxes = np.zeros((len(scales), 4))

    #TO-DO replace (512, 512) with something like
    # (img.shape[1], img.shape[1]) and figure out
    # configure img shape at run time
    crop_size = (TILE_SIZE, TILE_SIZE)

    for i, scale in enumerate(scales):
        x_start = y_start = 0.5 - (0.5 * scale)
        x_end = y_end = 0.5 + (0.5 * scale)
        boxes[i] = [x_start, y_start, x_end, y_end]

    def random_crop(img):
        # Create different crops for an image
        crops = crop_and_resize(
            [img], boxes=boxes, box_indices=np.zeros(len(scales)), crop_size=crop_size)
        # Return a random crop
        return crops[uniform(shape=[], minval=0, maxval=len(scales), dtype=int32)]

    choice = uniform(shape=[], minval=0., maxval=1., dtype=float32)

    # Only apply cropping 50% of the time
    return cond(choice < 0.5, lambda: img, lambda: random_crop(img))

def zoom_pair(img_a: Tensor, img_b: Tensor) -> Tensor:
    """Randomly magnify image from 1% to 3%"""
    # Generate 20 crop settings, ranging from a 1% to 3% crop.
    scales = list(np.arange(0.85, 1.0, 0.01))
    boxes = np.zeros((len(scales), 4))

    #TO-DO replace (512, 512) with something like
    # (img.shape[1], img.shape[1]) and figure out
    # configure img shape at run time
    crop_size = (TILE_SIZE, TILE_SIZE)

    for i, scale in enumerate(scales):
        x_start = y_start = 0.5 - (0.5 * scale)
        x_end = y_end = 0.5 + (0.5 * scale)
        boxes[i] = [x_start, y_start, x_end, y_end]

    def random_crop(img_a, img_b):
        # Create different crops for an image
        crops_a = crop_and_resize(
            [img_a], boxes=boxes, box_indices=np.zeros(len(scales)), crop_size=crop_size)
        crops_b = crop_and_resize(
            [img_b], boxes=boxes, box_indices=np.zeros(len(scales)), crop_size=crop_size)
        crop_index = uniform(shape=[], minval=0, maxval=len(scales), dtype=int32)
        # Return a random crop
        return crops_a[crop_index], crops_b[crop_index]

    choice = uniform(shape=[], minval=0., maxval=1., dtype=float32)

    #Apply 75% of the times
    return cond(choice < 0.75, lambda: random_crop(img_a, img_b), lambda: (img_a, img_b))
