"""Load image data from disk"""

import random
from copy import deepcopy
from os import listdir
from os.path import isfile, join
from time import time
from typing import Any, List, Mapping, Sequence

import tensorflow as tf
from numpy import arange

from utils.utils import get_paired_samples

from .augment_data import (color, flip, flip_pair, rotate, rotate_pair, zoom,
                           zoom_pair)
from .utils import load_jpeg, normalize


class DataLoader:
    """For dataset loading pipeline"""

    def __init__(
            self,
            config: Mapping[str, Any]
        ):
        self._batch_size = config['batch_size']
        self._augmentation_config = config.get('augmentation', None)
        self._val_tiles_percentage_per_sample = config['val_tiles_percentage_per_sample']
        self._tiles_percentage_per_sample = config['tiles_percentage_per_sample']
        self._train_source = config['train']['source']
        self._train_target = config['train']['target']
        self._ensure_batch_size_divisibility = config.get('multi_gpu', False)
        self._batch_size = config['batch_size']
        self._augmentation_functions = {
            'color': color,
            'flip': flip,
            'rotate': rotate,
            'zoom': zoom
        }

    def get_train_val_dataset(self, data_root: str):
        """Return train and val dataset"""
        source_train_dataset, source_val_dataset = self._get_datasets(
            sample_list=self._train_source,
            data_path=join(data_root, 'source'),
            tiles_percentage_per_sample=self._tiles_percentage_per_sample,
            val_tiles_percentage_per_sample=self._val_tiles_percentage_per_sample
        )
        target_train_dataset, target_val_dataset = self._get_datasets(
            sample_list=self._train_target,
            data_path=join(data_root, 'target'),
            tiles_percentage_per_sample=self._tiles_percentage_per_sample,
            val_tiles_percentage_per_sample=self._val_tiles_percentage_per_sample
        )
        if self._augmentation_config:
            source_train_dataset = self._augment_dataset(source_train_dataset)
            target_train_dataset = self._augment_dataset(target_train_dataset)

        source_train_dataset = self._configure_for_performance(
            source_train_dataset, self._batch_size)
        source_val_dataset = self._configure_for_performance(
            source_val_dataset, self._batch_size)
        target_train_dataset = self._configure_for_performance(
            target_train_dataset, self._batch_size)
        target_val_dataset = self._configure_for_performance(
            target_val_dataset, self._batch_size)

        return (tf.data.Dataset.zip((source_train_dataset, target_train_dataset)),
           tf.data.Dataset.zip((source_val_dataset, target_val_dataset)))

    def _get_datasets(
            self,
            sample_list: Sequence[str],
            data_path: str,
            tiles_percentage_per_sample: float,
            val_tiles_percentage_per_sample: float):
        """Return train and validation Datasets for a particular type of tiles"""
        random.seed(1984)
        train_tiles_path_list: List[str] = []
        val_tiles_path_list: List[str] = []
        for sample_name in sample_list:
            sample_tiles_path = join(data_path, sample_name.split('.')[0])
            sample_tiles_path_list = [join(sample_tiles_path, tile_name)\
                for tile_name in listdir(sample_tiles_path)\
                 if isfile(join(sample_tiles_path, tile_name))]
            random.shuffle(sample_tiles_path_list)
            train_tiles_count = int(len(sample_tiles_path_list)*(tiles_percentage_per_sample/100))
            val_tiles_count = int(len(sample_tiles_path_list)*(val_tiles_percentage_per_sample/100))
            train_tiles_path_list.extend(sample_tiles_path_list[:train_tiles_count])
            val_tiles_path_list.extend(
                sample_tiles_path_list[train_tiles_count:train_tiles_count + val_tiles_count])
            print('Training samples', train_tiles_count)
            print('Validation samples', val_tiles_count)
        random.shuffle(train_tiles_path_list)
        random.shuffle(val_tiles_path_list)
        print('Total training tiles: ', len(train_tiles_path_list))
        print('Total validation tiles: ', len(val_tiles_path_list))
        #TO-DO replace output type by output signature
        # and keep the output shape configurable

        if self._ensure_batch_size_divisibility:
            train_modulo = len(train_tiles_path_list) % self._batch_size
            if train_modulo != 0:
                train_tiles_path_list = train_tiles_path_list[:-train_modulo]

        train = tf.data.Dataset.from_generator(
            lambda: img_gen(train_tiles_path_list),
            (tf.float32)
        )
        val = tf.data.Dataset.from_generator(
            lambda: img_gen(val_tiles_path_list),
            (tf.float32)
        )
        return train, val

    def _augment_dataset(self, dataset: tf.data.Dataset):
        """Apply augmentation to dataset based on augmentation config"""
        fraction = self._augmentation_config['sample_percentage']/100.0
        for augmentation in self._augmentation_config['augmentation_types']:
            func = self._augmentation_functions[augmentation]
            dataset = dataset.map(
                lambda x: tf.cond(
                    tf.random.uniform([], 0, 1) < fraction,
                    lambda: func(x), #pylint: disable=cell-var-from-loop
                    lambda: x),
                num_parallel_calls=tf.data.experimental.AUTOTUNE
            )
        dataset = dataset.map(
            lambda x: tf.clip_by_value(x, -1, 1),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return dataset

    @staticmethod
    def _configure_for_performance(dataset: tf.data.Dataset, batch_size: int):
        """Fine tune data sets for performance"""
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return dataset

def img_gen(img_paths: Sequence[str]):
    """Generator for tf.data.Dataset source for single image generation"""
    indexes = arange(len(img_paths))
    random.shuffle(indexes)
    for ind in indexes:
        #file_path_tensor = convert_to_tensor(img_paths[ind])
        img = load_jpeg(img_paths[ind])
        img = normalize(img)
        yield img

class PairedDataLoader:
    """For paired dataset loading pipeline"""

    def __init__(
            self,
            config: Mapping[str, Any],
            dataset_type: str='train'
        ):
        self._batch_size = config['batch_size']
        self._augmentation_config = config.get('augmentation', None)
        if dataset_type == 'train':
            self._val_tiles_percentage_per_sample = config['val_tiles_percentage_per_sample']
            self._tiles_percentage_per_sample = config['tiles_percentage_per_sample']
            self._train_source = config['train']['source']
            self._train_target = config['train']['target']
            self._paired_data_percentage = config.get('paired_data_percentage', 100)
        elif dataset_type == 'val':
            self._val_tiles_percentage_per_sample = 5
            self._tiles_percentage_per_sample = config.get('val_tiles_percentage_per_sample', 50)
            self._train_source = config['val']['source']
            self._train_target = config['val']['target']
            self._paired_data_percentage = 100
        self._ensure_batch_size_divisibility = config.get('multi_gpu', False)
        self._batch_size = config['batch_size']
        self._augmentation_functions = {
            'flip': flip_pair,
            'rotate': rotate_pair,
            'zoom': zoom_pair
        }

    def get_train_val_dataset(self, data_root: str):
        """Return train and validation datasets"""
        paired_sample_list = get_paired_samples(
                unstained_list=self._train_source,
                stained_list=self._train_target
            )

        train_paired_dataset, val_paired_dataset = self._get_paired_tile_datasets(
                paired_sample_list=paired_sample_list,
                data_root=data_root,
                tiles_percentage_per_sample=self._tiles_percentage_per_sample,
                val_tiles_percentage_per_sample=self._val_tiles_percentage_per_sample
            )

        if self._augmentation_config:
            train_paired_dataset = self._augment_dataset(dataset=train_paired_dataset)

        train_paired_dataset = self._configure_for_performance(
            train_paired_dataset, self._batch_size)
        val_paired_dataset = self._configure_for_performance(
            val_paired_dataset, self._batch_size)

        return train_paired_dataset, val_paired_dataset

    def _get_paired_tile_datasets(
            self,
            paired_sample_list: Sequence[str],
            data_root: str,
            tiles_percentage_per_sample: float,
            val_tiles_percentage_per_sample: float
        ):
        """Return train and val Datasets for a particular type of tiles"""
        random.seed(1984)
        train_paired_tiles_path_list: List[str] = []
        val_paired_tiles_path_list: List[str] = []

        print(paired_sample_list)
        for sample_a, sample_b in paired_sample_list:
            sample_a_tiles_path_list = self._get_sample_tiles_path(
                join(data_root,'source'), sample_a)
            sample_b_tiles_path_list = self._get_sample_tiles_path(
                join(data_root,'target'), sample_b)
            sample_a_tiles_path_list.sort()
            sample_b_tiles_path_list.sort()
            if len(sample_a_tiles_path_list) != len(sample_b_tiles_path_list):
                raise RuntimeError(f'Unequal tiles in {sample_a} {len(sample_a_tiles_path_list)}'
                                   f'and {sample_b} {len(sample_b_tiles_path_list)}')
            paired_tiles_path_list = list(zip(sample_a_tiles_path_list,
                                              sample_b_tiles_path_list))
            random.shuffle(paired_tiles_path_list)
            train_tiles_count = int(len(paired_tiles_path_list)*(tiles_percentage_per_sample/100))
            val_tiles_count = int(len(paired_tiles_path_list)*(val_tiles_percentage_per_sample/100))
            
            print('Training samples', train_tiles_count)
            print('Validation samples', val_tiles_count)
            
            train_data_list = paired_tiles_path_list[:train_tiles_count]
            val_data_list = paired_tiles_path_list\
                [train_tiles_count:train_tiles_count + val_tiles_count]

            train_paired_tiles_path_list.extend(train_data_list)
            val_paired_tiles_path_list.extend(val_data_list)
        
        random.shuffle(train_paired_tiles_path_list)
        random.shuffle(val_paired_tiles_path_list)
        print('Total training tiles: ', len(train_paired_tiles_path_list))
        print('Total validation tiles: ', len(val_paired_tiles_path_list))
        
        if self._ensure_batch_size_divisibility:
            train_modulo = len(train_paired_tiles_path_list) % self._batch_size
            if train_modulo != 0:
                train_paired_tiles_path_list = train_paired_tiles_path_list[:-train_modulo]
        #TO-DO replace output type by output signature
        # and keep the output shape configurable
        train = tf.data.Dataset.from_generator(
            lambda: img_pair_gen(train_paired_tiles_path_list, self._paired_data_percentage),
            (tf.float32, tf.float32)
        )
        val = tf.data.Dataset.from_generator(
            lambda: img_pair_gen(val_paired_tiles_path_list, 100),
            (tf.float32, tf.float32)
        )
        return train, val

    def _augment_dataset(self, dataset: tf.data.Dataset):
        """Apply augmentation to dataset based on augmentation config"""
        fraction = self._augmentation_config['sample_percentage']/100.0
        for augmentation in self._augmentation_config['augmentation_types']:
            func = self._augmentation_functions[augmentation]
            dataset = dataset.map(
                lambda x, y: tf.cond(
                    tf.random.uniform([], 0, 1) < fraction,
                    lambda: func(x, y), #pylint: disable=cell-var-from-loop
                    lambda: (x, y)),
                num_parallel_calls=tf.data.experimental.AUTOTUNE
            )
        dataset = dataset.map(
            lambda x, y: (tf.clip_by_value(x, -1, 1), tf.clip_by_value(y, -1, 1)),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return dataset

    @staticmethod
    def _verify_tile_pairs_matching(tile_pairs: Sequence):
        """Ensure on the basis of index tiles are from the same location in two WSIs"""
        for tile_a, tile_b in tile_pairs:
            parts_a = tile_a[:-4].split('_')
            parts_b = tile_b[:-4].split('_')
            if parts_a[3] != parts_b[3] or parts_a[4] != parts_b[4]:
                raise RuntimeError(f'Tiles mismatch {tile_a} and {tile_b}')

    @staticmethod
    def _get_sample_tiles_path(data_path: str, sample_name: str):
        """Return a list of a WSI sample tiles paths"""
        sample_tiles_path = join(data_path, sample_name.split('.')[0])
        sample_tiles_path_list = [join(sample_tiles_path, tile_name)\
            for tile_name in listdir(sample_tiles_path)\
                if '.png' in tile_name]
        return sample_tiles_path_list

    @staticmethod
    def _configure_for_performance(dataset: tf.data.Dataset, batch_size: int):
        """Fine tune data sets for performance"""
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return dataset

def img_pair_gen(img_paths: Sequence, percentage: float):
    """Generator for tf.data.Dataset source for paired image generation"""
    percentage_paired_img_paths = get_percentage_paired_data(
        img_paths=img_paths, percentage=percentage) if percentage < 100\
            else deepcopy(img_paths)
    indexes = arange(len(percentage_paired_img_paths))
    random.seed(time())
    random.shuffle(indexes)
    for ind in indexes:
        img_a = load_jpeg(percentage_paired_img_paths[ind][1], channels=1)
        img_a = normalize(img_a)
        img_b = load_jpeg(percentage_paired_img_paths[ind][1])
        img_b = normalize(img_b)
        yield (img_a, img_b)

def get_percentage_paired_data(img_paths: Sequence, percentage: float):
    """Return partially paried data"""
    if percentage == 0:
        sample_a_path_list, sample_b_path_list = map(
        list, zip(*img_paths))
    else:
        count = len(img_paths)
        pair_count = int(count*(percentage/100))
        paired_tiles_path_list = img_paths[:pair_count]
        sample_a_path_list, sample_b_path_list = map(
            list, zip(*img_paths[pair_count:]))
    random.shuffle(sample_a_path_list)
    random.shuffle(sample_b_path_list)
    unpaired_tiles_path_list = list(zip(sample_a_path_list, sample_b_path_list))
    return unpaired_tiles_path_list if percentage == 0 else\
        paired_tiles_path_list + unpaired_tiles_path_list
