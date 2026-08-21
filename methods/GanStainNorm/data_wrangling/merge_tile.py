"""Various tile merging functions"""

from functools import partial
from itertools import product
from multiprocessing import Pool
from os import makedirs
from os.path import join, isfile
from time import perf_counter
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

from numpy import divide, expand_dims, floor
from numpy import max as np_max
from numpy import multiply, ndarray, zeros, ones
from tifffile import TiffFile, TiffWriter
from skimage import io
from tqdm import tqdm


class MergeTiles:
    """Merge inference generated tiles to create virtually stained WSI"""
    QUADS = ['first', 'second', 'third', 'fourth']
    def __init__(
            self,
            config: Mapping[str, Any],
            exp_path: str,
            multiprocess: bool,
            inference_root: Optional[str] = None
        ) -> None:
        self._data_source = config['data_source']
        self._tile_size = config['tile_size']
        self._stride = config['stride']
        self._merge_type = config['merge_type']
        self._tile_format = config['tile_format']
        self._paired_wsi = config.get('paired_wsi', False)
        self._multiprocess = multiprocess
        self._exp_path = exp_path
        self._inference_root = inference_root
        self._tile_dims = (self._tile_size, self._tile_size, 3)
        self._img_epoch_pairs = product(
            config['test']['source'], config['epochs'])
        self._wsi_arr: ndarray

    def __call__(self):

        start_time = perf_counter()
        if self._multiprocess:
            pool = Pool(10)
            pool.map(
                partial(
                    self._merge_tiles,
                    tile_size=self._tile_size,
                    stride=self._stride),
                self._img_epoch_pairs
            )
        else:
            for img_epoch_pair in tqdm(self._img_epoch_pairs):
                self._merge_tiles(
                    img_epoch_pair, self._tile_size, self._stride)
        elapsed = (perf_counter() - start_time)//60
        print(f'Merging tiles took {elapsed} minutes')

    def _merge_tiles(
            self,
            img_epoch_pair: Tuple[str, int],
            tile_size:int,
            stride:int,
        ):
        reference_source, epoch = img_epoch_pair
        if self._paired_wsi:
            img_name_components = reference_source.split('_')
            sample_number = f'{img_name_components[0]}_{img_name_components[1]}'
            wsi_path = join(self._data_source, sample_number, reference_source)
        else:
            wsi_path = join(self._data_source, reference_source)
            wsi_mask_path = join(self._data_source, f'mask_{reference_source}')

        with TiffFile(wsi_path) as img:
            img_arr = img.pages[0].asarray()
        img_h, img_w = img_arr.shape[0:2]
        del img_arr

        with TiffFile(wsi_mask_path) as img:
            mask = img.pages[0].asarray().astype('bool')

        mid = int(tile_size/2)
        rows = floor((img_h - tile_size + stride) / stride).astype("int")
        cols = floor((img_w - tile_size + stride) / stride).astype("int")

        height = img_h #(rows * stride) + tile_size
        width = img_w #(cols * stride) + tile_size
        self._wsi_arr = ones((height, width, 3), dtype='uint8')
        sample_name = reference_source.split('.')[0]
        tile_path = join(
                self._inference_root, f'epoch{epoch:03d}', sample_name
            ) if self._inference_root else join(
                self._exp_path, 'inference', f'epoch{epoch:03d}', sample_name)
        wsi_output = join(self._exp_path, 'wsi', f'epoch{epoch:03d}')
        weight_array = self._get_weight_array(tile_size=tile_size)
        makedirs(wsi_output, exist_ok=True)
        # Assigning all border tiles before weighted merging

        for i in range(rows):
            for j in range(cols):
                if i == 0 or i == rows - 1 or j == 0 or j == cols - 1:
                    h_start = i * stride
                    w_start = j * stride
                    tile_ij = self._get_tile(tile_path, reference_source, i, j)
                    if tile_ij is None:
                        continue
                    self._wsi_arr[h_start : h_start + tile_size,\
                                w_start : w_start + tile_size] = tile_ij

        for i in range(rows):
            for j in range(cols):
                h_start = i * stride
                w_start = j * stride
                tile_ij = self._get_tile(tile_path, reference_source, i, j)
                if tile_ij is None:
                    continue
                if self._merge_type == 'simple':
                    self._wsi_arr[h_start : h_start + tile_size,\
                                w_start : w_start + tile_size] = tile_ij

                if self._merge_type == 'weighted':
                    tile_dict:Dict[str, Union[ndarray, None]] = {}
                    tile_dict['ij'] = tile_ij
                    tile_dict['up'] = self._get_tile(
                        tile_path, reference_source, i-1, j, True)
                    tile_dict['up_after']  = self._get_tile(
                        tile_path, reference_source, i-1, j+1, True)
                    tile_dict['after'] = self._get_tile(
                        tile_path, reference_source, i, j+1)
                    tile_dict['down_after'] = self._get_tile(
                        tile_path, reference_source, i+1, j+1)
                    tile_dict['down'] = self._get_tile(
                        tile_path, reference_source, i+1, j)
                    tile_dict['down_before'] = self._get_tile(
                        tile_path, reference_source, i+1, j-1)
                    tile_dict['before'] = self._get_tile(
                        tile_path, reference_source, i, j-1, True)
                    tile_dict['up_before'] = self._get_tile(
                        tile_path, reference_source, i-1, j-1, True)

                    # Tile quadrants coordinate slices
                    quad_coords = self._get_tile_quad_coords(stride=stride, mid=mid)
                    wsi_quad_coords = self._get_wsi_quad_coords(
                        h_start=h_start, w_start=w_start, stride=stride, mid=mid)
                    final_quads = self._get_final_quads(
                            tile_dict=tile_dict,
                            quad_coords=quad_coords,
                            weight_array=weight_array
                        )

                    for quad in self.QUADS:
                        self._wsi_arr[wsi_quad_coords[quad]] = final_quads[quad].astype('uint8')
        tiff_name = join(wsi_output, reference_source)
        image_size_in_gb = (height * width * 3) / (1024**3)
        self._wsi_arr[mask < 1] = 255
        bigtiff = image_size_in_gb > 4
        with TiffWriter(tiff_name, bigtiff=bigtiff) as tiff_file:
            tiff_file.write(
            self._wsi_arr,
            photometric='rgb',
            planarconfig='contig',
            tile=(512, 512),
            compression='lzw'
        )

    @staticmethod
    def _get_tile_quad_coords(stride: int, mid: int) -> Dict[str, Tuple[slice, slice]]:
        """Return tile inner quadrants (cropping/excluding stride) coordinate slices"""
        quad_coords: Dict[str, Tuple[slice, slice]] = {}
        quad_coords['first'] = (slice(stride, mid), slice(stride, mid))
        quad_coords['second'] = (slice(stride, mid), slice(mid, -stride))
        quad_coords['third'] = (slice(mid, -stride), slice(stride, mid))
        quad_coords['fourth'] = (slice(mid, -stride), slice(mid, -stride))
        return quad_coords

    @staticmethod
    def _get_wsi_quad_coords(h_start: int, w_start: int, stride: int, mid: int):
        """Return cooresponding quadrant coordinates of the WSI"""
        quad_coords: Dict[str, Tuple[slice, slice]] = {}
        quad_coords['first'] = \
                (slice(h_start + stride, h_start + mid),
                slice(w_start + stride, w_start + mid))
        quad_coords['second'] = \
            (slice(h_start + stride, h_start + mid),
                slice(w_start + mid, w_start + mid + stride))
        quad_coords['third'] = \
            (slice(h_start + mid, h_start + mid + stride),
            slice(w_start + stride, w_start + mid))
        quad_coords['fourth'] = \
            (slice(h_start + mid, h_start + mid + stride),
            slice(w_start + mid, w_start + mid + stride))
        return quad_coords

    def _get_final_quads(
            self,
            tile_dict: Dict[str, Any],
            quad_coords: Dict[str, Tuple[slice, slice]],
            weight_array: ndarray
        ) -> Dict[str, ndarray]:
        """Return accumulated quadrant weighted sum"""
        final_quads: Dict[str, ndarray] = {}
        tile_quad_pairs: Dict[str, List[Tuple[str, str]]] = {
            'first': [('ij', 'first'), ('up', 'third'),
                      ('before', 'second'), ('up_before', 'fourth')],
            'second': [('ij', 'second'), ('up', 'fourth'),
                       ('after', 'first'), ('up_after', 'third')],
            'third': [('ij', 'third'), ('down', 'first'),
                      ('before', 'fourth'), ('down_before', 'second')],
            'fourth': [('ij', 'fourth'), ('down', 'second'),
                       ('after', 'third'), ('down_after', 'first')],
        }
        quad_size = int(weight_array.shape[0]/4)
        quad_sum = zeros((quad_size, quad_size, 3))
        weight_sum = zeros((quad_size, quad_size, 1))
        for quad in self.QUADS:
            for tile_quad_pair in tile_quad_pairs[quad]:
                if tile_dict[tile_quad_pair[0]] is not None:
                    weighted_tile = multiply(weight_array, tile_dict[tile_quad_pair[0]])
                    quad_sum += weighted_tile[quad_coords[tile_quad_pair[1]]]
                    weight_sum += weight_array[quad_coords[tile_quad_pair[1]]]
            final_quads[quad] = divide(quad_sum, weight_sum)
            quad_sum = zeros((quad_size, quad_size, 3))
            weight_sum = zeros((quad_size, quad_size, 1))
        return final_quads

    @staticmethod
    def _get_weight_array(tile_size: int) -> ndarray:
        """Return numpy array containing pixel weights calculated from the center"""
        weight_array = zeros((tile_size, tile_size))
        xcen, ycen = tile_size/2 - 1, tile_size/2 - 1
        for i in range(tile_size):
            for j in range(tile_size):
                weight_array[i, j] = min(abs(i - xcen), abs(j - ycen))
        normalized = weight_array/np_max(weight_array)
        #normalized += 1e-10
        normalized = abs(1 - normalized)
        return expand_dims(normalized, axis=2)

    def _get_tile(
            self,
            tile_path:str,
            reference:str,
            row:int,
            col:int,
            computed:bool=False
        ) -> Union[ndarray, None]:
        """Return tile read from disk as numpy array"""
        split_ref = reference.split('.')
        tile_name = f'{split_ref[0]}_{row}_{col}.{self._tile_format}'
        if isfile(join(tile_path, tile_name)):
            if row > 0 and col > 0 and computed is True:
                row_slice = slice(row * self._stride, (row * self._stride) + self._tile_size)
                col_slice = slice(col * self._stride, (col * self._stride) + self._tile_size)
                tile = self._wsi_arr[row_slice, col_slice]
            else:
                tile = io.imread(join(tile_path, tile_name), plugin="pil", as_gray=False)
        else:
            tile = None
        return tile
