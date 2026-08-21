"""Generates tiles from WSI files"""

import gc
import json
import tarfile
from argparse import ArgumentParser
from functools import partial
from io import BytesIO
from multiprocessing import Pool
from os import makedirs
from os.path import join
from typing import Generator, Sequence, Tuple, Union

from numpy import array, floor, ndarray
from numpy import sum as np_sum
from PIL import Image
from tifffile import TiffFile

class WsiTileGenerator:
    """Generates WSI tiles"""

    def __init__(
            self,
            data_source: str,
            output_root: str,
            tar_wrapper: bool,
            masked_tiles: bool = False,
            acceptable_tile_percentage: float= 50.0,
            paired_wsi: bool = False,
        ) -> None:
        self._data_source = data_source
        self._output_root = output_root
        self._tar_wrapper = tar_wrapper
        self._masked_tiles = masked_tiles
        self._acceptable_tile_percentage = acceptable_tile_percentage
        self._paired_wsi = paired_wsi

    def generate_wsi_tiles(
            self,
            img_list: Sequence[str],
            tile_size: int,
            stride: int,
            tile_format: str,
            multiprocess: bool=False
        ) -> None:
        """Generate tiles from wsi files"""
        for img_dict in img_list:
            output = join(
                self._output_root,
                img_dict['dataset_type'], # train/ test
                str(tile_size), # 512/ 1024/ 2048
                img_dict['image_type'] # unstained/ stained
            )
            makedirs(output, exist_ok=True)
            if multiprocess:
                pool = Pool(5)
                pool.map(
                    partial(
                        self._process_wsi,
                        output=output,
                        tile_format=tile_format,
                        tile_size=tile_size,
                        stride=stride
                    ),
                    img_dict['images']
                )
            else:
                for img_name in img_dict['images']:
                    self._process_wsi(
                        img_name=img_name,
                        output=output,
                        tile_format=tile_format,
                        tile_size=tile_size,
                        stride=stride
                    )

    def _process_wsi(
            self,
            img_name: str,
            output: str,
            tile_format: str,
            tile_size: int,
            stride: int
        ):
        """Process individual WIS for tiling"""
        if self._paired_wsi:
            img_name_components = img_name.split('_')
            sample_number = f'{img_name_components[0]}_{img_name_components[1]}'
            wsi_path = join(self._data_source, sample_number, img_name)
            mask_path = join(self._data_source, sample_number, f'{sample_number}_mask.tif')
        else:
            wsi_path = join(self._data_source, img_name)
            mask_path = join(self._data_source, f'mask_{img_name}')

        with TiffFile(wsi_path) as tif_img:
            img = tif_img.pages[0].asarray()
        if self._masked_tiles:
            with TiffFile(mask_path) as tif_mask:
                mask = array(tif_mask.pages[0].asarray()).astype('bool')
        else:
            mask = None
        tiles = self._tile_generator(
            img=img,
            mask=mask,
            tile_size=tile_size,
            stride=stride
        )
        if self._tar_wrapper:
            self._write_tiles_to_tar(
                img_name=img_name,
                output=output,
                tiles=tiles,
                tile_format=tile_format
            )
        else:
            self._write_tiles_to_disk(
                img_name=img_name,
                output=output,
                tiles=tiles,
                tile_format=tile_format
            )
        del img
        del mask
        gc.collect()

    def _write_tiles_to_disk(
            self,
            img_name: str,
            output: str,
            tiles: Generator[Tuple[Image.Image, int, int], None, None],
            tile_format: str
        ):
        """Write tile to case specific folder in the output"""
        for tile, row, col in tiles:
            output_img_folder = join(output, img_name.split('.')[0])
            makedirs(output_img_folder, exist_ok=True)
            tile_path = join(
                output_img_folder,
                self._get_tile_name(
                    img_name=img_name,
                    row_ind=row, col_ind=col,
                    tile_format=tile_format
                )
            )
            if tile_format == "png":
                tile.save(tile_path, format="png", compress_level=1)
            elif tile_format == "jpg":
                tile.save(tile_path, format="jpeg", quality=80)
            elif tile_format == "tif":
                tile.save(tile_path, format="tiff", compression='tiff_deflate')
            else:
                tile.save(tile_path, format=tile_format)

    def _write_tiles_to_tar(
            self,
            img_name: str,
            output: str,
            tiles: Generator[Tuple[Image.Image, int, int], None, None],
            tile_format: str
        ):
        """Write tiles to image specific tar file"""

        with tarfile.open(join(output, f'{img_name.split(".")[0]}.tar'), 'w') as tar:
            for tile, row, col in tiles:
                with BytesIO() as stream:
                    if tile_format == "png":
                        tile.save(stream, format="png", compress_level=1)
                    elif tile_format == "jpg":
                        tile.save(stream, format="jpeg", quality=80)
                    elif tile_format == "tif":
                        tile.save(stream, format="tiff", compression='tiff_deflate')
                    else:
                        tile.save(stream, format=tile_format)
                    tile.close()
                    tile_name = self._get_tile_name(
                        img_name=img_name, row_ind=row, col_ind=col, tile_format=tile_format)
                    tarinfo = tarfile.TarInfo(name=tile_name)
                    # Set number of bytes to write.
                    # Previously this only worked when subtracting -1 byte, but
                    # produced truncated JPEGs that trouble some software but not all.
                    tarinfo.size = stream.getbuffer().nbytes
                    # Write bytestream from the beginning to tar file.
                    stream.seek(0)
                    tar.addfile(tarinfo, fileobj=stream)

    def _tile_generator(
            self,
            img: ndarray,
            mask: Union[ndarray, None],
            tile_size: int,
            stride: int
        ) -> Tuple[Image.Image, int, int]:
        """A generator that returns tiles from tiff in a grid-like way"""
        img_h, img_w = img.shape[0:2]
        rows = floor((img_h - tile_size + stride) / stride).astype("int")
        cols = floor((img_w - tile_size + stride) / stride).astype("int")
        for row_ind in range(rows):
            for col_ind in range(cols):
                # Top-left coordinates of tile at full-res.
                h_start = row_ind * stride
                w_start = col_ind * stride
                # Bottom-right coordinates of tile at full-res.
                if mask is not None:
                    mask_tile = \
                        mask[h_start : h_start + tile_size, w_start : w_start + tile_size]
                    if np_sum(mask_tile) <\
                        tile_size * tile_size * (self._acceptable_tile_percentage/100):
                        continue

                tile = img[h_start : h_start + tile_size, w_start : w_start + tile_size]
                tile_img = Image.fromarray(tile)
                tile_img = tile_img.convert("RGB")
                yield tile_img, row_ind, col_ind

    @staticmethod
    def _save_tile(tile: ndarray, tile_name: str):
        """Save tile on disk"""
        tile_img = Image.fromarray(tile)
        tile_img = tile_img.convert("RGB")
        tile_img.save(tile_name)

    @staticmethod
    def _get_tile_name(img_name: str, row_ind: int, col_ind: int, tile_format: str) -> str:
        return f'{img_name.split(".")[0]}_{row_ind}_{col_ind}.{tile_format}'


def main():
    """The main function"""
    parser = ArgumentParser()
    parser.add_argument(
        '--config', help='Operation specific config', type=str, required=True)
    param = parser.parse_args()
    with open(param.config) as json_file:
        config = json.load(json_file)

    WsiTileGenerator(
        data_source=config['data_source'],
        output_root=config['output_root'],
        tar_wrapper=config['tar_wrapper'],
        masked_tiles=config['masked_tiles'],
        acceptable_tile_percentage=config['acceptable_tile_percentage']
    ).generate_wsi_tiles(
        img_list=config['image_list'],
        tile_size=config['tile_size'],
        stride=config['stride'],
        tile_format=config['tile_format']
    )

if __name__ == '__main__':
    main()
