from __future__ import division

from os.path import join
import numpy as np

from tifffile import TiffFile, TiffWriter
from PIL import Image

class CommonNormalize:
	
	def __init__(
			self,
			root_img_dir: str,
			ref_img_name: str,
			src_img_name: str,
			output_dir: str,
			tile_size:int =1024
		):
		""""""
		self.root_img_dir = root_img_dir
		self.output_dir = output_dir
		self.src_img_name = src_img_name
		self.ref_img_name = ref_img_name
		self.ref_wsi = self._read_img(join(self.root_img_dir, self.ref_img_name))
		self.src_wsi = self._read_img(join(self.root_img_dir, self.src_img_name))
		self.src_mask = self._read_img_mask(join(self.root_img_dir, f'mask_{self.src_img_name}'))
		self.tile_size = tile_size


	def _read_img(self, img_path: str) -> np.ndarray:
		"""Read wsi as a tif image"""
		with TiffFile(img_path) as wsi:
			wsi_arr = wsi.pages[0].asarray().astype('uint8')
		return wsi_arr


	def _read_img_mask(self, mask_path: str) -> np.ndarray:
		"""Read mask as a binary image"""
		with TiffFile(mask_path) as mask:
			mask_arr = mask.pages[0].asarray().astype('bool')
		return mask_arr


	def _write_output(self, normalized_img: np.ndarray, ds_factor:int=5):
		"""Write normlized output to disk along with downsampled image"""
		
		output_img_path = join(self.output_dir, self.src_img_name)
		with TiffWriter(output_img_path, bigtiff=False) as tiff_file:
			tiff_file.write(
            normalized_img,
            photometric='rgb',
            planarconfig='contig',
            tile=(512, 512),
            #compression='lzw'
        )
		
		ds_normalized = normalized_img[::ds_factor, ::ds_factor, :]
		pil_img = Image.fromarray(ds_normalized)
		pil_img = pil_img.convert('RGB')
		ds_img_path = join(self.output_dir, f'ds_{self.src_img_name.replace("tif","jpg")}')
		pil_img.save(ds_img_path, format="jpeg", quality=100)
		return

