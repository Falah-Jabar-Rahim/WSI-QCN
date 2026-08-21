'''Histogram matching based normalization'''

import argparse
from os import listdir

import numpy as np
from skimage.exposure import match_histograms

from common_normalize import CommonNormalize


class HistogramNormalize(CommonNormalize):
	'''Wrapper class for histgram matching based normalization'''

	def __init__(
			self,
			root_img_dir: str,
			ref_img_name: str,
			src_img_name: str,
			output_dir: str
			):
		super().__init__(
			root_img_dir=root_img_dir,
			ref_img_name=ref_img_name,
			src_img_name=src_img_name,
			output_dir=output_dir
		)

	def normalize(self):
		"""Normalize by histogram matching of source w.r.t reference image"""
		try:
			matched = match_histograms(self.src_wsi, self.ref_wsi, multichannel=True)
			matched = np.array(matched[:]).astype('uint8')
			print('matching done')
		except Exception as e:
			print('matching crashed')
			print(str(e))
			exit(0)
		
		# Clean-up the image
		matched[self.src_mask == False] = 255
		self._write_output(matched)
		return

def main():
	"""Executes normalization by processing input arguments"""
	parser = argparse.ArgumentParser()
	parser.add_argument("--root_img_dir", default="input/dataset/PanNuke_images_HE/source/",
			help="Path to WSIs and their masks")
	parser.add_argument( "--ref_img_name", default="tar.png",
			help="Name of the reference WSI")
	parser.add_argument("--src_img_name", default="input/dataset/PanNuke_images_HE/source/",
			help="Name of the source WSI")
	parser.add_argument("--output_dir", default="output",
			help="Path where normalized WSIs will be saved")
	
	params = parser.parse_args()

	if params.src_img_name:
		histogram_normalize = HistogramNormalize(
			root_img_dir=params.root_img_dir, 
			ref_img_name=params.ref_img_name,
			src_img_name=params.src_img_name,
			output_dir=params.output_dir
			)
		histogram_normalize.normalize()
	else:
		src_img_list = [src_img_name for src_img_name in listdir(params.root_img_dir) \
			        if '.tif' in src_img_name and 'mask_' not in src_img_name and \
						src_img_name != params.ref_img_name]
		for src_img_name in src_img_list:
			histogram_normalize = HistogramNormalize(
			root_img_dir=params.root_img_dir, 
			ref_img_name=params.ref_img_name,
			src_img_name=src_img_name,
			output_dir=params.output_dir
			)
			histogram_normalize.normalize()
		


if __name__ == '__main__':
	main()

