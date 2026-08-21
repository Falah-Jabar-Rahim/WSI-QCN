''' 
Normalize staining appearence of H&E stained images using Macenko method.
Reference:
M. Macenko et al., ‘A method for normalizing histology slides for quantitative analysis’, 
in 2009 IEEE International Symposium on Biomedical Imaging: From Nano to Macro, 2009, pp. 1107–1110.
'''

import argparse
from os import listdir

from .common_normalize import CommonNormalize
from .stainnorm_macenko import Normalizer


class MacenkoNormalize(CommonNormalize):
	'''Wrapper class for Vahadane Normalization'''
	def __init__(
			self,
			root_img_dir: str,
			ref_img_name: str,
			src_img_name: str,
			output_dir: str,
			tile_size:int =1024
			):
		super().__init__(
			root_img_dir=root_img_dir,
			ref_img_name=ref_img_name,
			src_img_name=src_img_name,
			output_dir=output_dir,
			tile_size=tile_size
		)


	def normalize(self):
		"""Macenko stain normalize source w.r.t reference image"""
		stain_normalizer = Normalizer()
		stain_normalizer.fit(self.ref_wsi)
		try:
			normalized = stain_normalizer.transform(self.src_wsi)
			print('transformation done')
		except Exception as e:
			print('transformation crashed')
			print(str(e))
			exit(0)				
		
		# Clean-up the image
		normalized[self.src_mask == False] = 255
		self._write_output(normalized)
		return


def main():
	"""Executes normalization by processing input arguments"""
	parser = argparse.ArgumentParser()
	parser.add_argument("--root_img_dir", required=True,
			help="Path to WSIs and their masks")
	parser.add_argument( "--ref_img_name", required=True,
			help="Name of the reference WSI")
	parser.add_argument("--src_img_name", required=False,
			help="Name of the source WSI")
	parser.add_argument("--output_dir", required=True,
			help="Path where normalized WSIs will be saved")
	
	params = parser.parse_args()
	if params.src_img_name:
		macenko_normalize = MacenkoNormalize(
			root_img_dir=params.root_img_dir, 
			ref_img_name=params.ref_img_name,
			src_img_name=params.src_img_name,
			output_dir=params.output_dir
			)
		macenko_normalize.normalize()
	else:
		src_img_list = [src_img_name for src_img_name in listdir(params.root_img_dir) \
			        if '.tif' in src_img_name and 'mask_' not in src_img_name and \
						src_img_name != params.ref_img_name]
		for src_img_name in src_img_list:
			macenko_normalize = MacenkoNormalize(
			root_img_dir=params.root_img_dir, 
			ref_img_name=params.ref_img_name,
			src_img_name=src_img_name,
			output_dir=params.output_dir
			)
			macenko_normalize.normalize()

if __name__ == '__main__':
	main()

