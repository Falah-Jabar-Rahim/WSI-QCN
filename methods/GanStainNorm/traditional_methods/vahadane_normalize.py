'''Normalize staining appearence of H&E stained images using Vahadane method.
Reference:
A. Vahadane et al., ‘Structure-Preserving Color Normalization and Sparse Stain Separation for Histological Images’, 
IEEE Transactions on Medical Imaging, vol. 35, no. 8, pp. 1962–1971, Aug. 2016.
'''

import argparse
from os import listdir

from .common_normalize import CommonNormalize
from .stainnorm_vahadane import Normalizer


class VahadaneNormalize(CommonNormalize):
	'''Wrapper class for Vahadane Normalization'''
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
		"""Vahadane stain normalize source w.r.t reference image"""
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
		vahadane_normalize = VahadaneNormalize(
			root_img_dir=params.root_img_dir, 
			ref_img_name=params.ref_img_name,
			src_img_name=params.src_img_name,
			output_dir=params.output_dir
			)
		vahadane_normalize.normalize()
	else:
		src_img_list = [src_img_name for src_img_name in listdir(params.root_img_dir) \
			            if '.tif' in src_img_name and 'mask_' not in src_img_name and \
						src_img_name != params.ref_img_name]
		for src_img_name in src_img_list:
			vahadane_normalize = VahadaneNormalize(
			root_img_dir=params.root_img_dir, 
			ref_img_name=params.ref_img_name,
			src_img_name=src_img_name,
			output_dir=params.output_dir
			)
			vahadane_normalize.normalize()

if __name__ == '__main__':
	main()
