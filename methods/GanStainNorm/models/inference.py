"""Generic inference implementation for different models"""

import json
import tarfile
import warnings
from io import BytesIO
from os import listdir, makedirs
from os.path import isfile, join
from typing import Any, Mapping, Optional, Sequence, Tuple

from numpy import expand_dims, uint8
from PIL import Image
from tensorflow.keras import Model
from tensorflow.keras.models import model_from_json
from tqdm import tqdm

from data_wrangling.utils import normalize, load_jpeg
from models.adv_pix2pix import build_dense_pix2pix
from models.cycle_gan import CycleGan, ReflectionPadding2D, build_cycle_gan
from models.pix2pix import (InstanceNormalization, Pix2Pix, UthGan,
                            build_pix2pix, build_uthgan)


class Inference:
    """Responsible for model inference related operations"""
    def __init__(
            self,
            exp_path: str,
            data_root: Optional[str] = None
        ) -> None:
        self._exp_path = exp_path
        self._data_root = data_root
        self._train_config = get_config(join(exp_path, 'config.json'))
        #self._train_config = get_config(join(exp_path))

        self._input_channel = 3 if self._train_config['exp_type'] == 'CycleGan' else 1

    def run_inference(self, config: Mapping[str, Any]):
        """Initiate model inference"""
        vs_model, layer_info = self._get_vs_model(
            model_type=self._train_config['exp_type'],
            input_img_dim=config['train_img_dim'],
            double_conv=self._train_config.get('double_conv', False),
            use_pix2pix_components=self._train_config.get('use_pix2pix_components', True)
        )
        for epoch in config['epochs']:
            weight_file = f'epoch.{epoch:03d}'
            weight_file_path = join(self._exp_path, 'checkpoint', weight_file)
            vs_model.load_weights(weight_file_path).expect_partial()
            gen_model = self._get_generator_model(vs_model=vs_model)
            if config['train_img_dim'] != config['inference_img_dim']:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    gen_model = self._modify_model_input(
                        model=gen_model,
                        new_input_shape=config['inference_img_dim'],
                        layer_info=layer_info)
            source_samples_path = join(self._data_root, 'source') if self._data_root\
                else join(config['data_root'], 'source')
            for sample_name in config['test']['source']:
                sample_name = sample_name.split('.')[0]
                tiles_path = join(source_samples_path, sample_name)
                tiles = [f for f in listdir(tiles_path) if isfile(join(tiles_path, f))]
                inference_output = join(
                    self._exp_path, 'inference', f'epoch{epoch:03d}')
                makedirs(inference_output, exist_ok=True)
                if config['tar_wrapper']:
                    self._write_tiles_to_tar(
                        model=gen_model,
                        tiles=tiles,
                        tiles_path=tiles_path,
                        inference_output=inference_output,
                        sample_name=sample_name
                    )
                else:
                    self._write_tiles_to_disk(
                        model=gen_model,
                        tiles=tiles,
                        tiles_path=tiles_path,
                        inference_output=inference_output,
                        sample_name=sample_name
                    )

    def _write_tiles_to_tar(
            self,
            model: Model,
            tiles: Sequence[str],
            tiles_path: str,
            inference_output: str,
            sample_name: str):
        with tarfile.open(join(inference_output, f'{sample_name}.tar'), 'w') as tar:
            for tile in tqdm(tiles):
                pil_tile = self._get_predicted_tile(
                tile=tile, tiles_path=tiles_path, model=model)
                stream = BytesIO()
                pil_tile.save(stream, format="jpeg", quality=80)
                tarinfo = tarfile.TarInfo(name=tile)
                # Set number of bytes to write.
                # Previously this only worked when subtracting -1 byte, but
                # produced truncated JPEGs that trouble some software but not all.
                tarinfo.size = stream.getbuffer().nbytes
                # Write bytestream from the beginning to tar file.
                stream.seek(0)
                tar.addfile(tarinfo, fileobj=stream)
                stream.close()

    def _write_tiles_to_disk(
            self,
            model: Model,
            tiles: Sequence[str],
            tiles_path: str,
            inference_output: str,
            sample_name: str):
        sample_output_path = join(inference_output, sample_name)
        makedirs(sample_output_path, exist_ok=True)
        for tile in tqdm(tiles):
            pil_tile = self._get_predicted_tile(
                tile=tile, tiles_path=tiles_path, model=model)
            pil_tile = pil_tile.convert("RGB")
            pil_tile.save(join(sample_output_path, f'{tile}'), format="jpeg", quality=80)

    def _get_predicted_tile(self, tile:str, tiles_path:str, model: Model):
        #tile_img = normalize(
        #    io.imread(join(tiles_path, f'{tile}'), plugin="pil", as_gray=True)
        #)
        tile_img = normalize(load_jpeg(join(tiles_path, f'{tile}'), self._input_channel))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            #prediction = model.predict(expand_dims(tile_img, axis=0))[0]
            prediction = model.predict(expand_dims(tile_img, axis=0))[0]
        prediction = (prediction * 127.5 + 127.5).astype(uint8)
        pil_tile = Image.fromarray(prediction)
        pil_tile = pil_tile.convert("RGB")
        return pil_tile

    @staticmethod
    def _modify_model_input(
            model: Model,
            new_input_shape: Sequence[int],
            layer_info: Mapping[str, Any]) -> Model:
        """Replace the model input shape with new_input_shape"""
        new_input_shape = [None] + new_input_shape
        model.layers[0]._batch_input_shape = tuple(new_input_shape)
        new_model = model_from_json(model.to_json(), layer_info)
        for layer in new_model.layers:
            try:
                layer.set_weights(model.get_layer(name=layer.name).get_weights())
            except ValueError as err:
                print(err)
                print(f'Could not transfer weight for layer: {layer.name}')
        return new_model

    @staticmethod
    def _get_generator_model(vs_model: Model) -> Model:
        """Extract generator model from GAN models"""
        if isinstance(vs_model, CycleGan):
            gen_model = vs_model.gen_g
        elif isinstance(vs_model, Pix2Pix) or isinstance(vs_model, UthGan):
            gen_model = vs_model.generator
        else:
            raise RuntimeError('Model not recognized')
        return gen_model

    @staticmethod
    def _get_vs_model(
            model_type: str,
            input_img_dim: Tuple[int, int, int],
            normalization: str='instancenorm',
            double_conv: bool=False,
            use_pix2pix_components: bool=True
        ) -> Tuple[Model, Mapping[str, Any]]:
        """Get virtual staining model with custom layer info dict"""
        if model_type == 'CycleGan':
            vs_model = build_cycle_gan(
                input_img_dim=input_img_dim,
                batch_size=4,
                normalization=normalization,
                use_pix2pix_components=use_pix2pix_components)
            layer_info = {'ReflectionPadding2D': ReflectionPadding2D}
        elif model_type == 'Pix2Pix':
            vs_model = build_pix2pix(
                input_img_dim=input_img_dim,
                batch_size=4,
                normalization=normalization,
                double_conv=double_conv)
            layer_info = {'InstanceNormalization': InstanceNormalization}
        elif model_type == 'UthGan':
            vs_model = build_uthgan(
                input_img_dim=input_img_dim, batch_size=4, normalization=normalization)
            layer_info = {'InstanceNormalization': InstanceNormalization}
        elif model_type == 'DensePix2Pix':
            vs_model = build_dense_pix2pix(
                input_img_dim=input_img_dim, batch_size=4, normalization=normalization)
            layer_info = {'InstanceNormalization': InstanceNormalization}
        else:
            raise RuntimeError(f'{model_type} model not found')
        return vs_model, layer_info


def get_config(config_path: str) -> Mapping[str, Any]:
    """Return config json"""
    with open(config_path) as json_file:
        config = json.load(json_file)
    return config
