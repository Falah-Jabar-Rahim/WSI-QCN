"""
CycleGAN / DensePix2Pix stain normalization.

Wraps the pretrained TensorFlow/Keras generators from the
`GanStainNorm` package (vendored from the "multi-lab-stain-
normalization" GAN pipeline: cycle_gan.py, adv_pix2pix.py,
pix2pix.py, custom_layers.py) for single-image and batched
inference, mirroring how the original models/inference.py runs a
trained experiment - just operating on in-memory RGB arrays
instead of writing tiles to disk/tar.

Both generators use InstanceNormalization (no cross-image batch
statistics), so batching several images through `model.predict()`
at once is numerically identical to running them one at a time.

Expected experiment layout (produced by the original training
pipeline's `execute.py train`):

    <exp_path>/
        config.json                     # training config
        checkpoint/
            epoch.040.index
            epoch.040.data-00000-of-00001
            ...

`config.json` must contain `exp_type` ("CycleGan" or
"DensePix2Pix") and, optionally, `normalization` and
`use_pix2pix_components` (only meaningful for CycleGan) - these
are read automatically so the generator is rebuilt exactly as it
was trained.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import tensorflow as tf

from methods.GanStainNorm.models.adv_pix2pix import build_dense_pix2pix
from methods.GanStainNorm.models.cycle_gan import build_cycle_gan

ImageInput = Union[str, Path, np.ndarray]

# Cache built + weight-loaded generators. CycleGan's ResNet
# generator has a fixed spatial input shape baked into its Input
# layer, so its cache key includes (H, W); DensePix2Pix's generator
# accepts any (H, W) (Input(shape=[None, None, channels])), so that
# part of the key is always None for it.
_GENERATOR_CACHE: Dict[tuple, tf.keras.Model] = {}

SUPPORTED_EXP_TYPES = {"CycleGan", "DensePix2Pix"}


def read_rgb_image(image: ImageInput) -> np.ndarray:
    """Load (or pass through) an RGB uint8 image."""
    if isinstance(image, (str, Path)):
        bgr = cv2.imread(str(image))
        if bgr is None:
            raise ValueError(f"Could not read image: {image}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    image = np.asarray(image)
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    return image.astype(np.uint8)


def _load_experiment_config(exp_path: Path) -> dict:
    config_path = exp_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Expected a training config at {config_path} - "
            "exp_path should point at an experiment folder "
            "produced by `execute.py train` (containing "
            "config.json and a checkpoint/ folder)."
        )
    with open(config_path) as config_file:
        return json.load(config_file)


def _load_generator(
    exp_path: Union[str, Path],
    epoch: int,
    image_hw: Optional[Tuple[int, int]] = None,
) -> Tuple[tf.keras.Model, str]:
    """
    Build (or fetch from cache) the trained generator for one
    experiment/epoch. Returns (generator, exp_type).
    """
    exp_path = Path(exp_path).expanduser().resolve()
    train_config = _load_experiment_config(exp_path)
    exp_type = train_config["exp_type"]

    if exp_type not in SUPPORTED_EXP_TYPES:
        raise ValueError(
            f"Unsupported exp_type '{exp_type}' in {exp_path}/"
            "config.json - only 'CycleGan' and 'DensePix2Pix' are "
            "wired into this pipeline."
        )

    normalization = train_config.get("normalization", "instancenorm")
    use_pix2pix_components = train_config.get(
        "use_pix2pix_components", False
    )

    # DensePix2Pix's generator has a fully dynamic (H, W) input, so
    # its cache entry doesn't depend on image size.
    cache_shape = image_hw if exp_type == "CycleGan" else None
    cache_key = (str(exp_path), epoch, exp_type, cache_shape)

    generator = _GENERATOR_CACHE.get(cache_key)
    if generator is not None:
        return generator, exp_type

    if exp_type == "CycleGan":
        if image_hw is None:
            raise ValueError(
                "image_hw is required to build the CycleGan "
                "generator (its input shape is fixed at build "
                "time)."
            )
        vs_model = build_cycle_gan(
            input_img_dim=(image_hw[0], image_hw[1], 3),
            batch_size=4,
            normalization=normalization,
            use_pix2pix_components=use_pix2pix_components,
        )
        generator = vs_model.gen_g

    else:  # DensePix2Pix
        vs_model = build_dense_pix2pix(
            input_img_dim=(None, None, 3),
            batch_size=4,
            normalization=normalization,
        )
        generator = vs_model.generator

    weight_path = exp_path / "checkpoint" / f"epoch.{epoch:03d}"
    vs_model.load_weights(str(weight_path)).expect_partial()
    generator.trainable = False

    _GENERATOR_CACHE[cache_key] = generator
    return generator, exp_type


def _preprocess_batch(
    images: List[np.ndarray],
    exp_type: str,
) -> np.ndarray:
    """
    Stack RGB uint8 images into a [-1, 1] float32 NHWC batch.

    CycleGan takes 3-channel RGB. DensePix2Pix takes 1-channel
    grayscale, matching how the original pipeline trained it
    (tf.io.decode_jpeg(..., channels=1)); tf.image.rgb_to_grayscale
    is the equivalent operation for already-decoded RGB arrays.
    """
    shapes = {image.shape[:2] for image in images}
    if len(shapes) != 1:
        raise ValueError(
            "All images in a batch must share the same (H, W), "
            f"got {sorted(shapes)}."
        )

    batch = np.stack(images, axis=0).astype(np.float32)

    if exp_type == "DensePix2Pix":
        batch = tf.image.rgb_to_grayscale(batch).numpy()

    return (batch / 127.5) - 1.0


def _postprocess_batch(predictions: np.ndarray) -> List[np.ndarray]:
    """
    Convert model output [-1, 1] NHWC back into a list of RGB
    uint8 images.
    """
    predictions = (predictions * 127.5 + 127.5)
    predictions = predictions.clip(0, 255).astype(np.uint8)

    return [
        np.ascontiguousarray(predictions[index])
        for index in range(predictions.shape[0])
    ]


def normalize_gan_tf(
    source: ImageInput,
    exp_path: Union[str, Path],
    epoch: int,
) -> dict:
    """
    Normalize a single image with a trained CycleGan/DensePix2Pix
    generator.
    """
    source_rgb = read_rgb_image(source)

    generator, exp_type = _load_generator(
        exp_path=exp_path,
        epoch=epoch,
        image_hw=source_rgb.shape[:2],
    )

    batch = _preprocess_batch([source_rgb], exp_type)
    prediction = generator.predict(batch, verbose=0)
    normalized = _postprocess_batch(prediction)[0]

    return {"normalized": normalized}


def normalize_gan_tf_batch(
    sources: List[ImageInput],
    exp_path: Union[str, Path],
    epoch: int,
) -> List[np.ndarray]:
    """
    Normalize a batch of same-shape images in a single
    `model.predict()` call.
    """
    images = [read_rgb_image(image) for image in sources]

    generator, exp_type = _load_generator(
        exp_path=exp_path,
        epoch=epoch,
        image_hw=images[0].shape[:2],
    )

    batch = _preprocess_batch(images, exp_type)
    predictions = generator.predict(batch, verbose=0)

    return _postprocess_batch(predictions)
