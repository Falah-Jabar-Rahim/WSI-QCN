import os

# Must be set before `torch` is imported below. On some driver
# setups, torch's default CUDA device-count check can hang
# indefinitely on first call; this switches it to a lightweight
# NVML-based check instead. setdefault() so an existing value
# (e.g. set via conda/PyCharm/the entry-point script) isn't
# overridden.
os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "1")

# `tiatoolbox` (imported below, needed only for its CPU-based
# Ruifrok/Vahadane normalizers) and the CycleGan/DensePix2Pix
# generators (via `tf_gan_normalizer`, wired in below) both pull
# in TensorFlow. By default TensorFlow reserves the *entire* GPU
# the moment it runs its first op, which can starve PyTorch of
# memory when both frameworks share a process. Enabling memory
# growth makes TensorFlow allocate incrementally instead, so it
# coexists with PyTorch's own allocator rather than grabbing
# everything up front.
#
# (Earlier this block fully disabled TensorFlow's GPU access as a
# defensive guess while debugging an unrelated CUDA hang. That
# hang turned out to be a driver/persistence-mode issue - a bare
# `torch.zeros(10).cuda()` hung with no TensorFlow involved at all
# - fixed by a reboot + `nvidia-smi -pm 1`, not by this guard. Now
# that CycleGan/DensePix2Pix genuinely need TensorFlow on the GPU,
# full GPU denial is replaced with memory growth instead.)
try:
    import tensorflow as _tensorflow
    for _gpu in _tensorflow.config.list_physical_devices("GPU"):
        _tensorflow.config.experimental.set_memory_growth(_gpu, True)
except ImportError:
    pass

from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torchstain
from skimage.exposure import match_histograms
from torchvision import transforms

from methods.StainNet.models import ResnetGenerator, StainNet

from sastaindiff_normalizer_v2 import (
    normalize_sastaindiff,
    normalize_sastaindiff_batch,
)
from tf_gan_normalizer import (
    normalize_gan_tf,
    normalize_gan_tf_batch,
)
from tiatoolbox.tools import stainnorm

ImageInput = Union[str, Path, np.ndarray]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TO_TENSOR_255 = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 255.0),
    ]
)

# Cache neural-network models so they are loaded only once.
_MODEL_CACHE: dict[tuple, torch.nn.Module] = {}

# Cache *fitted* classical normalizers (Macenko, Reinhard,
# Multi-Macenko, Ruifrok, Vahadane). Fitting estimates stain
# vectors from the reference/target image(s) and is the expensive,
# repeatable part of these methods. The target does not change
# from image to image within a run, so we fit once per unique
# target and reuse the fitted normalizer for every source image.
_FIT_CACHE: dict[tuple, object] = {}


def _target_cache_key(target) -> str:
    """
    Build a stable cache key identifying a reference target.

    File paths are keyed by their resolved string so the same
    target used across many images (or many methods) hits the
    same cache entry. In-memory NumPy arrays are keyed by object
    identity, which only helps when the exact same array object is
    reused across calls within one run.
    """
    if isinstance(target, (str, Path)):
        return str(Path(target).expanduser().resolve())

    return f"array-object:{id(target)}"


def _get_or_fit(cache_key: tuple, build_and_fit):
    """
    Return a cached fitted normalizer, building/fitting it once.
    """
    normalizer = _FIT_CACHE.get(cache_key)

    if normalizer is None:
        normalizer = build_and_fit()
        _FIT_CACHE[cache_key] = normalizer

    return normalizer


def ensure_rgb_uint8(image: np.ndarray) -> np.ndarray:

    """

    Convert an RGB image to contiguous uint8 format expected by TIAToolbox.

    """

    image = np.asarray(image)

    if image.ndim != 3 or image.shape[2] != 3:

        raise ValueError(

            f"Expected RGB image with shape HxWx3, got {image.shape}"

        )

    if image.dtype == np.uint8:

        return np.ascontiguousarray(image)

    image = image.astype(np.float32)

    # Handle floating-point images in [0, 1].

    if image.max() <= 1.0:

        image = image * 255.0

    image = np.clip(image, 0, 255).astype(np.uint8)

    return np.ascontiguousarray(image)
def normalize_ruifrok(
    source_rgb: np.ndarray,
    target_rgb: np.ndarray,
) -> np.ndarray:
    """
    Normalize an RGB source image using the Ruifrok & Johnston method.

    Parameters
    ----------
    source_rgb:
        Source image: a file path or an RGB NumPy array.
    target_rgb:
        Reference image: a file path or an RGB NumPy array.
        Fitting is cached per unique target (see `_FIT_CACHE`).

    Returns
    -------
    np.ndarray
        Normalized RGB uint8 image.
    """

    cache_key = ("ruifrok", _target_cache_key(target_rgb))

    def build():
        target_image = ensure_rgb_uint8(read_rgb_image(target_rgb))
        normalizer = stainnorm.RuifrokNormalizer()
        normalizer.fit(target_image)
        return normalizer

    normalizer = _get_or_fit(cache_key, build)

    source_rgb = ensure_rgb_uint8(read_rgb_image(source_rgb))
    normalized = normalizer.transform(source_rgb.copy())

    return ensure_rgb_uint8(normalized)

def normalize_vahadane(
    source_rgb: np.ndarray,
    target_rgb: np.ndarray,
) -> np.ndarray:
    """
    Normalize an RGB source image using the Vahadane method.

    Parameters
    ----------
    source_rgb:
        Source image: a file path or an RGB NumPy array.
    target_rgb:
        Reference image: a file path or an RGB NumPy array.
        Fitting is cached per unique target (see `_FIT_CACHE`).

    Returns
    -------
    np.ndarray
        Normalized RGB uint8 image.
    """
    cache_key = ("vahadane", _target_cache_key(target_rgb))

    def build():
        target_image = ensure_rgb_uint8(read_rgb_image(target_rgb))
        normalizer = stainnorm.VahadaneNormalizer()
        normalizer.fit(target_image)
        return normalizer

    normalizer = _get_or_fit(cache_key, build)

    source_rgb = ensure_rgb_uint8(read_rgb_image(source_rgb))
    normalized = normalizer.transform(source_rgb.copy())

    return ensure_rgb_uint8(normalized)

def imagenet_normalize(source_rgb):
    """
    ImageNet-style [-1, 1] normalization (as used by CellViT /
    CellViT++), stretched back to a viewable uint8 RGB image.

    This method needs no reference target: it only rescales the
    source image's own pixel statistics, so it operates directly
    on `source_rgb`.
    """
    image = read_rgb_image(source_rgb).astype(np.float32) / 255.0

    mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    std = np.array([0.5, 0.5, 0.5], dtype=np.float32)

    image = (image - mean) / std

    # Stretch back to [0, 1] for a viewable image.
    image -= image.min()
    image /= image.max() + 1e-8

    return (image * 255).astype(np.uint8)


def normalize_histogram_matching(source_rgb, target_rgb):
    """
    Match the source image's per-channel histogram to the target
    image's histogram (skimage.exposure.match_histograms).

    Unlike Ruifrok/Vahadane/Macenko/Reinhard, there's no separate
    "fit" step to cache here - match_histograms computes and
    applies the target's histogram directly from the two images
    each call, which is already cheap, so this runs one image at a
    time with no caching layer needed.
    """
    source_rgb = ensure_rgb_uint8(read_rgb_image(source_rgb))
    target_rgb = ensure_rgb_uint8(read_rgb_image(target_rgb))

    matched = match_histograms(
        source_rgb,
        target_rgb,
        channel_axis=-1,
    )

    return matched.clip(0, 255).astype(np.uint8)

# =========================================================
# General image utilities
# =========================================================

def read_rgb_image(image: ImageInput) -> np.ndarray:
    """
    Read an image path or validate an existing RGB NumPy image.

    NumPy inputs are assumed to already be RGB.

    Returns
    -------
    np.ndarray
        RGB image with shape [H, W, 3] and dtype uint8.
    """
    if isinstance(image, (str, Path)):
        image_path = str(image)

        bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)

        if bgr is None:
            raise FileNotFoundError(
                f"Could not read image: {image_path}"
            )

        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    elif isinstance(image, np.ndarray):
        image = image.copy()

    else:
        raise TypeError(
            "Image must be a file path or a NumPy array, "
            f"but received {type(image)}."
        )

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            "Expected image shape [H, W, 3], "
            f"but received {image.shape}."
        )

    if image.dtype != np.uint8:
        image = image.astype(np.float32)

        if image.size == 0:
            raise ValueError("The input image is empty.")

        if np.nanmax(image) <= 1.0:
            image = image * 255.0

        image = np.nan_to_num(
            image,
            nan=0.0,
            posinf=255.0,
            neginf=0.0,
        )

        image = np.clip(image, 0, 255).astype(np.uint8)

    return np.ascontiguousarray(image)


def tensor_to_rgb_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a torchstain output tensor into an RGB uint8 image.

    Supports:
      [C, H, W]
      [H, W, C]
      [1, C, H, W]
    """
    array = tensor.detach().cpu().numpy()

    if array.ndim == 4:
        if array.shape[0] != 1:
            raise ValueError(
                "Only batch size 1 is supported, "
                f"but received shape {array.shape}."
            )

        array = array[0]

    if array.ndim != 3:
        raise ValueError(
            "Expected a 3D or 4D tensor, "
            f"but received shape {array.shape}."
        )

    if array.shape[0] in (1, 3):
        array = np.moveaxis(array, 0, -1)

    if array.shape[-1] != 3:
        raise ValueError(
            "Expected three output channels, "
            f"but received shape {array.shape}."
        )

    return np.clip(array, 0, 255).astype(np.uint8)


def _to_numpy(value):
    """
    Convert optional tensor output to NumPy.
    """
    if value is None:
        return None

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()

    return np.asarray(value)


def _result(
    normalized: np.ndarray,
    hematoxylin=None,
    eosin=None,
) -> dict:
    """
    Create a consistent result dictionary.
    """
    return {
        "normalized": np.clip(
            normalized,
            0,
            255,
        ).astype(np.uint8),
        "hematoxylin": _to_numpy(hematoxylin),
        "eosin": _to_numpy(eosin),
    }


# =========================================================
# StainGAN and StainNet utilities
# =========================================================

def norm(image_rgb: np.ndarray) -> torch.Tensor:
    """
    Convert RGB uint8 [H,W,3] to BCHW float tensor in [-1,1].
    """
    image_rgb = read_rgb_image(image_rgb)

    tensor = image_rgb.astype(np.float32)
    tensor = np.moveaxis(tensor, -1, 0)
    tensor = tensor / 127.5 - 1.0
    tensor = tensor[np.newaxis, ...]

    return torch.from_numpy(tensor)


def un_norm(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert BCHW/CHW tensor in [-1,1] to RGB uint8.
    """
    if tensor.ndim == 4:
        if tensor.shape[0] != 1:
            raise ValueError(
                "Only batch size 1 is supported, "
                f"but received shape {tuple(tensor.shape)}."
            )

        tensor = tensor[0]

    if tensor.ndim != 3:
        raise ValueError(
            "Expected BCHW or CHW output, "
            f"but received shape {tuple(tensor.shape)}."
        )

    array = tensor.detach().cpu().float().numpy()
    array = np.moveaxis(array, 0, -1)
    array = (array + 1.0) * 127.5

    return np.clip(array, 0, 255).astype(np.uint8)


# =========================================================
# Batch helpers
#
# StainGAN and StainNet are fully convolutional (StainNet is a
# stack of 1x1 convs; StainGAN's ResnetGenerator uses
# InstanceNorm2d). Neither has any layer that mixes statistics
# across the batch dimension, so running N images through them at
# once is numerically identical to running them one at a time -
# it just costs one forward pass instead of N.
#
# All images going into a batch must share the same (H, W); use
# `images_to_batch_tensor` to stack them.
# =========================================================

def images_to_batch_tensor(images: list) -> torch.Tensor:
    """
    Stack same-shape RGB images into a BCHW float tensor in [-1, 1].
    """
    arrays = [read_rgb_image(image) for image in images]

    shapes = {array.shape for array in arrays}

    if len(shapes) != 1:
        raise ValueError(
            "All images in a batch must share the same shape, "
            f"but received: {sorted(shapes)}."
        )

    stacked = np.stack(arrays, axis=0).astype(np.float32)
    stacked = np.moveaxis(stacked, -1, 1)
    stacked = stacked / 127.5 - 1.0

    return torch.from_numpy(stacked)


def batch_tensor_to_images(tensor: torch.Tensor) -> list:
    """
    Convert a BCHW tensor in [-1, 1] into a list of RGB uint8 images.
    """
    array = tensor.detach().cpu().float().numpy()
    array = np.moveaxis(array, 1, -1)
    array = (array + 1.0) * 127.5
    array = np.clip(array, 0, 255).astype(np.uint8)

    return [
        np.ascontiguousarray(array[index])
        for index in range(array.shape[0])
    ]


def _extract_state_dict(checkpoint):
    """
    Extract a state dictionary without removing architecture-level
    prefixes such as 'model.'.
    """
    if not isinstance(checkpoint, dict):
        return checkpoint

    wrapper_keys = (
        "state_dict",
        "model_state_dict",
        "generator_state_dict",
        "generator",
        "netG",
        "net",
    )

    for wrapper_key in wrapper_keys:
        wrapped = checkpoint.get(wrapper_key)

        if isinstance(wrapped, dict):
            checkpoint = wrapped
            break

    cleaned_state = {}

    for key, value in checkpoint.items():
        cleaned_key = key

        # Remove only external training wrappers.
        for prefix in (
            "module.",
            "netG.",
            "generator.",
        ):
            if cleaned_key.startswith(prefix):
                cleaned_key = cleaned_key[len(prefix):]

        # Keep "model." because ResnetGenerator requires it.
        cleaned_state[cleaned_key] = value

    return cleaned_state

def load_staingan(
    checkpoint: Union[str, Path],
    device: torch.device = DEVICE,
) -> torch.nn.Module:
    """
    Load and cache a pretrained StainGAN generator.
    """
    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"StainGAN checkpoint not found: {checkpoint}"
        )

    cache_key = (
        "staingan",
        str(checkpoint),
        str(device),
    )

    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    model = ResnetGenerator(
        input_nc=3,
        output_nc=3,
        ngf=64,
        norm_layer=torch.nn.InstanceNorm2d,
        use_dropout=False,
        n_blocks=9,
    ).to(device)

    state = torch.load(
        checkpoint,
        map_location=device,
        weights_only=True,
    )

    state = _extract_state_dict(state)

    model_keys = set(model.state_dict().keys())
    state_keys = set(state.keys())

    if not state_keys.issubset(model_keys):
        state_with_model_prefix = {
            (
                key
                if key.startswith("model.")
                else f"model.{key}"
            ): value
            for key, value in state.items()
        }

        if set(state_with_model_prefix.keys()).issubset(model_keys):
            state = state_with_model_prefix

    model.load_state_dict(
        state,
        strict=True,
    )

    model.eval()
    model.requires_grad_(False)

    _MODEL_CACHE[cache_key] = model

    return model

def load_stainnet(
    checkpoint: Union[str, Path],
    device: torch.device = DEVICE,
    channels: int = 32,
    n_layer: int = 3,
) -> torch.nn.Module:
    checkpoint = Path(checkpoint).expanduser().resolve()

    if not checkpoint.exists():
        raise FileNotFoundError(
            f"StainNet checkpoint not found: {checkpoint}"
        )

    cache_key = (
        "stainnet",
        str(checkpoint),
        str(device),
        channels,
        n_layer,
    )

    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    try:
        model = StainNet(
            input_nc=3,
            output_nc=3,
            n_layer=n_layer,
            channels=channels,
        )
    except TypeError:
        model = StainNet()

    model = model.to(device)

    state = torch.load(
        checkpoint,
        map_location=device,
        weights_only=True,
    )

    state = _extract_state_dict(state)

    model.load_state_dict(
        state,
        strict=True,
    )

    model.eval()
    model.requires_grad_(False)

    _MODEL_CACHE[cache_key] = model

    return model
def normalize_staingan_torch(
    source: ImageInput,
    checkpoint: Union[str, Path],
    device: torch.device = DEVICE,
) -> dict:
    """
    Normalize an image with a pretrained StainGAN generator.
    """
    source_rgb = read_rgb_image(source)

    model = load_staingan(
        checkpoint=checkpoint,
        device=device,
    )

    input_tensor = norm(source_rgb).to(device)

    with torch.inference_mode():
        output_tensor = model(input_tensor)

    normalized = un_norm(output_tensor)

    return _result(normalized)


def normalize_stainnet_torch(
    source: ImageInput,
    checkpoint: Union[str, Path],
    device: torch.device = DEVICE,
    channels: int = 32,
    n_layer: int = 3,
) -> dict:
    """
    Normalize an image with a pretrained StainNet model.
    """
    source_rgb = read_rgb_image(source)

    model = load_stainnet(
        checkpoint=checkpoint,
        device=device,
        channels=channels,
        n_layer=n_layer,
    )

    input_tensor = norm(source_rgb).to(device)

    with torch.inference_mode():
        output_tensor = model(input_tensor)

    normalized = un_norm(output_tensor)

    return _result(normalized)


def normalize_staingan_torch_batch(
    sources: list,
    checkpoint: Union[str, Path],
    device: Optional[torch.device] = None,
) -> list:
    """
    Normalize a batch of same-shape images with a single StainGAN
    forward pass. See the "Batch helpers" note above for why this
    is safe.
    """
    selected_device = DEVICE if device is None else torch.device(device)

    model = load_staingan(
        checkpoint=checkpoint,
        device=selected_device,
    )

    input_tensor = images_to_batch_tensor(sources).to(selected_device)

    with torch.inference_mode():
        output_tensor = model(input_tensor)

    return batch_tensor_to_images(output_tensor)


def normalize_stainnet_torch_batch(
    sources: list,
    checkpoint: Union[str, Path],
    device: Optional[torch.device] = None,
    channels: int = 32,
    n_layer: int = 3,
) -> list:
    """
    Normalize a batch of same-shape images with a single StainNet
    forward pass. See the "Batch helpers" note above for why this
    is safe.
    """
    selected_device = DEVICE if device is None else torch.device(device)

    model = load_stainnet(
        checkpoint=checkpoint,
        device=selected_device,
        channels=channels,
        n_layer=n_layer,
    )

    input_tensor = images_to_batch_tensor(sources).to(selected_device)

    with torch.inference_mode():
        output_tensor = model(input_tensor)

    return batch_tensor_to_images(output_tensor)


# =========================================================
# Macenko normalization
# =========================================================

def normalize_macenko_numpy(
    source: ImageInput,
    target: ImageInput,
) -> dict:
    """
    Normalize an RGB source image using NumPy Macenko.
    """
    source_rgb = read_rgb_image(source)

    cache_key = ("macenko", "numpy", _target_cache_key(target))

    def build():
        normalizer = torchstain.normalizers.MacenkoNormalizer(
            backend="numpy"
        )
        normalizer.fit(read_rgb_image(target))
        return normalizer

    normalizer = _get_or_fit(cache_key, build)

    normalized, hematoxylin, eosin = normalizer.normalize(
        I=source_rgb,
        stains=True,
    )

    return _result(
        normalized=normalized,
        hematoxylin=hematoxylin,
        eosin=eosin,
    )


def normalize_macenko_torch(
    source: ImageInput,
    target: ImageInput,
) -> dict:
    """
    Normalize an RGB source image using PyTorch Macenko.
    """
    source_rgb = read_rgb_image(source)
    source_tensor = TO_TENSOR_255(source_rgb)

    cache_key = ("macenko", "torch", _target_cache_key(target))

    def build():
        target_tensor = TO_TENSOR_255(read_rgb_image(target))
        normalizer = torchstain.normalizers.MacenkoNormalizer(
            backend="torch"
        )
        normalizer.fit(target_tensor)
        return normalizer

    normalizer = _get_or_fit(cache_key, build)

    normalized, hematoxylin, eosin = normalizer.normalize(
        I=source_tensor,
        stains=True,
    )

    normalized_rgb = tensor_to_rgb_numpy(normalized)

    return _result(
        normalized=normalized_rgb,
        hematoxylin=hematoxylin,
        eosin=eosin,
    )


# =========================================================
# Reinhard normalization
# =========================================================

def normalize_reinhard_numpy(
    source: ImageInput,
    target: ImageInput,
) -> dict:
    """
    Normalize an RGB source image using NumPy Reinhard.
    """
    source_rgb = read_rgb_image(source)

    cache_key = ("reinhard", "numpy", _target_cache_key(target))

    def build():
        normalizer = torchstain.normalizers.ReinhardNormalizer(
            backend="numpy"
        )
        normalizer.fit(read_rgb_image(target))
        return normalizer

    normalizer = _get_or_fit(cache_key, build)

    normalized = normalizer.normalize(
        I=source_rgb
    )

    return _result(normalized)


def normalize_reinhard_torch(
    source: ImageInput,
    target: ImageInput,
) -> dict:
    """
    Normalize an RGB source image using PyTorch Reinhard.
    """
    source_rgb = read_rgb_image(source)
    source_tensor = TO_TENSOR_255(source_rgb)

    cache_key = ("reinhard", "torch", _target_cache_key(target))

    def build():
        target_tensor = TO_TENSOR_255(read_rgb_image(target))
        normalizer = torchstain.normalizers.ReinhardNormalizer(
            backend="torch"
        )
        normalizer.fit(target_tensor)
        return normalizer

    normalizer = _get_or_fit(cache_key, build)

    normalized = normalizer.normalize(
        I=source_tensor
    )

    normalized_rgb = tensor_to_rgb_numpy(normalized)

    return _result(normalized_rgb)


# =========================================================
# Multi-target Macenko normalization
# =========================================================

def normalize_multi_macenko_numpy(
    source: ImageInput,
    targets: list[ImageInput],
    norm_mode: str = "avg-post",
) -> dict:
    """
    Multi-target Macenko normalization using NumPy.
    """
    if not targets:
        raise ValueError(
            "At least one target image must be provided."
        )

    source_rgb = read_rgb_image(source)

    source_chw = np.moveaxis(
        source_rgb,
        -1,
        0,
    )

    cache_key = (
        "multi_macenko",
        "numpy",
        norm_mode,
        tuple(_target_cache_key(target) for target in targets),
    )

    def build():
        target_chw = [
            np.moveaxis(read_rgb_image(target), -1, 0)
            for target in targets
        ]
        normalizer = torchstain.normalizers.MultiMacenkoNormalizer(
            backend="numpy",
            norm_mode=norm_mode,
        )
        normalizer.fit(target_chw)
        return normalizer

    normalizer = _get_or_fit(cache_key, build)

    normalized, hematoxylin, eosin = normalizer.normalize(
        I=source_chw,
        stains=True,
    )

    if (
        normalized.ndim == 3
        and normalized.shape[0] in (1, 3)
    ):
        normalized = np.moveaxis(
            normalized,
            0,
            -1,
        )

    return _result(
        normalized=normalized,
        hematoxylin=hematoxylin,
        eosin=eosin,
    )


def normalize_multi_macenko_torch(
    source: ImageInput,
    targets: list[ImageInput],
    norm_mode: str = "avg-post",
) -> dict:
    """
    Multi-target Macenko normalization using PyTorch.
    """
    if not targets:
        raise ValueError(
            "At least one target image must be provided."
        )

    source_rgb = read_rgb_image(source)
    source_tensor = TO_TENSOR_255(source_rgb)

    cache_key = (
        "multi_macenko",
        "torch",
        norm_mode,
        tuple(_target_cache_key(target) for target in targets),
    )

    def build():
        target_tensors = [
            TO_TENSOR_255(read_rgb_image(target))
            for target in targets
        ]
        normalizer = torchstain.normalizers.MultiMacenkoNormalizer(
            backend="torch",
            norm_mode=norm_mode,
        )
        normalizer.fit(target_tensors)
        return normalizer

    normalizer = _get_or_fit(cache_key, build)

    normalized, hematoxylin, eosin = normalizer.normalize(
        I=source_tensor,
        stains=True,
    )

    normalized_rgb = tensor_to_rgb_numpy(normalized)

    return _result(
        normalized=normalized_rgb,
        hematoxylin=hematoxylin,
        eosin=eosin,
    )


# =========================================================
# Batch capability registry
#
# Classical / statistical methods (Macenko, Reinhard,
# Multi-Macenko, Ruifrok, Vahadane, mean/std) estimate stain
# statistics from each *source* image individually, so they always
# run one image at a time (their repeatable cost - fitting the
# target - is cached above instead).
#
# StainGAN, StainNet, and SAStainDiff are neural networks with no
# layer that mixes statistics across the batch dimension
# (InstanceNorm/GroupNorm/plain convs), so they can process several
# same-shape images in a single forward pass.
# =========================================================

METHOD_BATCH_SUPPORT: dict[str, bool] = {
    "ruifrok": False,
    "vahadane": False,
    "mean_std": False,
    "histogram_matching": False,
    "macenko": False,
    "reinhard": False,
    "multi_macenko": False,
    "staingan": True,
    "stainnet": True,
    "sastaindiff": True,
    "cyclegan": True,
    "densepix2pix": True,
}

# Methods that use torchstain (methods above) tell backend='torch',
# but the two TensorFlow/Keras GAN methods don't have a matching
# torchstain backend name at all - they're driven purely by
# 'exp_path'/'epoch', not the torch-style 'checkpoint'/'device'
# params used by everything else in this registry.
_TENSORFLOW_GAN_METHODS = {"cyclegan", "densepix2pix"}


def normalize_stain_batch(
    sources: list,
    method: str = "stainnet",
    backend: str = "torch",
    checkpoint: Optional[Union[str, Path]] = None,
    device: Optional[Union[str, torch.device]] = None,
    stainnet_channels: int = 32,
    stainnet_n_layer: int = 3,
    sastaindiff_image_size=256,
    sastaindiff_attention_resolutions="32,16,8",
    sastaindiff_num_channels=128,
    sastaindiff_num_res_blocks=3,
    sastaindiff_diffusion_steps=1000,
    sastaindiff_noise_schedule="linear",
    sastaindiff_use_ddim=True,
    sastaindiff_timestep_respacing="ddim50",
    sastaindiff_use_fp16=True,
    sastaindiff_use_anysize=False,
    sastaindiff_seed=None,
    gan_exp_path: Optional[Union[str, Path]] = None,
    gan_epoch: Optional[int] = None,
) -> list:
    """
    Batched counterpart to `normalize_stain`.

    Only methods marked `True` in METHOD_BATCH_SUPPORT are
    accepted. Every image in `sources` must share the same
    (H, W) shape - group images by shape before calling this
    (see `batch_runner.stream_shape_batches`).

    Returns
    -------
    list[dict]
        One result dict per input image, in the same order,
        each shaped like the output of `normalize_stain`.
    """
    method = method.lower().strip()
    backend = backend.lower().strip()

    if not METHOD_BATCH_SUPPORT.get(method, False):
        raise ValueError(
            f"Method '{method}' does not support batch processing. "
            "Use normalize_stain() and loop over images instead."
        )

    if method in _TENSORFLOW_GAN_METHODS:
        if gan_exp_path is None or gan_epoch is None:
            raise ValueError(
                f"'gan_exp_path' and 'gan_epoch' are required for "
                f"'{method}'."
            )

        images = normalize_gan_tf_batch(
            sources=sources,
            exp_path=gan_exp_path,
            epoch=gan_epoch,
        )

        return [_result(image) for image in images]

    if backend != "torch":
        raise ValueError(
            "Batch processing only supports backend='torch', "
            f"got '{backend}'."
        )

    if checkpoint is None:
        raise ValueError(
            f"'checkpoint' is required for '{method}'."
        )

    selected_device = (
        DEVICE
        if device is None
        else torch.device(device)
    )

    if method == "staingan":
        images = normalize_staingan_torch_batch(
            sources=sources,
            checkpoint=checkpoint,
            device=selected_device,
        )

    elif method == "stainnet":
        images = normalize_stainnet_torch_batch(
            sources=sources,
            checkpoint=checkpoint,
            device=selected_device,
            channels=stainnet_channels,
            n_layer=stainnet_n_layer,
        )

    elif method == "sastaindiff":
        images = normalize_sastaindiff_batch(
            images_rgb=sources,
            checkpoint=checkpoint,
            device=device,
            image_size=sastaindiff_image_size,
            attention_resolutions=(
                sastaindiff_attention_resolutions
            ),
            num_channels=sastaindiff_num_channels,
            num_res_blocks=sastaindiff_num_res_blocks,
            diffusion_steps=sastaindiff_diffusion_steps,
            noise_schedule=sastaindiff_noise_schedule,
            use_ddim=sastaindiff_use_ddim,
            timestep_respacing=(
                sastaindiff_timestep_respacing
            ),
            use_fp16=sastaindiff_use_fp16,
            use_anysize=sastaindiff_use_anysize,
            seed=sastaindiff_seed,
        )

    else:
        raise RuntimeError(
            "Unexpected batchable method configuration."
        )

    return [_result(image) for image in images]


# =========================================================
# Unified normalization interface
# =========================================================

def normalize_stain(
    source: ImageInput,
    target: Optional[ImageInput] = None,
    targets: Optional[list[ImageInput]] = None,
    method: str = "macenko",
    backend: str = "torch",
    norm_mode: str = "avg-post",
    checkpoint: Optional[Union[str, Path]] = None,
    device: Optional[Union[str, torch.device]] = None,
    stainnet_channels: int = 32,
    stainnet_n_layer: int = 3,
    sastaindiff_image_size=256,
    sastaindiff_attention_resolutions="32,16,8",
    sastaindiff_num_channels=128,
    sastaindiff_num_res_blocks=3,
    sastaindiff_diffusion_steps=1000,
    sastaindiff_noise_schedule="linear",
    sastaindiff_use_ddim=True,
    sastaindiff_timestep_respacing="ddim50",
    sastaindiff_use_fp16=True,
    sastaindiff_use_anysize=False,
    sastaindiff_seed=None,
    gan_exp_path: Optional[Union[str, Path]] = None,
    gan_epoch: Optional[int] = None,
) -> dict:
    """
    General stain-normalization interface.

    Supported methods
    -----------------
    - macenko
    - reinhard
    - multi_macenko
    - staingan
    - stainnet

    Supported backends
    ------------------
    - torch
    - numpy

    Notes
    -----
    StainGAN and StainNet support only the torch backend.

    Returns
    -------
    dict
        {
            "normalized": RGB uint8 image,
            "hematoxylin": optional stain output,
            "eosin": optional stain output,
        }
    """
    method = method.lower().strip()
    backend = backend.lower().strip()

    valid_methods = {
        "macenko",
        "reinhard",
        "multi_macenko",
        "staingan",
        "stainnet",
        "sastaindiff",
        "mean_std",
        "ruifrok",
        "vahadane",
        "histogram_matching",
        "cyclegan",
        "densepix2pix",
    }

    valid_backends = {
        "torch",
        "numpy",
    }

    if method in {"ruifrok", "vahadane"}:

        if target is None:
            raise ValueError(
                f"'target' is required for {method} normalization."
            )

        # Pass the *original* target (path or array) through so the
        # fit-cache in normalize_ruifrok/normalize_vahadane can key
        # off a stable identity instead of a freshly re-read array.
        if method == "ruifrok":
            normalized = normalize_ruifrok(
                source_rgb=source,
                target_rgb=target,
            )
        else:
            normalized = normalize_vahadane(
                source_rgb=source,
                target_rgb=target,
            )

        return {
            "normalized": normalized,
        }

    if method == "mean_std":
        normalized = imagenet_normalize(source)

        return {
            "normalized": normalized,
        }

    if method == "histogram_matching":
        if target is None:
            raise ValueError(
                "'target' is required for histogram_matching "
                "normalization."
            )

        normalized = normalize_histogram_matching(
            source_rgb=source,
            target_rgb=target,
        )

        return {
            "normalized": normalized,
        }

    if method in {"cyclegan", "densepix2pix"}:
        if gan_exp_path is None or gan_epoch is None:
            raise ValueError(
                f"'gan_exp_path' and 'gan_epoch' are required for "
                f"'{method}' normalization."
            )

        result = normalize_gan_tf(
            source=source,
            exp_path=gan_exp_path,
            epoch=gan_epoch,
        )

        return result


    if method not in valid_methods:
        raise ValueError(
            f"Unsupported method '{method}'. "
            f"Available methods: {sorted(valid_methods)}"
        )

    if backend not in valid_backends:
        raise ValueError(
            f"Unsupported backend '{backend}'. "
            f"Available backends: {sorted(valid_backends)}"
        )

    selected_device = (
        DEVICE
        if device is None
        else torch.device(device)
    )

    if method == "sastaindiff":
        if checkpoint is None:
            raise ValueError(
                "'checkpoint' is required for "
                "SAStainDiff normalization."
            )

        if backend != "torch":
            raise ValueError(
                "SAStainDiff only supports backend='torch'."
            )

        normalized = normalize_sastaindiff(
            image_rgb=source,
            checkpoint=checkpoint,
            device=device,
            image_size=sastaindiff_image_size,
            attention_resolutions=(
                sastaindiff_attention_resolutions
            ),
            num_channels=sastaindiff_num_channels,
            num_res_blocks=sastaindiff_num_res_blocks,
            diffusion_steps=sastaindiff_diffusion_steps,
            noise_schedule=sastaindiff_noise_schedule,
            use_ddim=sastaindiff_use_ddim,
            timestep_respacing=(
                sastaindiff_timestep_respacing
            ),
            use_fp16=sastaindiff_use_fp16,
            use_anysize=sastaindiff_use_anysize,
            seed=sastaindiff_seed,
        )

        return {
            "normalized": normalized,
            "method": "sastaindiff",
            "backend": "torch",
        }

    if method == "macenko":
        if target is None:
            raise ValueError(
                "'target' is required for Macenko normalization."
            )

        if backend == "torch":
            return normalize_macenko_torch(
                source=source,
                target=target,
            )

        return normalize_macenko_numpy(
            source=source,
            target=target,
        )

    if method == "reinhard":
        if target is None:
            raise ValueError(
                "'target' is required for Reinhard normalization."
            )

        if backend == "torch":
            return normalize_reinhard_torch(
                source=source,
                target=target,
            )

        return normalize_reinhard_numpy(
            source=source,
            target=target,
        )

    if method == "multi_macenko":
        if not targets:
            raise ValueError(
                "'targets' is required for multi-target "
                "Macenko normalization."
            )

        if backend == "torch":
            return normalize_multi_macenko_torch(
                source=source,
                targets=targets,
                norm_mode=norm_mode,
            )

        return normalize_multi_macenko_numpy(
            source=source,
            targets=targets,
            norm_mode=norm_mode,
        )

    if method == "staingan":
        if backend != "torch":
            raise ValueError(
                "StainGAN supports only backend='torch'."
            )

        if checkpoint is None:
            raise ValueError(
                "'checkpoint' is required for StainGAN."
            )

        return normalize_staingan_torch(
            source=source,
            checkpoint=checkpoint,
            device=selected_device,
        )

    if method == "stainnet":
        if backend != "torch":
            raise ValueError(
                "StainNet supports only backend='torch'."
            )

        if checkpoint is None:
            raise ValueError(
                "'checkpoint' is required for StainNet."
            )

        return normalize_stainnet_torch(
            source=source,
            checkpoint=checkpoint,
            device=selected_device,
            channels=stainnet_channels,
            n_layer=stainnet_n_layer,
        )

    raise RuntimeError(
        "Unexpected normalization configuration."
    )