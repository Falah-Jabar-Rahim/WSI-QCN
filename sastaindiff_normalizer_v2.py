from __future__ import annotations

import os

# Must be set before `torch` is imported below - see
# normalize_stain.py for the full explanation. Repeated here so
# this module is also safe to import on its own.
os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "1")

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

from methods.SAStainDiff.guided_diffusion_.guided_diffusion.script_util import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)

ImageInput = Union[str, Path, np.ndarray]

# Cache loaded models so repeated calls do not reload the checkpoint.
_MODEL_CACHE: Dict[
    Tuple,
    "SAStainDiffNormalizer",
] = {}


class SAStainDiffNormalizer:
    """
    Reusable SAStainDiff stain-normalization inference wrapper.

    Input:
        RGB uint8 NumPy image with shape [H, W, 3].

    Output:
        RGB uint8 NumPy image with shape [H, W, 3].
    """

    def __init__(
        self,
        checkpoint: str | Path,
        device: Optional[str | torch.device] = None,
        image_size: int = 256,
        attention_resolutions: str = "32,16,8",
        num_channels: int = 128,
        num_res_blocks: int = 3,
        use_scale_shift_norm: bool = True,
        diffusion_steps: int = 1000,
        noise_schedule: str = "linear",
        use_ddim: bool = True,
        timestep_respacing: str = "ddim50",
        use_fp16: bool = True,
        class_cond: bool = False,
        clip_denoised: bool = True,
        use_anysize: bool = False,
    ):
        self.checkpoint = Path(checkpoint)

        if not self.checkpoint.exists():
            raise FileNotFoundError(
                f"SAStainDiff checkpoint does not exist: "
                f"{self.checkpoint}"
            )

        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        if self.device.type == "cpu":
            # CPU inference generally should remain FP32.
            use_fp16 = False

        self.image_size = image_size
        self.use_ddim = use_ddim
        self.use_fp16 = use_fp16
        self.class_cond = class_cond
        self.clip_denoised = clip_denoised
        self.use_anysize = use_anysize

        model_config = model_and_diffusion_defaults()

        model_config.update(
            {
                "attention_resolutions": attention_resolutions,
                "image_size": image_size,
                "num_channels": num_channels,
                "num_res_blocks": num_res_blocks,
                "use_fp16": use_fp16,
                "use_scale_shift_norm": use_scale_shift_norm,
                "diffusion_steps": diffusion_steps,
                "noise_schedule": noise_schedule,
                "timestep_respacing": timestep_respacing,
                "class_cond": class_cond,
            }
        )

        self.model, self.diffusion = create_model_and_diffusion(
            **model_config
        )

        # Plain single-process checkpoint load. The original
        # guided-diffusion codebase used `dist_util.load_state_dict`,
        # which requires `mpi4py`/`torch.distributed` and can hang
        # or fail if no MPI runtime is set up - unnecessary for
        # single-process inference like this pipeline.
        state_dict = torch.load(
            str(self.checkpoint),
            map_location="cpu",
        )

        self.model.load_state_dict(
            state_dict,
            strict=True,
        )

        self.model.to(self.device)

        if self.use_fp16:
            self.model.convert_to_fp16()

        self.model.eval()

        if self.use_anysize:
            self.sample_function = (
                self.diffusion.ddim_sample_loop_any_size
                if self.use_ddim
                else self.diffusion.p_sample_loop_any_size
            )
        else:
            self.sample_function = (
                self.diffusion.ddim_sample_loop
                if self.use_ddim
                else self.diffusion.p_sample_loop
            )

    @staticmethod
    def _validate_rgb_image(
        image_rgb: np.ndarray,
    ) -> np.ndarray:
        image_rgb = np.asarray(image_rgb)

        if image_rgb.ndim != 3:
            raise ValueError(
                "SAStainDiff expects an image with shape "
                "[height, width, channels]."
            )

        if image_rgb.shape[2] != 3:
            raise ValueError(
                "SAStainDiff expects exactly three RGB channels. "
                f"Received shape: {image_rgb.shape}"
            )

        if image_rgb.dtype != np.uint8:
            if np.issubdtype(image_rgb.dtype, np.floating):
                maximum = float(image_rgb.max())

                if maximum <= 1.0:
                    image_rgb = image_rgb * 255.0

            image_rgb = np.clip(
                image_rgb,
                0,
                255,
            ).astype(np.uint8)

        return np.ascontiguousarray(image_rgb)

    @staticmethod
    def _preprocess(
        image_rgb: np.ndarray,
    ) -> torch.Tensor:
        """
        Convert RGB uint8 [H, W, 3] into normalized tensor
        [1, 3, H, W] in the range [-1, 1].
        """
        image_pil = Image.fromarray(
            image_rgb,
            mode="RGB",
        )

        tensor = pil_to_tensor(image_pil).float()
        tensor = tensor / 127.5 - 1.0

        return tensor.unsqueeze(0)

    @staticmethod
    def _postprocess(
        tensor: torch.Tensor,
    ) -> np.ndarray:
        """
        Convert model output [B, 3, H, W] in [-1, 1]
        into RGB uint8 [H, W, 3].
        """
        tensor = (
            (tensor + 1.0) * 127.5
        ).clamp(
            0,
            255,
        )

        tensor = tensor.to(torch.uint8)
        tensor = tensor.permute(0, 2, 3, 1)
        tensor = tensor.contiguous().cpu().numpy()

        return tensor[0]

    def _model_function(
        self,
        x: torch.Tensor,
        HE: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        ref_img: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del ref_img

        class_labels = y if self.class_cond else None

        return self.model(
            x,
            HE,
            t,
            class_labels,
        )

    @torch.inference_mode()
    def normalize(
        self,
        image_rgb: np.ndarray,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Normalize one RGB image.

        Parameters
        ----------
        image_rgb:
            RGB uint8 image.

        seed:
            Optional random seed. Set this for reproducible output.
        """
        image_rgb = self._validate_rgb_image(
            image_rgb
        )

        original_height, original_width = (
            image_rgb.shape[:2]
        )

        if not self.use_anysize:
            expected_shape = (
                self.image_size,
                self.image_size,
            )

            if (
                original_height,
                original_width,
            ) != expected_shape:
                raise ValueError(
                    "SAStainDiff standard sampling expects "
                    f"{self.image_size}×{self.image_size} images, "
                    f"but received "
                    f"{original_width}×{original_height}. "
                    "Set sastaindiff_use_anysize=True or resize/"
                    "tile the image before normalization."
                )

        if seed is not None:
            torch.manual_seed(seed)

            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        HE_input = self._preprocess(
            image_rgb
        ).to(
            self.device,
            non_blocking=True,
        )

        batch_size = HE_input.shape[0]

        final_timestep = torch.full(
            size=(batch_size,),
            fill_value=len(self.diffusion.betas) - 1,
            dtype=torch.long,
            device=self.device,
        )

        # Forward-diffuse the input image to the final timestep.
        initial_noise = self.diffusion.q_sample(
            HE_input,
            final_timestep,
        )

        sample = self.sample_function(
            self._model_function,
            HE_input.shape,
            HE=HE_input,
            clip_denoised=self.clip_denoised,
            device=self.device,
            progress=False,
            noise=initial_noise,
        )

        normalized = self._postprocess(
            sample
        )

        return normalized

    @staticmethod
    def _postprocess_batch(
        tensor: torch.Tensor,
    ) -> list:
        """
        Convert model output [B, 3, H, W] in [-1, 1]
        into a list of RGB uint8 [H, W, 3] images.
        """
        tensor = (
            (tensor + 1.0) * 127.5
        ).clamp(0, 255)

        tensor = tensor.to(torch.uint8)
        tensor = tensor.permute(0, 2, 3, 1)
        tensor = tensor.contiguous().cpu().numpy()

        return [
            np.ascontiguousarray(tensor[index])
            for index in range(tensor.shape[0])
        ]

    @torch.inference_mode()
    def normalize_batch(
        self,
        images_rgb: list,
        seed: Optional[int] = None,
    ) -> list:
        """
        Normalize a batch of RGB images that all share the same
        height and width in a single diffusion sampling pass.

        Numerically equivalent to calling `normalize()` once per
        image (the underlying UNet uses GroupNorm, which has no
        cross-sample statistics) - this just runs fewer, larger
        sampling passes.

        Parameters
        ----------
        images_rgb:
            List of RGB uint8 images, all with the same [H, W].

        seed:
            Optional random seed applied once for the whole batch.
        """
        validated = [
            self._validate_rgb_image(image)
            for image in images_rgb
        ]

        shapes = {image.shape[:2] for image in validated}

        if len(shapes) != 1:
            raise ValueError(
                "All images in a SAStainDiff batch must share the "
                f"same height/width, but received: {sorted(shapes)}."
            )

        original_height, original_width = next(iter(shapes))

        if not self.use_anysize:
            expected_shape = (
                self.image_size,
                self.image_size,
            )

            if (
                original_height,
                original_width,
            ) != expected_shape:
                raise ValueError(
                    "SAStainDiff standard sampling expects "
                    f"{self.image_size}×{self.image_size} images, "
                    f"but received "
                    f"{original_width}×{original_height}. "
                    "Set sastaindiff_use_anysize=True or resize/"
                    "tile the images before normalization."
                )

        if seed is not None:
            torch.manual_seed(seed)

            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        HE_input = torch.cat(
            [
                self._preprocess(image)
                for image in validated
            ],
            dim=0,
        ).to(
            self.device,
            non_blocking=True,
        )

        batch_size = HE_input.shape[0]

        final_timestep = torch.full(
            size=(batch_size,),
            fill_value=len(self.diffusion.betas) - 1,
            dtype=torch.long,
            device=self.device,
        )

        # Forward-diffuse the input images to the final timestep.
        initial_noise = self.diffusion.q_sample(
            HE_input,
            final_timestep,
        )

        sample = self.sample_function(
            self._model_function,
            HE_input.shape,
            HE=HE_input,
            clip_denoised=self.clip_denoised,
            device=self.device,
            progress=False,
            noise=initial_noise,
        )

        return self._postprocess_batch(sample)

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

def normalize_sastaindiff(
    image_rgb: np.ndarray,
    checkpoint: str | Path,
    device: Optional[str | torch.device] = None,
    image_size: int = 256,
    attention_resolutions: str = "32,16,8",
    num_channels: int = 128,
    num_res_blocks: int = 3,
    use_scale_shift_norm: bool = True,
    diffusion_steps: int = 1000,
    noise_schedule: str = "linear",
    use_ddim: bool = True,
    timestep_respacing: str = "ddim50",
    use_fp16: bool = True,
    class_cond: bool = False,
    clip_denoised: bool = True,
    use_anysize: bool = False,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Normalize an RGB image using a cached SAStainDiff model.
    """

    source_rgb = read_rgb_image(image_rgb)



    checkpoint = str(
        Path(checkpoint).expanduser().resolve()
    )

    resolved_device = (
        str(device)
        if device is not None
        else (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    )

    cache_key = (
        checkpoint,
        resolved_device,
        image_size,
        attention_resolutions,
        num_channels,
        num_res_blocks,
        use_scale_shift_norm,
        diffusion_steps,
        noise_schedule,
        use_ddim,
        timestep_respacing,
        use_fp16,
        class_cond,
        clip_denoised,
        use_anysize,
    )

    normalizer = _MODEL_CACHE.get(
        cache_key
    )
    if normalizer is None:
        normalizer = SAStainDiffNormalizer(
            checkpoint=checkpoint,
            device=resolved_device,
            image_size=image_size,
            attention_resolutions=attention_resolutions,
            num_channels=num_channels,
            num_res_blocks=num_res_blocks,
            use_scale_shift_norm=use_scale_shift_norm,
            diffusion_steps=diffusion_steps,
            noise_schedule=noise_schedule,
            use_ddim=use_ddim,
            timestep_respacing=timestep_respacing,
            use_fp16=use_fp16,
            class_cond=class_cond,
            clip_denoised=clip_denoised,
            use_anysize=use_anysize,
        )

        _MODEL_CACHE[cache_key] = normalizer

    return normalizer.normalize(
        image_rgb=source_rgb,
        seed=seed,
    )


def normalize_sastaindiff_batch(
    images_rgb: list,
    checkpoint: str | Path,
    device: Optional[str | torch.device] = None,
    image_size: int = 256,
    attention_resolutions: str = "32,16,8",
    num_channels: int = 128,
    num_res_blocks: int = 3,
    use_scale_shift_norm: bool = True,
    diffusion_steps: int = 1000,
    noise_schedule: str = "linear",
    use_ddim: bool = True,
    timestep_respacing: str = "ddim50",
    use_fp16: bool = True,
    class_cond: bool = False,
    clip_denoised: bool = True,
    use_anysize: bool = False,
    seed: Optional[int] = None,
) -> list:
    """
    Normalize a batch of same-shape RGB images using a cached
    SAStainDiff model in a single diffusion sampling pass.

    Mirrors `normalize_sastaindiff`, sharing the same model cache,
    so a run that mixes single-image and batched calls with the
    same configuration still only loads the checkpoint once.
    """
    source_images = [
        read_rgb_image(image)
        for image in images_rgb
    ]

    checkpoint = str(
        Path(checkpoint).expanduser().resolve()
    )

    resolved_device = (
        str(device)
        if device is not None
        else (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    )

    cache_key = (
        checkpoint,
        resolved_device,
        image_size,
        attention_resolutions,
        num_channels,
        num_res_blocks,
        use_scale_shift_norm,
        diffusion_steps,
        noise_schedule,
        use_ddim,
        timestep_respacing,
        use_fp16,
        class_cond,
        clip_denoised,
        use_anysize,
    )

    normalizer = _MODEL_CACHE.get(
        cache_key
    )
    if normalizer is None:
        normalizer = SAStainDiffNormalizer(
            checkpoint=checkpoint,
            device=resolved_device,
            image_size=image_size,
            attention_resolutions=attention_resolutions,
            num_channels=num_channels,
            num_res_blocks=num_res_blocks,
            use_scale_shift_norm=use_scale_shift_norm,
            diffusion_steps=diffusion_steps,
            noise_schedule=noise_schedule,
            use_ddim=use_ddim,
            timestep_respacing=timestep_respacing,
            use_fp16=use_fp16,
            class_cond=class_cond,
            clip_denoised=clip_denoised,
            use_anysize=use_anysize,
        )

        _MODEL_CACHE[cache_key] = normalizer

    return normalizer.normalize_batch(
        images_rgb=source_images,
        seed=seed,
    )