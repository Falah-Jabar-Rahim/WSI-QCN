"""
Batch stain-normalization runner.

Reads every image in `source_directory` and runs it through each
configured normalization method, saving results under:

    results/
        <method_name>/
            <original_image_name>

Two execution modes are used automatically, based on
`normalize_stain.METHOD_BATCH_SUPPORT`:

- "batch" methods (StainGAN, StainNet, SAStainDiff, CycleGan,
  DensePix2Pix) are fully convolutional networks with no
  cross-image statistics, so same-shape images are grouped and
  pushed through the model together in chunks of `batch_size` -
  one forward pass instead of one-per-image.
- "sequential" methods (Macenko, Reinhard, Multi-Macenko,
  Ruifrok, Vahadane, mean/std, histogram_matching) estimate stain
  statistics from each source image individually, so they run one
  image at a time. Their one-time cost - fitting the reference
  target - is cached internally, so the target is only fit once
  per method no matter how many images are processed.

CycleGan and DensePix2Pix are TensorFlow/Keras-based (the rest of
the pipeline is PyTorch) - see `tf_gan_normalizer.py` and the
vendored `GanStainNorm/` package. They expect an experiment folder
(`gan_exp_path`) produced by the original GAN pipeline's
`execute.py train`, containing `config.json` and a `checkpoint/`
folder; `gan_epoch` selects which saved epoch to load.
"""

import argparse
import os

# Must be set before `torch` is imported anywhere (including via
# batch_runner/normalize_stain below). On some driver setups,
# torch's default CUDA device-count check can hang indefinitely
# on first call; this switches it to a lightweight NVML-based
# check instead. setdefault() so an existing value (e.g. set via
# conda/PyCharm) isn't overridden.
os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "1")

from pathlib import Path

import cv2

from batch_runner import (
    MethodStats,
    Timer,
    stream_shape_batches,
)
from normalize_stain_v2 import (
    METHOD_BATCH_SUPPORT,
    normalize_stain,
    normalize_stain_batch,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run one or all configured stain-normalization "
            "methods over a folder of images."
        ),
    )
    parser.add_argument(
        "--method",
        default="all",
        help=(
            "Name of the method to run (matches a config's "
            "'name' in the `configurations` list below), or "
            "'all' to run every configured method. "
            "Use --list to see available names. Default: all."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=(
            "Override the batch size for batch-capable methods "
            "(staingan, stainnet, sastaindiff, cyclegan, "
            "densepix2pix). Ignored for sequential methods "
            "(macenko, reinhard, multi_macenko, ruifrok, "
            "vahadane, mean_std, histogram_matching). If "
            "omitted, each method's own default below is used."
        ),
    )
    parser.add_argument(
        "--input-dir",
        default=None,
        help="Override the input image folder (source_directory).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override the output folder (default: results/).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available method names and exit.",
    )
    return parser.parse_args()


def save_rgb_image(path, image_rgb):
    """
    Save an RGB NumPy image using OpenCV.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    image_bgr = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2BGR,
    )

    success = cv2.imwrite(
        str(path),
        image_bgr,
    )

    if not success:
        raise RuntimeError(
            f"Failed to save image: {path}"
        )


def get_image_files(folder):
    """
    Return all supported image files inside a folder.
    """
    folder = Path(folder)

    supported_extensions = {
        ".png",
        ".jpg",
        ".jpeg",
        ".tif",
        ".tiff",
        ".bmp",
    }

    image_files = [
        path
        for path in folder.iterdir()
        if path.is_file()
        and path.suffix.lower() in supported_extensions
    ]

    return sorted(image_files)


def run_sequential_config(name, config, image_files, output_dir):
    """
    Run one configuration one image at a time.

    Used for methods that estimate stain statistics per source
    image (Macenko, Reinhard, Multi-Macenko, Ruifrok, Vahadane,
    mean/std) - they cannot share a forward pass across images,
    but their target-fitting cost is cached internally.
    """
    stats = MethodStats(name=name, mode="sequential")
    total = len(image_files)

    for index, source_path in enumerate(image_files, start=1):
        stats.total_images += 1

        with Timer() as timer:
            try:
                result = normalize_stain(
                    source=str(source_path),
                    **config,
                )
                output_path = output_dir / name / source_path.name
                save_rgb_image(output_path, result["normalized"])
                stats.succeeded += 1
                status = "OK"
            except Exception as error:
                stats.failed += 1
                stats.failures.append(
                    f"{source_path.name}: "
                    f"{type(error).__name__}: {error}"
                )
                status = (
                    f"FAILED ({type(error).__name__}: {error})"
                )

        stats.elapsed_seconds += timer.elapsed
        print(
            f"    [{index}/{total}] {name}: {status} "
            f"({timer.elapsed:.3f}s)"
        )

    return stats


def run_batch_config(
    name,
    config,
    image_files,
    output_dir,
    batch_size,
):
    """
    Run one configuration in batches.

    Used for the fully convolutional / GroupNorm-based models
    (StainGAN, StainNet, SAStainDiff): images are grouped by shape
    so that each group can be pushed through the model together,
    in chunks of at most `batch_size`.
    """
    stats = MethodStats(name=name, mode="batch")

    for shape, chunk in stream_shape_batches(image_files, batch_size):
        stats.total_images += len(chunk)

        with Timer() as timer:
            try:
                results = normalize_stain_batch(
                    sources=[item.image for item in chunk],
                    **config,
                )

                for item, result in zip(chunk, results):
                    output_path = (
                        output_dir / name / item.path.name
                    )
                    save_rgb_image(
                        output_path,
                        result["normalized"],
                    )

                stats.succeeded += len(chunk)
                status = f"OK ({len(chunk)} images)"

            except Exception as error:
                stats.failed += len(chunk)
                for item in chunk:
                    stats.failures.append(
                        f"{item.path.name}: "
                        f"{type(error).__name__}: {error}"
                    )
                status = (
                    f"FAILED ({len(chunk)} images): "
                    f"{type(error).__name__}: {error}"
                )

        stats.elapsed_seconds += timer.elapsed
        print(
            f"    {name}: shape {shape} -> {status} "
            f"({timer.elapsed:.3f}s)"
        )

    return stats


def print_summary(all_stats, output_dir):
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(
        f"{'method':<22} {'mode':<10} {'images':<9} "
        f"{'time':<14} {'avg/img'}"
    )
    print("-" * 80)

    for stats in all_stats:
        print(stats.summary_line())

        for failure in stats.failures[:5]:
            print(f"      ! {failure}")

        remaining = len(stats.failures) - 5
        if remaining > 0:
            print(f"      ... and {remaining} more failure(s)")

    total_time = sum(stats.elapsed_seconds for stats in all_stats)
    total_ok = sum(stats.succeeded for stats in all_stats)
    total_failed = sum(stats.failed for stats in all_stats)

    print("-" * 80)
    print(
        f"Total: {total_ok} succeeded, {total_failed} failed, "
        f"{total_time:.2f}s"
    )
    print(f"Results saved to: {output_dir.resolve()}")


if __name__ == "__main__":

    args = parse_args()

    # =========================================================
    # Input folder containing source images
    # =========================================================
    source_directory = Path("data/input/test_example")

    # =========================================================
    # Reference images
    # =========================================================

    # Single reference image for Macenko, Reinhard, Ruifrok,
    # and Vahadane.
    target = "data/reference/tar.png"

    # Multiple reference images for Multi-Macenko.
    targets = [
        "data/reference/target_1.png",
        "data/reference/target_2.png",
        "data/reference/target_3.png",
        "data/reference/target_4.png",
        "data/reference/target_5.png",
        "data/reference/target_6.png",
        "data/reference/target_7.png",
    ]

    # =========================================================
    # Model checkpoints
    # =========================================================

    staingan_checkpoint = (
        "methods/StainNet/checkpoints/"
        "latest_net_G_A.pth"
    )

    stainnet_checkpoint = (
        "methods/StainNet/checkpoints/"
        "StainNet-Public_layer3_ch32.pth"
    )

    sastaindiff_checkpoint = (
        "methods/SAStainDiff/checkpoint/"
        "other_to_Aperio.pt"
    )

    # CycleGan and DensePix2Pix are trained/checkpointed via the
    # separate GAN pipeline's `execute.py train`. Each exp_path
    # folder is expected to contain `config.json` (written by that
    # pipeline - records exp_type/normalization/etc.) plus a
    # `checkpoint/` folder with `epoch.NNN.index` /
    # `epoch.NNN.data-00000-of-00001` files. Point these at your
    # trained experiment folders and pick the epoch to load.
    cyclegan_exp_path = "methods/GanStainNorm/checkpoints/CycleGan"
    cyclegan_epoch = 40

    densepix2pix_exp_path = "methods/GanStainNorm/checkpoints/pix2pix_dense_a"
    densepix2pix_epoch = 40

    # =========================================================
    # Output directory
    # =========================================================

    output_directory = Path("results")
    output_directory.mkdir(parents=True, exist_ok=True)

    # =========================================================
    # Normalization configurations
    #
    # "batch_size" is only meaningful for batch-capable methods
    # (staingan, stainnet, sastaindiff) - it caps how many
    # same-shape images are pushed through the model in one
    # forward pass. It's ignored for sequential methods.
    # =========================================================

    configurations = [
        {
            "name": "ruifrok",
            "method": "ruifrok",
            "backend": "numpy",
            "target": target,
        },
        {
            "name": "vahadane",
            "method": "vahadane",
            "backend": "numpy",
            "target": target,
        },
        {
            "name": "histogram_matching",
            "method": "histogram_matching",
            "backend": "numpy",
            "target": target,
        },
        {
            "name": "mean_std",
            "method": "mean_std",
            "backend": "numpy",
        },
        {
            "name": "macenko_torch",
            "method": "macenko",
            "backend": "torch",
            "target": target,
        },
        {
            "name": "reinhard_torch",
            "method": "reinhard",
            "backend": "torch",
            "target": target,
        },
        {
            "name": "multi_macenko_torch",
            "method": "multi_macenko",
            "backend": "torch",
            "targets": targets,
            "norm_mode": "avg-post",
        },
        {
            "name": "staingan",
            "method": "staingan",
            "backend": "torch",
            "checkpoint": staingan_checkpoint,
            "batch_size": 4,
        },
        {
            "name": "stainnet",
            "method": "stainnet",
            "backend": "torch",
            "checkpoint": stainnet_checkpoint,
            "stainnet_channels": 32,
            "stainnet_n_layer": 3,
            "batch_size": 4,
        },
        {
            "name": "sastaindiff",
            "method": "sastaindiff",
            "backend": "torch",
            "checkpoint": sastaindiff_checkpoint,
            "sastaindiff_image_size": 256,
            "sastaindiff_attention_resolutions": "32,16,8",
            "sastaindiff_num_channels": 128,
            "sastaindiff_num_res_blocks": 3,
            "sastaindiff_diffusion_steps": 500,
            "sastaindiff_noise_schedule": "linear",
            "sastaindiff_use_ddim": True,
            "sastaindiff_timestep_respacing": "ddim50",
            "sastaindiff_use_fp16": True,
            "sastaindiff_use_anysize": True,
            "sastaindiff_seed": 42,
            # Diffusion sampling is memory-hungry per image, so
            # its batch is kept small by default.
            "batch_size": 2,
        },
        {
            "name": "cyclegan",
            "method": "cyclegan",
            "gan_exp_path": cyclegan_exp_path,
            "gan_epoch": cyclegan_epoch,
            "batch_size": 8,
        },
        {
            "name": "densepix2pix",
            "method": "densepix2pix",
            "gan_exp_path": densepix2pix_exp_path,
            "gan_epoch": densepix2pix_epoch,
            "batch_size": 8,
        },
    ]

    # =========================================================
    # Apply CLI overrides
    # =========================================================

    if args.input_dir:
        source_directory = Path(args.input_dir)

    if args.output_dir:
        output_directory = Path(args.output_dir)
        output_directory.mkdir(parents=True, exist_ok=True)

    available_names = [config["name"] for config in configurations]

    if args.list:
        print("Available methods:")
        for config in configurations:
            is_batchable = METHOD_BATCH_SUPPORT.get(
                config["method"], False
            )
            mode = "batch" if is_batchable else "sequential"
            print(f"  - {config['name']:<20} ({mode})")
        raise SystemExit(0)

    if args.method != "all":
        if args.method not in available_names:
            raise SystemExit(
                f"Unknown method '{args.method}'. Available: "
                f"{', '.join(available_names)}, or 'all'."
            )
        configurations = [
            config for config in configurations
            if config["name"] == args.method
        ]

    # =========================================================
    # Find source images
    # =========================================================

    image_files = get_image_files(source_directory)

    if not image_files:
        raise RuntimeError(
            f"No images found in: {source_directory}"
        )

    print(f"Found {len(image_files)} images in {source_directory}")
    print("=" * 80)

    # =========================================================
    # Run each configuration over the whole folder
    # =========================================================

    all_stats = []

    for configuration in configurations:
        configuration = configuration.copy()

        name = configuration.pop("name")
        method = configuration["method"]
        default_batch_size = configuration.pop("batch_size", 8)
        batch_size = (
            args.batch_size
            if args.batch_size is not None
            else default_batch_size
        )

        is_batchable = METHOD_BATCH_SUPPORT.get(method, False)
        mode = "batch" if is_batchable else "sequential"

        print(f"\n[{mode}] Running '{name}' ({method}) ...")

        if is_batchable:
            stats = run_batch_config(
                name,
                configuration,
                image_files,
                output_directory,
                batch_size,
            )
        else:
            stats = run_sequential_config(
                name,
                configuration,
                image_files,
                output_directory,
            )

        all_stats.append(stats)

    print_summary(all_stats, output_directory)
