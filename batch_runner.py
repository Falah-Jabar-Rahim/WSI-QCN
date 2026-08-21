"""
Small utilities that support running normalization methods over a
whole folder of images: grouping images by shape (needed before
batching them through a model), chunking, timing, and collecting a
per-method report.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np

from normalize_stain_v2 import read_rgb_image


@dataclass
class LoadedImage:
    """
    An image already read from disk, paired with its source path
    so results can be written back to a matching output filename.
    """
    path: Path
    image: np.ndarray


def stream_shape_batches(
    image_files: list[Path],
    batch_size: int,
) -> Iterator[tuple[tuple, list[LoadedImage]]]:
    """
    Read images in folder order and yield same-shape batches of at
    most `batch_size` images as soon as a group fills up.

    Batch-capable models require every image in one forward pass
    to share the same spatial size, so images are bucketed by
    shape. Streaming (rather than reading the whole folder first)
    keeps memory bounded to roughly
    (distinct shapes seen so far x batch_size) images, which
    matters for large folders.

    Yields
    ------
    (shape, batch) pairs, where `batch` is a list of `LoadedImage`
    of length <= batch_size.
    """
    batch_size = max(1, batch_size)
    pending: dict[tuple, list[LoadedImage]] = defaultdict(list)

    for path in image_files:
        image = read_rgb_image(path)
        bucket = pending[image.shape]
        bucket.append(LoadedImage(path=path, image=image))

        if len(bucket) >= batch_size:
            yield image.shape, bucket
            pending[image.shape] = []

    # Flush any partially filled buckets left over at the end.
    for shape, bucket in pending.items():
        if bucket:
            yield shape, bucket


@dataclass
class MethodStats:
    """
    Tracks the outcome of running one normalization configuration
    over the whole input folder, for the end-of-run summary.
    """
    name: str
    mode: str  # "batch" or "sequential"
    total_images: int = 0
    succeeded: int = 0
    failed: int = 0
    elapsed_seconds: float = 0.0
    failures: list[str] = field(default_factory=list)

    @property
    def avg_seconds_per_image(self) -> float:
        if not self.total_images:
            return 0.0
        return self.elapsed_seconds / self.total_images

    def summary_line(self) -> str:
        return (
            f"{self.name:<22} "
            f"{self.mode:<10} "
            f"{self.succeeded:>4}/{self.total_images:<4} ok  "
            f"{self.elapsed_seconds:>8.2f}s total  "
            f"{self.avg_seconds_per_image:>7.3f}s/img"
        )


class Timer:
    """
    Context manager exposing wall-clock elapsed seconds as
    `.elapsed` once the `with` block exits.
    """
    elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc_info) -> None:
        self.elapsed = time.perf_counter() - self._start
