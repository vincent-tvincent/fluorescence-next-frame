"""One movie per directory, one single-channel grayscale TIFF per time point."""

from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
import random
import re

import numpy as np
import tifffile
import torch


@lru_cache(maxsize=2048)
def frame_shape(path: Path) -> tuple[int, ...]:
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        axes = series.axes
        if (len(tif.series) != 1 or len(series.shape) not in (2, 3)
                or not axes.endswith("YX") or any(axis in axes for axis in "TCS")
                or series.dtype.kind not in "uif"):
            raise ValueError(f"{path}: expected one grayscale YX image or ZYX stack, got {axes} {series.shape}")
        return tuple(series.shape)


def read_frame(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    if frame_shape(path) != shape:
        raise ValueError(f"{path}: expected frame shape {shape}, found {frame_shape(path)}")
    try:
        return tifffile.memmap(path, mode="r")
    except ValueError:
        return tifffile.imread(path)


@dataclass(frozen=True)
class Movie:
    directory: Path
    paths: tuple[Path, ...]
    shape: tuple[int, ...]

    @classmethod
    def open(cls, directory: Path):
        directory = directory.resolve()
        files = [p for p in directory.iterdir()
                 if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in (".tif", ".tiff")]
        if not files:
            raise ValueError(f"{directory}: no TIFF frames found")
        if len(files) > 1:
            indexed = {}
            stem = None
            for path in files:
                match = re.fullmatch(r"(.*?)(\d+)(?:\.ome)?", path.stem, flags=re.IGNORECASE)
                if match is None:
                    raise ValueError(f"{path}: expected a filename ending in a numeric frame index")
                prefix, index = match.group(1), int(match.group(2))
                if (stem is not None and prefix != stem) or index in indexed:
                    raise ValueError(f"{directory}: expected one filename series with unique frame indices")
                stem = prefix
                indexed[index] = path
            if max(indexed) - min(indexed) + 1 != len(indexed):
                raise ValueError(f"{directory}: gap in frame indices; split into contiguous movies")
            files = [indexed[i] for i in sorted(indexed)]
        return cls(directory, tuple(files), frame_shape(files[0]))


def split_movies(train_dirs, val_dirs, val_fraction):
    movies = [Movie.open(p) for p in train_dirs]
    if val_dirs:
        train, val = movies, [Movie.open(p) for p in val_dirs]
    elif len(movies) > 1:
        train, val = movies[:-1], movies[-1:]
    else:
        movie = movies[0]
        cut = len(movie.paths) - max(2, round(len(movie.paths) * val_fraction))
        train = [replace(movie, paths=movie.paths[:max(0, cut)])]
        val = [replace(movie, paths=movie.paths[max(0, cut):])]
    all_movies = train + val
    if any(len(m.paths) < 2 for m in all_movies):
        raise ValueError("each training and validation segment needs at least two distinct frames")
    if len({len(m.shape) for m in all_movies}) != 1:
        raise ValueError("use only 2D movies or only 3D movies in one run")
    seen = set()
    for movie in all_movies:
        paths = {p.resolve() for p in movie.paths}
        if seen.intersection(paths) or len(paths) != len(movie.paths):
            raise ValueError("a source frame appears more than once in the train/validation split")
        seen.update(paths)
    return train, val


@dataclass(frozen=True)
class IntensityScale:
    offset: float
    span: float

    @classmethod
    def fit(cls, movies):
        """Estimate one scale from training frames only, with bounded I/O and memory."""
        paths = [(path, m.shape) for m in movies for path in m.paths]
        selected = np.linspace(0, len(paths) - 1, min(24, len(paths)), dtype=int)
        rng = np.random.default_rng(0)
        samples = []
        for index in selected:
            path, shape = paths[index]
            raw = read_frame(path, shape)
            indices = rng.integers(raw.size, size=min(raw.size, 8192))
            indices.sort()  # Read sampled pages in disk order.
            samples.append(raw.flat[indices].astype(np.float64))
        values = np.concatenate(samples)
        if not np.isfinite(values).all():
            raise ValueError("training frames contain non-finite intensities")
        low, high = np.percentile(values, [1, 99.8])
        if high <= low:
            low, high = values.min(), values.max()
        return cls(float(low), float(high - low) if high > low else 1.0)

    def tensor(self, image, device):
        array = np.array(image, dtype=np.float32, copy=True, order="C")
        array -= self.offset
        array /= self.span
        if not np.isfinite(array).all():
            raise ValueError("frame contains non-finite normalized intensities")
        return torch.from_numpy(array)[None, None].to(device)


def patch_size(requested, ndim):
    if requested is None:
        return (96, 96) if ndim == 2 else (12, 96, 96)
    if len(requested) != ndim or min(requested) < 1:
        raise ValueError(f"patch or tile size must have {ndim} positive spatial dimensions")
    return tuple(requested)


def random_region(shape, size, rng: random.Random):
    starts = [rng.randrange(max(0, dim - width) + 1) for dim, width in zip(shape, size)]
    return tuple(slice(start, min(start + width, dim)) for start, width, dim in zip(starts, size, shape))


def clip_and_target(movie, target, history, region, scale, device):
    start = target - history
    if start < 0 or history < 1 or target >= len(movie.paths):
        raise ValueError("clip needs consecutive past observations and one distinct future frame")
    frames = [scale.tensor(read_frame(path, movie.shape)[region], device)
              for path in movie.paths[start:target + 1]]
    return torch.stack(frames[:-1], dim=1), frames[-1]


def training_batch(movies, history, batch, size, scale, device, rng):
    """Collate independent same-history, same-shape crops in RAM, then transfer once."""
    first = rng.choice(movies)
    shape = tuple(map(min, zip(first.shape, size)))
    compatible = [m for m in movies if tuple(map(min, zip(m.shape, size))) == shape]
    clips, targets = [], []
    for i in range(batch):
        movie = first if i == 0 else rng.choice(compatible)
        t = rng.randrange(history, len(movie.paths))
        region = random_region(movie.shape, size, rng)
        clip, target = clip_and_target(movie, t, history, region, scale, "cpu")
        clips.append(clip)
        targets.append(target)
    if batch == 1:
        return clips[0].to(device), targets[0].to(device)
    return torch.cat(clips).to(device), torch.cat(targets).to(device)
