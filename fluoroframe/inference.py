"""Forecast a complete frame with bounded tile size and arbitrary input history."""

from itertools import product

import numpy as np
import tifffile
import torch

from .data import Movie, patch_size, read_frame
from .experiment import load_checkpoint


def tile_regions(shape, core):
    halo = (16, 16) if len(shape) == 2 else (2, 16, 16)
    for start in product(*(range(0, dim, size) for dim, size in zip(shape, core))):
        end = [min(a + size, dim) for a, size, dim in zip(start, core, shape)]
        source = tuple(slice(max(0, a - pad), min(dim, b + pad))
                       for a, b, dim, pad in zip(start, end, shape, halo))
        destination = tuple(slice(a, b) for a, b in zip(start, end))
        center = tuple(slice(a - s.start, b - s.start) for a, b, s in zip(start, end, source))
        yield source, destination, center


@torch.inference_mode()
def predict_image(model, paths, shape, scale, core, device):
    if not paths:
        raise ValueError("a forecast needs at least one observed frame")
    tiles = list(tile_regions(shape, core))
    states = [None] * len(tiles)
    result = np.empty(shape, dtype=np.float32)
    for index, path in enumerate(paths):
        image = read_frame(path, shape)
        last = index == len(paths) - 1
        for i, (source, destination, center) in enumerate(tiles):
            frame = scale.tensor(image[source], device)
            state = states[i]
            if state is not None:
                state = state.to(device)
            state, skips = model.observe(frame, state)
            if last:
                normalized = model.decode(frame, state, skips)[0, 0][center].cpu().numpy()
                result[destination] = normalized * scale.span + scale.offset
                states[i] = None
            else:
                states[i] = state.cpu() if len(tiles) > 1 else state
    return result


def predict(args, device):
    if args.output.exists():
        raise FileExistsError(f"{args.output} already exists; choose a new output file")
    if args.output.suffix.lower() not in (".tif", ".tiff"):
        raise ValueError("use a .tif or .tiff output filename")
    model, scale, default_tile = load_checkpoint(args.checkpoint, device)
    movie = Movie.open(args.movie)
    if len(movie.shape) != model.ndim:
        raise ValueError(f"the checkpoint expects {model.ndim}D frames")
    paths = movie.paths
    if args.through is not None:
        names = [p.name for p in paths]
        if args.through not in names:
            raise ValueError(f"{args.through} is not a frame filename in {movie.directory}")
        paths = paths[:names.index(args.through) + 1]
    if args.history != "all":
        history = int(args.history)
        if not 1 <= history <= len(paths):
            raise ValueError(f"history must be between 1 and {len(paths)}, or 'all'")
        paths = paths[-history:]
    core = patch_size(args.tile or default_tile, model.ndim)
    image = predict_image(model, paths, movie.shape, scale, core, device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as output:
        try:
            tifffile.imwrite(output, image, photometric="minisblack",
                             metadata={"axes": "YX" if model.ndim == 2 else "ZYX"})
        except BaseException:
            args.output.unlink()
            raise
    print(f"{paths[0].name} through {paths[-1].name} ({len(paths)} observations) -> {args.output}")
