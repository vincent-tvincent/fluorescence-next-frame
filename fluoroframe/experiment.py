"""Training and paired validation against the copy-last-frame forecast."""

from dataclasses import asdict
import json
import math
from pathlib import Path
import random

import torch
from torch.nn import functional as F

from .data import IntensityScale, Movie, clip_and_target, patch_size, random_region, split_movies
from .model import FramePredictor


def available_histories(requested, movies):
    largest = max(len(movie.paths) - 1 for movie in movies)
    histories = sorted({h for h in requested if h <= largest})
    if not histories:
        raise ValueError(f"the requested history needs more frames; longest available history is {largest}")
    return histories


def validation_plan(movies, histories, size, samples, seed):
    rng = random.Random(seed)
    candidates = [(movie, t) for movie in movies for t in range(max(histories), len(movie.paths))]
    selected = rng.sample(candidates, min(samples, len(candidates)))
    return [(movie, t, random_region(movie.shape, size, rng)) for movie, t in selected]


@torch.inference_mode()
def measure(model, plan, histories, scale, device):
    model.eval()
    sums = {h: 0.0 for h in histories}
    copy_sum = 0.0
    voxels = 0
    for movie, t, region in plan:
        clip, target = clip_and_target(movie, t, max(histories), region, scale, device)
        voxels += target.numel()
        copy_sum += (clip[:, -1] - target).abs().sum().item()
        for h in histories:
            sums[h] += (model(clip[:, -h:]) - target).abs().sum().item()
    return {
        "target_crops": len(plan),
        "voxels": voxels,
        "copy_mae": scale.span * copy_sum / voxels,
        "model_mae": {str(h): scale.span * total / voxels for h, total in sums.items()},
    }


def describe(movies):
    return [{"directory": str(m.directory), "frames": len(m.paths),
             "first": m.paths[0].name, "last": m.paths[-1].name} for m in movies]


def train(args, device):
    if not 0 < args.val_fraction < 1 or not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("val-fraction must be between zero and one, and lr must be finite and positive")
    training, validation = split_movies(args.train, args.val, args.val_fraction)
    ndim = len(training[0].shape)
    size = patch_size(args.patch, ndim)
    histories = available_histories(args.histories, training)
    val_histories = available_histories([1, *histories], validation)
    plan = validation_plan(validation, val_histories, size, args.val_samples, args.seed)
    choices = {h: [movie for movie in training if len(movie.paths) > h] for h in histories}
    run = args.run
    if run.exists():
        raise FileExistsError(f"{run} already exists; choose a new run directory")
    scale = IntensityScale.fit(training)
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    model = FramePredictor(ndim, args.width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    config = {"ndim": ndim, "width": args.width, "patch": list(size), "scale": asdict(scale)}
    settings = {**config, "training": describe(training), "validation": describe(validation),
                "histories": histories, "validation_histories": val_histories, "seed": args.seed,
                "lr": args.lr, "epochs": args.epochs, "steps_per_epoch": args.steps_per_epoch,
                "val_samples": args.val_samples, "device": str(device)}
    run.mkdir(parents=True)
    (run / "run.json").write_text(json.dumps(settings, indent=2) + "\n")
    print(f"Training on {device}, {ndim}D patches {size}, histories {histories}. "
          f"Validation uses {len(plan)} fixed targets and histories {val_histories}.", flush=True)
    best = math.inf
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        for _ in range(args.steps_per_epoch):
            history = rng.choice(histories)
            movie = rng.choice(choices[history])
            t = rng.randrange(history, len(movie.paths))
            region = random_region(movie.shape, size, rng)
            clip, target = clip_and_target(movie, t, history, region, scale, device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.l1_loss(model(clip), target)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
        result = measure(model, plan, val_histories, scale, device)
        record = {"epoch": epoch, "train_mae": scale.span * loss_sum / args.steps_per_epoch, **result}
        score = result["model_mae"][str(max(val_histories))]
        if not math.isfinite(score):
            raise ValueError("validation MAE is non-finite; check the input intensities and learning rate")
        if score < best:
            tmp = run / "best.pt.tmp"
            torch.save({"format": 1, **config, "epoch": epoch, "model": model.state_dict()}, tmp)
            tmp.replace(run / "best.pt")
            best = score
        with (run / "metrics.jsonl").open("a") as log:
            log.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)


def load_checkpoint(path: Path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != 1:
        raise ValueError(f"{path}: unsupported checkpoint format")
    model = FramePredictor(checkpoint["ndim"], checkpoint["width"])
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), IntensityScale(**checkpoint["scale"]), tuple(checkpoint["patch"])


def evaluate(args, device):
    model, scale, default_patch = load_checkpoint(args.checkpoint, device)
    movies = [Movie.open(directory) for directory in args.movies]
    if any(len(m.shape) != model.ndim for m in movies):
        raise ValueError(f"the checkpoint expects {model.ndim}D frames")
    histories = available_histories(args.histories, movies)
    if set(histories) != set(args.histories):
        raise ValueError(f"only histories {histories} have enough observations and a held-out target")
    size = patch_size(args.patch or default_patch, model.ndim)
    plan = validation_plan(movies, histories, size, args.samples, args.seed)
    print(json.dumps(measure(model, plan, histories, scale, device), indent=2))
