"""Training and paired validation against the copy-last-frame forecast."""

from dataclasses import asdict
import json
import math
from pathlib import Path
import random
from time import monotonic

import torch

from .batching import configure_fp32, probe_cuda_batch, training_step
from .data import IntensityScale, Movie, clip_and_target, patch_size, random_region, split_movies, training_batch
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


def save_checkpoint(path, state):
    temporary = path.with_suffix(".pt.tmp")
    torch.save(state, temporary)
    temporary.replace(path)


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
    configure_fp32(device)
    batch = args.batch_size
    trials = []
    if batch == "auto":
        # This envelope covers every effective crop, including differently sized movies.
        shape = tuple(min(limit, max(m.shape[i] for m in training)) for i, limit in enumerate(size))
        print(f"Probing strict FP32 on {device}, crop {shape}, history {max(histories)}, "
              f"reserve {args.vram_reserve_gib:g} GiB.", flush=True)
        batch, trials = probe_cuda_batch(device, shape, max(histories), args.width, args.lr, args.vram_reserve_gib)
    print(f"Selected batch {batch}. Estimating the intensity scale from training frames.", flush=True)
    scale = IntensityScale.fit(training)
    rng = random.Random(args.seed)
    # Initialize a fresh model and optimizer after all disposable probe steps.
    torch.manual_seed(args.seed)
    model = FramePredictor(ndim, args.width).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    config = {"ndim": ndim, "width": args.width, "patch": list(size), "scale": asdict(scale), "precision": "fp32"}
    settings = {**config, "training": describe(training), "validation": describe(validation),
                "histories": histories, "validation_histories": val_histories, "seed": args.seed,
                "lr": args.lr, "epochs": args.epochs, "steps_per_epoch": args.steps_per_epoch,
                "val_samples": args.val_samples, "device": str(device), "batch_size": batch,
                "requested_batch_size": args.batch_size, "batch_probe": trials,
                "vram_reserve_gib": args.vram_reserve_gib if trials else None}
    run.mkdir(parents=True)
    (run / "run.json").write_text(json.dumps(settings, indent=2) + "\n")
    print(f"Training on {device} in FP32, batch {batch}, {ndim}D patches {size}, histories {histories}. "
          f"Validation uses {len(plan)} fixed targets and histories {val_histories}.", flush=True)
    best = math.inf
    for epoch in range(1, args.epochs + 1):
        started = monotonic()
        model.train()
        loss_sum = 0.0
        for step in range(1, args.steps_per_epoch + 1):
            history = rng.choice(histories)
            clip, target = training_batch(choices[history], history, batch, size, scale, device, rng)
            loss_sum += training_step(model, optimizer, clip, target).item()
            del clip, target
            if args.log_every and (step == 1 or step % args.log_every == 0):
                print(f"Epoch {epoch}/{args.epochs} step {step}/{args.steps_per_epoch} "
                      f"batch={batch} train_MAE={scale.span * loss_sum / step:.5f} "
                      f"elapsed={monotonic() - started:.1f}s", flush=True)
        result = measure(model, plan, val_histories, scale, device)
        record = {"epoch": epoch, "epoch_seconds": round(monotonic() - started, 3),
                  "train_mae": scale.span * loss_sum / args.steps_per_epoch, **result}
        score = result["model_mae"][str(max(val_histories))]
        if not math.isfinite(score):
            raise ValueError("validation MAE is non-finite; check the input intensities and learning rate")
        state = {"format": 1, **config, "epoch": epoch, "model": model.state_dict()}
        save_checkpoint(run / "last.pt", state)
        if score < best:
            save_checkpoint(run / "best.pt", state)
            best = score
        with (run / "metrics.jsonl").open("a") as log:
            log.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)


def load_checkpoint(path: Path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != 1:
        raise ValueError(f"{path}: unsupported checkpoint format")
    if checkpoint.get("precision") == "fp32":
        configure_fp32(device)
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
