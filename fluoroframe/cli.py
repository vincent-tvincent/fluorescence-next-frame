import argparse
from pathlib import Path

import torch

from .experiment import evaluate, train
from .inference import predict


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("expected an integer of at least 1")
    return number


def choose_device(requested):
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable in this PyTorch environment")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable in this PyTorch environment")
    return torch.device(requested)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Predict the next raw fluorescence image.")
    commands = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")

    p = commands.add_parser("train", parents=[common], help="train on raw time-lapse TIFFs")
    p.add_argument("--train", nargs="+", type=Path, required=True, metavar="MOVIE")
    p.add_argument("--val", nargs="+", type=Path, metavar="MOVIE", help="independent validation movies")
    p.add_argument("--val-fraction", type=float, default=0.2, help="held-out tail for a single movie")
    p.add_argument("--run", type=Path, required=True, help="new output directory")
    p.add_argument("--histories", nargs="+", type=positive, default=[1, 2, 4, 8],
                   help="observed clip lengths to sample during training")
    p.add_argument("--patch", nargs="+", type=positive, help="training crop in Y X or Z Y X order")
    p.add_argument("--width", type=positive, default=16, help="first encoder level channel count")
    p.add_argument("--epochs", type=positive, default=10)
    p.add_argument("--steps-per-epoch", type=positive, default=64)
    p.add_argument("--val-samples", type=positive, default=16, help="maximum fixed validation targets")
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(run_command=train)

    p = commands.add_parser("evaluate", parents=[common], help="compare histories and copy-last MAE")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--movies", nargs="+", type=Path, required=True, metavar="MOVIE")
    p.add_argument("--histories", nargs="+", type=positive, default=[1, 4])
    p.add_argument("--samples", type=positive, default=16, help="maximum number of target frames")
    p.add_argument("--patch", nargs="+", type=positive, help="evaluation crop, defaults to training size")
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(run_command=evaluate)

    p = commands.add_parser("predict", parents=[common], help="write the next full image as a TIFF")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--movie", type=Path, required=True)
    p.add_argument("--through", metavar="FILENAME", help="last observed TIFF, defaults to the final file")
    p.add_argument("--history", default="all", help="positive number of observed frames, or all")
    p.add_argument("--tile", nargs="+", type=positive, help="output tile size, defaults to training patch")
    p.add_argument("--output", type=Path, required=True, help="new .tif or .tiff file")
    p.set_defaults(run_command=predict)

    args = parser.parse_args(argv)
    try:
        args.run_command(args, choose_device(args.device))
    except (ValueError, OSError) as exc:
        parser.exit(2, f"fluoroframe: {exc}\n")
