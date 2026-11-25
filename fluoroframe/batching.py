"""Strict FP32 updates and CUDA batch-size selection for the longest training clip."""

import gc
import math

import torch
from torch.nn import functional as F

from .model import FramePredictor


GIB = 1024 ** 3


def configure_fp32(device):
    if device.type == "cuda":
        if hasattr(torch.backends.cudnn, "conv") and hasattr(torch.backends.cudnn.conv, "fp32_precision"):
            torch.backends.cuda.matmul.fp32_precision = "ieee"
            torch.backends.cudnn.conv.fp32_precision = "ieee"
        else:  # PyTorch 2.5–2.8
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False


def training_step(model, optimizer, clip, target):
    optimizer.zero_grad(set_to_none=True)
    loss = F.l1_loss(model(clip), target)
    loss.backward()
    optimizer.step()
    return loss.detach()


def largest_passing_batch(probe):
    """Bracket the memory limit, then test integers within the final bracket."""
    low, high = 0, 1
    while probe(high):
        low, high = high, 2 * high
    if low == 0:
        raise ValueError("batch 1 did not fit with the requested free VRAM; reduce the crop or history")
    while high - low > 1:
        candidate = (low + high) // 2
        if probe(candidate):
            low = candidate
        else:
            high = candidate
    return low


def probe_cuda_batch(device, shape, history, width, lr, reserve_gib):
    if device.type != "cuda":
        raise ValueError("automatic batch selection needs CUDA; use an integer batch size on this device")
    if not math.isfinite(reserve_gib) or reserve_gib < 0:
        raise ValueError("vram-reserve-gib must be finite and nonnegative")
    reserve = math.ceil(reserve_gib * GIB)
    trials = []

    def two_updates(batch):
        model = FramePredictor(len(shape), width).to(device=device, dtype=torch.float32)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        clip = torch.randn((batch, history, 1, *shape), device=device, dtype=torch.float32)
        target = torch.randn((batch, 1, *shape), device=device, dtype=torch.float32)
        for _ in range(2):  # Includes AdamW's lazily allocated state on the second update.
            loss = training_step(model, optimizer, clip, target)
        if not torch.isfinite(loss).item():
            raise ValueError("non-finite loss in the FP32 batch probe")
        torch.cuda.synchronize(device)

    def fits(batch):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        free, _ = torch.cuda.mem_get_info(device)
        available_at_start = free + torch.cuda.memory_reserved(device)
        try:
            two_updates(batch)
        except torch.cuda.OutOfMemoryError:
            oom = True
        else:
            oom = False
        peak = torch.cuda.max_memory_reserved(device)
        free, _ = torch.cuda.mem_get_info(device)
        available_at_end = free + torch.cuda.memory_reserved(device)
        headroom = min(available_at_start, available_at_end) - peak
        passed = not oom and headroom >= reserve
        trial = {"batch": batch, "passed": passed, "oom": oom, "peak_reserved_bytes": peak,
                 "estimated_free_at_peak_bytes": None if oom else headroom}
        trials.append(trial)
        status = "fits" if passed else ("allocator limit or OOM" if oom else "below headroom target")
        print(f"Probe batch {batch}: {status}, peak PyTorch reserved {peak / GIB:.2f} GiB", flush=True)
        gc.collect()
        torch.cuda.empty_cache()
        return passed

    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info(device)
    budget = free + torch.cuda.memory_reserved(device) - reserve
    if budget <= 0:
        raise ValueError("free VRAM is already below the requested reserve")
    # Bound scratch allocations before trying a candidate that may exceed the display GPU's capacity.
    getter = getattr(torch.cuda, "get_per_process_memory_fraction", None)
    previous_fraction = getter(device) if getter else 1.0
    limit = min(previous_fraction, budget / total)
    torch.cuda.set_per_process_memory_fraction(limit, device)
    try:
        size = largest_passing_batch(fits)
    finally:
        torch.cuda.set_per_process_memory_fraction(previous_fraction, device)
        gc.collect()
        torch.cuda.empty_cache()
    return size, trials
