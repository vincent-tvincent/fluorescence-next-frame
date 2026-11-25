"""One U-Net encoder and decoder with convolutional recurrence at the bottleneck."""

from collections.abc import Iterable

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ConvGRU(nn.Module):
    def __init__(self, conv, channels: int):
        super().__init__()
        self.gates = conv(2 * channels, 2 * channels, 3, padding=1)
        self.candidate = conv(2 * channels, channels, 3, padding=1)

    def forward(self, features: Tensor, state: Tensor | None) -> Tensor:
        if state is None:
            state = torch.zeros_like(features)
        reset, update = self.gates(torch.cat((features, state), dim=1)).sigmoid().chunk(2, dim=1)
        candidate = self.candidate(torch.cat((features, reset * state), dim=1)).tanh()
        return torch.lerp(state, candidate, update)


class FramePredictor(nn.Module):
    def __init__(self, ndim: int, width: int = 16):
        super().__init__()
        if ndim not in (2, 3) or width < 1:
            raise ValueError("ndim must be 2 or 3 and width must be positive")
        self.ndim = ndim
        self.width = width
        conv = nn.Conv2d if ndim == 2 else nn.Conv3d

        def block(in_channels, out_channels, stride=1):
            return nn.Sequential(
                conv(in_channels, out_channels, 3, stride=stride, padding=1),
                nn.ReLU(),
                conv(out_channels, out_channels, 3, padding=1),
                nn.ReLU(),
            )

        self.fine = block(1, width)
        self.middle = block(width, 2 * width, stride=2 if ndim == 2 else (1, 2, 2))
        self.coarse = block(2 * width, 4 * width, stride=2)
        self.memory = ConvGRU(conv, 4 * width)
        self.up_middle = block(6 * width, 2 * width)
        self.up_fine = block(3 * width, width)
        self.change = conv(width, 1, 1)
        # Start from the persistence forecast. Learning can add signed changes.
        nn.init.zeros_(self.change.weight)
        nn.init.zeros_(self.change.bias)

    def observe(self, frame: Tensor, state: Tensor | None = None):
        """Update spatial memory with one observed frame shaped (B, 1, [Z,] Y, X)."""
        fine = self.fine(frame)
        middle = self.middle(fine)
        state = self.memory(self.coarse(middle), state)
        return state, (fine, middle)

    def decode(self, last_frame: Tensor, state: Tensor, skips) -> Tensor:
        fine, middle = skips
        up = F.interpolate(state, size=middle.shape[2:], mode="nearest")
        up = self.up_middle(torch.cat((up, middle), dim=1))
        up = F.interpolate(up, size=fine.shape[2:], mode="nearest")
        up = self.up_fine(torch.cat((up, fine), dim=1))
        return last_frame + self.change(up)

    def forecast(self, observations: Iterable[Tensor]) -> Tensor:
        state = None
        for frame in observations:
            state, skips = self.observe(frame, state)
        if state is None:
            raise ValueError("a forecast needs at least one observed frame")
        return self.decode(frame, state, skips)

    def forward(self, clip: Tensor) -> Tensor:
        """Map (B, T, 1, [Z,] Y, X) to the next (B, 1, [Z,] Y, X) image."""
        if clip.ndim != self.ndim + 3 or clip.shape[2] != 1:
            raise ValueError("expected a single-channel clip shaped (B, T, 1, [Z,] Y, X)")
        return self.forecast(clip.unbind(dim=1))
