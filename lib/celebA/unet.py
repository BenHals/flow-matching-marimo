import math
import time
from enum import Enum

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.celebA.model_components import BlockType, PositionEncoding, ResidualBlock


class SimpleTimeEmbUnet(nn.Module):
    def __init__(
        self,
        image_channels: int = 1,
        n_embedding_blocks: int = 3,
        dim_step_size: int = 64,
        time_embedding_dim: int = 64,
        group_norm_channels: int = 32,
    ):
        super().__init__()

        self.time_embedder = nn.Sequential(
            PositionEncoding(time_embedding_dim),
            nn.Linear(time_embedding_dim, time_embedding_dim),
            nn.ReLU(),
            nn.Linear(time_embedding_dim, time_embedding_dim),
        )

        self.initial_channel_projection = nn.Conv2d(
            image_channels, dim_step_size, kernel_size=3, padding=1
        )
        self.encoder_blocks = nn.ModuleList(
            [
                ResidualBlock(
                    dim_step_size * 2**i,
                    dim_step_size * 2 ** (i + 1),
                    time_embedding_dim,
                    0,
                    BlockType.Encoder,
                    group_norm_channels=group_norm_channels,
                )
                for i in range(n_embedding_blocks)
            ]
        )

        self.decoder_blocks = nn.ModuleList(
            [
                ResidualBlock(
                    dim_step_size * 2 ** (n_embedding_blocks - (i)),
                    dim_step_size * 2 ** (n_embedding_blocks - (i + 1)),
                    time_embedding_dim,
                    0,
                    BlockType.DecoderWithSkip,
                    group_norm_channels=group_norm_channels,
                )
                for i in range(n_embedding_blocks)
            ]
        )

        self.output_channel_projection = nn.Conv2d(
            dim_step_size, image_channels, kernel_size=1
        )

    def forward(self, x_init: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embedder(t.squeeze(-1))
        x = self.initial_channel_projection(x_init)

        layer_residuals: list[torch.Tensor] = []
        for block in self.encoder_blocks:
            x = block(x, t_emb)
            layer_residuals.append(x)

        for i, block in enumerate(self.decoder_blocks):
            residual = layer_residuals.pop()
            x = torch.cat((x, residual), dim=1)
            x = block(x, t_emb)

        return self.output_channel_projection(x)

    def generate(
        self, n_steps: int = 50, image_dims: tuple[int, int, int] = (1, 28, 28)
    ) -> torch.Tensor:
        if n_steps < 1:
            raise ValueError("n_steps must be at least 1")
        t_steps = np.linspace(0.0, 1.0, n_steps + 1)
        step_size = t_steps[1] - t_steps[0]

        x_0 = torch.randn((1, *image_dims)).float()
        x_t = x_0
        for t in torch.from_numpy(t_steps).float():
            x_t += step_size * self(
                x_t,
                t.view(1, 1).float(),
            )

        return x_t.squeeze(0)

    def step(
        self, x_t: torch.Tensor, t_start: torch.Tensor, t_end: torch.Tensor
    ) -> torch.Tensor:
        t_start = t_start.view(1, 1).expand(x_t.shape[0], 1)
        return x_t + (t_end - t_start) * self(
            x_t + self(x_t, t_start) * (t_end - t_start) / 2,
            t_start + (t_end - t_start) / 2,
        )
