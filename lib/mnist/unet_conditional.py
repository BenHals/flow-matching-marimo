import math
import time
from enum import Enum

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionEncoding(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        emb = math.log(10000) / (self.embedding_dim // 2 - 1)

        emb = torch.exp(torch.arange(self.embedding_dim // 2, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class BlockType(Enum):
    Encoder = 0
    Decoder = 1


class ResidualBlock(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        time_embedding_dim: int,
        class_embedding_dim: int,
        block_type: BlockType,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        print(f"Created {block_type} from {input_channels} to {output_channels}")
        self.conditioning_projection = nn.Linear(
            time_embedding_dim + class_embedding_dim, output_channels
        )

        if block_type is BlockType.Decoder:
            # The decoder takes the residual input as well, so input channels is doubled
            n_input_channels = 2 * input_channels
            self.size_scaling = torch.nn.ConvTranspose2d(
                output_channels,
                output_channels,
                kernel_size=2,
                stride=2,
                padding=0,
                bias=False,
            )

            self.residual_channel_scaling = nn.ConvTranspose2d(
                n_input_channels, output_channels, kernel_size=2, stride=2, bias=False
            )
        else:
            n_input_channels = input_channels
            # This will halve the size
            self.size_scaling = nn.Conv2d(
                output_channels,
                output_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            )

            self.residual_channel_scaling = nn.Conv2d(
                n_input_channels, output_channels, kernel_size=1, stride=2, bias=False
            )

        self.channel_scaling = nn.Conv2d(
            n_input_channels,
            output_channels,
            kernel_size=3,
            padding=1,
            stride=1,
            bias=False,
        )
        self.conv = nn.Conv2d(
            output_channels, output_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn1 = nn.GroupNorm(8, output_channels)
        self.bn2 = nn.GroupNorm(8, output_channels)
        self.relu = nn.SiLU()

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, class_embedding: torch.Tensor
    ) -> torch.Tensor:
        batch_size = x.shape[0]
        residual = self.residual_channel_scaling(x)

        h = self.relu(self.bn1(self.channel_scaling(x)))
        conditioning_emb = self.conditioning_projection(
            torch.cat([t, class_embedding], dim=1)
        )
        conditioning_emb = conditioning_emb.view(
            (batch_size, self.output_channels, 1, 1)
        ).expand((batch_size, conditioning_emb.shape[1], h.shape[2], h.shape[3]))

        h = h + conditioning_emb

        h = self.relu(self.bn2(self.conv(h)))

        h = self.size_scaling(h)

        h = h + residual

        return h


class SimpleConditionalTimeEmbUnet(nn.Module):
    def __init__(
        self,
        image_channels: int = 1,
        n_embedding_blocks: int = 3,
        dim_step_size: int = 64,
        time_embedding_dim: int = 64,
        num_classes: int = 10,
        class_embedding_dim: int = 64,
        class_dropout_prob: float = 0.1,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.class_embedding_dim = class_embedding_dim
        self.class_dropout_prob = class_dropout_prob

        self.time_embedder = nn.Sequential(
            PositionEncoding(time_embedding_dim),
            nn.Linear(time_embedding_dim, time_embedding_dim),
            nn.ReLU(),
            nn.Linear(time_embedding_dim, time_embedding_dim),
        )

        self.class_codebook = nn.Embedding(num_classes + 1, class_embedding_dim)
        self.class_embedder = nn.Sequential(
            nn.Linear(class_embedding_dim, class_embedding_dim),
            nn.ReLU(),
            nn.Linear(class_embedding_dim, class_embedding_dim),
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
                    class_embedding_dim,
                    BlockType.Encoder,
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
                    class_embedding_dim,
                    BlockType.Decoder,
                )
                for i in range(n_embedding_blocks)
            ]
        )

        self.output_channel_projection = nn.Conv2d(
            dim_step_size, image_channels, kernel_size=1
        )

    def forward(
        self, x_init: torch.Tensor, t: torch.Tensor, class_labels: torch.Tensor | None
    ) -> torch.Tensor:
        """
        x_init.shape = [B, C, H, W]
        t.shape = [B, 1]
        class_labels.shape = [B,]
        """
        device = x_init.device
        batch_size = x_init.shape[0]
        t_emb = self.time_embedder(t.squeeze(-1))

        null_class_value = self.num_classes
        null_labels = torch.full(
            (batch_size,), null_class_value, device=device, dtype=torch.long
        )
        if class_labels is None:
            class_labels = null_labels
        else:
            if self.training and self.class_dropout_prob > 0.0:
                dropout_mask = (
                    torch.rand(batch_size, device=device) < self.class_dropout_prob
                )
                class_labels = torch.where(dropout_mask, null_labels, class_labels)

        class_emb = self.class_embedder(self.class_codebook(class_labels))

        x = self.initial_channel_projection(x_init)

        layer_residuals: list[torch.Tensor] = []
        for block in self.encoder_blocks:
            x = block(x, t_emb, class_emb)
            layer_residuals.append(x)

        for i, block in enumerate(self.decoder_blocks):
            residual = layer_residuals.pop()
            x = torch.cat((x, residual), dim=1)
            x = block(x, t_emb, class_emb)

        return self.output_channel_projection(x)

    def generate(
        self,
        class_label: int | None,
        n_steps: int = 50,
        classifier_free_guidance_mix: float = 3.0,
    ) -> torch.Tensor:
        if n_steps < 1:
            raise ValueError("n_steps must be at least 1")

        self.eval()
        t_steps = np.linspace(0.0, 1.0, n_steps + 1)
        step_size = t_steps[1] - t_steps[0]

        x_0 = torch.randn((1, 1, 28, 28)).float()
        x_t = x_0
        class_tensor = torch.tensor([class_label]) if class_label is not None else None
        null_class = torch.tensor([self.num_classes])
        for t in torch.from_numpy(t_steps).float():
            t_tensor = t.view(1, 1).float()
            with torch.no_grad():
                unconditional_prediction = self(x_t, t_tensor, null_class)

                if class_tensor is not None:
                    conditional_prediction = self(x_t, t_tensor, class_tensor)
                    pred = unconditional_prediction + classifier_free_guidance_mix * (
                        conditional_prediction - unconditional_prediction
                    )
                else:
                    pred = unconditional_prediction
            x_t += step_size * pred

        return x_t.squeeze(0)

    def step(
        self, x_t: torch.Tensor, t_start: torch.Tensor, t_end: torch.Tensor
    ) -> torch.Tensor:
        t_start = t_start.view(1, 1).expand(x_t.shape[0], 1)
        return x_t + (t_end - t_start) * self(
            x_t + self(x_t, t_start) * (t_end - t_start) / 2,
            t_start + (t_end - t_start) / 2,
        )
