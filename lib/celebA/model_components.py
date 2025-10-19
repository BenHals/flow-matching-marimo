import math
from enum import Enum

import torch
import torch.nn as nn


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
    DecoderWithSkip = 1
    Decoder = 2
    Same = 3


class ResidualBlock(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        time_embedding_dim: int,
        class_embedding_dim: int,
        block_type: BlockType,
        group_norm_channels: int = 32,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        print(f"Created {block_type} from {input_channels} to {output_channels}")

        time_embedding_dim = max(0, time_embedding_dim)
        class_embedding_dim = max(0, class_embedding_dim)
        self.has_condition = time_embedding_dim > 0 or class_embedding_dim > 0
        if self.has_condition:
            self.conditioning_projection = nn.Linear(
                time_embedding_dim + class_embedding_dim, output_channels
            )
        else:
            self.conditioning_projection = nn.Identity()

        n_input_channels = input_channels
        if block_type is BlockType.Decoder or block_type is BlockType.DecoderWithSkip:
            # The decoder takes the residual input as well, so input channels is doubled
            if block_type is BlockType.DecoderWithSkip:
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
        elif block_type is BlockType.Encoder:
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
        else:
            self.size_scaling = nn.Identity()
            self.residual_channel_scaling = (
                nn.Identity()
                if input_channels == output_channels
                else nn.Conv2d(
                    input_channels, output_channels, kernel_size=1, stride=1, padding=0
                )
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
        self.bn1 = nn.GroupNorm(group_norm_channels, output_channels)
        self.bn2 = nn.GroupNorm(group_norm_channels, output_channels)
        self.relu = nn.SiLU()

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor | None = None,
        class_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = x.shape[0]
        residual = self.residual_channel_scaling(x)

        h = self.relu(self.bn1(self.channel_scaling(x)))

        if self.has_condition:
            if t is not None and class_embedding is not None:
                conditioning_emb = self.conditioning_projection(
                    torch.cat([t, class_embedding], dim=1)
                )
            elif t is not None:
                conditioning_emb = self.conditioning_projection(t)
            elif class_embedding is not None:
                conditioning_emb = self.conditioning_projection(class_embedding)
            else:
                raise ValueError(
                    "time_embedding_dim or class_embedding_dim was above 0, but no inputs passed"
                )
            conditioning_emb = conditioning_emb.view(
                (batch_size, self.output_channels, 1, 1)
            ).expand((batch_size, conditioning_emb.shape[1], h.shape[2], h.shape[3]))

            h = h + conditioning_emb

        h = self.relu(self.bn2(self.conv(h)))

        h = self.size_scaling(h)

        h = h + residual

        return h
