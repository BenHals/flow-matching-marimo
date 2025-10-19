import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from lib.celebA.model_components import BlockType, ResidualBlock


class Quantizer(nn.Module):
    def __init__(self, n_embeddings: int, embedding_dim: int):
        super().__init__()
        self.embedding_codebook = nn.Embedding(n_embeddings, embedding_dim)

    def forward(
        self, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, C, H, W = z.shape

        # Get X into the shape B, HxW, C
        # Can think of this as, for each batch item we have HxW tensors of dim C
        # This format is what torch.cdist wants, where the second to last dim is the 'rows', i.e. indexes tensors
        # and the last dim indexes the embedding dim
        x = rearrange(z, "b c h w -> b (h w) c")

        # Calculate this distance between each pair of tensors in the input and output
        # This will have shape B (HxW) E
        # i.e. for each batch, the complete set of distances between each pair of vectors in the input (HxW) and codebook (E)
        # First we need to expand the codebook across the batch dim
        batch_expanded_codebook = self.embedding_codebook.weight.unsqueeze(0).expand(
            (z.shape[0], *self.embedding_codebook.weight.shape)
        )
        distances = torch.cdist(x, batch_expanded_codebook)
        # For each vector in the input (HxW), find the index of the minimum distance pair with E
        nearest_neighbors = torch.argmin(distances, dim=-1)

        # For each tensor in the input (B (HxW)), get the closest vector in the codebook
        # This returns a flattened list of shape: BxHxW C
        flattened_quantized_z = torch.index_select(
            self.embedding_codebook.weight, 0, nearest_neighbors.view(-1)
        )

        flattened_z = rearrange(z, "b c h w -> (b h w) c")

        # Loss on the embedding network, pushing z close to the codebook
        # Detech codebook vectors, so loss doesn't flow
        commitment_loss = torch.mean(
            (flattened_quantized_z.detach() - flattened_z) ** 2
        )

        # Loss on the codebook vectors, pushing closer to z
        # Detach embeddings so loss doesn't flow
        codebook_loss = torch.mean((flattened_quantized_z - flattened_z.detach()) ** 2)

        # This just equals the quantized z, but by doing it this way, the loss on z is
        # transfered directly from the loss on the decoder input, which is what we want.
        flattened_z_out = flattened_z + (flattened_quantized_z - flattened_z).detach()

        # Rearrange from
        # BxHxW C to B H W C then to B C H W
        z_out = flattened_z_out.reshape((B, H, W, C)).permute((0, 3, 1, 2))
        return z_out, commitment_loss, codebook_loss


class VQVAEAutoEncoder(nn.Module):
    def __init__(
        self,
        image_channels: int = 1,
        image_resolution: int = 32,
        n_embedding_blocks: int = 3,
        n_middle_blocks: int = 2,
        z_dim: int = 32,
        dim_step_size: int = 64,
        commitment_loss_weight: float = 0.2,
        n_codebook_embeddings: int = 256,
        group_norm_channels: int = 32,
    ) -> None:
        super().__init__()
        self.commitment_loss_weight = commitment_loss_weight

        self.initial_channel_projection = nn.Conv2d(
            image_channels, dim_step_size, kernel_size=3, padding=1
        )
        self.encoder_blocks = nn.ModuleList(
            [
                ResidualBlock(
                    dim_step_size * 2**i,
                    dim_step_size * 2 ** (i + 1),
                    -1,
                    -1,
                    BlockType.Encoder,
                    group_norm_channels=group_norm_channels,
                )
                for i in range(n_embedding_blocks)
            ]
        )

        middle_block_dim = dim_step_size * 2**n_embedding_blocks

        self.middle_blocks_enc = nn.ModuleList(
            [
                ResidualBlock(
                    middle_block_dim,
                    middle_block_dim,
                    -1,
                    -1,
                    BlockType.Same,
                    group_norm_channels=group_norm_channels,
                )
                for _ in range(n_middle_blocks)
            ]
        )

        self.encoder_norm = nn.GroupNorm(group_norm_channels, middle_block_dim)
        self.z_out = nn.Conv2d(
            middle_block_dim, z_dim, kernel_size=3, stride=1, padding=1
        )
        self.pre_quantization_conv = nn.Conv2d(
            z_dim, z_dim, kernel_size=3, stride=1, padding=1
        )
        self.quantizer = Quantizer(n_codebook_embeddings, z_dim)

        self.post_quantization_conv = nn.Conv2d(
            z_dim, z_dim, kernel_size=3, stride=1, padding=1
        )
        self.z_in = nn.Conv2d(
            z_dim, middle_block_dim, kernel_size=3, stride=1, padding=1
        )

        self.middle_blocks_dec = nn.ModuleList(
            [
                ResidualBlock(
                    middle_block_dim,
                    middle_block_dim,
                    -1,
                    -1,
                    BlockType.Same,
                    group_norm_channels=group_norm_channels,
                )
                for _ in range(n_middle_blocks)
            ]
        )

        self.decoder_blocks = nn.ModuleList(
            [
                ResidualBlock(
                    dim_step_size * 2 ** (n_embedding_blocks - (i)),
                    dim_step_size * 2 ** (n_embedding_blocks - (i + 1)),
                    -1,
                    -1,
                    BlockType.Decoder,
                    group_norm_channels=group_norm_channels,
                )
                for i in range(n_embedding_blocks)
            ]
        )

        self.decoder_norm = nn.GroupNorm(group_norm_channels, dim_step_size)
        self.output_channel_projection = nn.Conv2d(
            dim_step_size, image_channels, kernel_size=3, padding=1
        )

    def forward_encode(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.initial_channel_projection(x)
        for b in self.encoder_blocks:
            x = b(x)
        for b in self.middle_blocks_enc:
            x = b(x)

        x = F.relu(self.encoder_norm(x))
        z = self.z_out(x)
        z = self.pre_quantization_conv(z)
        (
            zq,
            commitment_loss,
            codebook_loss,
        ) = self.quantizer(z)
        return zq, commitment_loss, codebook_loss

    def forward_encode_sample(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.forward_encode(x)

    def forward_decode(self, z: torch.Tensor) -> torch.Tensor:
        z = self.post_quantization_conv(z)
        x = self.z_in(z)
        for b in self.middle_blocks_dec:
            x = b(x)
        for b in self.decoder_blocks:
            x = b(x)
        x = F.relu(self.decoder_norm(x))
        return self.output_channel_projection(x)

    def forward_decode_quantize(self, z: torch.Tensor) -> torch.Tensor:
        (
            zq,
            commitment_loss,
            codebook_loss,
        ) = self.quantizer(z)
        return self.forward_decode(zq)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z, commitment_loss, codebook_loss = self.forward_encode(x)
        loss = codebook_loss + self.commitment_loss_weight * commitment_loss
        x_out = self.forward_decode(z)
        return x_out, loss
