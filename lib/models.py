import pathlib

import torch

from lib.celebA.unet import SimpleTimeEmbUnet
from lib.celebA.vqvae import VQVAEAutoEncoder
from lib.mnist.unet_conditional import SimpleConditionalTimeEmbUnet
from lib.spiral_model.model import FlowModelSimple


def load_pretrained_spiral_model():
    base_path = pathlib.Path("lib/spiral_model")
    weights_path = base_path / "model.pth"

    model = FlowModelSimple()
    model.load_state_dict(torch.load(str(weights_path), weights_only=True))
    model.to("cpu")
    model.eval()
    return model


def load_pretrained_mnist_model():
    base_path = pathlib.Path("lib/mnist")
    weights_path = base_path / "model.pth"
    model = SimpleConditionalTimeEmbUnet(n_embedding_blocks=2, dim_step_size=32)
    model.load_state_dict(torch.load(str(weights_path), weights_only=True))
    model.to("cpu")
    model.eval()
    return model


def load_pretrained_celebA_model():
    base_path = pathlib.Path("lib/celebA")
    base_latent_model_weights_path = base_path / "vqvae-small-model-celebA-data-5.pth"
    base_flow_model_weights_path = base_path / "mnist_latent-model-12.pth"

    latent_encoder = VQVAEAutoEncoder(
        n_embedding_blocks=4,
        dim_step_size=16,
        z_dim=4,
        image_channels=3,
        n_codebook_embeddings=8192,
        group_norm_channels=16,
    )
    latent_encoder.load_state_dict(
        torch.load(base_latent_model_weights_path, weights_only=True)
    )

    model = SimpleTimeEmbUnet(n_embedding_blocks=2, dim_step_size=64, image_channels=4)
    model.load_state_dict(torch.load(base_flow_model_weights_path, weights_only=True))
    model.to("cpu")
    model.eval()
    latent_encoder.to("cpu")
    latent_encoder.eval()

    return latent_encoder, model
