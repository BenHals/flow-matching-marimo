import pathlib

import torch

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
