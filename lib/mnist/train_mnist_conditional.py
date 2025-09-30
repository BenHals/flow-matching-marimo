import functools

import matplotlib.pyplot as plt
import torch
import torch.nn.functional
import torch.optim
import torchvision.datasets
import torchvision.transforms
from loguru import logger
from prettytable import PrettyTable

from unet_conditional import SimpleConditionalTimeEmbUnet


def count_parameters_by_layer(model):
    """
    Counts and displays the number of trainable parameters for each layer
    in a PyTorch model.
    """
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Parameters: {total_params}")
    return total_params


def train():
    logger.info("Running Training!")
    train_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
        ]
    )

    mnist = torchvision.datasets.MNIST(
        train=True, download=True, root="data", transform=train_transform
    )
    ds = torch.utils.data.DataLoader(
        mnist, batch_size=512, shuffle=True, num_workers=8, persistent_workers=True
    )
    model_name = "mnist_conditional"
    version = 6

    # u = UnetFlowModelSimple(h=96)
    u = SimpleConditionalTimeEmbUnet(n_embedding_blocks=2, dim_step_size=96)
    u.to(device="cuda")
    load = True
    count_parameters_by_layer(u)
    if not load:
        optimizer = torch.optim.Adam(u.parameters(), 1e-4)

        epoch_losses = []
        for epoch in range(25):
            losses = []
            for batch, label in ds:
                x_0 = torch.randn_like(batch)
                t_values = torch.rand(batch.shape[0], 1)
                t = t_values.view(
                    [batch.shape[0], *[1 for _ in range(len(batch.shape) - 1)]]
                ).expand(batch.shape)
                x_t = (1 - t) * x_0 + t * batch

                v = batch - x_0

                optimizer.zero_grad()
                p = u(
                    x_t.to(device="cuda"),
                    t_values.to(device="cuda"),
                    label.to(device="cuda"),
                )
                p = p.reshape(v.shape)
                loss = torch.nn.functional.mse_loss(p, v.to(device="cuda"))
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            epoch_loss = sum(losses) / len(losses)
            epoch_losses.append(epoch_loss)
            logger.info(f"Epoch {epoch} - loss: {epoch_loss}")

        torch.save(u.state_dict(), f"{model_name}-model-{version}.pth")
        plt.plot(epoch_losses)
        plt.show()
    else:
        u.load_state_dict(
            torch.load(f"{model_name}-model-{version}.pth", weights_only=True)
        )

    u.eval()

    u.to(device="cpu")

    for c in [*range(10), None]:
        x_out = u.generate(n_steps=50, class_label=c).detach()

        plt.imshow(x_out.reshape((28, 28)), cmap="gray")
        plt.show()


if __name__ == "__main__":
    train()
