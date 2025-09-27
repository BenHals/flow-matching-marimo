import matplotlib.pyplot as plt
import torch
import torch.nn.functional
import torch.optim
from loguru import logger
from sklearn.datasets import make_moons

import data
from model import FlowModelSimple


def train():
    logger.info("Running Training!")
    ds = data.get_spiral_dataloader(n_samples=1000000 // 2048, batch_size=5096 * 2)
    # ds = data.get_moon_dataloader(n_samples=10000, batch_size=256)

    u = FlowModelSimple()
    u = u.to(device="cuda")
    load = True
    if not load:
        optimizer = torch.optim.Adam(u.parameters(), 1e-4)

        epoch_losses = []
        for epoch in range(1024):
            losses = []
            for batch in ds:
                x_0 = torch.randn_like(batch)
                t = torch.rand(batch.shape[0], 1)
                x_t = (1 - t) * x_0 + t * batch

                v = batch - x_0

                optimizer.zero_grad()
                p = u(x_t.to(device="cuda"), t.to(device="cuda"))
                loss = torch.nn.functional.mse_loss(p, v.to(device="cuda"))
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            epoch_loss = sum(losses) / len(losses)
            epoch_losses.append(epoch_loss)
            logger.info(f"Epoch {epoch} - loss: {epoch_loss}")

        torch.save(u.state_dict(), "model.pth")
        plt.plot(epoch_losses)
        plt.show()
    else:
        u.load_state_dict(torch.load("model.pth", weights_only=True))

    u.eval()

    u = u.to(device="cpu")
    samples = [u.generate() for _ in range(1000)]
    plt.scatter([i[0].item() for i in samples], [i[1].item() for i in samples])
    plt.show()

    x = torch.randn(300, 2)
    n_steps = 8
    fig, axes = plt.subplots(1, n_steps + 1, figsize=(30, 4), sharex=True, sharey=True)
    time_steps = torch.linspace(0, 1.0, n_steps + 1)

    axes[0].scatter(x.detach()[:, 0], x.detach()[:, 1], s=10)
    axes[0].set_title(f"t = {time_steps[0]:.2f}")
    axes[0].set_xlim(-3.0, 3.0)
    axes[0].set_ylim(-3.0, 3.0)

    for i in range(n_steps):
        x = u.step(x, time_steps[i], time_steps[i + 1])
        axes[i + 1].scatter(x.detach()[:, 0], x.detach()[:, 1], s=10)
        axes[i + 1].set_title(f"t = {time_steps[i + 1]:.2f}")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    train()
