import math

import torch
from sklearn.datasets import make_moons
from torch.utils.data import DataLoader, Dataset


class SpiralDataSet(Dataset):
    def __init__(self, n_arms: int, n_samples: int, batch_size: int) -> None:
        self.n_arms = n_arms
        self.n_samples = n_samples
        self.batch_size = batch_size

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        t = torch.rand(self.batch_size)
        x = torch.sin(t * 2 * math.pi * self.n_arms) * t
        y = torch.cos(t * 2 * math.pi * self.n_arms) * t
        r = torch.stack((x, y), dim=-1)

        return r


class MoonDataSet(Dataset):
    def __init__(self, n_samples: int) -> None:
        self.n_samples = n_samples
        self.data = torch.Tensor(make_moons(n_samples, noise=0.05)[0])

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        idx = int(torch.randint(0, self.n_samples - 1, size=(1,))[0])
        r = self.data[idx]

        return r


def get_spiral_dataloader(
    n_arms: int = 3, n_samples: int = 1000, batch_size: int = 8
) -> DataLoader:
    return DataLoader(
        SpiralDataSet(n_arms=n_arms, n_samples=n_samples, batch_size=batch_size),
        batch_size=None,
    )


def get_moon_dataloader(n_samples: int = 1000, batch_size: int = 8) -> DataLoader:
    return DataLoader(MoonDataSet(n_samples=n_samples), batch_size=batch_size)
