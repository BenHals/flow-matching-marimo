import numpy as np
import torch
import torch.functional as f
import torch.nn


class FlowModelSimple(torch.nn.Module):
    def __init__(self, d: int = 2, h: int = 64):
        super().__init__()
        self.d = d
        self.h = h
        self.net = torch.nn.Sequential(
            *[
                torch.nn.Linear(d + 1, h),
                torch.nn.ReLU(),
                torch.nn.Linear(h, h),
                torch.nn.ReLU(),
                torch.nn.Linear(h, h),
                torch.nn.ReLU(),
                torch.nn.Linear(h, d),
            ]
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((x, t), dim=-1))

    def generate(self, n_steps: int = 50) -> torch.Tensor:
        if n_steps < 1:
            raise ValueError("n_steps must be at least 1")
        t_steps = np.linspace(0.0, 1.0, n_steps + 1)
        step_size = t_steps[1] - t_steps[0]

        x_0 = torch.randn(self.d)
        x_t = x_0
        for t in torch.from_numpy(t_steps):
            x_t += step_size * self(
                x_t.unsqueeze(0).float(), t.unsqueeze(0).unsqueeze(0).float()
            ).squeeze(0)

        return x_t

    def step(
        self, x_t: torch.Tensor, t_start: torch.Tensor, t_end: torch.Tensor
    ) -> torch.Tensor:
        t_start = t_start.view(1, 1).expand(x_t.shape[0], 1)
        return x_t + (t_end - t_start) * self(
            x_t + self(x_t, t_start) * (t_end - t_start) / 2,
            t_start + (t_end - t_start) / 2,
        )
