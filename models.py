import torch
import torch.nn as nn


class Network(nn.Module):
    def __init__(self, state_dim: int, action_dim: int) -> None:
        super(Network, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)
