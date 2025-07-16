import torch


class EventGenerator(torch.nn.Module):
    def __init__(self, noise_dim: int, output_dim: int):
        super(EventGenerator, self).__init__()
        self.noise_dim = noise_dim
        self.network = torch.nn.Sequential(
            torch.nn.Linear(noise_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, output_dim),
            torch.nn.ReLU(),
        )

    def forward(self, x) -> torch.Tensor:
        return self.network(x)


class ConstrainedGenerator(torch.nn.Module):
    def __init__(self, param_dim: int, noise_dim: int, output_dim: int) -> None:
        super(ConstrainedGenerator, self).__init__()
        self.noise_dim = noise_dim
        self.network = torch.nn.Sequential(
            torch.nn.Linear(param_dim + noise_dim, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(128),
            torch.nn.Linear(128, output_dim),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(output_dim),
        )

    def forward(self, x) -> torch.Tensor:
        return self.network(x)
