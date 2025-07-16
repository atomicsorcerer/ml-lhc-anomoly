import torch


class GeneralDiscriminator(torch.nn.Module):
    def __init__(self, input_dim: int) -> None:
        super(GeneralDiscriminator, self).__init__()
        self.input_dim = input_dim
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1),
        )

    def forward(self, x) -> torch.Tensor:
        return self.network(x)


class ConstrainedDiscriminator(torch.nn.Module):
    def __init__(self, param_dim: int, input_dim: int) -> None:
        super(ConstrainedDiscriminator, self).__init__()
        self.input_dim = input_dim
        self.param_dim = param_dim
        self.network = torch.nn.Sequential(
            torch.nn.Linear(param_dim + input_dim, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1),
            torch.nn.BatchNorm1d(1),
        )

    def forward(self, x) -> torch.Tensor:
        return self.network(x)
