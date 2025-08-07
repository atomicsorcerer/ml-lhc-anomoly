import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
import polars as pl
from settings import RANDOM_SEED


class EventDataset(Dataset):
    def __init__(
        self,
        bg_file_path: str,
        signal_file_path: str,
        limit: int = 10_000,
        signal_proportion: float = 0.5,
        normalize: bool = False,
    ) -> None:
        # Import the CSV files and add a column for background/signal (denoted as 0 or 1, respectively)
        bg_dataset = pl.read_csv(bg_file_path).with_columns(pl.lit(0.0).alias("label"))
        signal_dataset = pl.read_csv(signal_file_path).with_columns(
            pl.lit(1.0).alias("label")
        )

        # Sample the dataset
        if (limit * signal_proportion) % 1 != 0:
            raise ValueError("Limit times the signal proportion must be an integer.")

        signal_dataset = signal_dataset.sample(
            n=int(limit * signal_proportion),
            shuffle=True,
            seed=RANDOM_SEED,
        )
        bg_dataset = bg_dataset.sample(
            n=int(limit * (1 - signal_proportion)),
            shuffle=True,
            seed=RANDOM_SEED,
        )

        dataset = pl.concat((bg_dataset, signal_dataset))

        # Select the necessary columns for training, split dataset into features and labels
        features = dataset.select(
            # ["energy_1", "px_1", "py_1", "pz_1", "energy_2", "px_2", "py_2", "pz_2"]
            ["mass"]
        )
        labels = dataset.select(["label"])

        # Convert dataset type to torch.Tensor and reshape it
        features = features.to_torch().type(torch.float32)
        features = features.reshape((-1, 1))
        if normalize:
            features = features / features.abs().max()
        features.requires_grad_()
        self.features = features

        self.mass = (
            dataset.select(["mass"]).to_torch().type(torch.float32).reshape((-1, 1))
        )
        self.labels = labels.to_torch().type(torch.float32)
        self.dataframe = dataset

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


class ParameterizedBackgroundDataset(Dataset):
    def __init__(
        self,
        start: int,
        stop: int,
        bins: int,
        params: list[tuple[float, float, float, float]],
        noise_scale: float = 1.0,
    ) -> None:
        self.start = start
        self.stop = stop
        self.bins = bins
        self.params = params

        step = (stop - start) / bins
        params_dataset = []
        mass_dataset = []
        for theta_0, theta_1, theta_2, theta_3 in params:
            f = (
                lambda x: theta_0
                * (1 - x) ** theta_1
                * x**theta_2
                * x ** (theta_3 * np.log(x))
            )  # General form of a smooth mass background distribution

            for i in range(bins):
                val = start + (i + 0.5) * step  # Middle of the bin interval
                n_in_bin = round(f(val))
                params_dataset.extend(
                    [theta_0, theta_1, theta_2, theta_3] * n_in_bin
                )  # Append params for each in the smooth distribution function
                mass_dataset.extend(
                    [val] * n_in_bin
                )  # Append 'val' for each in the smooth distribution function

        mass_dataset = torch.Tensor(mass_dataset).reshape(-1, 1)
        entry_noise = (
            (
                torch.rand(
                    mass_dataset.shape,
                    generator=torch.Generator().manual_seed(RANDOM_SEED),
                )
                - 0.5
            )
            * step
            * noise_scale
        )
        self.mass_dataset = mass_dataset + entry_noise
        self.params_dataset = torch.Tensor(params_dataset).reshape(-1, 4)

    def __len__(self) -> int:
        return len(self.mass_dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.params_dataset[idx], self.mass_dataset[idx]


if __name__ == "__main__":
    data_1 = ParameterizedBackgroundDataset(
        0,
        1000,
        100,
        [(1000.0, 0.0, -0.08, 0.0)],
    )
    data_0 = ParameterizedBackgroundDataset(
        0,
        1000,
        100,
        [(1000.0, 0.0, -0.04, 0.0)],
    )
    data_2 = ParameterizedBackgroundDataset(
        0,
        1000,
        100,
        [(1000.0, 0.0, -0.12, 0.0)],
    )
    figure, axis = plt.subplots(1, 3, sharex=True, sharey=True)
    axis[0].hist(
        [data_0.mass_dataset.flatten()],
        bins=100,
        histtype="bar",
        color="tab:red",
        label="$theta_2$=-0.04",
        range=(0, 1000),
    )
    axis[1].hist(
        [data_1.mass_dataset.flatten()],
        bins=100,
        histtype="bar",
        color="tab:blue",
        label="$theta_2$=-0.08",
        range=(0, 1000),
    )
    axis[2].hist(
        [data_2.mass_dataset.flatten()],
        bins=100,
        histtype="bar",
        color="black",
        label="$theta_2$=-0.12",
        range=(0, 1000),
    )
    figure.supxlabel("Mass")
    figure.supylabel("Entries")
    figure.legend()
    plt.show()
