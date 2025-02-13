import torch
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
            ["energy_1", "px_1", "py_1", "pz_1", "energy_2", "px_2", "py_2", "pz_2"]
        )
        labels = dataset.select(["label"])

        # Convert dataset type to torch.Tensor and reshape it
        features = features.to_torch().type(torch.float32)
        self.features = features.reshape((-1, 2, 4))
        self.labels = labels.to_torch().type(torch.float32)
        self.dataframe = dataset

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]
