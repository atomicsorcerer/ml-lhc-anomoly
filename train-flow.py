import torch
from torch.utils.data import DataLoader, random_split
from torcheval.metrics import BinaryAUROC

from nflows.flows.base import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import *
from nflows.transforms.permutations import ReversePermutation

import matplotlib.pyplot as plt
import numpy as np

from data import EventDataset
from models.discriminators import GeneralDiscriminator
from models.generators import EventGenerator
from settings import TEST_PROPORTION, RANDOM_SEED


# Set training parameters
BATCH_SIZE = 128
LEARNING_RATE = 0.0001
EPOCHS = 5

# Prepare dataset
data = EventDataset(
    "data/background.csv", "data/signal.csv", 100_000, signal_proportion=0.01
)
train_data, test_data = random_split(
    data,
    [1 - TEST_PROPORTION, TEST_PROPORTION],
    generator=torch.Generator().manual_seed(RANDOM_SEED),
)
train_dataloader = DataLoader(train_data, BATCH_SIZE)
test_dataloader = DataLoader(test_data, BATCH_SIZE)

# Load and prepare model
transforms = [
    MaskedAffineAutoregressiveTransform(
        features=1, hidden_features=64, use_batch_norm=True
    ),
    # ReversePermutation(features=1),
    MaskedAffineAutoregressiveTransform(
        features=1, hidden_features=64, use_batch_norm=True
    ),
    # ReversePermutation(features=1),
    MaskedAffineAutoregressiveTransform(
        features=1, hidden_features=64, use_batch_norm=True
    ),
    # ReversePermutation(features=1),
    MaskedAffineAutoregressiveTransform(
        features=1, hidden_features=64, use_batch_norm=True
    ),
    # ReversePermutation(features=1),
    MaskedAffineAutoregressiveTransform(
        features=1, hidden_features=64, use_batch_norm=True
    ),
    # ReversePermutation(features=1),
]
composite_transform = CompositeTransform(transforms)

base_dist = StandardNormal(shape=[1])

flow = Flow(composite_transform, base_dist)

# Set up training optimizer
optimizer = torch.optim.Adam(flow.parameters(), lr=LEARNING_RATE)

# Train the flow
for epoch in range(EPOCHS):
    print(f"Epoch {epoch + 1}\n-------------------------------")
    for batch, (X, sb_truth) in enumerate(
        train_dataloader
    ):  # sb_truth = actual s/b label (1/0 respectively)
        optimizer.zero_grad()

        negative_log_likelihood = -flow.log_prob(X)

        loss = negative_log_likelihood.mean()
        loss.backward()
        optimizer.step()

        # Display training metrics
        if batch % 100 == 0:
            print(f"{batch} - Loss: {loss}")

# Test the flow
with torch.no_grad():
    test_samples = 100_000
    bins = 50
    # limit = [0, 3000]

    fake_events = flow.sample(test_samples).numpy()

    figure, axis = plt.subplots(1, 2, sharex=True, sharey=True)
    axis[0].hist(
        [data.features.flatten()],
        bins=bins,
        histtype="bar",
        color="black",
        label="Original",
        # range=limit,
    )
    axis[1].hist(
        [fake_events.flatten()],
        bins=bins,
        histtype="bar",
        color="tab:red",
        label="Generated",
        # range=limit,
    )
    figure.supxlabel("Mass")
    figure.supylabel("Entries")
    figure.legend()
    plt.show()
