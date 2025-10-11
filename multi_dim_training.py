import time
import sys

from torch.utils.data import DataLoader, random_split

from nflows.transforms.autoregressive import *
import matplotlib.pyplot as plt
import polars as pl

from data.event_dataset import EventDataset
from utils.loss import (
    calculate_first_order_non_smoothness_penalty,
    calculate_outlier_gradient_penalty,
    calculate_outlier_gradient_penalty_with_preprocess,
    calculate_second_order_non_smoothness_penalty,
    calculate_outlier_gradient_penalty_with_preprocess_mod_z_scores,
)
from models.flows import create_spline_flow
from settings import TEST_PROPORTION, RANDOM_SEED


# Set training parameters
BATCH_SIZE = int(sys.argv[1])
LEARNING_RATE = float(sys.argv[2])
WEIGHT_DECAY = float(sys.argv[3])
EPOCHS = int(sys.argv[4])
DATASET_SIZE = int(sys.argv[5])
SIGNAL_PROPORTION = float(sys.argv[6])

SMOOTHNESS_PENALTY_FACTOR = float(sys.argv[7])

# Information from pre-processing
settings = pl.read_csv("pre_process_results/1d_unconstrained_full.csv")
GRAD_MEDIAN = settings.get_column("first_order_median").item()
GRAD_MAD = settings.get_column("first_order_mad").item()

# Prepare dataset
data = EventDataset(
    "data/background.csv",
    "data/signal.csv",
    ["mass"],
    DATASET_SIZE,
    SIGNAL_PROPORTION,
    mass_region=(500.0, None),
    normalize=True,
    norm_type="one_dim",
)
train_data, test_data = random_split(
    data,
    [1 - TEST_PROPORTION, TEST_PROPORTION],
    generator=torch.Generator().manual_seed(RANDOM_SEED),
)
train_dataloader = DataLoader(train_data, BATCH_SIZE)
test_dataloader = DataLoader(test_data, BATCH_SIZE)

# Load and prepare model
flow = create_spline_flow(10, 1, 32, 64, 4.0)

# Set up training optimizer
optimizer = torch.optim.AdamW(
    flow.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
)

# Establish performance metrics
loss_per_epoch = []

# Train the flow
for epoch in range(EPOCHS):
    print(f"Epoch {epoch + 1}\n-------------------------------")
    for batch, (X, sb_truth) in enumerate(
        train_dataloader
    ):  # sb_truth = actual s/b label (1/0 respectively)
        optimizer.zero_grad()

        # Calculate the log probability and the log absolute determinant
        z, logabsdet = flow._transform.forward(X)
        log_prob_z = flow._distribution.log_prob(z)
        log_prob = log_prob_z + logabsdet

        # Calculate negative log likelihood for distribution
        loss = -log_prob.mean()

        # Calculate un-smoothness penalty
        if SMOOTHNESS_PENALTY_FACTOR > 0.0:
            loss += calculate_outlier_gradient_penalty_with_preprocess_mod_z_scores(
                log_prob,
                X,
                GRAD_MEDIAN,
                GRAD_MAD,
                loss_cut_off=0.0,
                alpha=SMOOTHNESS_PENALTY_FACTOR,
            )

        loss.backward()
        optimizer.step()

        # Display training metrics
        if batch % 100 == 0:
            print(f"{batch} - loss: {loss}")

    with torch.no_grad():
        test_loss = 0
        for X, y in test_dataloader:
            loss = -flow.log_prob(X).mean()
            test_loss += loss / len(test_dataloader)

        loss_per_epoch.append(test_loss)
        print(f"Testing loss: {test_loss}")

# Plot the loss over time
plt.plot(loss_per_epoch, color="tab:blue")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

# Save model
model_save_name = input("Saved model file name [time/date]: ")
if model_save_name.strip() == "":
    model_save_name = round(time.time())
torch.save(flow.state_dict(), f"cluster/saved_models_cluster/{model_save_name}.pth")
