import time

from torch.utils.data import DataLoader, random_split

from nflows.transforms.autoregressive import *

import matplotlib.pyplot as plt

from data import EventDataset
from utils.loss import calculate_non_smoothness_penalty_1d
from models.flows import create_spline_flow
from settings import TEST_PROPORTION, RANDOM_SEED


# Set training parameters
BATCH_SIZE = 128
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 0.01
EPOCHS = 20
DATASET_SIZE = 500_000
SIGNAL_PROPORTION = 0.02

SMOOTHNESS_PENALTY_FACTOR = 0.001
SMOOTHNESS_PENALTY_N_KNOTS = 500

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

        # Calculate log_likelihood for distribution
        loss = -flow.log_prob(X).mean()

        # Calculate un-smoothness penalty
        if SMOOTHNESS_PENALTY_FACTOR > 0.0:
            smoothness_penalty = calculate_non_smoothness_penalty_1d(
                flow, -4.0, 4.0, SMOOTHNESS_PENALTY_N_KNOTS, SMOOTHNESS_PENALTY_FACTOR
            )
            loss += smoothness_penalty

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

# Test the flow's generation
with torch.no_grad():
    bins = 100
    limit = [-4, 4]

    fake_events = flow.sample(DATASET_SIZE)

    figure, axis = plt.subplots(1, 2, sharex=True, sharey=True)
    axis[0].hist(
        [data.features.flatten()],
        bins=bins,
        histtype="bar",
        color="black",
        label="Original",
        range=limit,
    )
    axis[1].hist(
        [fake_events.flatten()],
        bins=bins,
        histtype="bar",
        color="tab:red",
        label="Generated",
        range=limit,
    )
    figure.supxlabel("Mass")
    figure.supylabel("Entries")
    figure.legend()
    plt.show()

    X = torch.linspace(-4.0, 4.0, 100).reshape((-1, 1))
    Y = flow.log_prob(X).exp()
    plt.plot(X.numpy().flatten(), Y.numpy().flatten())
    plt.xlabel("Normalized mass")
    plt.ylabel("Predicted density")
    plt.show()

# Save model
model_save_name = input("Saved model file name [time/date]: ")
if model_save_name.strip() == "":
    model_save_name = round(time.time())
torch.save(flow.state_dict(), f"saved_models_1d/{model_save_name}.pth")
