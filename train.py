import torch
from torch.utils.data import DataLoader, random_split
from torcheval.metrics import BinaryAUROC

import matplotlib.pyplot as plt
import numpy as np

from data import EventDataset
from models import ParticleFlowNetwork
from utils import train, test
from settings import TEST_PROPORTION, RANDOM_SEED


# Set training parameters
BATCH_SIZE = 64
LEARNING_RATE = 0.01
WEIGHT_DECAY = 0.01
EPOCHS = 10

# Import dataset; split and prepare data
data = EventDataset(
    "data/background.csv", "data/signal.csv", 10_000, signal_proportion=0.01
)
train_data, test_data = random_split(
    data,
    [1 - TEST_PROPORTION, TEST_PROPORTION],
    generator=torch.Generator().manual_seed(RANDOM_SEED),
)
train_dataloader = DataLoader(train_data, BATCH_SIZE)
test_dataloader = DataLoader(test_data, BATCH_SIZE)

# Load the model
model = ParticleFlowNetwork(3, [64, 64], [64, 64])

# Set up training optimizer and loss function
loss_function = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(
    model.parameters(), LEARNING_RATE, weight_decay=WEIGHT_DECAY
)

# Configure training metrics
auc = []
acc = []
metric = BinaryAUROC()

# Train the model
for i in range(EPOCHS):
    print(f"Epoch {i + 1}\n-------------------------------")
    train(train_dataloader, model, loss_function, optimizer, True)
    loss, acc, auc_metric = test(test_dataloader, model, loss_function, metric, True)

# Graph the latent space representation with the prediction
X = model.p_map(data.features).detach().numpy()
Y = model(data.features)
Y = torch.nn.functional.sigmoid(Y).flatten().detach().numpy()

fig = plt.figure()
ax = fig.add_subplot(projection="3d")

p = ax.scatter(X[..., 0], X[..., 1], X[..., 2], c=Y, cmap="viridis")
fig.colorbar(p)
plt.show()
