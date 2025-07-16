import torch
from torch.utils.data import DataLoader, random_split
from torcheval.metrics import BinaryAUROC

import matplotlib.pyplot as plt
import numpy as np

from data import EventDataset
from models.discriminators import GeneralDiscriminator
from models.generators import EventGenerator
from settings import TEST_PROPORTION, RANDOM_SEED


# Set training parameters
BATCH_SIZE = 128
DISCRIMINATOR_LEARNING_RATE = 0.01
GENERATOR_LEARNING_RATE = 0.01
WEIGHT_DECAY = 0.01
EPOCHS = 10
NOISE_DIM = 16

# Import dataset; split and prepare data
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

# Load the models
generator = EventGenerator(noise_dim=NOISE_DIM, output_dim=1)
# Add constrained generator/discriminator here
discriminator = GeneralDiscriminator(input_dim=1)

# Set up training optimizer and loss function
loss_function = torch.nn.BCEWithLogitsLoss()
generator_optimizer = torch.optim.AdamW(
    generator.parameters(), GENERATOR_LEARNING_RATE, weight_decay=WEIGHT_DECAY
)
discriminator_optimizer = torch.optim.AdamW(
    discriminator.parameters(), DISCRIMINATOR_LEARNING_RATE, weight_decay=WEIGHT_DECAY
)

# Configure training metrics
auc = []
acc = []
metric = BinaryAUROC()

# Train the GAN (both generator and discriminator)
for epoch in range(EPOCHS):
    print(f"Epoch {epoch + 1}\n-------------------------------")
    for batch, (real_data, sb_truth) in enumerate(
        train_dataloader
    ):  # sb_truth = actual s/b label (1/0 respectively)
        # Train the discriminator
        discriminator_optimizer.zero_grad()
        real_data_labels = torch.ones((len(real_data), 1))
        real_data_discriminator_outputs = discriminator(real_data)
        real_data_discriminator_loss = loss_function(
            real_data_discriminator_outputs, real_data_labels
        )
        real_data_discriminator_loss.backward()

        noise = torch.randn((len(real_data), NOISE_DIM))
        fake_data = generator(noise)
        fake_data_labels = torch.zeros((len(real_data), 1))
        fake_data_discriminator_outputs = discriminator(fake_data.detach())
        fake_data_discriminator_loss = loss_function(
            fake_data_discriminator_outputs, fake_data_labels
        )
        fake_data_discriminator_loss.backward()
        discriminator_optimizer.step()

        # Train the optimizer
        generator_optimizer.zero_grad()
        fake_data_labels_for_gen = torch.ones((len(real_data), 1))
        fake_data_discrim_outputs_for_gen = discriminator(fake_data)
        generator_loss = loss_function(
            fake_data_discrim_outputs_for_gen, fake_data_labels_for_gen
        )
        generator_loss.backward()
        generator_optimizer.step()

        # Display training metrics
        if batch % 100 == 0:
            print(
                f"Discriminator loss: {real_data_discriminator_loss + fake_data_discriminator_loss}"
            )
            print(f"Generator loss: {generator_loss}")

# Test the models
with torch.no_grad():
    test_samples = 100_000
    bins = 10
    limit = [0, 3000]

    noise = torch.randn(test_samples, NOISE_DIM)
    fake_events = generator(noise).detach().numpy()

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

    discrim_output = discriminator(torch.linspace(0, 3000, 1000).reshape((-1, 1)))
    plt.plot(np.linspace(0, 3000, 1000), discrim_output.detach().numpy())
    figure.supxlabel("Mass")
    figure.supylabel("Discriminator output")
    plt.show()
