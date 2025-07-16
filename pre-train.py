import torch
from torch.utils.data import DataLoader, random_split
from torcheval.metrics import BinaryAUROC

import matplotlib.pyplot as plt
import numpy as np

from data.event_dataset import ParameterizedBackgroundDataset
from models.discriminators import ConstrainedDiscriminator
from models.generators import ConstrainedGenerator
from settings import TEST_PROPORTION, RANDOM_SEED


# Set training parameters
BATCH_SIZE = 128
DISCRIMINATOR_LEARNING_RATE = 0.005
GENERATOR_LEARNING_RATE = 0.005
WEIGHT_DECAY = 0.001
EPOCHS = 5
NOISE_DIM = 64

# Import dataset; split and prepare data
data = ParameterizedBackgroundDataset(
    0,
    1000,
    100,
    [
        (1000.0, 0.0, -0.5, 0.0),
        (1000.0, 0.0, -0.04, 0.0),
        (1000.0, 0.0, -0.08, 0.0),
        (1000.0, 0.0, -0.16, 0.0),
    ],
)
train_dataloader = DataLoader(data, BATCH_SIZE, shuffle=True)

# Load the models
generator = ConstrainedGenerator(param_dim=4, noise_dim=NOISE_DIM, output_dim=1)
discriminator = ConstrainedDiscriminator(param_dim=4, input_dim=1)

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

# Train the constrained GAN (both generator and discriminator)
for epoch in range(EPOCHS):
    print(f"Epoch {epoch + 1}\n-------------------------------")
    for batch, (params, real_masses) in enumerate(train_dataloader):
        # Train the discriminator
        concat_real_data = torch.concat([params, real_masses], dim=1)
        discriminator_optimizer.zero_grad()
        real_data_labels = torch.ones((len(real_masses), 1))
        real_data_discriminator_outputs = discriminator(concat_real_data)
        real_data_discriminator_loss = loss_function(
            real_data_discriminator_outputs, real_data_labels
        )
        real_data_discriminator_loss.backward()

        noise = torch.randn((len(real_masses), NOISE_DIM))
        concat_fake_input = torch.concat([params, noise], dim=1)
        fake_data = generator(concat_fake_input)
        fake_data_labels = torch.zeros((len(real_masses), 1))
        concat_fake_data = torch.concat([params, fake_data], dim=1)
        fake_data_discriminator_outputs = discriminator(concat_fake_data.detach())
        fake_data_discriminator_loss = loss_function(
            fake_data_discriminator_outputs, fake_data_labels
        )
        fake_data_discriminator_loss.backward()
        discriminator_optimizer.step()

        # Train the optimizer
        generator_optimizer.zero_grad()
        fake_data_labels_for_gen = torch.ones((len(real_masses), 1))
        fake_data_discrim_outputs_for_gen = discriminator(concat_fake_data)
        generator_loss = loss_function(
            fake_data_discrim_outputs_for_gen, fake_data_labels_for_gen
        )
        generator_loss.backward()
        generator_optimizer.step()

        # Display training metrics
        if batch % 100 == 0:
            print(
                f"{batch} - Discrim loss: {real_data_discriminator_loss + fake_data_discriminator_loss}, Generator loss: {generator_loss}"
            )

# Test the models
with torch.no_grad():
    test_samples = 100_000
    bins = 25
    limit = [0, 250]

    params = torch.Tensor([1000.0, 0.0, -0.2, 0.0] * test_samples).reshape(-1, 4)
    params_2 = torch.Tensor([1000.0, 0.0, -0.04, 0.0] * test_samples).reshape(-1, 4)
    noise = torch.randn(test_samples, NOISE_DIM)
    noise_2 = torch.randn(test_samples, NOISE_DIM)
    fake_events_0 = generator(torch.concat([params, noise], dim=1)).detach().numpy()
    fake_events_1 = generator(torch.concat([params_2, noise_2], dim=1)).detach().numpy()

    figure, axis = plt.subplots(1, 2, sharex=True, sharey=True)
    axis[0].hist(
        [fake_events_0.flatten()],
        bins=bins,
        histtype="bar",
        color="black",
        label="$theta_2$=-0.20",
        range=limit,
    )
    axis[1].hist(
        [fake_events_1.flatten()],
        bins=bins,
        histtype="bar",
        color="tab:red",
        label="$theta_2$=-0.04",
        range=limit,
    )
    figure.supxlabel("Mass")
    figure.supylabel("Entries")
    figure.legend()
    plt.show()
