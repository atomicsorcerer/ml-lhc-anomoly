import torch


# Search space
smoothness_penalty_thresholds = torch.linspace(-4, 4, 17)
penalty_factors = torch.tensor([10**i for i in range(-4, 3)])

# Set general training parameters
BATCH_SIZE = 128
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.01
EPOCHS = 10
DATASET_SIZE = 500_000
