import numpy as np
import torch
from matplotlib import pyplot as plt

from data import EventDataset
from models.flows import create_spline_flow


X = torch.linspace(0.0, 1.0, 100).reshape((-1, 1))

unconstrained_flow = create_spline_flow(10, 1, 32, 64, 4.0)
unconstrained_flow.load_state_dict(torch.load("saved_models/unconstrained_1.pth"))
unconstrained_Y = unconstrained_flow.log_prob(X).exp().detach().numpy().flatten()

constrained_flow = create_spline_flow(10, 1, 32, 64, 4.0)
constrained_flow.load_state_dict(torch.load("saved_models/constrained_0.pth"))
constrained_Y = constrained_flow.log_prob(X).exp().detach().numpy().flatten()

density_difference = constrained_Y - unconstrained_Y
density_difference = np.abs(density_difference)

likelihood_ratio = unconstrained_Y / constrained_Y

# Graph the difference in flow PDFs
X = X.flatten()
plt.plot(X, unconstrained_Y, c="black", label="Unconstrained", alpha=0.5)
plt.plot(X, constrained_Y, c="tab:blue", label="Constrained", alpha=0.5)
plt.plot(X, density_difference, c="tab:red", label="Difference b/w models")
plt.xlabel("Normalized mass")
plt.ylabel("Predicted density")
plt.legend()
plt.show()

# Graph the difference in PDF compared to the true signal histogram
db = EventDataset(
    "data/background.csv",
    "data/signal.csv",
    100_000,
    signal_proportion=0.1,
    normalize=True,
)
signal_db = db.features[db.labels == 1.0]

fig, ax1 = plt.subplots()
ax1.hist(
    signal_db.detach().numpy().flatten(),
    bins=100,
    range=(0.0, 1.0),
    color="black",
    label="Binned signal masses",
    alpha=0.7,
)
ax1.set_ylabel("Signal entries")

ax2 = ax1.twinx()
ax2.plot(X, density_difference, c="tab:red", label="Density difference")
ax2.set_ylabel("Predicted signal density")

fig.legend()
fig.supxlabel("Normalized mass")
plt.show()
