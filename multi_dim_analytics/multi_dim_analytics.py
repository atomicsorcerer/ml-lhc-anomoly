import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn import metrics

from data import EventDataset
from models.flows import create_spline_flow

db = EventDataset(
    "../data/background.csv",
    "data/signal.csv",
    ["energy_1", "px_1", "py_1", "pz_1", "energy_2", "px_2", "py_2", "pz_2"],
    100_000,
    signal_proportion=0.1,
    normalize=True,
)

unconstrained_flow = create_spline_flow(10, 8, 32, 64, 4.0)
unconstrained_flow.load_state_dict(
    torch.load("../saved_models_multi_dim/unconstrained_0.pth")
)
s_and_bg_densities = unconstrained_flow.log_prob(db.features.detach()).exp()

constrained_flow = create_spline_flow(10, 8, 32, 64, 4.0)
constrained_flow.load_state_dict(torch.load("../saved_models_multi_dim/constrained_0.pth"))
bg_densities = constrained_flow.log_prob(db.features.detach()).exp()

likelihood_ratios = s_and_bg_densities / bg_densities
likelihood_ratios = likelihood_ratios.detach().numpy()
signal_likelihood_ratios = likelihood_ratios[db.labels.flatten() == 1.0]
bg_likelihood_ratios = likelihood_ratios[db.labels.flatten() == 0.0]

fpr, tpr, thresholds = metrics.roc_curve(db.labels.detach().numpy(), likelihood_ratios)

plt.plot(fpr, tpr, label="Our model")
plt.plot(
    np.linspace(0.0, 1.0, 100), np.linspace(0.0, 1.0, 100), label="Random classifier"
)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()
