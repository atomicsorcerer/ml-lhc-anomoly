import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn import metrics

from data import EventDataset
from models.flows import create_spline_flow


db = EventDataset(
    "../data/background.csv",
    "../data/signal.csv",
    ["mass"],
    500_000,
    0.02,
    mass_region=(500.0, None),
    normalize=True,
    norm_type="one_dim",
)
unconstrained_flow = create_spline_flow(10, 1, 32, 64, 4.0)
unconstrained_flow.load_state_dict(
    torch.load("../saved_models_1d/unconstrained_large_1.pth")
)
s_and_bg_densities = unconstrained_flow.log_prob(db.features.detach()).exp()

constrained_flow = create_spline_flow(10, 1, 32, 64, 4.0)
constrained_flow.load_state_dict(
    torch.load("../saved_models_1d/constrained_500_knots_s0001.pth")
)
bg_densities = constrained_flow.log_prob(db.features.detach()).exp()

likelihood_ratios = s_and_bg_densities / bg_densities
likelihood_ratios = likelihood_ratios.detach().numpy()
likelihood_ratios = likelihood_ratios / likelihood_ratios.max()

fpr, tpr, thresholds = metrics.roc_curve(db.labels.detach().numpy(), likelihood_ratios)
auc = metrics.roc_auc_score(db.labels.detach().numpy(), likelihood_ratios)

print(auc)

plt.plot(fpr, tpr, label="Our model (1d)")
plt.plot(
    np.linspace(0.0, 1.0, 100), np.linspace(0.0, 1.0, 100), label="Random classifier"
)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()
