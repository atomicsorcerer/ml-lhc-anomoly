import numpy as np
import torch
from matplotlib import pyplot as plt

from models.flows import create_spline_flow


X = torch.linspace(0.0, 1.0, 100).reshape((-1, 1))

unconstrained_flow = create_spline_flow(10, 1, 32, 64, 4.0)
unconstrained_flow.load_state_dict(torch.load("saved_models/unconstrained_0.pth"))
unconstrained_Y = unconstrained_flow.log_prob(X).exp().detach().numpy().flatten()

constrained_flow = create_spline_flow(10, 1, 32, 64, 4.0)
constrained_flow.load_state_dict(torch.load("saved_models/constrained_0.pth"))
constrained_Y = constrained_flow.log_prob(X).exp().detach().numpy().flatten()

density_difference = constrained_Y - unconstrained_Y
density_difference = np.abs(density_difference)

# Graph
X = X.flatten()
plt.plot(X, unconstrained_Y, c="black", label="Unconstrained", alpha=0.5)
plt.plot(X, constrained_Y, c="tab:blue", label="Constrained", alpha=0.5)
plt.plot(X, density_difference, c="tab:red", label="Difference b/w models")
plt.xlabel("Normalized mass")
plt.ylabel("Predicted density")
plt.legend()
plt.show()
