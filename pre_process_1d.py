import numpy as np
import torch
from matplotlib import pyplot as plt
import pandas as pd

from data import EventDataset
from models.flows import create_spline_flow


bins = 200
z_bins = 50

data = EventDataset(
    "data/background.csv",
    "data/signal.csv",
    ["mass"],
    10_000,
    0.1,
    mass_region=(500.0, None),
    normalize=True,
    norm_type="one_dim",
)
X = data.features
Y = data.labels

unconstrained_flow = create_spline_flow(10, 1, 32, 64, 4.0)
unconstrained_flow.load_state_dict(
    torch.load("saved_models_1d/unconstrained_2_50_epochs.pth")
)
unconstrained_Y = unconstrained_flow.log_prob(X)

grad_log_prob_first = torch.autograd.grad(
    outputs=unconstrained_Y.sum(), inputs=X, create_graph=True
)[0]
grad_log_prob_second = torch.autograd.grad(
    outputs=grad_log_prob_first.sum(), inputs=X, create_graph=True
)[0]

gradients_first_order = torch.norm(grad_log_prob_first, dim=1)
grad_std_dev_first_order = torch.std(gradients_first_order)
grad_mean_first_order = torch.mean(gradients_first_order)
z_scores_first_order = (
    gradients_first_order - grad_mean_first_order
) / grad_std_dev_first_order

first_order_median = gradients_first_order.median()
first_order_mad = abs(gradients_first_order - first_order_median).median()
modified_z_score_first_order = (
    0.6745 * (gradients_first_order - first_order_median) / first_order_mad
)

gradients_second_order = torch.norm(grad_log_prob_second, dim=1)
grad_std_dev_second_order = torch.std(gradients_second_order)
grad_mean_second_order = torch.mean(gradients_second_order)
z_scores_second_order = (
    gradients_second_order - grad_mean_second_order
) / grad_std_dev_second_order

second_order_median = gradients_second_order.median()
second_order_mad = abs(gradients_second_order - second_order_median).median()
modified_z_score_second_order = (
    0.6745 * (gradients_second_order - second_order_median) / second_order_mad
)

# Modified z-score histogram
plt.hist(
    [
        modified_z_score_first_order[Y == 0.0].detach().numpy(),
        modified_z_score_first_order[Y == 1.0].detach().numpy(),
    ],
    bins=z_bins,
    color=["tab:blue", "tab:red"],
    label=["Background", "Signal"],
    range=(-4, 4),
    histtype="barstacked",
)
plt.xlabel("Normalized magnitude (modified z-scores)")
plt.ylabel("Entries")
plt.legend()
plt.title("1D Unconstrained model, first-order-gradient magnitudes")
plt.show()

settings = {
    "first_order_std_dev": grad_std_dev_first_order.item(),
    "first_order_mean": grad_mean_first_order.item(),
    "second_order_std_dev": grad_std_dev_second_order.item(),
    "second_order_mean": grad_mean_second_order.item(),
    "first_order_median": first_order_median.item(),
    "first_order_mad": first_order_mad.item(),
    "second_order_median": second_order_median.item(),
    "second_order_mad": second_order_mad.item(),
}
print(settings)
settings = pd.DataFrame(settings)
save_name = input("Save settings as: ")
if save_name.strip() != "":
    settings.write_csv(f"pre_process_results/1d_{save_name}.csv")
