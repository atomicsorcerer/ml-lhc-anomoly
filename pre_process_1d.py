import numpy as np
import torch
from matplotlib import pyplot as plt
import polars as pl

from data import EventDataset
from models.flows import create_spline_flow


bins = 200
z_bins = 50

data = EventDataset(
    "data/background.csv",
    "data/signal.csv",
    ["mass"],
    100_000,
    0.1,
    mass_region=(500.0, None),
    normalize=True,
    norm_type="one_dim",
)
X = data.features
# X = torch.linspace(-4, 4, 10000).reshape(-1, 1).requires_grad_()
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

# Z-score histogram
plt.hist(
    [
        modified_z_score_second_order[Y == 0.0].detach().numpy(),
        modified_z_score_second_order[Y == 1.0].detach().numpy(),
    ],
    bins=z_bins,
    color=["tab:blue", "tab:red"],
    range=(-4, 4),
    histtype="barstacked",
)
plt.xlabel("Normalized magnitude (z-scores)")
plt.ylabel("Entries")
plt.title("Unconstrained model, second-order-gradient magnitudes")
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
settings = pl.DataFrame(settings)
save_name = input("Save settings as: ")
if save_name.strip() != "":
    settings.write_csv(f"pre_process_results/{save_name}.csv")

exit()

# # Gradient magnitude histogram
plt.hist(gradients.detach().numpy(), bins=bins, color="black")
# plt.axvspan(0, grad_std_dev.item(), color="tab:green", alpha=0.3, label="1 sigma")
# plt.axvspan(
#     grad_std_dev.item(),
#     grad_std_dev.item() * 2,
#     color="tab:orange",
#     alpha=0.3,
#     label="2 sigma",
# )
# plt.axvspan(
#     grad_std_dev.item() * 2,
#     grad_std_dev.item() * 3,
#     color="tab:red",
#     alpha=0.3,
#     label="3 sigma",
# )
# plt.axvspan(
#     grad_std_dev.item() * 3,
#     grad_std_dev.item() * 4,
#     color="gray",
#     alpha=0.3,
#     label="4 sigma",
# )
# plt.legend()
plt.xlabel("Gradient magnitude")
plt.xlim(0, grad_std_dev * 4)
plt.ylabel("Entries")
plt.title("Unconstrained model, gradient magnitudes")
plt.show()
