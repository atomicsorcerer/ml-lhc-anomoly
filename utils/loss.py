import torch
from nflows.flows import Flow

from utils.physics import calculate_squared_dijet_mass


def calculate_non_smoothness_penalty(
    flow: Flow, n_knots: int, alpha: float
) -> torch.Tensor:
    knots = torch.linspace(0.0, 1.0, n_knots).reshape((-1, 1))
    densities = flow.log_prob(knots).exp()
    slopes = densities[1:] - densities[:-1]
    slopes = slopes.abs()
    slope_differences = slopes[1:] - slopes[:-1]
    slope_differences = slope_differences.abs()
    normalized_slope_differences = slope_differences / slope_differences.max()
    smoothness_penalty = normalized_slope_differences.mean()
    smoothness_penalty = smoothness_penalty * alpha

    return smoothness_penalty


def calculate_impossible_mass_penalty(
    flow: Flow, n_samples: int, alpha: float
) -> torch.Tensor:
    base_samples = flow._distribution.sample(len(n_samples))
    generated_samples, _ = flow._transform.inverse(base_samples)
    generated_masses = calculate_squared_dijet_mass(generated_samples)
    negative_mass_penalty = (
        torch.nn.functional.softplus(-generated_masses).mean() * alpha
    )

    return negative_mass_penalty


# Calculate and generate 1d mass distribution for penalties
# base_samples = flow._distribution.sample(len(X))
# generated_samples, _ = flow._transform.inverse(base_samples)
# generated_masses = calculate_squared_dijet_mass(generated_samples)
# target_masses = calculate_squared_dijet_mass(X)

# Calculate penalty term (KL divergence) for 1d mass distribution
# kl_generated_masses = (
#     generated_masses + generated_masses.min().abs()
# )  # Shift distribution to start at zero
# kl_target_masses = (
#     target_masses + generated_masses.min().abs()
# )  # Match up with the shift
# kl_div = torch.nn.functional.kl_div(
#     torch.nn.functional.log_softmax(kl_generated_masses, dim=0),
#     torch.nn.functional.log_softmax(kl_target_masses, dim=0),
#     reduction="batchmean",
#     log_target=True,
# )
# kl_div = torch.nn.functional.relu(kl_div)  # Keep loss positive
