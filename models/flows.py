from nflows.flows.base import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import *
from nflows.transforms import RandomPermutation, ReversePermutation


def create_spline_flow(
    num_layers: int, features: int, hidden_features: int, num_bins: int, tail_bound
):
    transforms = []
    for i in range(num_layers):
        transforms.append(
            MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                features=features,
                hidden_features=hidden_features,
                num_bins=num_bins,
                tails="linear",
                tail_bound=tail_bound,
                use_batch_norm=True,
            ),
        )
        transforms.append(ReversePermutation(features=features))

    composite_transform = CompositeTransform(transforms)
    base_dist = StandardNormal(shape=[features])
    flow = Flow(composite_transform, base_dist)

    return flow
