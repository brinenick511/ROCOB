"""GSEM allocation under an at-most parameter budget."""

import math
from collections.abc import Sequence

import torch
from torch import Tensor


@torch.no_grad()
def allocate_gsem_pac_rank_increments(
    *,
    pac_spectra: Sequence[Tensor],
    per_rank_parameter_costs: Sequence[int],
    total_parameter_budget: int,
    base_pac_rank: int,
    ogc_rank: int,
    max_pac_rank_increment: int,
) -> list[int]:
    """Return one GSEM PAC rank increment per module in input order.

    Each PAC spectrum contains descending singular values. Each rank costs
    in_features + out_features parameters. The total budget includes the
    fixed base PAC and OGC ranks. On ties, the smallest total cost wins.
    """
    if len(pac_spectra) != len(per_rank_parameter_costs) or any(
        parameter_cost <= 0 for parameter_cost in per_rank_parameter_costs
    ):
        raise ValueError("provide one positive rank cost per spectrum")
    if min(total_parameter_budget, base_pac_rank, ogc_rank, max_pac_rank_increment) < 0:
        raise ValueError("budget and rank bounds must be nonnegative")
    fixed_parameter_cost = (base_pac_rank + ogc_rank) * sum(per_rank_parameter_costs)
    if total_parameter_budget < fixed_parameter_cost:
        raise ValueError("total budget cannot cover the fixed PAC and OGC ranks")
    if not pac_spectra:
        return []

    cumulative_energy_gains = []
    for pac_spectrum in pac_spectra:
        if pac_spectrum.ndim != 1 or len(pac_spectrum) < max(base_pac_rank, ogc_rank):
            raise ValueError("each spectrum must contain the available fixed ranks")
        singular_values = pac_spectrum.to(device="cpu", dtype=torch.float64)
        if (
            not bool(torch.isfinite(singular_values).all())
            or bool((singular_values < 0).any())
            or bool((singular_values[1:] > singular_values[:-1]).any())
        ):
            raise ValueError("spectra must be finite, nonnegative, and descending")
        increment_singular_values = singular_values[
            base_pac_rank : base_pac_rank + max_pac_rank_increment
        ]
        cumulative_energy_gains.append(
            torch.cat(
                (singular_values.new_zeros(1), increment_singular_values.square().cumsum(dim=0))
            )
        )

    cost_unit = math.gcd(*per_rank_parameter_costs)
    compressed_rank_costs = [
        parameter_cost // cost_unit for parameter_cost in per_rank_parameter_costs
    ]
    increment_parameter_budget = total_parameter_budget - fixed_parameter_cost
    compressed_capacity = increment_parameter_budget // cost_unit
    compressed_capacity = min(
        compressed_capacity,
        sum(
            compressed_rank_cost * (len(energy_by_increment) - 1)
            for compressed_rank_cost, energy_by_increment in zip(
                compressed_rank_costs, cumulative_energy_gains
            )
        ),
    )

    # Track the best energy at each exact compressed cost.
    previous_energy = torch.full(
        (compressed_capacity + 1,), -torch.inf, dtype=torch.float64, device="cpu"
    )
    previous_energy[0] = 0.0
    increment_choices = []
    for compressed_rank_cost, energy_by_increment in zip(
        compressed_rank_costs, cumulative_energy_gains
    ):
        best_energy = previous_energy.clone()
        increment_by_cost = torch.zeros(compressed_capacity + 1, dtype=torch.int32, device="cpu")
        feasible_increment_limit = min(
            len(energy_by_increment) - 1, compressed_capacity // compressed_rank_cost
        )
        for pac_rank_increment in range(1, feasible_increment_limit + 1):
            increment_cost = compressed_rank_cost * pac_rank_increment
            candidate_energy = previous_energy[:-increment_cost] + energy_by_increment[
                pac_rank_increment
            ]
            improves_energy = candidate_energy > best_energy[increment_cost:]
            best_energy[increment_cost:] = torch.where(
                improves_energy, candidate_energy, best_energy[increment_cost:]
            )
            increment_by_cost[increment_cost:].masked_fill_(improves_energy, pac_rank_increment)
        previous_energy = best_energy
        increment_choices.append(increment_by_cost)

    # argmax chooses the first maximum, so unused budget is allowed.
    remaining_cost = int(previous_energy.argmax())
    pac_rank_increments = [0] * len(compressed_rank_costs)
    for module_index in reversed(range(len(compressed_rank_costs))):
        pac_rank_increment = int(increment_choices[module_index][remaining_cost])
        pac_rank_increments[module_index] = pac_rank_increment
        remaining_cost -= compressed_rank_costs[module_index] * pac_rank_increment
    return pac_rank_increments
