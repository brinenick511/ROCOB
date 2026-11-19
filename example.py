"""Build ROCOB factors for two synthetic linear modules on CPU."""

import argparse
import json
import math
from pathlib import Path

import torch

from budget import allocate_gsem_pac_rank_increments
from compensation import calibration_second_moment, compute_rocob_compensation, pac_spectrum


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    arguments = parser.parse_args()
    config = json.loads(arguments.config.read_text(encoding="utf-8"))
    rank_allocation = config["rank_allocation"]
    ogc_rank = config["ogc_rank"] if "ogc_rank" in config else config["OGC_RANK"]
    damp_percent = config["damp_percent"]
    modules = []
    for module_name, residual_diagonal, in_features, observed_features in (
        ("square", [9, 6, 5, 2, 1], 5, 3),
        ("wide", [10, 7, 6, 1], 8, 8),
    ):
        out_features = len(residual_diagonal)
        dequantized_weight = torch.eye(out_features, in_features, dtype=torch.float32)
        full_weight = dequantized_weight.clone()
        full_weight.diagonal().add_(torch.tensor(residual_diagonal, dtype=torch.float32))
        delta_weight = full_weight - dequantized_weight
        activations = torch.eye(in_features, dtype=torch.float64)[:observed_features]
        activations = activations * math.sqrt(observed_features)
        second_moment = calibration_second_moment(activations)
        modules.append((module_name, delta_weight, second_moment))

    if rank_allocation == "fixed":
        total_parameter_budget = None
        pac_rank = config["pac_rank"] if "pac_rank" in config else config["PAC_RANK"]
        pac_ranks = [pac_rank] * len(modules)
    elif rank_allocation == "gsem":
        base_pac_rank = config["base_pac_rank"]
        total_parameter_budget = config["total_parameter_budget"]
        pac_spectra = [
            pac_spectrum(delta_weight, second_moment, damp_percent=damp_percent)
            for _, delta_weight, second_moment in modules
        ]
        per_rank_parameter_costs = [sum(delta_weight.shape) for _, delta_weight, _ in modules]
        pac_rank_increments = allocate_gsem_pac_rank_increments(
            pac_spectra=pac_spectra,
            per_rank_parameter_costs=per_rank_parameter_costs,
            total_parameter_budget=total_parameter_budget,
            base_pac_rank=base_pac_rank,
            ogc_rank=ogc_rank,
            max_pac_rank_increment=config["max_pac_rank_increment"],
        )
        pac_ranks = [
            base_pac_rank + pac_rank_increment for pac_rank_increment in pac_rank_increments
        ]
    else:
        raise ValueError("rank_allocation must be 'fixed' or 'gsem'")

    module_results = {}
    parameter_count = 0
    for (module_name, delta_weight, second_moment), pac_rank in zip(modules, pac_ranks):
        input_factor, output_factor = compute_rocob_compensation(
            delta_weight,
            second_moment,
            pac_rank=pac_rank,
            ogc_rank=ogc_rank,
            damp_percent=damp_percent,
        )
        parameter_count += input_factor.numel() + output_factor.numel()
        residual_after = delta_weight - output_factor @ input_factor
        module_results[module_name] = {
            "pac_rank": pac_rank,
            "ogc_rank": ogc_rank,
            "residual_before": round(delta_weight.square().sum().item(), 4),
            "residual_after": round(residual_after.square().sum().item(), 4),
        }
    print(
        json.dumps(
            {
                "budget": total_parameter_budget,
                "parameters": parameter_count,
                "modules": module_results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
