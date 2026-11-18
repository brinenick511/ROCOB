# SPDX-FileCopyrightText: 2024-2025 NVIDIA CORPORATION
# SPDX-FileCopyrightText: 2025 ModelCloud.ai (qubitium@modelcloud.ai)
# SPDX-License-Identifier: Apache-2.0
# EoRA: https://arxiv.org/abs/2410.21271
# EoRA official repository: https://github.com/NVlabs/EoRA
# Adapted from GPTQModel's EoRA implementation for standalone ROCOB use.

"""PAC and OGC for linear weights in PyTorch's (out, in) layout."""

import math

import torch
from torch import Tensor


@torch.no_grad()
def calibration_second_moment(activations: Tensor) -> Tensor:
    """Compute X.T @ X / N over valid token rows, without centering.

    The last dimension holds input features. Remove padding before calling.
    """
    if activations.ndim < 2 or activations.numel() == 0:
        raise ValueError("activations must have nonempty sample and feature axes")
    token_rows = activations.flatten(end_dim=-2).to(dtype=torch.float64)
    return token_rows.T @ token_rows / token_rows.shape[0]


def _pac_geometry(
    delta_weight: Tensor, second_moment: Tensor, damp_percent: float
) -> tuple[Tensor, Tensor]:
    in_features = delta_weight.shape[1]
    if second_moment.shape != (in_features, in_features):
        raise ValueError("second_moment must have shape (in_features, in_features)")
    if not math.isfinite(damp_percent) or damp_percent < 0:
        raise ValueError("damp_percent must be finite and nonnegative")

    symmetric_moment = second_moment.to(device=delta_weight.device, dtype=torch.float64)
    symmetric_moment = (symmetric_moment + symmetric_moment.T) * 0.5
    symmetric_moment.diagonal().add_(damp_percent * symmetric_moment.diagonal().mean())
    eigenvalues, eigenvectors = torch.linalg.eigh(symmetric_moment)

    # Invert only the numerical support of the PSD second moment.
    eigenvalue_cutoff = in_features * torch.finfo(eigenvalues.dtype).eps * eigenvalues.abs().amax()
    support_mask = eigenvalues > eigenvalue_cutoff
    sqrt_eigenvalues = eigenvalues.clamp_min(0).sqrt().masked_fill(~support_mask, 0)
    inverse_sqrt_eigenvalues = (
        sqrt_eigenvalues.masked_fill(~support_mask, 1).reciprocal().masked_fill(~support_mask, 0)
    )

    moment_factor = (eigenvectors * sqrt_eigenvalues).to(dtype=delta_weight.dtype)
    inverse_moment_factor = (inverse_sqrt_eigenvalues[:, None] * eigenvectors.T).to(
        dtype=delta_weight.dtype
    )
    return delta_weight @ moment_factor, inverse_moment_factor


@torch.no_grad()
def pac_spectrum(
    delta_weight: Tensor, second_moment: Tensor, *, damp_percent: float = 0.0
) -> Tensor:
    """Return the descending singular values used to score GSEM increments."""
    if delta_weight.ndim != 2 or delta_weight.numel() == 0:
        raise ValueError("delta_weight must be a nonempty weight matrix")
    weighted_residual, _ = _pac_geometry(delta_weight.float(), second_moment, damp_percent)
    return torch.linalg.svdvals(weighted_residual)


def _factors(
    matrix: Tensor, component_rank: int, inverse_moment_factor: Tensor | None = None
) -> tuple[Tensor, Tensor]:
    if component_rank == 0:
        return matrix.new_zeros((0, matrix.shape[1])), matrix.new_zeros((matrix.shape[0], 0))

    left_vectors, singular_values, right_vectors_transposed = torch.linalg.svd(matrix, full_matrices=False)
    sqrt_singular_values = singular_values[:component_rank].sqrt()
    input_factor = sqrt_singular_values[:, None] * right_vectors_transposed[:component_rank]
    if inverse_moment_factor is not None:
        input_factor = input_factor @ inverse_moment_factor
    output_factor = left_vectors[:, :component_rank] * sqrt_singular_values
    return input_factor, output_factor


@torch.no_grad()
def compute_rocob_compensation(
    delta_weight: Tensor,
    second_moment: Tensor,
    *,
    pac_rank: int,
    ogc_rank: int,
    damp_percent: float = 0.0,
    dtype: torch.dtype | None = None,
) -> tuple[Tensor, Tensor]:
    """Return (A, B) for the additive weight correction B @ A.

    Form delta_weight as W.float() - Q_dequantized.float(). Fit PAC at the
    supplied pac_rank and OGC at ogc_rank on the residual after PAC. The
    rank axis of each returned factor has length pac_rank + ogc_rank.
    """
    if delta_weight.ndim != 2 or delta_weight.numel() == 0:
        raise ValueError("delta_weight must be a nonempty weight matrix")
    max_component_rank = min(delta_weight.shape)
    if not (0 <= pac_rank <= max_component_rank and 0 <= ogc_rank <= max_component_rank):
        raise ValueError("each component rank must lie in [0, min(out_features, in_features)]")

    weight_residual = delta_weight.float()
    if pac_rank:
        weighted_residual, inverse_moment_factor = _pac_geometry(
            weight_residual, second_moment, damp_percent
        )
        pac_input_factor, pac_output_factor = _factors(
            weighted_residual, pac_rank, inverse_moment_factor
        )
        residual_after_pac = weight_residual - pac_output_factor @ pac_input_factor
    else:
        pac_input_factor, pac_output_factor = _factors(weight_residual, 0)
        residual_after_pac = weight_residual
    ogc_input_factor, ogc_output_factor = _factors(residual_after_pac, ogc_rank)

    target_dtype = dtype if dtype is not None else delta_weight.dtype
    input_factor = torch.cat((pac_input_factor, ogc_input_factor), dim=0).to(dtype=target_dtype)
    output_factor = torch.cat((pac_output_factor, ogc_output_factor), dim=1).to(dtype=target_dtype)
    return input_factor, output_factor
