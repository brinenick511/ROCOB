"""Quantize, profile, allocate and build a ROCOB adapter for dense Qwen3."""

import argparse
import math
from pathlib import Path

from artifacts import ADAPTER_FILE, ALLOCATION_FILE, BASE_FILE, PROFILE_FILE
from artifacts import cache_path, calibration_data, inventory, module_spec, native_identity
from artifacts import new_output, packed_files, read_artifact, read_base, read_json, sha256, stamp, write_json
from ranks import factor_ranks


def quantize_command(args):
    from gptq_replay import quantize

    native, config = native_identity(args.model)
    calibration, calibration_info = calibration_data(args.calibration, config)
    with new_output(args.output) as output:
        modules = quantize(args.model, calibration, output)
        write_json(output / BASE_FILE, {
            **stamp("base"), "native": native, "quantization_calibration": calibration_info,
            "modules": modules, "files": packed_files(output),
        })


def profile_command(args):
    from gptq_replay import profile

    base, base_hash = read_base(args.base)
    native, config = native_identity(args.model)
    if native != base["native"]:
        raise ValueError("model revision or snapshot files differ from the quantization input")
    calibration, calibration_info = calibration_data(args.calibration, config)
    damp = args.pac_damp
    if not math.isfinite(damp) or damp < 0:
        raise ValueError("PAC damping must be finite and nonnegative")
    with new_output(args.output) as output:
        entries = profile(
            args.model, args.base, calibration, calibration_info["tokens"],
            base["modules"], damp, output,
        )
        write_json(output / PROFILE_FILE, {
            **stamp("profile"), "base_sha256": base_hash, "calibration": calibration_info,
            "pac_damp": damp, "modules": entries,
        })


def read_profile(directory):
    profile, digest = read_artifact(directory / PROFILE_FILE, "profile")
    entries = profile["modules"]
    if not entries or not math.isfinite(profile["pac_damp"]) or profile["pac_damp"] < 0:
        raise ValueError("profile requires modules and finite, nonnegative PAC damping")
    specs = {}
    for name, entry in entries.items():
        cache_path(directory, name)  # Validate full names used as filenames.
        spec = module_spec(entry["shape"], entry["bits"])
        spectrum = entry["pac_spectrum"]
        if (
            spec["cost_per_rank"] != entry["cost_per_rank"]
            or len(spectrum) != min(spec["shape"])
            or any(not math.isfinite(s) or s < 0 for s in spectrum)
            or any(a < b for a, b in zip(spectrum, spectrum[1:]))
        ):
            raise ValueError(f"{name}: invalid PAC spectrum or rank cost")
        specs[name] = spec
    return profile, digest, specs


def dynamic_ranks(specs, base_pac, ogc, increments, max_increment, budget):
    if increments.keys() != specs.keys():
        raise ValueError("GSEM increments must cover exactly the profiled module keys")
    if type(max_increment) is not int or max_increment < 0 or type(budget) is not int or budget < 0:
        raise ValueError("GSEM increment limit and total budget must be nonnegative integers")
    if base_pac == 0 and ogc == 0:
        raise ValueError("the fixed PAC floor and OGC must allow a positive LoRA rank at zero increment")
    ranks = {}
    for name, spec in specs.items():
        increment = increments[name]
        ranks[name] = factor_ranks(
            name, spec["shape"], base_pac_rank=base_pac, ogc_rank=ogc, pac_increment=increment,
        )
        if increment > max_increment:
            raise ValueError(f"{name}: increment exceeds the GSEM limit")
    used = sum(specs[name]["cost_per_rank"] * sum(ranks[name]) for name in specs)
    if used > budget:
        raise ValueError("GSEM ranks exceed the total parameter budget")
    return ranks, used


def allocate_command(args):
    import torch

    from budget import allocate_gsem_pac_rank_increments

    profile, profile_hash, specs = read_profile(args.profile)
    names = sorted(specs)
    costs = [specs[name]["cost_per_rank"] for name in names]
    budget = args.total_params if args.total_params is not None else args.budget_rank * sum(costs)
    fixed_cost = sum(costs) * (args.base_pac + args.ogc)
    dynamic_ranks(specs, args.base_pac, args.ogc, dict.fromkeys(names, 0), args.max_increment, budget)
    spectra = [torch.tensor(profile["modules"][name]["pac_spectrum"], dtype=torch.float64) for name in names]
    increments = allocate_gsem_pac_rank_increments(
        pac_spectra=spectra,
        per_rank_parameter_costs=costs,
        total_parameter_budget=budget,
        base_pac_rank=args.base_pac,
        ogc_rank=args.ogc,
        max_pac_rank_increment=args.max_increment,
    )
    by_name = dict(zip(names, increments, strict=True))
    _, used = dynamic_ranks(specs, args.base_pac, args.ogc, by_name, args.max_increment, budget)
    with new_output(args.output) as output:
        write_json(output / ALLOCATION_FILE, {
            **stamp("allocation"), "profile_sha256": profile_hash, "base_sha256": profile["base_sha256"],
            "base_pac": args.base_pac, "ogc": args.ogc, "max_increment": args.max_increment,
            "total_params": budget, "fixed_params": fixed_cost, "used_params": used,
            "increments": by_name,
        })


def adapter_command(args):
    import torch
    from safetensors.torch import load_file, save_file

    from compensation import compute_rocob_compensation
    from gptq_replay import cuda_device

    base, base_hash = read_base(args.base)
    profile, profile_hash, specs = read_profile(args.profile)
    if profile["base_sha256"] != base_hash or specs != base["modules"]:
        raise ValueError("profile does not match the packed base and module inventory")

    allocation_hash = None
    if args.allocation:
        plan, allocation_hash = read_artifact(args.allocation, "allocation")
        if plan["profile_sha256"] != profile_hash or plan["base_sha256"] != base_hash:
            raise ValueError("GSEM allocation belongs to a different profile or GPTQ base")
        ranks, used = dynamic_ranks(
            specs, plan["base_pac"], plan["ogc"], plan["increments"], plan["max_increment"], plan["total_params"],
        )
        default_rank = plan["base_pac"] + plan["ogc"]
        fixed_params = sum(spec["cost_per_rank"] for spec in specs.values()) * default_rank
        if used != plan["used_params"] or fixed_params != plan["fixed_params"]:
            raise ValueError("GSEM allocation parameter counts do not match its ranks")
    else:
        ranks = {
            name: factor_ranks(name, spec["shape"], base_pac_rank=args.pac, ogc_rank=args.ogc)
            for name, spec in specs.items()
        }
        default_rank = args.pac + args.ogc
        used = sum(default_rank * spec["cost_per_rank"] for spec in specs.values())

    device = cuda_device()
    weights = {}
    rank_pattern = {}
    for name, (pac_rank, ogc_rank) in ranks.items():
        cache = cache_path(args.profile, name)
        if sha256(cache) != profile["modules"][name]["cache_sha256"]:
            raise ValueError(f"{name}: profiling cache has changed")
        inputs = load_file(cache, device=str(device))
        out_features, in_features = specs[name]["shape"]
        if (
            inputs.keys() != {"residual", "moment"}
            or inputs["residual"].shape != (out_features, in_features)
            or inputs["moment"].shape != (in_features, in_features)
            or any(t.dtype != torch.float32 for t in inputs.values())
        ):
            raise ValueError(f"{name}: incompatible residual or calibration moment in the cache")
        A, B = compute_rocob_compensation(
            inputs["residual"], inputs["moment"], pac_rank=pac_rank, ogc_rank=ogc_rank,
            damp_percent=profile["pac_damp"], dtype=torch.float16,
        )
        rank = pac_rank + ogc_rank
        if A.shape != (rank, in_features) or B.shape != (out_features, rank):
            raise ValueError(f"{name}: adapter factor axes disagree with selected ranks")
        if not torch.isfinite(A).all() or not torch.isfinite(B).all():
            raise ValueError(f"{name}: factors are non-finite in FP16")
        weights[f"base_model.model.{name}.lora_A.weight"] = A.cpu().contiguous()
        weights[f"base_model.model.{name}.lora_B.weight"] = B.cpu().contiguous()
        if rank != default_rank:
            rank_pattern[name] = rank

    if sum(t.numel() for t in weights.values()) != used:
        raise ValueError("saved factor counts disagree with the rank costs")
    with new_output(args.output) as output:
        save_file(weights, output / "adapter_model.safetensors", metadata={"format": "pt"})
        write_json(output / "adapter_config.json", {
            "base_model_name_or_path": str(args.base.resolve()),
            "peft_type": "LORA", "task_type": "CAUSAL_LM", "inference_mode": True,
            "target_modules": sorted(specs), "bias": "none", "use_rslora": False,
            "r": default_rank, "lora_alpha": default_rank,
            "rank_pattern": rank_pattern, "alpha_pattern": rank_pattern,
        })
        write_json(output / ADAPTER_FILE, {
            **stamp("adapter"), "base_sha256": base_hash, "profile_sha256": profile_hash,
            "allocation_sha256": allocation_hash, "component_ranks": ranks,
            "used_params": used, "files": inventory(output),
        })


def load_adapter(base_dir, adapter_dir):
    """Load the bound adapter and its packed GPTQ base on a single CUDA device."""
    import torch

    from gptq_replay import cuda_device, load_upstream, packed_modules

    base_dir, adapter_dir = Path(base_dir), Path(adapter_dir).resolve()
    base, base_hash = read_base(base_dir)
    manifest, _ = read_artifact(adapter_dir / ADAPTER_FILE, "adapter")
    ranks = manifest["component_ranks"]
    if (
        manifest["base_sha256"] != base_hash
        or ranks.keys() != base["modules"].keys()
        or manifest["files"] != inventory(adapter_dir, excluded=(ADAPTER_FILE,))
    ):
        raise ValueError("adapter files or module inventory do not match the packed base")
    config = read_json(adapter_dir / "adapter_config.json")
    default_rank = config["r"]
    expected_pattern = {name: sum(pair) for name, pair in ranks.items() if sum(pair) != default_rank}
    if (
        type(default_rank) is not int or default_rank <= 0
        or config["lora_alpha"] != default_rank
        or config["rank_pattern"] != expected_pattern
        or config["alpha_pattern"] != expected_pattern
        or config["use_rslora"]
    ):
        raise ValueError("adapter rank or unit scaling metadata does not match its factors")

    device = cuda_device()
    GPTQModel = load_upstream()
    from gptqmodel.adapter.adapter import AdapterCache, Lora

    adapter_path = str(adapter_dir)
    AdapterCache.remove(adapter_path)
    try:
        model = GPTQModel.load(
            str(base_dir), adapter=Lora(rank=default_rank, path=adapter_path), backend="torch",
            device=str(device), dtype=torch.float16, trust_remote_code=False, local_files_only=True,
        )
    finally:
        AdapterCache.remove(adapter_path)
    modules, specs = packed_modules(model, adapted=True)
    if specs != base["modules"]:
        raise ValueError("loaded GPTQ modules differ from the base manifest")
    for name, module in modules.items():
        out_features, in_features = specs[name]["shape"]
        pac, ogc = ranks[name]
        factor_ranks(name, specs[name]["shape"], base_pac_rank=pac, ogc_rank=ogc)
        rank = pac + ogc
        lora = module.adapter
        if (
            not isinstance(lora, Lora) or lora.rank != rank
            or lora.lora_A.shape != (in_features, rank)
            or lora.lora_B.shape != (rank, out_features)
            or any(t.device != device or t.dtype != torch.float16 for t in (lora.lora_A, lora.lora_B))
        ):
            raise ValueError(f"{name}: loaded GPTQModel adapter disagrees with the rank or factor axes")
        # The upstream cached-weight forward bypasses the adapter contribution.
        module.enable_weight_cache(False)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="stage", required=True)

    quantize = commands.add_parser("quantize")
    quantize.add_argument("model", type=Path)
    quantize.add_argument("calibration", type=Path)

    profile = commands.add_parser("profile")
    profile.add_argument("model", type=Path)
    profile.add_argument("base", type=Path)
    profile.add_argument("calibration", type=Path)
    profile.add_argument("--pac-damp", type=float, default=0.05)

    allocate = commands.add_parser("allocate")
    allocate.add_argument("profile", type=Path)
    allocate.add_argument("--base-pac", type=int, required=True)
    allocate.add_argument("--ogc", type=int, required=True)
    allocate.add_argument("--max-increment", type=int, required=True)
    budget = allocate.add_mutually_exclusive_group(required=True)
    budget.add_argument("--total-params", type=int)
    budget.add_argument("--budget-rank", type=int)

    adapter = commands.add_parser("adapter")
    adapter.add_argument("base", type=Path)
    adapter.add_argument("profile", type=Path)
    mode = adapter.add_mutually_exclusive_group(required=True)
    mode.add_argument("--allocation", type=Path)
    mode.add_argument("--pac", type=int)
    adapter.add_argument("--ogc", type=int)

    for command, run in (
        (quantize, quantize_command), (profile, profile_command),
        (allocate, allocate_command), (adapter, adapter_command),
    ):
        command.add_argument("output", type=Path)
        command.set_defaults(run=run)
    args = parser.parse_args()
    if args.stage == "adapter" and ((args.allocation is None) != (args.ogc is not None)):
        parser.error("provide --pac and --ogc together, or --allocation alone")
    try:
        args.run(args)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"rocob {args.stage}: {exc}\n")


if __name__ == "__main__":
    main()
