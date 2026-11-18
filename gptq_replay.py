"""Native GPTQ and dense dequantized replay at the pinned GPTQModel revision."""

import json
import os
from importlib.metadata import PackageNotFoundError, distribution

from artifacts import BITS, GROUP_SIZE, UPSTREAM_REVISION, UPSTREAM_URL
from artifacts import cache_path, module_spec, sha256


def cuda_device():
    import torch

    if (
        torch.version.cuda is None
        or torch.cuda.device_count() != 1
        or torch.distributed.is_initialized()
        or os.environ.get("WORLD_SIZE", "1") != "1"
    ):
        raise ValueError("ROCOB requires one visible NVIDIA CUDA device in a single process")
    torch.set_float32_matmul_precision("highest")
    return torch.device("cuda:0")


def load_upstream():
    try:
        provenance = distribution("GPTQModel").read_text("direct_url.json")
    except PackageNotFoundError:
        provenance = None
    source = json.loads(provenance) if provenance else {}
    if (
        source.get("url", "").removesuffix(".git").lower() != UPSTREAM_URL.removesuffix(".git").lower()
        or source.get("vcs_info", {}).get("commit_id") != UPSTREAM_REVISION
    ):
        raise ValueError(f"install GPTQModel from {UPSTREAM_URL} at {UPSTREAM_REVISION}")
    from gptqmodel import GPTQModel

    return GPTQModel


def native_model(GPTQModel, snapshot, qcfg):
    import torch

    model = GPTQModel.load(
        str(snapshot),
        quantize_config=qcfg,
        dtype=torch.float16,
        trust_remote_code=False,
        use_safetensors=True,
        local_files_only=True,
    )
    if model.quantized:
        raise ValueError("native Qwen3 source unexpectedly loaded as quantized")
    return model


def native_modules(model):
    import torch
    from torch.nn import Linear
    from gptqmodel.utils.model import get_module_by_name_prefix

    layers, prefix = get_module_by_name_prefix(model.model, model.extract_layers_node())
    subsets = model.simple_layer_modules(model.model.config, model.quantize_config)
    specs = {}
    for layer_index, layer in enumerate(layers):
        for subset in subsets:
            for name in subset:
                full_name = f"{prefix}.{layer_index}.{name}"
                linear = layer.get_submodule(name)
                if not isinstance(linear, Linear) or linear.weight.dtype != torch.float16 or full_name in specs:
                    raise ValueError(f"unexpected native projection: {full_name}")
                specs[full_name] = module_spec(tuple(linear.weight.shape), BITS)
    if not specs:
        raise ValueError("native model has no GPTQ decoder projections")
    return specs


def packed_modules(model, *, adapted=False):
    import torch
    from gptqmodel.nn_modules.qlinear.torch import TorchQuantLinear
    from gptqmodel.utils.model import find_modules

    cfg = model.quantize_config
    if (
        not model.quantized
        or cfg.quant_method != "gptq"
        or cfg.format != "gptq"
        or cfg.bits != BITS
        or cfg.group_size != GROUP_SIZE
        or not cfg.sym
        or not cfg.true_sequential
        or cfg.gptaq
        or cfg.meta_get("gptaq") is True
        or cfg.dynamic
        or cfg.lm_head
        or (cfg.adapter is not None) != adapted
        or cfg.rotation
        or cfg.mock_quantization
        or cfg.pack_dtype != torch.int32
    ):
        raise ValueError("unexpected GPTQ W4A16 base or adapter configuration")
    modules = find_modules(model.model, layers=[TorchQuantLinear])
    specs = {}
    for name, module in modules.items():
        out_features, in_features = module.out_features, module.in_features
        specs[name] = module_spec((out_features, in_features), module.bits)
        groups = in_features // GROUP_SIZE
        pack_factor = 32 // BITS
        buffers = (
            (module.qweight, (in_features // pack_factor, out_features), torch.int32),
            (module.qzeros, (groups, out_features // pack_factor), torch.int32),
            (module.scales, (groups, out_features), torch.float16),
            (module.g_idx, (in_features,), torch.int32),
        )
        if module.group_size != GROUP_SIZE or any(t.shape != s or t.dtype != d for t, s, d in buffers):
            raise ValueError(f"{name}: unexpected packed group size, tensor dtype or shape")
    return modules, specs


def quantize(snapshot, calibration, output):
    cuda_device()
    GPTQModel = load_upstream()
    from gptqmodel import QuantizeConfig

    cfg = QuantizeConfig(
        bits=BITS,
        group_size=GROUP_SIZE,
        sym=True,
        true_sequential=True,
        quant_method="gptq",
        gptaq=False,
        damp_percent=0.05,
        damp_auto_increment=0.01,
        device="cuda",
        pack_impl="gpu",
        offload_to_disk=False,
    )
    model = native_model(GPTQModel, snapshot, cfg)
    expected = native_modules(model)
    model.quantize(calibration=calibration, calibration_sort=None, batch_size=1, backend="torch")
    _, packed = packed_modules(model)
    if packed != expected:
        raise ValueError("native and packed module keys, shapes or bit widths differ")
    model.save(str(output))
    return packed


def profile(snapshot, packed, calibration, token_count, expected, damp_percent, output):
    import torch
    from safetensors.torch import save_file

    device = cuda_device()
    GPTQModel = load_upstream()
    from gptqmodel.adapter.adapter import Lora
    from gptqmodel.looper.dequantize_processor import DequantizeProcessor
    from gptqmodel.looper.eora_processor import EoraProcessor
    from gptqmodel.looper.module_looper import ModuleLooper
    from gptqmodel.looper.tensorparallel_weight_processor import TensorParallelWeightProcessor

    from compensation import pac_spectrum

    base = GPTQModel.load(
        str(packed), backend="torch", device="cpu", dtype=torch.float16, local_files_only=True
    )
    qcfg = base.quantize_config
    qmodules, packed_specs = packed_modules(base)
    if packed_specs != expected:
        raise ValueError("packed model modules differ from the base manifest")
    del base
    qcfg.device = "cuda"
    qcfg.offload_to_disk = False
    model = native_model(GPTQModel, snapshot, qcfg)
    if native_modules(model) != expected:
        raise ValueError("source model modules differ from the base manifest")
    # EoraProcessor.preprocess expects a Lora object for module selection.
    qcfg.adapter = Lora(rank=1)
    (output / "cache").mkdir()
    entries = {}

    class ProfileProcessor(EoraProcessor):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.eora_process_input = self.gram_contribution
            self.token_counts = {}

        def preprocess(self, module, **kwargs):
            super().preprocess(module, **kwargs)
            self.token_counts[module.name] = 0

        def gram_contribution(self, input, name, sample_size, device):
            rows = input[0].flatten(end_dim=-2).to(device=device, dtype=torch.float32)
            with self.lock:
                self.token_counts[name] += rows.shape[0]
            # A unit scale makes the upstream segment merge an unweighted sum.
            return 1, (rows.T @ rows) / token_count, 1.0

        def process(self, module, **kwargs):
            name = module.full_name
            spec = expected[name]
            if module.state.get("tp_pad_info") or module.weight.device != device:
                raise ValueError(f"{name}: profiling requires unpadded weights on cuda:0")
            if self.token_counts.pop(module.name) != token_count:
                raise ValueError(f"{name}: calibrated token count differs from the input")
            residual = module.state.pop("w_wq_diff")
            wq = module.state.pop("wq")
            moment = self._finalize_eigen_scaling_matrix(module.name)
            out_features, in_features = spec["shape"]
            if (
                any(x.shape != (out_features, in_features) for x in (module.weight, residual, wq))
                or moment.shape != (in_features, in_features)
                or wq.dtype != module.weight.dtype
                or wq.dtype != torch.float16
                or residual.dtype != torch.float32
                or moment.dtype != torch.float32
            ):
                raise ValueError(f"{name}: unexpected projection, residual or moment layout")
            if not torch.isfinite(moment).all() or not torch.isfinite(residual).all():
                raise ValueError(f"{name}: non-finite calibration moment or residual")
            spectrum = pac_spectrum(residual, moment, damp_percent=damp_percent).cpu().tolist()
            cache = cache_path(output, name)
            save_file({"residual": residual.cpu().contiguous(), "moment": moment.cpu().contiguous()}, cache)
            with self.lock:
                if name in entries:
                    raise ValueError(f"{name}: duplicate profiling result")
                entries[name] = {**spec, "pac_spectrum": spectrum, "cache_sha256": sha256(cache)}
            # The remaining subsets and next layer replay only the native GPTQ weights.
            module.weight.data.copy_(wq)

        def submodule_finalize(self, module, model, **kwargs):
            pass  # Profiling creates no upstream EoRA adapter state to finalize.

    processor_args = dict(
        tokenizer=model.tokenizer,
        qcfg=qcfg,
        calibration=calibration,
        prepare_dataset_func=model.prepare_dataset,
        calibration_concat_size=None,
        calibration_sort=None,
        batch_size=1,
    )
    dequantizer = DequantizeProcessor(qmodules)
    dequantizer.fwd_after_process = False  # This processor only writes residuals and dequantized weights.
    processors = [
        TensorParallelWeightProcessor(**processor_args),
        dequantizer,
        ProfileProcessor(**processor_args),
    ]
    ModuleLooper(model, processors=processors).loop()
    if qmodules or entries.keys() != expected.keys():
        raise ValueError("calibration replay did not profile exactly the packed module inventory")
    return entries
