"""Files and identity checks shared by the four GPTQ stages."""

import hashlib
import json
import re
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory


UPSTREAM_URL = "https://github.com/ModelCloud/GPTQModel.git"
UPSTREAM_REVISION = "6c3e279867a473e39be0329555215fd6bcffae55"
BASE_FILE = "rocob-base.json"
PROFILE_FILE = "profile.json"
ALLOCATION_FILE = "allocation.json"
ADAPTER_FILE = "rocob-adapter.json"
BITS = 4
GROUP_SIZE = 128


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(path.read_bytes(), object_pairs_hook=_unique_keys)


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def write_json(path, info):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(info, stream, sort_keys=True, allow_nan=False, separators=(",", ":"))
        stream.write("\n")


@contextmanager
def new_output(path):
    path = path.absolute()
    if path.exists() or path.is_symlink():
        raise ValueError(f"output already exists: {path}")
    with TemporaryDirectory(prefix=f".{path.name}-", dir=path.parent) as temp:
        staging = Path(temp) / "output"
        staging.mkdir()
        yield staging
        if path.exists() or path.is_symlink():
            raise ValueError(f"output already exists: {path}")
        staging.rename(path)


def inventory(path, *, excluded=()):
    return {
        str(file.relative_to(path)): sha256(file)
        for file in sorted(path.rglob("*"))
        if file.is_file() and str(file.relative_to(path)) not in excluded
    }


def native_identity(snapshot):
    if not snapshot.is_dir() or not re.fullmatch(r"[0-9a-f]{40}", snapshot.name):
        raise ValueError("model must be a local snapshot directory named for its full 40-hex revision")
    config = read_json(snapshot / "config.json")
    if config.get("model_type") != "qwen3" or "quantization_config" in config:
        raise ValueError("model must be a native dense Qwen3 checkpoint")
    files = inventory(snapshot)
    if not any(name.endswith(".safetensors") for name in files):
        raise ValueError("model snapshot must contain safetensors weights")
    return {"revision": snapshot.name, "files": files}, config


def calibration_data(path, config):
    raw = path.read_bytes()
    rows = json.loads(raw)
    if not isinstance(rows, list) or not rows:
        raise ValueError("calibration must be a nonempty JSON array of unpadded token ID arrays")
    for index, row in enumerate(rows):
        if (
            not isinstance(row, list)
            or not (10 < len(row) <= config["max_position_embeddings"])
            or any(type(token) is not int or not 0 <= token < config["vocab_size"] for token in row)
        ):
            raise ValueError(f"invalid calibration token IDs or sequence length in row {index}")
    return [{"input_ids": row, "attention_mask": [1] * len(row)} for row in rows], {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "sequences": len(rows),
        "tokens": sum(map(len, rows)),
    }


def read_artifact(path, stage):
    raw = path.read_bytes()
    info = json.loads(raw, object_pairs_hook=_unique_keys)
    if info.get("schema") != f"rocob/{stage}/1" or info.get("gptqmodel_revision") != UPSTREAM_REVISION:
        raise ValueError(f"{path}: wrong artifact type or GPTQModel revision")
    return info, hashlib.sha256(raw).hexdigest()


def module_spec(shape, bits):
    if type(bits) is not int or bits != BITS or len(shape) != 2 or any(type(n) is not int or n <= 0 for n in shape):
        raise ValueError("invalid W4 linear module dimensions or bit width")
    if shape[1] % GROUP_SIZE or shape[0] % 8:
        raise ValueError(f"projection {shape} needs group/packing padding, outside the supported scope")
    return {"shape": list(shape), "bits": bits, "cost_per_rank": sum(shape)}


def cache_path(directory, full_name):
    if not re.fullmatch(r"[a-z0-9_]+(?:\.[a-z0-9_]+)+", full_name):
        raise ValueError(f"invalid module full name: {full_name}")
    return directory / "cache" / f"{full_name}.safetensors"


def read_base(directory):
    base, digest = read_artifact(directory / BASE_FILE, "base")
    if not base["modules"] or base["files"] != packed_files(directory):
        raise ValueError("invalid packed GPTQ directory or files changed since quantization")
    return base, digest


def packed_files(directory):
    files = inventory(directory, excluded=(BASE_FILE,))
    config = read_json(directory / "quantize_config.json")
    if (
        "config.json" not in files
        or not any(name.endswith(".safetensors") for name in files)
        or config.get("quant_method") != "gptq"
        or config.get("checkpoint_format") != "gptq"
        or config.get("bits") != BITS
        or config.get("group_size") != GROUP_SIZE
        or config.get("sym") is not True
        or config.get("meta", {}).get("true_sequential") is not True
        or config.get("meta", {}).get("gptaq") is not False
        or config.get("pack_dtype") != "int32"
        or config.get("dynamic")
        or config.get("lm_head")
    ):
        raise ValueError("expected saved symmetric, sequential GPTQ W4A16 safetensors")
    return files


def stamp(stage):
    return {"schema": f"rocob/{stage}/1", "gptqmodel_revision": UPSTREAM_REVISION}
