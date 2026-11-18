# ROCOB for GPTQ

ROCOB builds a training-free low-rank adapter for a saved GPTQ checkpoint. PAC fits the weight residual in the calibration geometry. OGC fits the residual left by PAC. GSEM assigns extra PAC rank under a shared parameter budget. [Paper](https://aclanthology.org/2026.aacl-long.158/)

## Qwen3-1.7B W4A16

Use one visible NVIDIA CUDA GPU and install the upstream GPTQModel build dependencies, including a CUDA build of PyTorch. Install the pinned source in that environment. The CLI checks the commit recorded by pip.

```sh
pip install --no-build-isolation 'GPTQModel @ git+https://github.com/ModelCloud/GPTQModel.git@6c3e279867a473e39be0329555215fd6bcffae55'
```

Set `MODEL` to your local `Qwen/Qwen3-1.7B` safetensors snapshot directory, named for its full 40-character commit SHA. Use the same snapshot for both commands. Each calibration JSON file contains one array of unpadded token ID arrays from that snapshot's tokenizer in input order. Use sequences longer than 10 tokens. The two files may contain different calibration sets.

```sh
export CUDA_VISIBLE_DEVICES=0
python gptq.py quantize "$MODEL" gptq-ids.json q4
python gptq.py profile "$MODEL" q4 pac-ids.json prof
python gptq.py allocate prof plan --base-pac 10 --ogc 2 --max-increment 8 --budget-rank 16
python gptq.py adapter q4 prof rocob-gsem --allocation plan/allocation.json
```

For fixed PAC and OGC ranks, use the same profile.

```sh
python gptq.py adapter q4 prof rocob-fixed --pac 14 --ogc 2
```

The example uses symmetric 4-bit GPTQ with group size 128 and requests GPU packing, following the original GPTQ entrypoint. The paper appendix gives quantizer damping `0.05` and retry increment `0.01`. Profiling applies PAC damping `0.05` once, configurable with `--pac-damp`.

The input snapshot SHA and token files are supplied locally. FP16, sequential quantization, batch size 1 and the PAC/OGC split here define the demo. For an example calibration mix, prepare WikiText-2 for quantization and C4 plus GSM8K training data for profiling. `--budget-rank 16` caps total adapter parameters at `16 * sum(in_features + out_features)` over the actual projections. Use `--total-params N` for an explicit parameter cap. The example is pending hardware validation.

## Artifacts and loading

`q4/` contains the packed GPTQ checkpoint and `rocob-base.json`. `prof/` contains unsquared PAC spectra, module costs and one FP32 residual/moment cache per projection. Budget disk space for both matrices per projection. `plan/allocation.json` binds the per-module increments to that profile. Each adapter directory contains `adapter_config.json`, `adapter_model.safetensors` and the base/profile binding in `rocob-adapter.json`.

Run from this directory to load the generated adapter with its GPTQ base.

```python
from gptq import load_adapter

model = load_adapter("q4", "rocob-gsem")
```

Profiling replays the saved GPTQ weights in dense FP16 and records calibration statistics before adding any correction. Both rank modes fit the exact cached residual and moment. The loader checks the packed base and transposed adapter shapes and disables the pinned Torch backend's dequantized weight cache for adapted projections.

The release scope is dense Qwen3, homogeneous GPTQ W4A16, one GPU, unpadded projections and positive total LoRA rank on every projection.

## Paper and licenses

Haoqi Yang, Teng Xiao, Zuchao Li, Hao Huang, Lefei Zhang, and Hai Zhao. 2026. *ROCOB: Dual-Manifold Alignment with Optimal Budgeting for Training-Free PTQ Recovery*. IJCNLP-AACL 2026. [Paper](https://aclanthology.org/2026.aacl-long.158/)

The integration targets [GPTQModel at `6c3e2798`](https://github.com/ModelCloud/GPTQModel/tree/6c3e279867a473e39be0329555215fd6bcffae55). The compensation core carries the NVIDIA and ModelCloud EoRA notices. [MIT license](LICENSE) · [GPTQModel Apache 2.0 terms](LICENSE-GPTQModel)
