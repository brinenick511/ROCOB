# ROCOB

[![ACL Anthology](https://img.shields.io/badge/ACL_Anthology-2026.aacl--long.158-1f6feb?style=flat)](https://aclanthology.org/2026.aacl-long.158/)

Code for *ROCOB: Dual-Manifold Alignment with Optimal Budgeting for Training-Free PTQ Recovery* (IJCNLP–AACL 2026). ROCOB builds a low-rank adapter for a saved GPTQ checkpoint without training. PAC and OGC fit successive weight residuals, and GSEM allocates extra PAC ranks across projections under a shared parameter budget. [Paper PDF](docs/paper.pdf) · [Poster PDF](docs/poster.pdf) · [SpinQuant experiment](experiments/spinquant/README.md)

## Qwen3-1.7B W4A16

Use one NVIDIA CUDA GPU and install GPTQModel's build dependencies, including a CUDA build of PyTorch. Install the pinned source below.

```sh
pip install --no-build-isolation 'GPTQModel @ git+https://github.com/ModelCloud/GPTQModel.git@6c3e279867a473e39be0329555215fd6bcffae55'
```

Set `MODEL` to a local `Qwen/Qwen3-1.7B` safetensors snapshot directory named for its full 40-character commit SHA. Each calibration file is a JSON array of unpadded token ID sequences, each longer than 10 tokens and encoded with the snapshot's tokenizer. Quantization and profiling may use different calibration sets.

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

This example uses symmetric 4-bit GPTQ with group size 128. Profiling uses PAC damping `0.05` by default (`--pac-damp`). GSEM adds a per-projection increment of up to `--max-increment` to `--base-pac`. Here `--budget-rank 16` caps the total adapter parameters at `16 * sum(in_features + out_features)` across projections. Use `--total-params N` for a direct parameter cap. Hardware validation is pending.

## Artifacts and loading

`q4/` contains the packed GPTQ checkpoint and `rocob-base.json`. `prof/` contains unsquared PAC spectra, module costs and one FP32 residual/moment cache per projection. Budget disk space for both matrices per projection. `plan/allocation.json` binds the per-module increments to that profile. Each adapter directory contains `adapter_config.json`, `adapter_model.safetensors` and the base/profile binding in `rocob-adapter.json`.

Run from this directory to load the generated adapter with its GPTQ base.

```python
from gptq import load_adapter

model = load_adapter("q4", "rocob-gsem")
```

Both rank modes fit the cached residuals and moments from the dense FP16 replay. The adapter loader checks the packed base and adapter shapes. This workflow supports dense Qwen3, homogeneous GPTQ W4A16, unpadded projections and a positive total LoRA rank on every projection.

## SpinQuant evaluation

Evaluate Llama-3.2-1B-Instruct on WikiText-2 at W2 or W3 using a learned rotation, GPTQ and fixed PAC/OGC ranks. The SpinQuant patch and launcher are distributed under CC BY-NC 4.0. [Experiment](experiments/spinquant/README.md) · [SpinQuant license](experiments/spinquant/LICENSE)

## Licenses

The compensation code retains the NVIDIA and ModelCloud EoRA notices. [ROCOB MIT license](LICENSE) · [GPTQModel Apache 2.0 terms](LICENSE-GPTQModel) · [GPTQModel source at `6c3e2798`](https://github.com/ModelCloud/GPTQModel/tree/6c3e279867a473e39be0329555215fd6bcffae55)

The published paper is available under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). [Publication record](https://aclanthology.org/2026.aacl-long.158/)

## Citation

```bibtex
@inproceedings{yang-etal-2026-rocob,
  title     = {{ROCOB}: Dual-Manifold Alignment with Optimal Budgeting for Training-Free {PTQ} Recovery},
  author    = {Yang, Haoqi and Xiao, Teng and Li, Zuchao and
               Huang, Hao and Zhang, Lefei and Zhao, Hai},
  booktitle = {Proceedings of the 15th International Joint Conference on Natural Language Processing
               and the 5th Conference of the Asia-Pacific Chapter of the Association for
               Computational Linguistics (Volume 1: Long Papers)},
  year      = {2026},
  pages     = {3491--3512},
  doi       = {10.18653/2026.aacl-long.158},
  url       = {https://aclanthology.org/2026.aacl-long.158/}
}
```
