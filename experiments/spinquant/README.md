# ROCOB with SpinQuant

Evaluate the appendix’s Llama-3.2-1B-Instruct WikiText-2 setting at nominal W2A16KV16 or W3A16KV16. The workflow loads a learned rotation, runs the archived SpinQuant GPTQ path with fixed-rank DMA in each decoder block, and prints test perplexity.

## Run

Use a CUDA environment with BF16 support and enough memory for 64 calibration sequences per forward. Install PyTorch with CUDA, the pinned checkout’s `requirement.txt`, and `fast-hadamard-transform` in that environment. Stage a local model and tokenizer snapshot and a trusted SpinQuant `R.bin` for the same model and bit setting. The state contains `R1` and `model.layers.<i>.self_attn.R2` for every decoder layer. [Upstream setup](https://github.com/facebookresearch/SpinQuant/blob/8f47aa3f00e8662caf1a484153920a07e5281c3a/README.md#run) · [Model](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)

From this experiment directory, use a fresh checkout of the pinned upstream source.

```bash
git clone https://github.com/facebookresearch/SpinQuant.git SpinQuant
git -C SpinQuant checkout --detach 8f47aa3f00e8662caf1a484153920a07e5281c3a
git -C SpinQuant apply --check ../rocob.patch
git -C SpinQuant apply ../rocob.patch
```

Set `MODEL_DIR` to the snapshot directory and `ROTATION_W2` to the W2 state. Both commands use the same inputs. The archived `29/3` pair maps to `PAC_RANK=29 OGC_RANK=3`. The `0/0` run is the SpinQuant reference with DMA disabled.

```bash
W_BITS=2 PAC_RANK=29 OGC_RANK=3 bash run.sh SpinQuant "$MODEL_DIR" "$ROTATION_W2"
W_BITS=2 PAC_RANK=0  OGC_RANK=0 bash run.sh SpinQuant "$MODEL_DIR" "$ROTATION_W2"
```

For W3, set `W_BITS=3` and use the W3 state for both runs. The launcher defaults to GPU 0 and uses the 1B recipe's calibration batch of 64. SpinQuant samples 128 length-2048 windows from the raw WikiText-2 train split with seed 0. The inherited `--w_asym --w_clip` configuration fits asymmetric scales and zero points, while its GPTQ fake-quantization step uses the fitted scale and signed clipping without the zero point. Read the final `wiki2 ppl is:` line for test PPL on contiguous 2048-token blocks. [WikiText-2](https://huggingface.co/datasets/Salesforce/wikitext)

The archived W2 and W3 rotation files have SHA-256 values `b624ea488624bacbe293a48bbde713582fa0fa7b0a113eee2ec067f54063e328` and `9f2efb5f4d5bf8d6d1c480b62689f8fa11074538636296d31583450d21d88bb9`. These identify the recovered files. The historical run summaries lack exact run commands, model and tokenizer revisions, and per-result rotation digests.

### Generate a rotation state

To generate a fresh W2 state, set `MODEL_DIR` and `ROTATION_DIR` to absolute paths. This example uses one GPU and an effective training batch of eight. Use the resulting `ROTATION_DIR/R.bin` for both W2 runs. Generate a separate state with `--w_bits 3` for W3. [Upstream rotation trainer](https://github.com/facebookresearch/SpinQuant/blob/8f47aa3f00e8662caf1a484153920a07e5281c3a/optimize_rotation.py)

```bash
(
  cd SpinQuant || exit 1
  CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 optimize_rotation.py \
    --input_model "$MODEL_DIR" --output_rotation_path "$ROTATION_DIR" \
    --output_dir "$ROTATION_DIR/trainer" --do_train True --model_max_length 2048 \
    --bf16 True --gradient_checkpointing True --save_strategy no --report_to none \
    --per_device_train_batch_size 4 --gradient_accumulation_steps 2 \
    --max_steps 100 --learning_rate 1.5 --lr_scheduler_type cosine \
    --w_bits 2 --a_bits 16 --k_bits 16 --v_bits 16 --w_asym --w_clip
)
```

## Fixed-rank DMA

The patch captures the rotated weight `W` and raw calibration Gram `H` before GPTQ writes `Q`. The archived SpinQuant PAC variant truncates `float(W - Q) @ H` and solves on the right by `H`. This full-Gram weighting gives an idealized `H²` metric for symmetric `H`; the square-root weighting in the paper's main method gives an `H` metric. After the PAC update is merged in the weight dtype, OGC fits the remaining residual and adds its update in the same dtype. GPTQ calibrates in `[K,V,Q]`, `[O]`, `[up,gate]`, `[down]` order, with each group seeing the preceding groups’ merged weights. The appendix path uses DMA alone and evaluates the resulting dense model. [ROCOB paper](https://aclanthology.org/2026.aacl-long.158/)

## Attribution

The patch adapts Meta Platforms, Inc. and affiliates’ SpinQuant code at `8f47aa3f00e8662caf1a484153920a07e5281c3a`. SpinQuant is by Zechun Liu, Changsheng Zhao, Igor Fedorov, Bilge Soran, Dhruv Choudhary, Raghuraman Krishnamoorthi, Vikas Chandra, Yuandong Tian, and Tijmen Blankevoort. The ROCOB modifications add fixed-rank DMA, calibration batching and adaptive GPTQ damping. [SpinQuant paper](https://arxiv.org/abs/2405.16406) · [Upstream source and credits](https://github.com/facebookresearch/SpinQuant/tree/8f47aa3f00e8662caf1a484153920a07e5281c3a)

The experiment patch and launcher use CC BY-NC 4.0. Section 5 of the included SpinQuant license gives the warranty disclaimer. The GPTQ patch preserves the source notice for QuaRot, which carries Apache 2.0 terms. [SpinQuant license](LICENSE) · [QuaRot license](LICENSE-QuaRot) · [QuaRot source](https://github.com/spcl/QuaRot)
