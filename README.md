<div align="center">
  <h1>BioNeMo × ModelOpt Bridge</h1>
  <h4>Post-Training Quantization for BioNeMo Biological Foundation Models</h4>
</div>

## What This Branch Does

This branch (`feature/modelopt-bridge-quantization`) bridges **NVIDIA BioNeMo Framework** with **NVIDIA ModelOpt** to enable post-training quantization (PTQ) on biological foundation models. It provides:

- An **adapter-pattern bridge** that connects BioNeMo model APIs to ModelOpt's `mtq.quantize()` pipeline
- **22 quantization methods** from ModelOpt applied to biological models
- **Automated quality evaluation** (cosine similarity, top-k agreement, MSE) comparing BF16 baseline vs. quantized
- **Subprocess isolation** for Evo2's non-deterministic Hyena/SSM architecture

## Bridged Quantization Methods (22 from ModelOpt)

All 22 post-training quantization configurations from `modelopt.torch.quantization` are bridged:

| Family | Methods | Weight Bits | Activation Bits | Calibration | Description |
|--------|:-------:|:-----------:|:---------------:|:-----------:|-------------|
| **FP8** | 5 | 8 | 8 / — | No | E4M3 floating point (per-tensor, per-channel, KV cache, blockwise) |
| **INT8** | 2 | 8 | 8 | Partial | Symmetric + SmoothQuant (migrates outlier difficulty W↔A) |
| **INT4** | 2 | 4 | 16 / — | Partial | AWQ (activation-aware) + blockwise weight-only |
| **NVFP4** | 8 | 4 | 8 | Partial | NVIDIA FP4 with AWQ lite/clip/full, KV/affine-KV, SVD variants |
| **MX** | 4 | 4–8 | 4–8 | No | OCP Microscaling (FP4, FP6, FP8, INT8 block-scaled) |
| **Mixed** | 1 | 4 | 8 | Yes | W4A8 AWQ beta (4-bit weights + 8-bit activations) |

## Supported Models

| Model | Architecture | Domain | Parameters | Checkpoint (NGC) |
|-------|-------------|--------|:----------:|-------------------|
| **ESM-2** | Transformer encoder | Protein sequences | 8M | `esm2/8m:2.0` |
| **Geneformer** | Transformer encoder | Gene expression | 10M | `geneformer/10M_241113:2.0` |
| **Evo2** | Hyena/Mamba SSM | DNA sequences | 7B | `evo2/7b-8k:1.0` |

## Quantization Results Summary

### ESM-2 (8M, Protein Transformer)

All 22 methods produce **perfect or near-perfect** quality preservation:

| Method Family | CosSim | Top-1 Agree | Status |
|--------------|:------:|:-----------:|:------:|
| FP8 (5 methods) | 1.0000 | 100% | ✅ All Pass |
| INT8 (2 methods) | 1.0000 | 100% | ✅ All Pass |
| INT4 (2 methods) | 1.0000 | 100% | ✅ All Pass |
| NVFP4 (8 methods) | 1.0000 | 100% | ✅ All Pass |
| MX (4 methods) | 1.0000 | 100% | ✅ All Pass |
| W4A8 (1 method) | 1.0000 | 100% | ✅ All Pass |

### Geneformer (10M, Gene Expression Transformer)

Similarly, all methods maintain quality:

| Method Family | CosSim | Top-1 Agree | Status |
|--------------|:------:|:-----------:|:------:|
| FP8 (5 methods) | 1.0000 | 100% | ✅ All Pass |
| INT8 (2 methods) | 1.0000 | 100% | ✅ All Pass |
| INT4 (2 methods) | 1.0000 | 100% | ✅ All Pass |
| NVFP4 (8 methods) | 1.0000 | 100% | ✅ All Pass |
| MX (4 methods) | 1.0000 | 100% | ✅ All Pass |
| W4A8 (1 method) | 1.0000 | 100% | ✅ All Pass |

### Evo2 (7B, Hyena/Mamba SSM)

Evo2's unique SSM architecture requires special handling:

| Mode | Scope | CosSim | Top-1 | Status |
|------|-------|:------:|:-----:|:------:|
| **Default** | 5 attention layers only (TE Linear) | 1.0000 | 100% | ✅ All 22 Pass |
| **All-MLP** | 32 MLP layers (27 Hyena + 5 Attn) | 1.0000 | 100% | ✅ FP8/INT8/INT4 Pass |

> **Note**: HyenaFilter internal `nn.Linear` layers and filter parameters (`gamma`, `R`, `p`, `h`) are not quantizable by ModelOpt — they reside inside `nn.Sequential` or are `nn.Parameter`, not `nn.Linear` modules.

## Quick Start

```bash
# 1. Start BioNeMo container
bash quantization/docker/start_container.sh

# 2. Download models (inside container)
bash scripts/download_models.sh

# 3. Quick smoke test (6 methods × 3 models)
bash scripts/run_quantization.sh all-quick

# 4. Full sweep on one model
python tests/test_all_methods.py --model esm2

# 5. Evo2 with subprocess isolation
python tests/test_evo2_subprocess.py --quant FP8_DEFAULT_CFG,INT8_DEFAULT_CFG
```

## Project Structure

```
quantization/
├── src/
│   ├── adapters.py       # Adapter pattern bridge (ESM-2, Geneformer, Evo2)
│   ├── quantize.py       # ModelOpt mtq.quantize() wrapper (22 methods)
│   └── metrics.py        # Quality metrics (cosine sim, top-k, MSE)
├── tests/
│   ├── test_single_model.py      # One model × one method
│   ├── test_all_methods.py       # One model × all 22 methods
│   └── test_evo2_subprocess.py   # Evo2 subprocess isolation
├── scripts/
│   ├── run_quantization.sh       # Master config & runner
│   └── download_models.sh        # NGC model downloader
├── docker/
│   └── start_container.sh        # BioNeMo container launcher
├── configs/
│   └── quant_methods.yaml        # All 22 methods documented
└── README.md
```

## Prerequisites

- NVIDIA GPU (Hopper+ recommended for FP8/NVFP4 hardware acceleration)
- Docker with NVIDIA Container Toolkit
- NGC access for model checkpoint downloads
- BioNeMo Framework container v2.6.3+
