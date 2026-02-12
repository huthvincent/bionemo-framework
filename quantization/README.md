# BioNeMo-ModelOpt Bridge: Quantization Testing

Quantization experiments for NVIDIA BioNeMo Framework models using [NVIDIA ModelOpt](https://github.com/NVIDIA/TensorRT-Model-Optimizer).

## Overview

This project provides a bridge between BioNeMo biological foundation models and ModelOpt's post-training quantization (PTQ) toolkit. It tests **22 quantization methods** across **3 model architectures**:

| Model | Architecture | Domain | Parameters | Checkpoint |
|-------|-------------|--------|------------|------------|
| **ESM-2** | Transformer encoder | Protein sequences | 8M | NGC `esm2/8m:2.0` |
| **Geneformer** | Transformer encoder | Gene expression | 10M | NGC `geneformer/10M_241113:2.0` |
| **Evo2** | Hyena/Mamba SSM | DNA sequences | 7B | NGC `evo2/7b-8k:1.0` |

## Project Structure

```
quantization/
├── src/                           # Core modules (adapter pattern)
│   ├── adapters.py                # Model adapters: load, tokenize, forward
│   ├── quantize.py                # ModelOpt quantization wrapper
│   └── metrics.py                 # Quality metrics (cosine sim, top-k, MSE)
├── tests/
│   ├── test_single_model.py       # Test one model × one method
│   ├── test_all_methods.py        # Test one model × all 22 methods
│   └── test_evo2_subprocess.py    # Evo2 subprocess isolation test
├── scripts/
│   ├── run_quantization.sh        # Master config & runner (all details)
│   └── download_models.sh         # Download pretrained checkpoints
├── docker/
│   └── start_container.sh         # Launch BioNeMo container
├── configs/
│   └── quant_methods.yaml         # All 22 methods documented
└── results/                       # CSV output directory
```

## Quick Start

### 1. Start the BioNeMo Container

```bash
bash docker/start_container.sh
```

### 2. Download Models

```bash
# Inside the container:
bash scripts/download_models.sh          # All models
bash scripts/download_models.sh esm2     # ESM-2 only
```

### 3. Run Quantization Tests

```bash
# Quick smoke test (6 representative methods × 3 models)
bash scripts/run_quantization.sh all-quick

# Full sweep on one model
python tests/test_all_methods.py --model esm2

# Single method test
python tests/test_single_model.py --model esm2 --quant FP8_DEFAULT_CFG

# Evo2 with subprocess isolation (required for Evo2)
python tests/test_evo2_subprocess.py --quant FP8_DEFAULT_CFG,INT8_DEFAULT_CFG

# Evo2 with all-MLP quantization (Hyena + Attention layers)
python tests/test_evo2_subprocess.py --all-mlp
```

## Architecture: Adapter Pattern

Each model has a dedicated adapter class (`src/adapters.py`) that encapsulates:

```
ModelAdapter (ABC)
├── download_checkpoint()     → NGC download
├── load_model()              → Model + tokenizer on CUDA
├── generate_test_data()      → Synthetic calibration data
├── run_forward()             → Forward pass + logit extraction
└── build_calibration_args()  → Forward args for mtq.quantize()

Concrete adapters:
├── ESM2Adapter        (protein, Transformer)
├── GeneformerAdapter  (gene expression, Transformer)
└── Evo2Adapter        (DNA, Hyena/Mamba SSM)
```

## Quantization Methods (22 total)

See [`configs/quant_methods.yaml`](configs/quant_methods.yaml) for full details.

| Family | Methods | Weight Bits | Act Bits | Calibration |
|--------|---------|:-----------:|:--------:|:-----------:|
| **FP8** | 5 configs | 8 | 8/— | No |
| **INT8** | 2 configs (default, SmoothQuant) | 8 | 8 | Partial |
| **INT4** | 2 configs (AWQ, blockwise) | 4 | 16/— | Partial |
| **NVFP4** | 8 configs (various calibrations) | 4 | 8 | Partial |
| **MX** | 4 configs (FP4/6/8, INT8) | 4–8 | 4–8 | No |
| **Mixed** | 1 config (W4A8 AWQ) | 4 | 8 | Yes |

## Quality Metrics

For each quantization method, we compare BF16 vs. quantized outputs:

- **Cosine Similarity**: Directional alignment of logit vectors (1.0 = perfect)
- **Top-1 Agreement**: Does the predicted token match? (practical correctness)
- **Top-5 Overlap**: How many top-5 predictions overlap? (ranking stability)
- **MSE**: Raw numerical difference between logits

Pass criteria: `cos_sim ≥ 0.90` and `top1_agree ≥ 50%`

## Known Issues

### Evo2 Non-Determinism
Evo2's Hyena/SSM layers have non-deterministic initialization. `configure_model()` initializes some weights randomly, and `load_state_dict()` doesn't fully override all parameters. **Solution**: Each quant method runs in a separate subprocess (`test_evo2_subprocess.py`) where BF16 baseline and quantized model come from the same load.

### Evo2 Layer Coverage
- **Default mode**: Only 5 attention layers are quantized (TE Linear only)
- **All-MLP mode** (`--all-mlp`): All 32 MLP layers quantized (27 Hyena + 5 Attention)
- **Not quantizable**: HyenaFilter `nn.Linear` (inside `nn.Sequential`, not traversed by MTQ), filter parameters (`gamma`, `R`, `p`, `h`), and conv kernels

## Prerequisites

- NVIDIA GPU with CUDA support (Hopper+ recommended for FP8/NVFP4)
- Docker with NVIDIA Container Toolkit
- NGC access for model downloads
- BioNeMo Framework container (v2.6.3+)

## License

This project follows the BioNeMo Framework license. See the root repository for details.
