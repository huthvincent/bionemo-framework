#!/bin/bash
# ==============================================================================
#  BioNeMo Quantization Runner — Master Configuration & Execution Script
# ==============================================================================
#
# This script is the SINGLE SOURCE OF TRUTH for all quantization experiments.
# It defines:
#   1. Which models to quantize (ESM-2, Geneformer, Evo2)
#   2. Which quantization methods to use (22 ModelOpt configs)
#   3. Which layers to target (TE-only vs all-MLP)
#   4. Calibration parameters (batch size, sequence length, iterations)
#   5. Output paths for results
#
# Usage:
#   bash scripts/run_quantization.sh                    # Run all experiments
#   bash scripts/run_quantization.sh esm2               # Run ESM-2 only
#   bash scripts/run_quantization.sh evo2               # Run Evo2 only
#   bash scripts/run_quantization.sh evo2 --all-mlp     # Evo2 with all MLPs
#   bash scripts/run_quantization.sh all-quick           # Quick smoke test
#
# Must run INSIDE the BioNeMo Docker container.
# ==============================================================================

set -euo pipefail

# ==========================================================
#  SECTION 1: MODEL CONFIGURATION
# ==========================================================

# ---------- Model: ESM-2 ----------
# Architecture:  Transformer encoder
# Domain:        Protein sequences (amino acids)
# Checkpoint:    NGC esm2/8m:2.0 (8M parameters)
# Layers:        6 Transformer layers, each with:
#                  - TELayerNormColumnParallelLinear (QKV projection)
#                  - TERowParallelLinear (output projection)
#                  - MLP (ColumnParallelLinear + RowParallelLinear)
# Quantizable:   All TE Linear layers (default ModelOpt behavior)
ESM2_TAG="esm2/8m:2.0"
ESM2_SEQ_LENGTH=128
ESM2_BATCH_SIZE=4
ESM2_CALIB_BATCHES=4

# ---------- Model: Geneformer ----------
# Architecture:  Transformer encoder
# Domain:        Single-cell gene expression (gene rank tokens)
# Checkpoint:    NGC geneformer/10M_241113:2.0 (10M parameters)
# Layers:        6 Transformer layers (same structure as ESM-2)
# Quantizable:   All TE Linear layers (default ModelOpt behavior)
# Note:          Requires patching load_settings_from_checkpoint
GENEFORMER_TAG="geneformer/10M_241113:2.0"
GENEFORMER_SEQ_LENGTH=128
GENEFORMER_BATCH_SIZE=4
GENEFORMER_CALIB_BATCHES=4

# ---------- Model: Evo2 ----------
# Architecture:  Hyena/Mamba SSM + Attention hybrid
# Domain:        DNA nucleotide sequences (A/C/G/T)
# Checkpoint:    NGC evo2/7b-8k:1.0 (7B parameters)
# Layers:        32 layers total:
#                  - 27 HyenaLayer: Hyena convolution + MLP
#                  - 5 TransformerLayer: Self-attention + MLP
# Quantizable layers by mode:
#   default:     Only TE Linear in 5 attention layers (~10% of model)
#   all-mlp:     All 32 MLP layers (27 Hyena + 5 Attention, using QuantMLP)
# NOT quantizable (always bypassed):
#   - HyenaFilter MLP (nn.Linear inside nn.Sequential, not traversed by MTQ)
#   - ImplicitModalFilter parameters (gamma, R, p) — nn.Parameter, not Linear
#   - ExplicitSingleDecayFilter parameter (h) — nn.Parameter, not Linear
#   - ParallelCausalDepthwiseConv1d — excluded by default config
#   - FFT convolution kernels — operator-level, not quantizable modules
# IMPORTANT: Evo2 requires subprocess isolation for quantization testing
#   due to non-deterministic model initialization in Hyena/SSM layers.
EVO2_TAG="evo2/7b-8k:1.0"
EVO2_SEQ_LENGTH=256
EVO2_BATCH_SIZE=2
EVO2_CALIB_BATCHES=4


# ==========================================================
#  SECTION 2: QUANTIZATION METHODS
# ==========================================================

# All 22 ModelOpt quantization configurations, organized by family.
# Each method is a key in modelopt.torch.quantization (e.g., mtq.FP8_DEFAULT_CFG).

# --- FP8 family (E4M3 / E5M2 floating point) ---
# Highest precision among quantized formats. Minimal quality loss.
# [Weight bits] / [Activation bits]
FP8_METHODS=(
    "FP8_DEFAULT_CFG"                     # FP8 E4M3, per-tensor           (8/8)
    "FP8_PER_CHANNEL_PER_TOKEN_CFG"       # FP8, per-channel W, per-token A (8/8)
    "FP8_KV_CFG"                          # FP8 with KV cache quantization  (8/8)
    "FP8_AFFINE_KV_CFG"                   # FP8 with affine KV quantization (8/8)
    "FP8_2D_BLOCKWISE_WEIGHT_ONLY_CFG"    # FP8 2D blockwise, weight-only   (8/-)
)

# --- INT8 family ---
# Good balance of compression and quality. SmoothQuant handles outliers.
INT8_METHODS=(
    "INT8_DEFAULT_CFG"                    # INT8 symmetric, per-tensor       (8/8)
    "INT8_SMOOTHQUANT_CFG"                # INT8 SmoothQuant (migrates difficulty) (8/8)
)

# --- INT4 family ---
# Higher compression, may show quality degradation on sensitive layers.
INT4_METHODS=(
    "INT4_AWQ_CFG"                        # INT4 AWQ (activation-aware)      (4/16)
    "INT4_BLOCKWISE_WEIGHT_ONLY_CFG"      # INT4 blockwise, weight-only      (4/-)
)

# --- NVFP4 family (NVIDIA FP4 format) ---
# NVIDIA's custom 4-bit floating point. Various calibration strategies.
NVFP4_METHODS=(
    "NVFP4_DEFAULT_CFG"                   # NVFP4 default                    (4/8)
    "NVFP4_AWQ_LITE_CFG"                  # NVFP4 with AWQ lite calibration  (4/8)
    "NVFP4_AWQ_CLIP_CFG"                  # NVFP4 with AWQ clip calibration  (4/8)
    "NVFP4_AWQ_FULL_CFG"                  # NVFP4 with full AWQ calibration  (4/8)
    "NVFP4_KV_CFG"                        # NVFP4 with KV cache quant        (4/8)
    "NVFP4_KV_ROTATE_CFG"                 # NVFP4 with rotated KV quant      (4/8)
    "NVFP4_AFFINE_KV_CFG"                 # NVFP4 with affine KV quant       (4/8)
    "NVFP4_SVDQUANT_DEFAULT_CFG"          # NVFP4 with SVD quantization      (4/8)
)

# --- MX (Microscaling) family ---
# Block-scaled formats. Efficient for hardware with MX support.
MX_METHODS=(
    "MXFP4_DEFAULT_CFG"                   # Microscaling FP4                 (4/4)
    "MXFP6_DEFAULT_CFG"                   # Microscaling FP6                 (6/6)
    "MXFP8_DEFAULT_CFG"                   # Microscaling FP8                 (8/8)
    "MXINT8_DEFAULT_CFG"                  # Microscaling INT8                (8/8)
)

# --- Mixed precision ---
W4A8_METHODS=(
    "W4A8_AWQ_BETA_CFG"                   # 4-bit weights + 8-bit activations (4/8)
)

# Combined: all 22 methods
ALL_METHODS=(
    "${FP8_METHODS[@]}"
    "${INT8_METHODS[@]}"
    "${INT4_METHODS[@]}"
    "${NVFP4_METHODS[@]}"
    "${MX_METHODS[@]}"
    "${W4A8_METHODS[@]}"
)

# Quick smoke test: one representative from each family
QUICK_METHODS=(
    "FP8_DEFAULT_CFG"
    "INT8_DEFAULT_CFG"
    "INT4_AWQ_CFG"
    "NVFP4_DEFAULT_CFG"
    "MXFP8_DEFAULT_CFG"
    "W4A8_AWQ_BETA_CFG"
)


# ==========================================================
#  SECTION 3: OUTPUT CONFIGURATION
# ==========================================================

RESULTS_DIR="results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p "${RESULTS_DIR}"


# ==========================================================
#  SECTION 4: EXECUTION
# ==========================================================

run_esm2() {
    local methods_str="${1:-$(IFS=,; echo "${ALL_METHODS[*]}")}"
    echo ""
    echo "============================================================"
    echo "  Running ESM-2 quantization: $(echo "$methods_str" | tr ',' '\n' | wc -l) methods"
    echo "============================================================"
    python tests/test_all_methods.py \
        --model esm2 \
        --quant "$methods_str" \
        --output "${RESULTS_DIR}/esm2_${TIMESTAMP}.csv"
}

run_geneformer() {
    local methods_str="${1:-$(IFS=,; echo "${ALL_METHODS[*]}")}"
    echo ""
    echo "============================================================"
    echo "  Running Geneformer quantization: $(echo "$methods_str" | tr ',' '\n' | wc -l) methods"
    echo "============================================================"
    python tests/test_all_methods.py \
        --model geneformer \
        --quant "$methods_str" \
        --output "${RESULTS_DIR}/geneformer_${TIMESTAMP}.csv"
}

run_evo2() {
    local methods_str="${1:-$(IFS=,; echo "${ALL_METHODS[*]}")}"
    local extra_args="${2:-}"
    echo ""
    echo "============================================================"
    echo "  Running Evo2 quantization (subprocess isolation)"
    echo "  Methods: $(echo "$methods_str" | tr ',' '\n' | wc -l)"
    echo "  Extra args: ${extra_args:-none}"
    echo "============================================================"
    python tests/test_evo2_subprocess.py \
        --quant "$methods_str" \
        ${extra_args} \
        --output "${RESULTS_DIR}/evo2_${TIMESTAMP}.csv"
}

# ---------- Parse CLI arguments ----------
TARGET="${1:-all}"
EXTRA="${2:-}"

case "$TARGET" in
    esm2)
        run_esm2
        ;;
    geneformer)
        run_geneformer
        ;;
    evo2)
        run_evo2 "$(IFS=,; echo "${ALL_METHODS[*]}")" "$EXTRA"
        ;;
    all)
        run_esm2
        run_geneformer
        run_evo2
        ;;
    all-quick)
        methods_str=$(IFS=,; echo "${QUICK_METHODS[*]}")
        run_esm2 "$methods_str"
        run_geneformer "$methods_str"
        run_evo2 "$methods_str"
        ;;
    *)
        echo "Usage: $0 [esm2|geneformer|evo2|all|all-quick] [--all-mlp|--attention-only]"
        echo ""
        echo "Options:"
        echo "  esm2        Test ESM-2 only"
        echo "  geneformer  Test Geneformer only"
        echo "  evo2        Test Evo2 only (with optional --all-mlp or --attention-only)"
        echo "  all         Test all 3 models with all 22 methods"
        echo "  all-quick   Quick smoke test (6 representative methods per model)"
        exit 1
        ;;
esac

echo ""
echo "============================================================"
echo "  ✅ All quantization experiments complete!"
echo "  📄 Results saved to: ${RESULTS_DIR}/"
echo "============================================================"
