#!/usr/bin/env python3
"""Test ALL 22 ModelOpt quantization methods on one or more BioNeMo models.

Iterates over every quantization config and tests each on the specified model.
For non-Evo2 models, reloads the model for each method to ensure clean state.
For Evo2, use test_evo2_subprocess.py instead (subprocess isolation needed).

Usage:
    # Test all 22 methods on ESM-2
    python tests/test_all_methods.py --model esm2

    # Test all methods on Geneformer
    python tests/test_all_methods.py --model geneformer

    # Test subset of methods
    python tests/test_all_methods.py --model esm2 \
        --quant FP8_DEFAULT_CFG,INT8_DEFAULT_CFG

    # Test all models (except Evo2 — use subprocess test for that)
    python tests/test_all_methods.py --model all
"""

import argparse
import gc
import os
import sys
import time
import traceback
from datetime import datetime

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.adapters import get_adapter, list_models
from src.quantize import quantize_model, ALL_QUANT_METHODS
from src.metrics import compute_metrics, print_summary, save_csv


def test_model_all_methods(model_name, methods, enable_all_mlp=False):
    """Test all specified quantization methods on a single model.

    For each method, reloads the model fresh to ensure no quantizer leakage
    between methods.

    Args:
        model_name: Model identifier.
        methods: List of quantization method names.
        enable_all_mlp: Whether to quantize all MLPs.

    Returns:
        List of result dicts.
    """
    adapter = get_adapter(model_name)

    print(f"\n{'=' * 80}")
    print(f"  {adapter.description}")
    print(f"  Testing {len(methods)} quantization methods")
    print(f"{'=' * 80}")

    # 1. Download & load for baseline
    ckpt_path = adapter.download_checkpoint()
    print(f"\n  Loading {model_name} for BF16 baseline...")
    t0 = time.time()
    model, tokenizer, input_type = adapter.load_model(ckpt_path)
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  ✓ Loaded: {num_params:.1f}M params in {time.time() - t0:.1f}s")

    # 2. Generate test data
    seq_length = 256 if model_name == "evo2" else 128
    input_ids, attention_mask = adapter.generate_test_data(
        tokenizer, batch_size=4, seq_length=seq_length,
    )
    print(f"  ✓ Test data: {input_ids.shape}")

    # 3. BF16 baseline
    bf16_logits = adapter.run_forward(model, input_ids, attention_mask)
    print(f"  ✓ BF16 output: {bf16_logits.shape}")

    # Free baseline model
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # 4. Test each method
    results = []
    for i, qname in enumerate(methods):
        print(f"\n  [{i+1}/{len(methods)}] {qname}...", end=" ", flush=True)

        result = {
            "model": model_name, "quant_method": qname, "status": "ERROR",
            "error": "", "cos_sim_avg": 0.0, "cos_sim_min": 0.0,
            "top1_agree": 0.0, "top5_overlap": 0.0, "mse": 0.0,
            "quant_time_s": 0.0, "infer_time_ms": 0.0, "num_quantizers": 0,
            "num_params_m": num_params,
        }

        model_copy = None
        try:
            model_copy, _, _ = adapter.load_model(ckpt_path)

            q_info = quantize_model(
                model_copy, qname, adapter, tokenizer,
                enable_all_mlp=enable_all_mlp,
                calib_seq_length=seq_length,
            )
            result.update(q_info)

            t0 = time.time()
            q_logits = adapter.run_forward(model_copy, input_ids, attention_mask)
            result["infer_time_ms"] = (time.time() - t0) * 1000

            if q_logits is not None:
                metrics = compute_metrics(bf16_logits, q_logits)
                result.update(metrics)
                result["status"] = (
                    "PASS" if metrics["cos_sim_avg"] >= 0.90
                    and metrics["top1_agree"] >= 0.50 else "FAIL"
                )
            else:
                result["error"] = "Forward returned None"

        except Exception:
            result["error"] = traceback.format_exc()[-200:]

        finally:
            if model_copy is not None:
                del model_copy
            gc.collect()
            torch.cuda.empty_cache()

        icon = {"PASS": "✅", "FAIL": "❌"}.get(result["status"], "⚠️")
        if result["status"] in ("PASS", "FAIL"):
            print(f"{icon} cos={result['cos_sim_avg']:.4f} "
                  f"top1={result['top1_agree']*100:.1f}% "
                  f"q={result['num_quantizers']} t={result['quant_time_s']:.1f}s")
        else:
            print(f"{icon} {result['error'][:80]}")
        results.append(result)

    print_summary(results, f"{model_name.upper()} — All Methods Summary")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Test all quantization methods on BioNeMo models",
    )
    parser.add_argument(
        "--model", type=str, required=True,
        choices=list_models() + ["all"],
        help="Model to test, or 'all' for ESM-2 + Geneformer",
    )
    parser.add_argument(
        "--quant", type=str, default=None,
        help="Comma-separated list (default: all 22 methods)",
    )
    parser.add_argument(
        "--all-mlp", action="store_true",
        help="Quantize all MLPs (Evo2 only)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="CSV output path",
    )
    args = parser.parse_args()

    methods = args.quant.split(",") if args.quant else ALL_QUANT_METHODS

    # "all" = ESM-2 + Geneformer (Evo2 needs subprocess isolation)
    if args.model == "all":
        models = ["esm2", "geneformer"]
    else:
        models = [args.model]

    all_results = []
    for model_name in models:
        try:
            results = test_model_all_methods(
                model_name, methods, enable_all_mlp=args.all_mlp,
            )
            all_results.extend(results)
        except Exception as e:
            print(f"\n  ❌ {model_name} completely failed: {e}")
            traceback.print_exc()

    # Save CSV
    output_path = args.output or f"results/all_methods_{datetime.now():%Y%m%d_%H%M%S}.csv"
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    save_csv(all_results, output_path)

    # Grand summary
    if len(models) > 1:
        print(f"\n\n{'=' * 100}")
        print(f"  GRAND SUMMARY — All Models × All Methods")
        print(f"{'=' * 100}")
        for r in all_results:
            icon = {"PASS": "✅", "FAIL": "❌"}.get(r["status"], "⚠️")
            print(f"  {r['model']:>12} | {r['quant_method']:<35} {icon}")
        print(f"{'=' * 100}\n")


if __name__ == "__main__":
    main()
