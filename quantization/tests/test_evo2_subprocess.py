#!/usr/bin/env python3
"""Test quantization methods on Evo2 using SUBPROCESS ISOLATION.

Why subprocess isolation for Evo2?
==================================
Evo2's Hyena/SSM layers have NON-DETERMINISTIC initialization:
  1. configure_model() initializes weights randomly
  2. load_state_dict() doesn't fully override all parameters
  3. Different CUDA RNG states between loads → different model weights
  4. mtq.quantize() refuses to re-quantize an already-quantized model

Solution: Each quant method runs as a SEPARATE PYTHON PROCESS:
  1. Fresh process → Fresh model load → Deterministic baseline
  2. Quantize in-place within the same process
  3. Compare quantized vs. baseline (same model instance)
  4. Report results via stdout JSON

This ensures each method is tested against its own BF16 baseline from
the same model load, eliminating cross-load non-determinism.

Supported modes:
  --attention-only:  Only quantize the 5 attention layers (default behavior)
  --all-mlp:         Quantize all 32 MLP layers (27 Hyena + 5 Attention)

Usage:
    # Test all 22 methods (default: attention-only)
    python tests/test_evo2_subprocess.py

    # Test specific methods
    python tests/test_evo2_subprocess.py --quant FP8_DEFAULT_CFG,INT8_DEFAULT_CFG

    # Test with all-MLP quantization (includes Hyena layers)
    python tests/test_evo2_subprocess.py --all-mlp

    # Attention-only mode
    python tests/test_evo2_subprocess.py --attention-only
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime


# All 22 ModelOpt quantization methods
ALL_QUANT_METHODS = [
    "FP8_DEFAULT_CFG",
    "FP8_2D_BLOCKWISE_WEIGHT_ONLY_CFG",
    "FP8_AFFINE_KV_CFG",
    "FP8_KV_CFG",
    "FP8_PER_CHANNEL_PER_TOKEN_CFG",
    "INT4_AWQ_CFG",
    "INT4_BLOCKWISE_WEIGHT_ONLY_CFG",
    "INT8_DEFAULT_CFG",
    "INT8_SMOOTHQUANT_CFG",
    "MXFP4_DEFAULT_CFG",
    "MXFP6_DEFAULT_CFG",
    "MXFP8_DEFAULT_CFG",
    "MXINT8_DEFAULT_CFG",
    "NVFP4_DEFAULT_CFG",
    "NVFP4_AWQ_LITE_CFG",
    "NVFP4_AWQ_CLIP_CFG",
    "NVFP4_AWQ_FULL_CFG",
    "NVFP4_AFFINE_KV_CFG",
    "NVFP4_KV_CFG",
    "NVFP4_KV_ROTATE_CFG",
    "NVFP4_SVDQUANT_DEFAULT_CFG",
    "W4A8_AWQ_BETA_CFG",
]


# ============================================================================
# Worker script: runs as a SUBPROCESS for ONE quantization method
# ============================================================================

WORKER_SCRIPT = r'''
"""Subprocess worker: load Evo2, get BF16 baseline, quantize, compare.

This runs as a standalone Python script in its own process to ensure:
  1. Fresh CUDA RNG state → deterministic model initialization
  2. Clean model → no leftover quantizers from other methods
"""
import copy, json, os, sys, time, random
import torch, torch.nn.functional as F

# Parse arguments
quant_name = sys.argv[1]
enable_all_mlp = "--all-mlp" in sys.argv

# ---- Initialize distributed environment ----
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29599")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("LOCAL_RANK", "0")

import torch.distributed as dist
if not dist.is_initialized():
    dist.init_process_group(backend="nccl", world_size=1, rank=0)
from megatron.core import parallel_state
if not parallel_state.is_initialized():
    parallel_state.initialize_model_parallel(1, 1)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
model_parallel_cuda_manual_seed(42)

# ---- Load Evo2 model ----
import nemo.lightning as nl
from bionemo.core.data.load import load as bionemo_load
from bionemo.evo2.data.tokenizer import Evo2Tokenizer

ckpt = str(bionemo_load("evo2/7b-8k:1.0", source="ngc"))
ctx = nl.io.load_context(ckpt)
config = ctx.model.config
config.initial_ckpt_path = ckpt
tok = Evo2Tokenizer()
tok.vocab_size = tok.tokenizer.vocab_size
model = config.configure_model(tok)
model = model.bfloat16().cuda().eval()

# ---- Generate deterministic test data ----
rng = random.Random(42)
seqs = ["".join(rng.choices("ACGT", k=254)) for _ in range(4)]
all_ids = []
for s in seqs:
    ids = tok.tokenize(s)
    if isinstance(ids, list) and len(ids) > 0 and isinstance(ids[0], list):
        ids = ids[0]
    ids = ids[:256] + [0] * (256 - len(ids[:256]))
    all_ids.append(ids)
input_ids = torch.tensor(all_ids, dtype=torch.long).cuda()
attention_mask = (input_ids != 0).long().cuda()
position_ids = torch.arange(256, device="cuda").unsqueeze(0).expand(4, -1)

# ---- BF16 baseline (BEFORE quantization, same model instance) ----
with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
    out = model(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask)
    bf16_logits = out if isinstance(out, torch.Tensor) else getattr(out, "token_logits", getattr(out, "logits", out))

# ---- Quantize in-place ----
import modelopt.torch.quantization as mtq

cfg = copy.deepcopy(getattr(mtq, quant_name, None))
result = {
    "quant_method": quant_name, "status": "ERROR", "error": "",
    "cos_sim_avg": 0.0, "cos_sim_min": 0.0, "top1_agree": 0.0,
    "top5_overlap": 0.0, "mse": 0.0, "quant_time_s": 0.0,
    "num_quantizers": 0, "num_quant_modules": 0,
    "num_params_m": sum(p.numel() for p in model.parameters()) / 1e6,
    "mode": "all-mlp" if enable_all_mlp else "default",
}

if cfg is None:
    result["error"] = f"Config {quant_name} not found"
    print("RESULT:" + json.dumps(result))
    sys.exit(0)

try:
    # Optionally enable all-MLP quantization
    if enable_all_mlp:
        quant_cfg = cfg.get("quant_cfg", {})
        if "default" in quant_cfg:
            del quant_cfg["default"]

    # Calibration forward loop
    def forward_loop(m):
        for _ in range(4):
            cids = torch.randint(0, 512, (2, 256), dtype=torch.long).cuda()
            cmask = torch.ones_like(cids)
            cpos = torch.arange(256, device="cuda").unsqueeze(0).expand(2, -1)
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
                try:
                    m(input_ids=cids, position_ids=cpos, attention_mask=cmask)
                except Exception:
                    pass

    t0 = time.time()
    num_q = mtq.quantize(model, cfg, forward_loop=forward_loop)
    result["quant_time_s"] = time.time() - t0
    result["num_quantizers"] = num_q if isinstance(num_q, int) else 0

    # Count quantized modules
    for name, m in model.named_modules():
        if "Quant" in type(m).__name__:
            result["num_quant_modules"] += 1

    # ---- Forward pass with quantized model ----
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        out2 = model(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask)
        q_logits = out2 if isinstance(out2, torch.Tensor) else getattr(out2, "token_logits", getattr(out2, "logits", out2))

    # ---- Compute quality metrics ----
    bf = bf16_logits.reshape(-1, bf16_logits.shape[-1]).float()
    qf = q_logits.reshape(-1, q_logits.shape[-1]).float()
    cos = F.cosine_similarity(bf, qf, dim=-1)
    top1_bf = bf.argmax(dim=-1)
    top1_q = qf.argmax(dim=-1)
    top5_bf = bf.topk(5, dim=-1).indices
    top5_q = qf.topk(5, dim=-1).indices
    overlap = sum(len(set(top5_bf[i].tolist()) & set(top5_q[i].tolist())) / 5.0
                  for i in range(top5_bf.shape[0])) / top5_bf.shape[0]

    result["cos_sim_avg"] = cos.mean().item()
    result["cos_sim_min"] = cos.min().item()
    result["top1_agree"] = (top1_bf == top1_q).float().mean().item()
    result["top5_overlap"] = overlap
    result["mse"] = F.mse_loss(bf, qf).item()
    result["status"] = "PASS" if result["cos_sim_avg"] >= 0.90 and result["top1_agree"] >= 0.50 else "FAIL"

except Exception as e:
    import traceback
    result["error"] = traceback.format_exc()[-300:]

print("RESULT:" + json.dumps(result))
'''


# ============================================================================
# Main driver: launch one subprocess per quantization method
# ============================================================================

def run_single_quant(quant_name, port, enable_all_mlp=False):
    """Run one quantization method as an isolated subprocess.

    Args:
        quant_name: ModelOpt quantization config name.
        port: MASTER_PORT for this subprocess (must be unique).
        enable_all_mlp: Whether to enable all-MLP mode.

    Returns:
        Result dict parsed from the subprocess's stdout JSON.
    """
    env = os.environ.copy()
    env["MASTER_PORT"] = str(port)

    cmd = [sys.executable, "-c", WORKER_SCRIPT, quant_name]
    if enable_all_mlp:
        cmd.append("--all-mlp")

    try:
        proc = subprocess.run(
            cmd, env=env, capture_output=True, text=True, timeout=600,
        )
        for line in proc.stdout.splitlines():
            if line.startswith("RESULT:"):
                return json.loads(line[7:])

        error_msg = (proc.stderr or proc.stdout or "No output")[-200:]
        return _error_result(quant_name, error_msg)

    except subprocess.TimeoutExpired:
        return _error_result(quant_name, "Timeout (10 min)")
    except Exception as e:
        return _error_result(quant_name, str(e)[:200])


def _error_result(quant_name, error):
    """Create a default error result dict."""
    return {
        "quant_method": quant_name, "status": "ERROR", "error": error,
        "cos_sim_avg": 0, "cos_sim_min": 0, "top1_agree": 0,
        "top5_overlap": 0, "mse": 0, "quant_time_s": 0,
        "num_quantizers": 0, "num_quant_modules": 0, "num_params_m": 0,
        "mode": "",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Test Evo2 quantization with subprocess isolation",
    )
    parser.add_argument(
        "--quant", type=str, default=None,
        help="Comma-separated list of methods (default: all 22)",
    )
    parser.add_argument(
        "--attention-only", action="store_true",
        help="Only quantize attention layers (default ModelOpt behavior)",
    )
    parser.add_argument(
        "--all-mlp", action="store_true",
        help="Quantize all 32 MLP layers (27 Hyena + 5 Attention)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="CSV output path",
    )
    args = parser.parse_args()

    methods = args.quant.split(",") if args.quant else ALL_QUANT_METHODS
    mode = "all-mlp" if args.all_mlp else "default"

    print(f"\n{'=' * 80}")
    print(f"  Evo2 Quantization Test (Subprocess Isolation)")
    print(f"  Methods: {len(methods)}")
    print(f"  Mode: {mode}")
    print(f"  {'all-mlp: quantize all 32 MLP layers (27 Hyena + 5 Attention)' if args.all_mlp else 'default: only TE Linear layers in 5 attention layers'}")
    print(f"{'=' * 80}")

    results = []
    base_port = 29600
    for i, qname in enumerate(methods):
        print(f"\n  [{i+1}/{len(methods)}] {qname}...", end=" ", flush=True)
        t_start = time.time()
        r = run_single_quant(qname, base_port + i, enable_all_mlp=args.all_mlp)
        wall = time.time() - t_start

        icon = {"PASS": "✅", "FAIL": "❌"}.get(r["status"], "⚠️")
        if r["status"] in ("PASS", "FAIL"):
            print(f"{icon} cos={r['cos_sim_avg']:.4f} "
                  f"top1={r['top1_agree']*100:.1f}% "
                  f"qmod={r.get('num_quant_modules', 0)} "
                  f"t={r['quant_time_s']:.1f}s (wall={wall:.0f}s)")
        else:
            print(f"{icon} {r.get('error', '')[:80]} (wall={wall:.0f}s)")
        results.append(r)

    # Summary table
    print(f"\n\n{'=' * 100}")
    print(f"  EVO2 — {mode} mode — Results")
    print(f"{'=' * 100}")
    print(f"  {'Method':<35} {'St':>4} {'CosSim':>8} {'Top1%':>7} {'Top5%':>7} {'MSE':>10} {'#QM':>4}")
    print(f"  {'-'*35} {'-'*4} {'-'*8} {'-'*7} {'-'*7} {'-'*10} {'-'*4}")
    for r in results:
        if r["status"] in ("PASS", "FAIL"):
            icon = "✅" if r["status"] == "PASS" else "❌"
            print(f"  {r['quant_method']:<35} {icon:>4} {r['cos_sim_avg']:>8.4f} "
                  f"{r['top1_agree']*100:>6.1f}% {r.get('top5_overlap',0)*100:>6.1f}% "
                  f"{r['mse']:>10.6f} {r.get('num_quant_modules',0):>4}")
        else:
            print(f"  {r['quant_method']:<35} {'⚠️':>4} {'—':>8} {'—':>7} {'—':>7} {'—':>10} {'—':>4}")
    print(f"{'=' * 100}")

    n_p = sum(1 for r in results if r["status"] == "PASS")
    n_f = sum(1 for r in results if r["status"] == "FAIL")
    n_e = sum(1 for r in results if r["status"] == "ERROR")
    print(f"  Total: {n_p} PASS, {n_f} FAIL, {n_e} ERROR out of {len(results)}")

    # Save CSV
    output_path = args.output or f"results/evo2_{mode}_{datetime.now():%Y%m%d_%H%M%S}.csv"
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fieldnames = [
        "quant_method", "status", "mode", "cos_sim_avg", "cos_sim_min",
        "top1_agree", "top5_overlap", "mse", "num_quantizers",
        "num_quant_modules", "quant_time_s", "num_params_m", "error",
    ]
    with open(output_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    print(f"\n  📄 Results saved to: {output_path}")


if __name__ == "__main__":
    main()
