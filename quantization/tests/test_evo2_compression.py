#!/usr/bin/env python3
"""Test Evo2 real weight compression (INT8, FP8, INT4).

Runs in a subprocess because Evo2 requires Megatron distributed init.
Tests that compressed inference produces quality above threshold.

Usage (inside BioNeMo Docker):
  python tests/test_evo2_compression.py
  python tests/test_evo2_compression.py --precision int8
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile


WORKER_SCRIPT = r'''
import gc, json, os, sys, time
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", os.environ.get("WORKER_PORT", "29500"))
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("LOCAL_RANK", "0")

import torch
import torch.nn.functional as F
import torch.distributed as dist
if not dist.is_initialized():
    dist.init_process_group(backend="nccl", world_size=1, rank=0)
from megatron.core import parallel_state
if not parallel_state.is_initialized():
    parallel_state.initialize_model_parallel(1, 1)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
model_parallel_cuda_manual_seed(42)

import nemo.lightning as nl
from bionemo.core.data.load import load as bionemo_load
from bionemo.evo2.data.tokenizer import Evo2Tokenizer

sys.path.insert(0, "/workspace/quantization/src")
from compress_model import compress_model

precision = sys.argv[1]
output_file = sys.argv[2]

result = {"precision": precision, "status": "ERROR", "error": ""}

try:
    ckpt = str(bionemo_load("evo2/7b-8k:1.0", source="ngc"))
    ctx = nl.io.load_context(ckpt)
    config = ctx.model.config
    config.initial_ckpt_path = ckpt
    tokenizer = Evo2Tokenizer()
    tokenizer.vocab_size = tokenizer.tokenizer.vocab_size
    model = config.configure_model(tokenizer)
    model = model.bfloat16().cuda().eval()

    # BF16 baseline
    input_ids = torch.randint(0, 512, (1, 1024), dtype=torch.long).cuda()
    position_ids = torch.arange(1024, device="cuda").unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        bf16_out = model(input_ids=input_ids, position_ids=position_ids,
                         attention_mask=attention_mask)

    bf16_logits = bf16_out if isinstance(bf16_out, torch.Tensor) else \
        getattr(bf16_out, "token_logits", getattr(bf16_out, "logits", bf16_out))
    bf16_logits = bf16_logits.detach().clone()
    del bf16_out

    # Compress
    stats = compress_model(model, precision=precision, verbose=True)
    result["mem_saved_pct"] = stats["mem_saved_pct"]
    result["compressed_linears"] = stats["compressed_linears"]

    # Compressed inference
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        q_out = model(input_ids=input_ids, position_ids=position_ids,
                      attention_mask=attention_mask)

    q_logits = q_out if isinstance(q_out, torch.Tensor) else \
        getattr(q_out, "token_logits", getattr(q_out, "logits", q_out))

    bf = bf16_logits.reshape(-1, bf16_logits.shape[-1]).float()
    qf = q_logits.reshape(-1, q_logits.shape[-1]).float()
    cos = F.cosine_similarity(bf, qf, dim=-1)
    result["cos_sim_avg"] = cos.mean().item()
    result["top1_agree"] = (bf.argmax(-1) == qf.argmax(-1)).float().mean().item()
    result["mse"] = F.mse_loss(bf, qf).item()
    result["status"] = "PASS"

except Exception as e:
    import traceback
    result["error"] = traceback.format_exc()[-600:]

with open(output_file, "w") as f:
    json.dump(result, f, indent=2)
'''


def run_compression_test(precision: str, port: int) -> dict:
    """Run one compression test in isolated subprocess."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as wf:
        wf.write(WORKER_SCRIPT)
        worker_path = wf.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as of:
        output_path = of.name

    env = os.environ.copy()
    env["WORKER_PORT"] = str(port)
    env["CUDA_VISIBLE_DEVICES"] = "0"

    cmd = [sys.executable, worker_path, precision, output_path]
    proc = subprocess.run(cmd, env=env, capture_output=False, timeout=600)

    if os.path.exists(output_path):
        with open(output_path) as f:
            result = json.load(f)
    else:
        result = {"precision": precision, "status": "CRASH", "error": "No output"}

    os.unlink(worker_path)
    if os.path.exists(output_path):
        os.unlink(output_path)

    return result


# Quality thresholds per precision
THRESHOLDS = {
    "int8": {"cos_sim": 0.99, "top1": 0.99},
    "fp8":  {"cos_sim": 0.10, "top1": 0.30},   # FP8 dequant path has lower quality
    "int4": {"cos_sim": 0.05, "top1": 0.50},   # INT4 expected degradation
}


def main():
    parser = argparse.ArgumentParser(description="Evo2 Compression Test")
    parser.add_argument("--precision", default="int8",
                        choices=["fp8", "int8", "int4"],
                        help="Precision to test")
    args = parser.parse_args()

    print(f"\n{'=' * 60}")
    print(f"  Evo2 Real Weight Compression Test: {args.precision.upper()}")
    print(f"{'=' * 60}\n")

    result = run_compression_test(args.precision, port=29900)

    status = result.get("status", "?")
    cos = result.get("cos_sim_avg", 0)
    top1 = result.get("top1_agree", 0)
    mse = result.get("mse", 0)
    saved = result.get("mem_saved_pct", 0)

    print(f"  Status: {status}")
    print(f"  Memory saved: {saved:.1f}%")
    print(f"  CosSim: {cos:.6f}")
    print(f"  Top-1:  {top1:.4f}")
    print(f"  MSE:    {mse:.6f}")

    if result.get("error"):
        print(f"  Error:  {result['error'][:200]}")

    # Check thresholds
    thresh = THRESHOLDS.get(args.precision, {"cos_sim": 0.05, "top1": 0.30})
    passed = (status == "PASS"
              and cos >= thresh["cos_sim"]
              and top1 >= thresh["top1"])

    if passed:
        print(f"\n  ✅ PASSED (cos_sim={cos:.4f} >= {thresh['cos_sim']}, "
              f"top1={top1:.4f} >= {thresh['top1']})")
    else:
        print(f"\n  ❌ FAILED")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
