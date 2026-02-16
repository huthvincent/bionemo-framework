#!/usr/bin/env python3
"""Benchmark KV Cache Compression for Evo2 — Comprehensive Comparison.

Tests 6 configurations:
  1. BF16 baseline (no compression)
  2. Weight INT8 only
  3. KV FP8 only
  4. KV INT8 only
  5. Weight INT8 + KV FP8 (combined)
  6. Weight INT8 + KV INT8 (combined)

Each config runs in isolated subprocess for clean memory measurement.

Usage (inside BioNeMo Docker):
  python /workspace/quantization/scripts/benchmark_kv_compress.py
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time


# === Subprocess worker ===
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

# Add src to path
sys.path.insert(0, "/workspace/quantization/src")
from compress_model import compress_model
from kv_cache_compress import apply_kv_compression

# Parse args
config_name = sys.argv[1]     # e.g. "bf16", "weight_int8", "kv_fp8", "weight_int8+kv_fp8"
eval_seq_len = int(sys.argv[2])
max_seqlen_search = sys.argv[3] == "1"
output_file = sys.argv[4]

result = {
    "config": config_name,
    "eval_seq_len": eval_seq_len,
    "status": "ERROR", "error": "",
    # Memory
    "mem_model_mb": 0, "mem_peak_eval_mb": 0,
    # Quality
    "cos_sim_avg": 0.0, "cos_sim_min": 0.0, "top1_agree": 0.0, "mse": 0.0,
    # Speed
    "infer_ms": 0.0, "bf16_infer_ms": 0.0,
    # Max sequence length
    "max_seqlen": 0, "max_seqlen_peak_gb": 0.0,
}

try:
    # Load model
    ckpt = str(bionemo_load("evo2/7b-8k:1.0", source="ngc"))
    ctx = nl.io.load_context(ckpt)
    config = ctx.model.config
    config.initial_ckpt_path = ckpt
    tokenizer = Evo2Tokenizer()
    tokenizer.vocab_size = tokenizer.tokenizer.vocab_size
    model = config.configure_model(tokenizer)
    model = model.bfloat16().cuda().eval()

    # Generate eval input
    input_ids = torch.randint(0, 512, (1, eval_seq_len), dtype=torch.long).cuda()
    position_ids = torch.arange(eval_seq_len, device="cuda").unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)

    # BF16 baseline
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        # Warmup
        _ = model(input_ids=input_ids[:, :256], position_ids=position_ids[:, :256],
                  attention_mask=attention_mask[:, :256])
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        bf16_out = model(input_ids=input_ids, position_ids=position_ids,
                         attention_mask=attention_mask)
        torch.cuda.synchronize()
        result["bf16_infer_ms"] = (time.perf_counter() - t0) * 1000

    bf16_logits = bf16_out if isinstance(bf16_out, torch.Tensor) else \
        getattr(bf16_out, "token_logits", getattr(bf16_out, "logits", bf16_out))
    bf16_logits = bf16_logits.detach().clone()

    del bf16_out
    gc.collect()
    torch.cuda.empty_cache()

    # Apply compression(s) based on config
    kv_compressor = None
    weight_stats = None

    if "bf16" == config_name:
        pass  # no compression
    else:
        # Weight compression
        if "weight_int8" in config_name:
            print(f"  Applying weight INT8 compression...")
            weight_stats = compress_model(model, precision="int8", verbose=True)

        if "weight_int4" in config_name:
            print(f"  Applying weight INT4 compression...")
            weight_stats = compress_model(model, precision="int4", verbose=True)

        if "weight_fp8" in config_name:
            print(f"  Applying weight FP8 compression...")
            weight_stats = compress_model(model, precision="fp8", verbose=True)

        # KV cache compression
        if "kv_fp8" in config_name:
            print(f"  Applying KV FP8 compression...")
            kv_compressor = apply_kv_compression(model, precision="fp8_kv")
        elif "kv_int8" in config_name:
            print(f"  Applying KV INT8 compression...")
            kv_compressor = apply_kv_compression(model, precision="int8_kv")

    result["mem_model_mb"] = torch.cuda.memory_allocated() / 1e6

    # Run compressed inference
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        t0 = time.perf_counter()
        q_out = model(input_ids=input_ids, position_ids=position_ids,
                      attention_mask=attention_mask)
        torch.cuda.synchronize()
        result["infer_ms"] = (time.perf_counter() - t0) * 1000

    result["mem_peak_eval_mb"] = torch.cuda.max_memory_allocated() / 1e6

    q_logits = q_out if isinstance(q_out, torch.Tensor) else \
        getattr(q_out, "token_logits", getattr(q_out, "logits", q_out))

    # Quality metrics (compare against BF16 baseline)
    bf = bf16_logits.reshape(-1, bf16_logits.shape[-1]).float()
    qf = q_logits.reshape(-1, q_logits.shape[-1]).float()
    cos = F.cosine_similarity(bf, qf, dim=-1)
    result["cos_sim_avg"] = cos.mean().item()
    result["cos_sim_min"] = cos.min().item()
    top1_bf = bf.argmax(dim=-1)
    top1_q = qf.argmax(dim=-1)
    result["top1_agree"] = (top1_bf == top1_q).float().mean().item()
    result["mse"] = F.mse_loss(bf, qf).item()

    # Free eval tensors
    del bf16_logits, q_logits, q_out, bf, qf, input_ids, position_ids, attention_mask
    gc.collect()
    torch.cuda.empty_cache()

    # Max sequence length search
    if max_seqlen_search:
        print("  Searching max sequence length...")
        test_lens = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768,
                     65536, 131072, 262144, 524288]
        best_len = 0
        best_peak = 0.0

        for slen in test_lens:
            try:
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

                ids = torch.randint(0, 512, (1, slen), dtype=torch.long).cuda()
                pos = torch.arange(slen, device="cuda").unsqueeze(0)
                mask = torch.ones_like(ids)

                with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    _ = model(input_ids=ids, position_ids=pos, attention_mask=mask)
                    torch.cuda.synchronize()

                peak = torch.cuda.max_memory_allocated() / 1e9
                print(f"    seq_len={slen:>8d}: OK (peak={peak:.1f} GB)")
                best_len = slen
                best_peak = peak
                del ids, pos, mask
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                err_str = str(e).lower()
                if "out of memory" in err_str:
                    print(f"    seq_len={slen:>8d}: OOM")
                elif "canuse32bit" in err_str:
                    print(f"    seq_len={slen:>8d}: CUDA index limit")
                else:
                    print(f"    seq_len={slen:>8d}: Error: {str(e)[:100]}")
                try:
                    del ids, pos, mask
                except:
                    pass
                gc.collect()
                torch.cuda.empty_cache()
                break

        result["max_seqlen"] = best_len
        result["max_seqlen_peak_gb"] = best_peak

    result["status"] = "PASS"

except Exception as e:
    import traceback
    result["error"] = traceback.format_exc()[-800:]

# Cleanup
if kv_compressor is not None:
    kv_compressor.remove()

with open(output_file, "w") as f:
    json.dump(result, f, indent=2)
print(f"  Result saved to {output_file}")
'''


def run_test(config_name, eval_seq_len, max_seqlen_search, port, output_dir):
    """Run one test in isolated subprocess."""
    output_file = os.path.join(output_dir, f"result_kv_{config_name}.json")
    worker_file = os.path.join(output_dir, "_worker_kv.py")

    with open(worker_file, "w") as f:
        f.write(WORKER_SCRIPT)

    env = os.environ.copy()
    env["WORKER_PORT"] = str(port)
    env["CUDA_VISIBLE_DEVICES"] = "0"

    cmd = [
        sys.executable, worker_file,
        config_name, str(eval_seq_len),
        "1" if max_seqlen_search else "0",
        output_file,
    ]

    print(f"\n{'─' * 70}")
    print(f"  Config: {config_name}")
    print(f"{'─' * 70}")

    t0 = time.time()
    proc = subprocess.run(cmd, env=env, capture_output=False, timeout=1200)
    elapsed = time.time() - t0

    if os.path.exists(output_file):
        with open(output_file) as f:
            result = json.load(f)
        result["wall_time_s"] = elapsed
        return result
    else:
        return {
            "config": config_name, "status": "CRASH",
            "error": "No output file", "wall_time_s": elapsed,
        }


def main():
    parser = argparse.ArgumentParser(description="Evo2 KV Cache Compression Benchmark")
    parser.add_argument("--configs", nargs="+",
                        default=["bf16", "kv_fp8", "kv_int8",
                                 "weight_int8", "weight_int8+kv_fp8", "weight_int8+kv_int8"],
                        help="Configurations to test")
    parser.add_argument("--eval-seq-len", type=int, default=2048)
    parser.add_argument("--max-seqlen", action="store_true", default=True)
    parser.add_argument("--output-dir", default="/workspace/quantization/results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("  Evo2 KV Cache Compression Benchmark")
    print(f"  Configs: {args.configs}")
    print(f"  Eval seq_len: {args.eval_seq_len}")
    print("=" * 80)

    all_results = []
    port = 29800

    for config_name in args.configs:
        port += 1
        result = run_test(config_name, args.eval_seq_len,
                          args.max_seqlen, port, args.output_dir)

        status = result.get("status", "?")
        mem = result.get("mem_model_mb", 0)
        peak = result.get("mem_peak_eval_mb", 0)
        cos = result.get("cos_sim_avg", 0)
        top1 = result.get("top1_agree", 0)
        ms = result.get("infer_ms", 0)
        max_sl = result.get("max_seqlen", 0)

        print(f"\n  ► {status}: {config_name}")
        print(f"    Model memory: {mem:.0f} MB, Peak: {peak:.0f} MB")
        print(f"    Quality: CosSim={cos:.6f}  Top1={top1:.4f}")
        print(f"    Inference: {ms:.1f} ms")
        if max_sl > 0:
            print(f"    Max SeqLen: {max_sl:,}")

        if result.get("error"):
            print(f"    Error: {result['error'][:300]}")

        all_results.append(result)

    # Save CSV
    csv_path = os.path.join(args.output_dir, "kv_compress_benchmark.csv")
    if all_results:
        keys = sorted(set(k for r in all_results for k in r.keys()))
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in all_results:
                w.writerow({k: r.get(k, "") for k in keys})

    # Summary table
    print(f"\n{'=' * 100}")
    print("  COMPREHENSIVE COMPRESSION RESULTS")
    print(f"{'=' * 100}")
    print(f"  {'Config':<28s} {'ModelMB':>8s} {'PeakMB':>8s} {'CosSim':>8s} "
          f"{'Top1':>6s} {'MSE':>10s} {'InferMs':>8s} {'MaxSeqLen':>10s}")
    print(f"  {'-' * 95}")
    for r in all_results:
        print(f"  {r.get('config','?'):<28s} "
              f"{r.get('mem_model_mb',0):>8.0f} "
              f"{r.get('mem_peak_eval_mb',0):>8.0f} "
              f"{r.get('cos_sim_avg',0):>8.6f} "
              f"{r.get('top1_agree',0):>6.4f} "
              f"{r.get('mse',0):>10.6f} "
              f"{r.get('infer_ms',0):>8.1f} "
              f"{r.get('max_seqlen',0):>10,}")

    print(f"\n✅ Results saved to: {csv_path}")
    print("Done.")


if __name__ == "__main__":
    main()
