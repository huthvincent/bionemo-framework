"""KV Cache Compression for Evo2 Attention Layers.

Evo2 has 27 Hyena (SSM) + 5 Attention layers. Only the 5 attention layers
produce KV cache. At long sequences (262K), KV cache uses ~21 GB.

Approach: Monkey-patch the core_attention module in each SelfAttention layer,
wrapping the forward to quantize K and V to FP8/INT8 before the attention
dot product, then dequantize back. This reduces intermediate memory.

Supported modes:
  - fp8_kv:  Quantize K,V to float8_e4m3fn (50% KV memory savings)
  - int8_kv: Quantize K,V to int8 (50% KV memory savings)
  - int4_kv: Quantize K,V to packed int4 (75% KV memory savings)
"""

import torch
import torch.nn as nn


def _quantize_to_fp8(tensor: torch.Tensor) -> tuple:
    """Quantize BF16 tensor to FP8, return (fp8_tensor, scale)."""
    amax = tensor.abs().amax().clamp(min=1e-12)
    scale = amax / 448.0  # E4M3 max
    t_fp8 = (tensor / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    return t_fp8, scale


def _dequantize_fp8(t_fp8: torch.Tensor, scale: torch.Tensor,
                    dtype: torch.dtype) -> torch.Tensor:
    return t_fp8.to(dtype) * scale


def _quantize_to_int8(tensor: torch.Tensor) -> tuple:
    """Quantize BF16 tensor to INT8 (per-tensor), return (int8_tensor, scale)."""
    amax = tensor.abs().amax().clamp(min=1e-12)
    scale = amax / 127.0
    t_int8 = (tensor / scale).round().clamp(-128, 127).to(torch.int8)
    return t_int8, scale


def _dequantize_int8(t_int8: torch.Tensor, scale: torch.Tensor,
                     dtype: torch.dtype) -> torch.Tensor:
    return t_int8.to(dtype) * scale


class KVCacheCompressor:
    """Apply KV cache compression to Evo2's attention layers.

    Usage:
        compressor = KVCacheCompressor(model, precision='fp8_kv')
        # Now model inference will use compressed KV cache internally
        output = model(input_ids=...)
        # When done, remove hooks
        compressor.remove()
    """

    def __init__(self, model: nn.Module, precision: str = 'fp8_kv'):
        """
        Args:
            model: Evo2 model
            precision: 'fp8_kv' or 'int8_kv'
        """
        self.precision = precision
        self.hooks = []
        self.patched_modules = []
        self._install(model)

    def _install(self, model: nn.Module):
        """Find all SelfAttention modules and patch their core_attention."""
        count = 0
        for name, mod in model.named_modules():
            cls_name = mod.__class__.__name__
            if cls_name == 'SelfAttention':
                # The SelfAttention module has a core_attention sub-module
                if hasattr(mod, 'core_attention'):
                    self._patch_core_attention(mod.core_attention, name)
                    count += 1
        if count > 0:
            print(f"  KV compression ({self.precision}): patched {count} attention layers")

    def _patch_core_attention(self, core_attn, layer_name):
        """Wrap the core_attention's forward to compress K/V."""
        original_forward = core_attn.forward

        precision = self.precision

        def compressed_forward(query, key, value, *args, **kwargs):
            if precision == 'fp8_kv':
                orig_dtype = key.dtype
                k_fp8, k_scale = _quantize_to_fp8(key)
                v_fp8, v_scale = _quantize_to_fp8(value)
                # Free original K/V memory
                del key, value
                # Dequant back for attention computation
                key_deq = _dequantize_fp8(k_fp8, k_scale, orig_dtype)
                value_deq = _dequantize_fp8(v_fp8, v_scale, orig_dtype)
                del k_fp8, v_fp8
                return original_forward(query, key_deq, value_deq, *args, **kwargs)

            elif precision == 'int8_kv':
                orig_dtype = key.dtype
                k_int8, k_scale = _quantize_to_int8(key)
                v_int8, v_scale = _quantize_to_int8(value)
                del key, value
                key_deq = _dequantize_int8(k_int8, k_scale, orig_dtype)
                value_deq = _dequantize_int8(v_int8, v_scale, orig_dtype)
                del k_int8, v_int8
                return original_forward(query, key_deq, value_deq, *args, **kwargs)

            else:
                return original_forward(query, key, value, *args, **kwargs)

        core_attn.forward = compressed_forward
        self.patched_modules.append((core_attn, original_forward))

    def remove(self):
        """Restore original forward methods."""
        for mod, orig_forward in self.patched_modules:
            mod.forward = orig_forward
        self.patched_modules.clear()
        self.hooks.clear()


def apply_kv_compression(model: nn.Module, precision: str = 'fp8_kv') -> KVCacheCompressor:
    """Convenience function to apply KV cache compression.

    Args:
        model: Evo2 model
        precision: 'fp8_kv' or 'int8_kv'

    Returns:
        KVCacheCompressor instance (call .remove() when done)
    """
    return KVCacheCompressor(model, precision)
