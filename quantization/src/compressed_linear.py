"""Compressed Linear modules: real low-precision weight storage for Evo2.

Provides drop-in replacements for nn.Linear that store weights in FP8/INT8/INT4,
achieving real memory savings (50-75%) while maintaining inference quality.

Precision families:
  FP8  → torch.float8_e4m3fn, hardware-accelerated via torch._scaled_mm
  INT8 → torch.int8, dequant-on-the-fly to BF16
  INT4 → 2×INT4 packed into uint8, dequant-on-the-fly to BF16
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FP8Linear(nn.Module):
    """Linear layer with weights stored as float8_e4m3fn (E4M3).

    Uses torch._scaled_mm for hardware-accelerated FP8 matmul on H100/H200.
    Falls back to dequant + BF16 matmul if _scaled_mm fails.
    """

    def __init__(self, weight_fp8: torch.Tensor, scale: torch.Tensor,
                 bias: torch.Tensor = None, out_features: int = 0, in_features: int = 0,
                 return_bias: bool = True):
        super().__init__()
        self.out_features = out_features or weight_fp8.shape[0]
        self.in_features = in_features or weight_fp8.shape[1]
        self.return_bias = return_bias
        self.register_buffer("weight_fp8", weight_fp8)  # float8_e4m3fn
        self.register_buffer("scale", scale)             # scalar or per-channel
        if bias is not None:
            self.register_buffer("bias", bias)
        else:
            self.bias = None
        self._use_scaled_mm = True  # try hardware path first

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        orig_shape = x.shape

        if self._use_scaled_mm:
            try:
                out = self._forward_scaled_mm(x, orig_dtype, orig_shape)
                return (out, None) if self.return_bias else out
            except Exception:
                self._use_scaled_mm = False  # fallback permanently

        out = self._forward_dequant(x, orig_dtype)
        return (out, None) if self.return_bias else out

    def _forward_scaled_mm(self, x, orig_dtype, orig_shape):
        """Hardware-accelerated FP8 matmul via torch._scaled_mm."""
        # Flatten to 2D: (batch*seq, hidden)
        x_2d = x.reshape(-1, x.shape[-1])

        # Scale input to FP8
        x_abs_max = x_2d.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
        x_scale = (x_abs_max / 448.0)  # E4M3 max = 448
        x_fp8 = (x_2d / x_scale).to(torch.float8_e4m3fn)

        # FP8 matmul: (M, K) @ (K, N) = (M, N)
        # _scaled_mm expects: input (M,K), weight (N,K).T → (K,N)
        x_scale_scalar = x_scale.squeeze(-1).max()  # per-tensor scale
        w_scale_scalar = self.scale.max() if self.scale.numel() > 1 else self.scale

        out = torch._scaled_mm(
            x_fp8,
            self.weight_fp8.t(),
            scale_a=x_scale_scalar.float(),
            scale_b=w_scale_scalar.float(),
            out_dtype=orig_dtype,
        )

        if self.bias is not None:
            out = out + self.bias

        return out.reshape(*orig_shape[:-1], self.out_features)

    def _forward_dequant(self, x, orig_dtype):
        """Fallback: dequantize FP8 weights to BF16, then standard matmul."""
        w = self.weight_fp8.to(orig_dtype) * self.scale.to(orig_dtype)
        out = F.linear(x, w, self.bias)
        return out

    def extra_repr(self):
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"bias={self.bias is not None}, dtype=float8_e4m3fn")

    @staticmethod
    def from_linear(linear: nn.Linear, scale: torch.Tensor = None, return_bias: bool = True):
        """Convert nn.Linear to FP8Linear. Quantizes on CPU to save GPU memory."""
        device = linear.weight.data.device
        w = linear.weight.data.cpu().float()
        if scale is None:
            amax = w.abs().amax()
            scale = (amax / 448.0).clamp(min=1e-12)
        w_scaled = (w / scale).clamp(-448, 448)
        w_fp8 = w_scaled.to(torch.float8_e4m3fn).to(device)
        scale = scale.to(device)
        bias = linear.bias.data.to(device) if linear.bias is not None else None
        return FP8Linear(w_fp8, scale, bias, linear.out_features, linear.in_features,
                         return_bias=return_bias)


class INT8Linear(nn.Module):
    """Linear layer with weights stored as int8.

    Dequantizes to BF16 on-the-fly for matmul.
    Per-channel quantization preserves accuracy.
    """

    def __init__(self, weight_int8: torch.Tensor, scale: torch.Tensor,
                 bias: torch.Tensor = None, out_features: int = 0, in_features: int = 0,
                 return_bias: bool = True):
        super().__init__()
        self.out_features = out_features or weight_int8.shape[0]
        self.in_features = in_features or weight_int8.shape[1]
        self.return_bias = return_bias
        self.register_buffer("weight_int8", weight_int8)  # torch.int8
        self.register_buffer("scale", scale)               # per-channel: (out_features, 1)
        if bias is not None:
            self.register_buffer("bias", bias)
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dequantize: int8 → BF16
        w = self.weight_int8.to(x.dtype) * self.scale.to(x.dtype)
        out = F.linear(x, w, self.bias)
        return (out, None) if self.return_bias else out

    def extra_repr(self):
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"bias={self.bias is not None}, dtype=int8")

    @staticmethod
    def from_linear(linear: nn.Linear, scale: torch.Tensor = None, per_channel: bool = True,
                    return_bias: bool = True):
        """Convert nn.Linear to INT8Linear. Quantizes on CPU to save GPU memory."""
        device = linear.weight.data.device
        w = linear.weight.data.cpu().float()
        if scale is None:
            if per_channel:
                amax = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
            else:
                amax = w.abs().amax().unsqueeze(0).unsqueeze(0).clamp(min=1e-12)
            scale = amax / 127.0
        w_int8 = (w / scale).round().clamp(-128, 127).to(torch.int8).to(device)
        scale = scale.to(device)
        bias = linear.bias.data.to(device) if linear.bias is not None else None
        return INT8Linear(w_int8, scale, bias, linear.out_features, linear.in_features,
                          return_bias=return_bias)


class INT4Linear(nn.Module):
    """Linear layer with weights stored as packed int4 (2 values per uint8 byte).

    Per-group quantization for better accuracy.
    Dequantizes to BF16 on-the-fly for matmul.
    """

    def __init__(self, weight_packed: torch.Tensor, scale: torch.Tensor,
                 zeros: torch.Tensor, bias: torch.Tensor = None,
                 out_features: int = 0, in_features: int = 0, group_size: int = 128,
                 return_bias: bool = True):
        super().__init__()
        self.out_features = out_features
        self.in_features = in_features
        self.group_size = group_size
        self.return_bias = return_bias
        self.register_buffer("weight_packed", weight_packed)
        self.register_buffer("scale", scale)
        self.register_buffer("zeros", zeros)
        if bias is not None:
            self.register_buffer("bias", bias)
        else:
            self.bias = None

    def _unpack_and_dequant(self, dtype):
        """Unpack int4 weights and dequantize to target dtype."""
        # Unpack: each uint8 holds 2 int4 values
        packed = self.weight_packed
        low = (packed & 0x0F).to(torch.int8) - 8   # low nibble → signed [-8, 7]
        high = ((packed >> 4) & 0x0F).to(torch.int8) - 8  # high nibble → signed

        # Interleave: (out, in_features)
        w_int4 = torch.stack([low, high], dim=-1).reshape(self.out_features, self.in_features)

        # Per-group dequantize
        w_float = w_int4.to(dtype)
        n_groups = self.in_features // self.group_size
        w_float = w_float.reshape(self.out_features, n_groups, self.group_size)
        scale = self.scale.to(dtype).unsqueeze(-1)  # (out, n_groups, 1)
        zeros = self.zeros.to(dtype).unsqueeze(-1)
        w_float = w_float * scale + zeros
        return w_float.reshape(self.out_features, self.in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self._unpack_and_dequant(x.dtype)
        out = F.linear(x, w, self.bias)
        return (out, None) if self.return_bias else out

    def extra_repr(self):
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"bias={self.bias is not None}, dtype=int4, group_size={self.group_size}")

    @staticmethod
    def from_linear(linear: nn.Linear, group_size: int = 128):
        """Convert nn.Linear to INT4Linear. Quantizes on CPU to save GPU memory."""
        device = linear.weight.data.device
        w = linear.weight.data.cpu().float()
        out_features, in_features = w.shape

        # Pad in_features to multiple of group_size
        if in_features % group_size != 0:
            pad = group_size - (in_features % group_size)
            w = F.pad(w, (0, pad))
            in_features_padded = in_features + pad
        else:
            in_features_padded = in_features

        n_groups = in_features_padded // group_size
        w_grouped = w.reshape(out_features, n_groups, group_size)

        # Per-group scale and zero-point (asymmetric)
        w_min = w_grouped.amin(dim=-1)
        w_max = w_grouped.amax(dim=-1)
        scale = (w_max - w_min) / 15.0
        scale = scale.clamp(min=1e-12)
        zeros = w_min + 8.0 * scale

        # Quantize to int4 range [-8, 7]
        w_q = ((w_grouped - zeros.unsqueeze(-1)) / scale.unsqueeze(-1)).round().clamp(-8, 7)
        w_q = (w_q + 8).to(torch.uint8)

        # Pack: 2 int4 values into 1 uint8
        w_q_flat = w_q.reshape(out_features, in_features_padded)
        low = w_q_flat[:, 0::2]
        high = w_q_flat[:, 1::2]
        packed = (low | (high << 4)).to(device)

        bias = linear.bias.data.to(device) if linear.bias is not None else None
        return INT4Linear(packed, scale.to(device), (zeros - 8.0 * scale).to(device),
                          bias, out_features, in_features_padded, group_size,
                          return_bias=True)


# === Factory function ===

PRECISION_MAP = {
    "fp8": FP8Linear,
    "int8": INT8Linear,
    "int4": INT4Linear,
}

# Map 22 ModelOpt method names to precision families
METHOD_TO_PRECISION = {
    "FP8_DEFAULT_CFG": "fp8",
    "FP8_KV_CFG": "fp8",
    "INT8_DEFAULT_CFG": "int8",
    "INT8_SMOOTHQUANT_CFG": "int8",
    "INT8_KV_CFG": "int8",
    "INT4_AWQ_CFG": "int4",
    "W4A8_AWQ_CFG": "int4",
    "INT4_BLOCKWISE_WEIGHT_ONLY_CFG": "int4",
    "FP8_INT4_BLOCKWISE_CFG": "int4",
    "FP8_INT4_BLOCKWISE_KV_CFG": "int4",
    "NVFP4_DEFAULT_CFG": "int4",
    "NVFP4_KV_CFG": "int4",
    "NVFP4_KV_ROTATE_CFG": "int4",
    "MX_DEFAULT_CFG": "int8",
    "MX_FP8_CFG": "fp8",
    "MX_FP4_CFG": "int4",
    "MX_INT8_CFG": "int8",
    "MXFP8_DEFAULT_CFG": "fp8",
    "MXFP6E2M3_DEFAULT_CFG": "int8",
    "MXFP6E3M2_DEFAULT_CFG": "int8",
    "MXFP4_DEFAULT_CFG": "int4",
    "MXINT8_DEFAULT_CFG": "int8",
}


def compress_linear(linear: nn.Linear, precision: str = "fp8", **kwargs):
    """Convert nn.Linear to compressed format.

    Args:
        linear: Original linear layer
        precision: One of "fp8", "int8", "int4"
        **kwargs: Extra args passed to the compressed linear constructor
    """
    cls = PRECISION_MAP.get(precision)
    if cls is None:
        raise ValueError(f"Unknown precision: {precision}. Choose from: {list(PRECISION_MAP.keys())}")
    return cls.from_linear(linear, **kwargs)
