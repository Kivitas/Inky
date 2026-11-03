"""Weight-only quantization for inference: int8 per output channel, or packed int4 group-wise.

Weights are stored as integer buffers with float scales and dequantized on the fly in ``forward``.
Activations, norms, embeddings and the decision engine stay in float.
"""

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_SKIP_NAMES = ("embed_tokens", "arlo", "halting", "step_embedding")

# Upper bound on weight elements materialised at once by the fused (cache-off) forward path.
_FUSED_CHUNK_ELEMS = 1 << 20


def _group_size_for(in_features: int, group_size: int) -> int:
    """Largest divisor of ``in_features`` that is <= ``group_size``."""
    g = max(1, min(int(group_size), in_features))
    while in_features % g != 0:
        g -= 1
    return g


class QuantLinear(nn.Module):
    """Linear layer with int8 (per-row) or packed int4 (group-wise) weights.

    int8: ``qweight`` is int8 [out, in]; ``scale`` is float [out, 1], with scale = max|w| / 127 per row.
    int4: ``qweight`` is uint8 [out, ceil(in / 2)], two unsigned nibbles per byte (value + 8, so the
    range is -8..7). ``scale`` is float [out, in / group_size], with scale = max|w| / 7 per group.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bits: int = 8,
        group_size: int = 64,
        bias: bool = False,
        cache_dequantized: bool = False,
    ):
        super().__init__()
        if bits not in (4, 8):
            raise ValueError(f"bits must be 4 or 8, got {bits}")
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self._weight_dtype = torch.float32  # dtype of the float layer this one replaces
        # When True the full float weight is dequantized once per dtype and kept (float memory cost).
        # When False, forward dequantizes one block of output rows at a time and never holds the full weight.
        self.cache_dequantized = bool(cache_dequantized)
        self._dequant_cache: Optional[torch.Tensor] = None  # plain attribute, not a buffer

        if bits == 8:
            self.group_size = in_features
            self.register_buffer("qweight", torch.zeros(out_features, in_features, dtype=torch.int8))
            self.register_buffer("scale", torch.ones(out_features, 1))
        else:
            self.group_size = _group_size_for(in_features, group_size)
            n_groups = in_features // self.group_size
            self.register_buffer("qweight", torch.zeros(out_features, (in_features + 1) // 2, dtype=torch.uint8))
            self.register_buffer("scale", torch.ones(out_features, n_groups))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_linear(
        cls, linear: nn.Linear, bits: int = 8, group_size: int = 64, cache_dequantized: bool = False
    ) -> "QuantLinear":
        out_f, in_f = linear.weight.shape
        layer = cls(
            in_f,
            out_f,
            bits=bits,
            group_size=group_size,
            bias=linear.bias is not None,
            cache_dequantized=cache_dequantized,
        )
        w = linear.weight.detach().float()
        with torch.no_grad():
            if bits == 8:
                scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / 127.0
                q = torch.round(w / scale).clamp(-127, 127).to(torch.int8)
                layer.qweight.copy_(q)
                layer.scale.copy_(scale)
            else:
                g = layer.group_size
                wg = w.reshape(out_f, in_f // g, g)
                scale = wg.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / 7.0
                q = torch.round(wg / scale).clamp(-8, 7).to(torch.int16) + 8  # 0..15
                q = q.reshape(out_f, in_f).to(torch.uint8)
                if in_f % 2:
                    q = torch.cat([q, torch.full((out_f, 1), 8, dtype=torch.uint8)], dim=1)
                packed = q[:, 0::2] | (q[:, 1::2] << 4)
                layer.qweight.copy_(packed)
                layer.scale.copy_(scale.squeeze(-1))
            if linear.bias is not None:
                layer.bias.copy_(linear.bias.detach().float())
        layer._weight_dtype = linear.weight.dtype
        layer.to(linear.weight.device)
        return layer

    def _dequantize_rows(self, start: int, end: int, dtype: torch.dtype) -> torch.Tensor:
        """Reconstruct output rows [start, end) of the weight matrix in ``dtype``."""
        rows = end - start
        if self.bits == 8:
            w = self.qweight[start:end].float() * self.scale[start:end]
        else:
            packed = self.qweight[start:end]
            lo = (packed & 0x0F).to(torch.int16)
            hi = (packed >> 4).to(torch.int16)
            q = torch.stack([lo, hi], dim=-1).reshape(rows, -1)[:, : self.in_features]
            wg = (q.float() - 8.0).reshape(rows, -1, self.group_size)
            w = (wg * self.scale[start:end].unsqueeze(-1)).reshape(rows, self.in_features)
        return w.to(dtype)

    def dequantize(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Reconstruct the full weight matrix [out, in] in ``dtype``."""
        return self._dequantize_rows(0, self.out_features, dtype)

    def clear_dequant_cache(self) -> None:
        """Drop the cached dequantized weight (call after changing qweight/scale in place)."""
        self._dequant_cache = None

    def _cached_weight(self, dtype: torch.dtype) -> torch.Tensor:
        cached = self._dequant_cache
        if cached is None or cached.dtype != dtype:
            cached = self.dequantize(dtype)
            self._dequant_cache = cached
        return cached

    def _apply(self, *args, **kwargs):
        # Device/dtype moves invalidate the cache; it is rebuilt lazily on the next forward.
        self._dequant_cache = None
        return super()._apply(*args, **kwargs)

    def _load_from_state_dict(self, *args, **kwargs):
        self._dequant_cache = None
        return super()._load_from_state_dict(*args, **kwargs)

    @property
    def weight(self) -> torch.Tensor:
        """Dequantized weight, so code that reads ``.weight`` directly (e.g. MLA decoding) still works."""
        if self.cache_dequantized:
            return self._cached_weight(self._weight_dtype)
        return self.dequantize(self._weight_dtype)

    def _forward_fused(self, x: torch.Tensor, bias: Optional[torch.Tensor]) -> torch.Tensor:
        """Dequantize block-by-block over output rows so the full float weight is never materialised."""
        rows = max(1, _FUSED_CHUNK_ELEMS // self.in_features)
        if rows >= self.out_features:
            return F.linear(x, self.dequantize(x.dtype), bias)
        outs = []
        for start in range(0, self.out_features, rows):
            end = min(start + rows, self.out_features)
            outs.append(F.linear(x, self._dequantize_rows(start, end, x.dtype)))
        y = torch.cat(outs, dim=-1)
        return y if bias is None else y + bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = None if self.bias is None else self.bias.to(x.dtype)
        if self.cache_dequantized:
            return F.linear(x, self._cached_weight(x.dtype), b)
        return self._forward_fused(x, b)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bits={self.bits}, group_size={self.group_size}"


def quantize_model(
    model: nn.Module,
    bits: int = 8,
    group_size: int = 64,
    skip_names: Sequence[str] = DEFAULT_SKIP_NAMES,
    cache_dequantized: bool = False,
) -> nn.Module:
    """Replace every nn.Linear whose qualified name contains no skip name with a QuantLinear, in place.

    ``cache_dequantized=True`` keeps a float copy of each weight (faster decoding, float memory cost).
    The default keeps the quantized storage only and dequantizes block-wise in forward.
    Decode speed needs ``cache_dequantized=True``: with the default, every decoding step dequantizes
    each weight again (GEN-04). Use the default only when memory matters more than speed.
    """
    targets = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and not any(s in name for s in skip_names):
            targets.append((name, module))
    with torch.no_grad():
        for name, linear in targets:
            parent_name, _, child = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(
                parent,
                child,
                QuantLinear.from_linear(linear, bits=bits, group_size=group_size, cache_dequantized=cache_dequantized),
            )
    return model


def model_nbytes(model: nn.Module) -> int:
    """Total bytes of all parameters and buffers."""
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    total += sum(b.numel() * b.element_size() for b in model.buffers())
    return int(total)
