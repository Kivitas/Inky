"""Isolated External Memory Bank (B) with Sub-Linear Product-Key Memory (PKM) Indexing.

Implements the Decoupled Knowledge-Reasoning Architecture for Inky:
- ``QuantizedValueTable``: stores M = S^2 memory slots of width d_mem in float32, per-row
  quantized int8, or packed 2-values-per-byte int4 on CPU (or disk mmap), gathering and
  dequantizing strictly the K winning slots per token onto the target compute device.
- ``ProductKeyMemoryBank``: performs O(S * d_k) Cartesian sub-key decomposition (or O(1)
  multi-head rolling n-gram hash routing), sparse value fetch, softmax attention aggregation,
  and gated residual injection into the recurrent core at designated loop steps.
"""

import math
import os
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import InkyConfig
from .layers import RMSNorm

VALID_STORAGE_DTYPES = ("int8", "int4", "float32")
VALID_ROUTING_MODES = ("pkm", "hash")


class QuantizedValueTable(nn.Module):
    """Sparse-fetched external value table storing M = S^2 slots of dimension ``mem_dim``.

    Supports three storage modes:
    - ``"float32"``: standard ``torch.float32`` tensor ``[M, d_mem]``.
    - ``"int8"``: per-row symmetric quantized ``torch.int8`` table ``[M, d_mem]`` plus
      per-row ``scales`` ``[M, 1]`` (``float16``/``float32``).
    - ``"int4"``: packed 2-values-per-byte ``torch.uint8`` table ``[M, d_mem // 2]`` plus
      per-row ``scales`` ``[M, 1]``, unpacking and dequantizing only the K gathered slots.
    """

    def __init__(
        self,
        num_slots: int,
        mem_dim: int,
        storage_dtype: str = "int8",
        scale_dtype: torch.dtype = torch.float16,
        mmap_path: Optional[str] = None,
        pin_memory: bool = False,
        keep_on_cpu: bool = True,
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        if num_slots < 1:
            raise ValueError(f"num_slots must be >= 1, got {num_slots}")
        if mem_dim < 2:
            raise ValueError(f"mem_dim must be >= 2, got {mem_dim}")
        if storage_dtype not in VALID_STORAGE_DTYPES:
            raise ValueError(
                f"storage_dtype must be one of {VALID_STORAGE_DTYPES}, got {storage_dtype!r}"
            )
        if storage_dtype == "int4" and mem_dim % 2 != 0:
            raise ValueError(f"mem_dim must be even for storage_dtype='int4', got {mem_dim}")

        self.num_slots = int(num_slots)
        self.mem_dim = int(mem_dim)
        self.storage_dtype = storage_dtype
        self.scale_dtype = scale_dtype
        self.mmap_path = mmap_path
        self.pin_memory = bool(pin_memory)
        self.keep_on_cpu = bool(keep_on_cpu)

        table, scales = self._allocate_storage(init_std=float(init_std))
        self.register_buffer("table", table)
        if scales is not None:
            self.register_buffer("scales", scales)
        else:
            self.register_buffer("scales", None)

    def _allocate_storage(
        self, init_std: float
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.storage_dtype == "float32":
            shape = (self.num_slots, self.mem_dim)
            table = self._create_tensor(shape, torch.float32, np.float32)
            if self.mmap_path is None:
                if init_std > 0.0:
                    table.normal_(0.0, init_std)
                else:
                    table.zero_()
            return table, None

        if self.storage_dtype == "int8":
            shape = (self.num_slots, self.mem_dim)
            table = self._create_tensor(shape, torch.int8, np.int8)
            scales = torch.ones((self.num_slots, 1), dtype=self.scale_dtype)
            if self.mmap_path is None:
                if init_std > 0.0:
                    table.random_(-64, 65)
                    scales.fill_(max(init_std / 32.0, 1e-6))
                else:
                    table.zero_()
            return table, scales

        # storage_dtype == "int4": 2 signed 4-bit values [-8, 7] offset by +8 into [0, 15] per byte
        shape = (self.num_slots, self.mem_dim // 2)
        table = self._create_tensor(shape, torch.uint8, np.uint8)
        scales = torch.ones((self.num_slots, 1), dtype=self.scale_dtype)
        if self.mmap_path is None:
            if init_std > 0.0:
                table.random_(0, 256)
                scales.fill_(max(init_std / 4.0, 1e-6))
            else:
                # 0x88 encodes (8 - 8 = 0) in both low and high nibbles
                table.fill_(0x88)
        return table, scales

    def _create_tensor(
        self,
        shape: Tuple[int, int],
        torch_dtype: torch.dtype,
        np_dtype: Any,
    ) -> torch.Tensor:
        if self.mmap_path is not None:
            mode = "r+" if os.path.exists(self.mmap_path) else "w+"
            mmap_arr = np.memmap(self.mmap_path, dtype=np_dtype, mode=mode, shape=shape)
            return torch.from_numpy(mmap_arr)
        tensor = torch.empty(shape, dtype=torch_dtype, device="cpu")
        if self.pin_memory and torch.cuda.is_available():
            tensor = tensor.pin_memory()
        return tensor

    def _apply(self, fn, recurse: bool = True):
        """Keep the cold value table on CPU when ``keep_on_cpu`` is True."""
        if not self.keep_on_cpu:
            return super()._apply(fn, recurse=recurse)
        saved_table = self._buffers.get("table", None)
        saved_scales = self._buffers.get("scales", None)
        super()._apply(fn, recurse=recurse)
        if saved_table is not None:
            self._buffers["table"] = saved_table
        if saved_scales is not None:
            self._buffers["scales"] = saved_scales
        return self

    @property
    def storage_bytes(self) -> int:
        """Total bytes consumed by the value table and its quantization scales."""
        total = self.table.numel() * self.table.element_size()
        if self.scales is not None:
            total += self.scales.numel() * self.scales.element_size()
        return total

    @torch.no_grad()
    def write_slots(self, indices: torch.Tensor, values: torch.Tensor) -> None:
        """Populate or update specific memory slots in ``float32``, ``int8``, or ``int4``."""
        idx = indices.detach().to(device=self.table.device, dtype=torch.long).reshape(-1)
        if idx.numel() == 0:
            return
        if bool((idx < 0).any() or (idx >= self.num_slots).any()):
            raise IndexError(f"Slot indices out of bounds for [0, {self.num_slots})")

        val = values.detach().to(device=self.table.device, dtype=torch.float32)
        val = val.reshape(idx.numel(), self.mem_dim)

        if self.storage_dtype == "float32":
            self.table[idx] = val
            return

        if self.storage_dtype == "int8":
            max_abs = val.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
            scale = max_abs / 127.0
            q = torch.clamp(torch.round(val / scale), -127.0, 127.0).to(torch.int8)
            self.table[idx] = q
            assert self.scales is not None
            self.scales[idx] = scale.to(dtype=self.scales.dtype, device=self.scales.device)
            return

        # storage_dtype == "int4"
        max_abs = val.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        scale = max_abs / 7.0
        q_signed = torch.clamp(torch.round(val / scale), -8.0, 7.0).to(torch.int16)
        q_unsigned = (q_signed + 8).to(torch.uint8)
        low = q_unsigned[:, 0::2] & 0x0F
        high = (q_unsigned[:, 1::2] & 0x0F) << 4
        self.table[idx] = low | high
        assert self.scales is not None
        self.scales[idx] = scale.to(dtype=self.scales.dtype, device=self.scales.device)

    def fetch(
        self,
        indices: torch.Tensor,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Sparse-gather only the K winning rows per token and dequantize on ``device``.

        Args:
            indices: Integer tensor of shape ``[..., K]`` with values in ``[0, M)``.
            device: Target compute device for the dequantized values.
            dtype: Target floating-point dtype for the returned values.

        Returns:
            Tensor of shape ``[..., K, mem_dim]`` on ``device`` with dtype ``dtype``.
        """
        target_device = indices.device if device is None else device
        gather_idx = indices.detach().to(device=self.table.device, dtype=torch.long)

        if self.storage_dtype == "float32":
            gathered = self.table[gather_idx]
            return gathered.to(device=target_device, dtype=dtype, non_blocking=True)

        assert self.scales is not None
        gathered_scales = self.scales[gather_idx].to(
            device=target_device, dtype=dtype, non_blocking=True
        )

        if self.storage_dtype == "int8":
            gathered_q = self.table[gather_idx].to(device=target_device, non_blocking=True)
            return gathered_q.to(dtype=dtype) * gathered_scales

        # storage_dtype == "int4": transfer packed bytes [..., K, d_mem // 2] then unpack on device
        gathered_packed = self.table[gather_idx].to(device=target_device, non_blocking=True)
        low = (gathered_packed & 0x0F).to(torch.int8) - 8
        high = ((gathered_packed >> 4) & 0x0F).to(torch.int8) - 8
        unpacked = torch.stack((low, high), dim=-1).reshape(*indices.shape, self.mem_dim)
        return unpacked.to(dtype=dtype) * gathered_scales


class ProductKeyMemoryBank(nn.Module):
    """Isolated External Memory Bank (B) with Sub-Linear Product-Key Memory (PKM) Indexing.

    Decomposes M = S^2 memory slots into two compact sub-key codebooks C^(1), C^(2) of size
    ``[S, d_k / 2]``, reducing lookup complexity from O(M * d_k) to O(S * d_k) (~1,000x
    fewer FLOPs at M = 4.19M). Also supports O(1) deterministic multi-head rolling n-gram
    hash routing (``routing_mode="hash"``). Injects retrieved vectors back into the looped
    recurrent state via a sigmoid-gated residual projection at specified loop steps.
    """

    def __init__(
        self,
        hidden_size: Union[int, InkyConfig] = 384,
        sub_keys: int = 256,
        key_dim: int = 128,
        mem_dim: int = 256,
        top_k: int = 8,
        query_steps: Sequence[int] = (1,),
        routing_mode: str = "pkm",
        hash_ngram: int = 3,
        storage_dtype: str = "int8",
        scale_dtype: torch.dtype = torch.float16,
        rms_norm_eps: float = 1e-6,
        mmap_path: Optional[str] = None,
        keep_table_on_cpu: bool = True,
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        if isinstance(hidden_size, InkyConfig):
            cfg = hidden_size
            hidden_size = int(cfg.hidden_size)
            sub_keys = int(getattr(cfg, "pkm_sub_keys", getattr(cfg, "memory_bank_sub_keys", sub_keys)))
            key_dim = int(getattr(cfg, "pkm_key_dim", getattr(cfg, "memory_bank_key_dim", key_dim)))
            mem_dim = int(getattr(cfg, "pkm_mem_dim", getattr(cfg, "memory_bank_mem_dim", mem_dim)))
            top_k = int(getattr(cfg, "pkm_top_k", getattr(cfg, "memory_bank_top_k", top_k)))
            query_steps = getattr(cfg, "pkm_query_steps", getattr(cfg, "memory_bank_query_steps", query_steps))
            routing_mode = str(getattr(cfg, "pkm_routing_mode", getattr(cfg, "memory_bank_routing_mode", routing_mode)))
            hash_ngram = int(getattr(cfg, "pkm_hash_ngram", getattr(cfg, "memory_bank_hash_ngram", hash_ngram)))
            storage_dtype = str(
                getattr(cfg, "pkm_storage_dtype", getattr(cfg, "memory_bank_storage_dtype", storage_dtype))
            )
            rms_norm_eps = float(getattr(cfg, "rms_norm_eps", rms_norm_eps))
            init_std = float(getattr(cfg, "initializer_range", init_std))

        if int(hidden_size) < 1:
            raise ValueError(f"hidden_size must be >= 1, got {hidden_size}")
        if int(sub_keys) < 1:
            raise ValueError(f"sub_keys must be >= 1, got {sub_keys}")
        if int(key_dim) < 2 or int(key_dim) % 2 != 0:
            raise ValueError(f"key_dim must be an even integer >= 2, got {key_dim}")
        if int(top_k) < 1 or int(top_k) > int(sub_keys):
            raise ValueError(f"top_k must be in [1, sub_keys={sub_keys}], got {top_k}")
        if routing_mode not in VALID_ROUTING_MODES:
            raise ValueError(
                f"routing_mode must be one of {VALID_ROUTING_MODES}, got {routing_mode!r}"
            )
        if int(hash_ngram) < 1:
            raise ValueError(f"hash_ngram must be >= 1, got {hash_ngram}")

        self.hidden_size = int(hidden_size)
        self.sub_keys = int(sub_keys)
        self.num_slots = self.sub_keys * self.sub_keys
        self.key_dim = int(key_dim)
        self.mem_dim = int(mem_dim)
        self.top_k = int(top_k)
        self.query_steps = {int(s) for s in query_steps}
        self.routing_mode = routing_mode
        self.hash_ngram = int(hash_ngram)

        half_d = self.key_dim // 2

        # Pre-norm and query projection W_q in R^{d_k x d_model}
        self.norm = RMSNorm(self.hidden_size, eps=rms_norm_eps)
        self.q_proj = nn.Linear(self.hidden_size, self.key_dim, bias=False)

        # Compact sub-key codebooks C^(1), C^(2) in R^{S x (d_k / 2)}
        self.c1 = nn.Parameter(torch.randn(self.sub_keys, half_d) * (half_d**-0.5))
        self.c2 = nn.Parameter(torch.randn(self.sub_keys, half_d) * (half_d**-0.5))

        # Cold / quantized value store V in R^{S^2 x d_mem}
        self.value_table = QuantizedValueTable(
            num_slots=self.num_slots,
            mem_dim=self.mem_dim,
            storage_dtype=storage_dtype,
            scale_dtype=scale_dtype,
            mmap_path=mmap_path,
            keep_on_cpu=keep_table_on_cpu,
            init_std=init_std,
        )

        # Output projection W_out and Gated Residual Injection W_g
        self.out_proj = nn.Linear(self.mem_dim, self.hidden_size, bias=False)
        self.gate_proj = nn.Linear(2 * self.hidden_size, self.hidden_size, bias=True)

        # Prime multipliers for deterministic multi-head rolling n-gram hash routing
        primes = torch.arange(1, self.top_k + 1, dtype=torch.long) * 2654435761 + 104729
        self.register_buffer("hash_primes", primes, persistent=False)

    def pkm_route(self, h_norm: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute O(S * d_k) Product-Key Top-K slot indices and unnormalized scores.

        Args:
            h_norm: Normalized hidden states of shape ``[B, L, hidden_size]``.

        Returns:
            Tuple ``(top_indices, top_scores)`` each of shape ``[B, L, K]``, where flat
            slot indices lie in ``[0, S^2)`` via ``a * S + b``.
        """
        q = self.q_proj(h_norm)
        q1, q2 = q.chunk(2, dim=-1)

        s1 = F.linear(q1, self.c1.to(dtype=q1.dtype))  # [B, L, S]
        s2 = F.linear(q2, self.c2.to(dtype=q2.dtype))  # [B, L, S]

        v1, idx1 = torch.topk(s1, k=self.top_k, dim=-1)  # [B, L, K]
        v2, idx2 = torch.topk(s2, k=self.top_k, dim=-1)  # [B, L, K]

        leading = h_norm.shape[:-1]
        k_sq = self.top_k * self.top_k

        # Form K x K Cartesian sum grid s^(1)_a + s^(2)_b and flat indices a * S + b
        grid_scores = (v1.unsqueeze(-1) + v2.unsqueeze(-2)).reshape(*leading, k_sq)
        grid_indices = (idx1.unsqueeze(-1) * self.sub_keys + idx2.unsqueeze(-2)).reshape(
            *leading, k_sq
        )

        top_scores, sub_idx = torch.topk(grid_scores, k=self.top_k, dim=-1)
        top_indices = torch.gather(grid_indices, dim=-1, index=sub_idx)
        return top_indices, top_scores

    def hash_route(
        self, input_ids: torch.Tensor, ngram: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Deterministic O(1) multi-head rolling n-gram hash routing without query dot-products.

        Args:
            input_ids: Token IDs of shape ``[B, L]``.
            ngram: Rolling window length ``m`` (defaults to ``self.hash_ngram``).

        Returns:
            Tuple ``(top_indices, top_scores)`` of shape ``[B, L, K]``.
        """
        m = self.hash_ngram if ngram is None else int(ngram)
        if m < 1:
            raise ValueError(f"ngram must be >= 1, got {m}")
        ids = input_ids.to(dtype=torch.long)
        bsz, seq_len = ids.shape
        primes = self.hash_primes.to(device=ids.device).view(1, 1, self.top_k)

        acc = torch.zeros((bsz, seq_len, self.top_k), dtype=torch.long, device=ids.device)
        prime_pow = primes
        for offset in range(m):
            if offset == 0:
                shifted = ids
            elif offset < seq_len:
                pad = torch.zeros((bsz, offset), dtype=torch.long, device=ids.device)
                shifted = torch.cat([pad, ids[:, :-offset]], dim=1)
            else:
                shifted = torch.zeros_like(ids)
            acc = acc + shifted.unsqueeze(-1) * prime_pow
            prime_pow = prime_pow * primes

        # Mask sign bit before modulo so int64 overflow always maps to [0, M)
        top_indices = (acc & 0x7FFFFFFFFFFFFFFF) % self.num_slots
        top_scores = torch.zeros(
            (bsz, seq_len, self.top_k), dtype=torch.float32, device=ids.device
        )
        return top_indices, top_scores

    def write_slots(self, indices: torch.Tensor, values: torch.Tensor) -> None:
        """Convenience wrapper around ``self.value_table.write_slots(indices, values)``."""
        self.value_table.write_slots(indices, values)

    def forward(
        self,
        h: torch.Tensor,
        step: int = 1,
        input_ids: Optional[torch.Tensor] = None,
        fetch_gate: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Query the external memory bank and apply gated residual injection.

        Args:
            h: Hidden state tensor ``[B, L, hidden_size]`` at loop iteration ``step``.
            step: Current recurrent loop step index (queries bank only if ``step in query_steps``).
            input_ids: Optional token IDs ``[B, L]`` required when ``routing_mode == 'hash'``.
            fetch_gate: Optional external executive gate (e.g. from JEPA), broadcastable to
                ``[B, L, hidden_size]`` (shapes ``[B, L]``, ``[B, L, 1]``, or scalar).

        Returns:
            Tuple ``(u_t, info)`` where ``u_t`` has shape ``[B, L, hidden_size]`` and ``info``
            contains ``{"indices", "weights", "gate_mean", "queried"}``.
        """
        if step not in self.query_steps:
            return h, {
                "indices": None,
                "weights": None,
                "gate_mean": torch.zeros((), device=h.device, dtype=h.dtype),
                "queried": False,
            }

        h_norm = self.norm(h)

        if self.routing_mode == "hash":
            if input_ids is None:
                raise ValueError("input_ids must be provided when routing_mode='hash'")
            top_indices, top_scores = self.hash_route(input_ids)
            top_scores = top_scores.to(device=h.device, dtype=h.dtype)
        else:
            top_indices, top_scores = self.pkm_route(h_norm)

        # Sparse value fetch: transfers only [B, L, K, d_mem] to h.device
        values = self.value_table.fetch(top_indices, device=h.device, dtype=h.dtype)

        # Softmax attention aggregation over the K winning slots
        alpha = F.softmax(top_scores.float() / math.sqrt(self.key_dim), dim=-1).to(dtype=h.dtype)
        m_t = torch.einsum("blk,blkd->bld", alpha, values)  # [B, L, d_mem]

        # Gated residual injection into the recurrent state
        m_hat = self.out_proj(m_t)  # [B, L, hidden_size]
        gate = torch.sigmoid(self.gate_proj(torch.cat([h_norm, m_hat], dim=-1)))

        if fetch_gate is not None:
            fg = fetch_gate.to(device=gate.device, dtype=gate.dtype)
            if fg.dim() == gate.dim() - 1:
                fg = fg.unsqueeze(-1)
            gate = gate * fg

        u_t = h + gate * m_hat
        info: Dict[str, Any] = {
            "indices": top_indices,
            "weights": alpha,
            "gate_mean": gate.mean().detach(),
            "queried": True,
        }
        return u_t, info
