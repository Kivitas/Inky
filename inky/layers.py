"""Building blocks for Inky: norms, RoPE, latent-KV attention, shared-expert MoE, recurrent core."""

import math
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from .config import InkyConfig

# Per-layer decoding cache. Latent attention: (c_kv, k_pe), the compressed KV latent [B, T, kv_lora_rank]
# and shared RoPE key [B, 1, T, rope_dim] (a LatentKV, which unpacks like that pair). SSM mixer:
# (state, empty), state [B, hidden, state_dim].
KVEntry = Tuple[torch.Tensor, torch.Tensor]


class LatentKV(tuple):
    """Latent attention cache entry: a growable buffer plus a valid length.

    Unpacks as ``(c_kv, k_pe)`` exactly like the plain tuple it replaces: ``c_kv`` [B, T, rank] and
    ``k_pe`` [B, 1, T, rope] are *views* of the first ``length`` positions of ``c_buf`` [B, cap, rank] and
    ``k_buf`` [B, 1, cap, rope]. Appending a token writes into the spare capacity instead of concatenating
    the whole cache, and the buffers grow geometrically (see ``MultiLatentAttention._append``), so a decode
    step costs O(new tokens) of copying, not O(cache).

    Aliasing contract. An entry never changes after it is built: an append returns a *new* entry that shares
    the buffer. Data beyond ``length`` is scratch. In-place appends are done only when ``writable`` is set,
    which ``InkyModel`` does for entries owned by the cache being extended (``owner``). Any other entry
    (a fork, a snapshot kept by another object, a hand-built tuple, a layer-level call) is copied on its
    first append, so it can never overwrite data another cache still reads.
    """

    def __new__(cls, c_buf: torch.Tensor, k_buf: torch.Tensor, length: int, writable: bool = False):
        obj = super().__new__(cls, (c_buf[:, :length], k_buf[:, :, :length]))
        obj.c_buf = c_buf
        obj.k_buf = k_buf
        obj.length = int(length)
        obj.writable = bool(writable)
        obj.owner = None          # weakref to the InkyCache that may extend this entry in place
        return obj

    def __reduce__(self):
        return (LatentKV, (self.c_buf, self.k_buf, self.length))

    @property
    def capacity(self) -> int:
        return self.c_buf.shape[1]

    def alias_writable(self) -> "LatentKV":
        """Same buffers and length, marked as extendable in place (for the cache that owns the entry)."""
        out = LatentKV(self.c_buf, self.k_buf, self.length, writable=True)
        out.owner = self.owner
        return out

    def truncate(self, new_len: int) -> "LatentKV":
        """Return a view truncated to ``new_len`` positions without copying buffers."""
        if new_len < 0 or new_len > self.length:
            raise ValueError(f"cannot truncate LatentKV of length {self.length} to {new_len}")
        out = LatentKV(self.c_buf, self.k_buf, new_len, writable=False)
        out.owner = self.owner
        return out


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_f = x.float()
        out = x_f * torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + self.eps)
        # Keep the output in the input dtype even if the weight is fp32 (e.g. under autocast), so that
        # latent-KV cache entries are stored in the activation dtype.
        return out.to(x.dtype) * self.weight.to(x.dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_positions: int, base: float):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        freqs = torch.outer(torch.arange(max_positions, dtype=torch.float32), inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        # Persistent so that from_pretrained restores the exact tables instead of rebuilding them.
        self.register_buffer("cos", emb.cos())
        self.register_buffer("sin", emb.sin())

    def _apply(self, fn, recurse: bool = True):
        """Moves and casts the module as usual, but keeps the tables in float32.

        ``model.bfloat16()`` would otherwise round every cos/sin entry to 8 significant bits, which is
        coarse at large positions. apply_rope does the rotation in at least float32, so the tables are
        kept in float32 and only moved to the new device.
        """
        tables = {name: getattr(self, name) for name in ("cos", "sin")}
        super()._apply(fn, recurse)
        for name, table in tables.items():
            self._buffers[name] = table.to(device=getattr(self, name).device, dtype=torch.float32)
        return self

    def forward(self, start: int, length: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.cos[start:start + length], self.sin[start:start + length]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate the last dimension of x (broadcast over batch and heads); cos/sin are [S, D].

    The rotation is computed in at least float32 whatever the dtype of cos/sin, and the result is returned
    in the dtype of x. For float32 inputs this is the plain formula. For bf16 or fp16 inputs it avoids
    the silent promotion to float32 that an fp32 table used to cause (and the dtype mismatch in attention).
    """
    calc = torch.promote_types(x.dtype, torch.float32)
    xf = x.to(calc)
    return (xf * cos.to(calc) + _rotate_half(xf) * sin.to(calc)).to(x.dtype)


def causal_band_mask(S: int, T: int, past_len: int, window: Optional[int], device=None) -> torch.Tensor:
    """Bool mask [S, T] (True = may attend) for S new queries after ``past_len`` cached positions (T = past_len + S).

    Query i sits at absolute position ``past_len + i`` and may attend to key j when ``j <= past_len + i`` and, with
    a ``window``, ``past_len + i - j < window`` (the last ``window`` positions, itself included)."""
    mask = torch.ones(S, T, dtype=torch.bool, device=device).tril(diagonal=past_len)
    if window is not None:
        mask = mask & torch.ones(S, T, dtype=torch.bool, device=device).triu(diagonal=past_len - window + 1)
    return mask


class MultiLatentAttention(nn.Module):
    """Causal multi-latent attention.

    Queries go through a low-rank bottleneck. Keys and values share one compressed latent
    per token, and the RoPE part of the key is shared across heads. The cache stores only
    the latent and the RoPE key, which is much smaller than full per-head K and V.

    QK-norm (``qk_norm=True``) has two modes (``config.qk_norm_mode``):

    * ``"head"``: per-head RMSNorm of q_nope, q_pe, k_nope and k_pe. The key norm acts on the up-projected
      k_nope = W_uk c, which is nonlinear in c, so the absorbed cached-decode path cannot fold W_uk into
      the query. Each decode step up-projects the whole latent cache to per-head k_nope
      (O(T * H * nope * kv_rank) work and a [B, H, T, nope] temporary), which gives up most of the
      decode-time benefit of absorption. The cache itself stays small (latent + RoPE key).
    * ``"latent"``: norms on q_nope, q_pe and k_pe only. There is no per-head k_nope norm: the key's
      magnitude is already controlled by the RMSNorm ``kv_a_norm`` on the latent c. The nope score is
      q_nope . (W_uk c) = (W_uk^T q_nope) . c, which is linear in c, so folding W_uk into the (already
      normed) query is exact and the absorbed decode path equals the full path up to float rounding.
      A further norm on the absorbed query q_c = W_uk^T q_nope is deliberately NOT applied: the full path
      would then need per-head latent-width keys (rank instead of nope) to match it, losing the very
      saving that absorption gives. Decode cost is the same as with ``qk_norm=False``.

    Cache. Entries are ``LatentKV`` buffers with spare capacity; a new token is written into the buffer
    rather than concatenated (see ``_append``).
    """

    def __init__(self, config: InkyConfig):
        super().__init__()
        d = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.nope_dim = config.qk_nope_head_dim
        self.rope_dim = config.qk_rope_head_dim
        self.qk_dim = config.qk_head_dim
        self.v_dim = config.v_head_dim
        self.kv_rank = config.kv_lora_rank

        self.q_a_proj = nn.Linear(d, config.q_lora_rank, bias=False)
        self.q_a_norm = RMSNorm(config.q_lora_rank, config.rms_norm_eps)
        self.q_b_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.qk_dim, bias=False)

        self.kv_a_proj = nn.Linear(d, self.kv_rank + self.rope_dim, bias=False)
        self.kv_a_norm = RMSNorm(self.kv_rank, config.rms_norm_eps)
        self.kv_b_proj = nn.Linear(self.kv_rank, self.num_heads * (self.nope_dim + self.v_dim), bias=False)

        self.o_proj = nn.Linear(self.num_heads * self.v_dim, d, bias=False)
        self.scale = self.qk_dim ** -0.5
        self.max_len = int(config.max_position_embeddings)   # cache capacity ceiling (see _append)
        # Optional attention output gate (config.attention_gate): one sigmoid per head, from the block input.
        self.out_gate = nn.Linear(d, self.num_heads) if config.attention_gate else None

        # Optional QK-norm (per head, over the nope and rope parts separately). Created only when enabled,
        # so the default parameter set is unchanged. In "head" mode keys are normalised after
        # up-projection, so the absorbed path has to materialise k_nope (see _absorbed_attention). In
        # "latent" mode there is no k_nope norm and absorption stays exact.
        self.qk_norm = bool(config.qk_norm)
        self.qk_norm_mode = str(getattr(config, "qk_norm_mode", "head"))
        self.head_k_norm = self.qk_norm and self.qk_norm_mode == "head"
        if self.qk_norm:
            self.q_nope_norm = RMSNorm(self.nope_dim, config.rms_norm_eps)
            self.q_pe_norm = RMSNorm(self.rope_dim, config.rms_norm_eps)
            if self.head_k_norm:
                self.k_nope_norm = RMSNorm(self.nope_dim, config.rms_norm_eps)
            self.k_pe_norm = RMSNorm(self.rope_dim, config.rms_norm_eps)

    def _split_q(self, q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Splits the projected queries [B, H, S, qk] into nope and rotated rope parts."""
        q_nope, q_pe = q.split([self.nope_dim, self.rope_dim], dim=-1)
        if self.qk_norm:
            q_nope = self.q_nope_norm(q_nope)
            q_pe = self.q_pe_norm(q_pe)
        return q_nope, apply_rope(q_pe, cos, sin)

    def _prep_k_pe(self, k_pe: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Shared RoPE key [B, 1, S, rope], normalised (if enabled) and rotated before it is cached."""
        if self.qk_norm:
            k_pe = self.k_pe_norm(k_pe)
        return apply_rope(k_pe, cos, sin)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past: Optional[KVEntry] = None,
        attn_mask: Optional[torch.Tensor] = None,
        segment_ids: Optional[torch.Tensor] = None,
        window: Optional[int] = None,
    ) -> Tuple[torch.Tensor, KVEntry]:
        """``attn_mask`` (bool, broadcastable to [B, 1, S, past+S] or [B, S, past+S]; True = may attend)
        and ``segment_ids`` (int [B, S]; a query attends only to keys with the same id, prefill only)
        are optional and are combined with the causal mask. Both None gives the default behaviour.
        ``window`` (default None = global) limits each query to the last ``window`` positions, itself included;
        the cache entry still holds every position, a one-token decode just reads the last ``window`` of them."""
        B, S, _ = x.shape
        H = self.num_heads

        q = self.q_b_proj(self.q_a_norm(self.q_a_proj(x))).view(B, S, H, self.qk_dim).transpose(1, 2)
        q_nope, q_pe = self._split_q(q, cos, sin)

        c_kv, k_pe = self.kv_a_proj(x).split([self.kv_rank, self.rope_dim], dim=-1)
        c_kv = self.kv_a_norm(c_kv)
        k_pe = self._prep_k_pe(k_pe.unsqueeze(1), cos, sin)  # [B, 1, S, rope]

        if past is not None:
            present = self._append(past, c_kv, k_pe)
            c_kv, k_pe = present
        else:
            present = LatentKV(c_kv, k_pe, S)
        past_len = c_kv.shape[1] - S
        windowed = window is not None and past_len + S > window          # a window that covers everything is global

        if attn_mask is None and segment_ids is None and (S == 1 or not windowed):
            if S == 1:
                if windowed:                                             # the new token sees the last `window` keys
                    c_kv, k_pe = c_kv[:, -window:], k_pe[:, :, -window:]
                out = self._decode_absorbed(q_nope, q_pe, c_kv, k_pe)
            else:
                out = self._full_attention(q_nope, q_pe, c_kv, k_pe, past_len)
        else:
            extra = self._extra_mask(B, S, past_len, past is not None, attn_mask, segment_ids, x.device)
            out = self._full_attention(q_nope, q_pe, c_kv, k_pe, past_len, extra_mask=extra,
                                       window=window if windowed else None)

        if self.out_gate is not None:                                    # out [B, H, S, v], gate [B, S, H]
            out = out * torch.sigmoid(self.out_gate(x)).transpose(1, 2).unsqueeze(-1).to(out.dtype)
        out = out.transpose(1, 2).reshape(B, S, H * self.v_dim)
        return self.o_proj(out), present

    def _grown_capacity(self, need: int, cap: int) -> int:
        """Next buffer capacity: double, but never past ``max_position_embeddings`` (unless ``need`` is larger)."""
        return max(need, min(2 * cap, max(self.max_len, need)))

    def _append(self, past: KVEntry, c_new: torch.Tensor, k_new: torch.Tensor) -> LatentKV:
        """Returns the cache entry holding ``past`` followed by the new latents (c_new [B, S, rank],
        k_new [B, 1, S, rope]), without concatenating the cache.

        The new latents are written into the spare capacity of the entry's buffer, which is possible
        when the entry is ``writable`` (see LatentKV) and has room. Otherwise a buffer is allocated
        (geometric growth, bounded by ``max_position_embeddings``) and the valid prefix is copied once.
        A plain tuple ``past`` is accepted and copied on the first append. When autograd must track the
        cache, or the dtypes differ, the original concatenation is used so results stay identical.
        """
        entry = past if isinstance(past, LatentKV) else LatentKV(past[0], past[1], past[0].shape[1])
        S = c_new.shape[1]
        n = entry.length
        need = n + S
        c_buf, k_buf = entry.c_buf, entry.k_buf
        needs_grad = torch.is_grad_enabled() and (
            c_new.requires_grad or k_new.requires_grad or c_buf.requires_grad or k_buf.requires_grad
        )
        in_place_ok = (
            not needs_grad
            and c_buf.dtype == c_new.dtype and k_buf.dtype == k_new.dtype
            and c_buf.device == c_new.device
            and not (c_buf.is_inference() and not torch.is_inference_mode_enabled())
        )
        if not in_place_ok:
            return LatentKV(torch.cat([entry[0], c_new], dim=1), torch.cat([entry[1], k_new], dim=2), need)
        cap = c_buf.shape[1]
        if need > cap or not entry.writable:
            new_cap = self._grown_capacity(need, cap) if need > cap else cap
            B = c_buf.shape[0]
            c_big = c_buf.new_empty(B, new_cap, c_buf.shape[2])
            k_big = k_buf.new_empty(B, k_buf.shape[1], new_cap, k_buf.shape[3])
            c_big[:, :n] = entry[0]
            k_big[:, :, :n] = entry[1]
            c_buf, k_buf = c_big, k_big
        c_buf[:, n:need] = c_new
        k_buf[:, :, n:need] = k_new
        return LatentKV(c_buf, k_buf, need)

    @staticmethod
    def _extra_mask(
        B: int,
        S: int,
        past_len: int,
        has_past: bool,
        attn_mask: Optional[torch.Tensor],
        segment_ids: Optional[torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        """Bool mask [B or 1, 1 or H, S, past_len + S] (True = may attend), before the causal part."""
        mask = None
        if segment_ids is not None:
            if has_past:
                raise ValueError("segment_ids is only supported for prefill (no past cache)")
            if tuple(segment_ids.shape) != (B, S):
                raise ValueError(f"segment_ids must have shape {(B, S)}, got {tuple(segment_ids.shape)}")
            seg = segment_ids.to(device)
            mask = (seg.unsqueeze(1).unsqueeze(3) == seg.unsqueeze(1).unsqueeze(2))  # [B, 1, S, S]
        if attn_mask is not None:
            m = attn_mask.to(device=device, dtype=torch.bool)
            if m.dim() == 3:
                m = m.unsqueeze(1)
            mask = m if mask is None else (mask & m)
        return mask

    def _full_attention(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        c_kv: torch.Tensor,
        k_pe: torch.Tensor,
        past_len: int,
        extra_mask: Optional[torch.Tensor] = None,
        window: Optional[int] = None,
    ) -> torch.Tensor:
        B, H, S, _ = q_nope.shape
        T = c_kv.shape[1]
        kv = self.kv_b_proj(c_kv).view(B, T, H, self.nope_dim + self.v_dim).transpose(1, 2)
        k_nope, v = kv.split([self.nope_dim, self.v_dim], dim=-1)
        if self.head_k_norm:
            k_nope = self.k_nope_norm(k_nope)
        k = torch.cat([k_nope, k_pe.expand(B, H, T, self.rope_dim)], dim=-1)
        q = torch.cat([q_nope, q_pe], dim=-1)
        if extra_mask is None and window is None:
            if past_len == 0:
                return F.scaled_dot_product_attention(q, k, v, is_causal=True)
        # Query i sits at absolute position past_len + i and may attend to keys j <= past_len + i (and, with a
        # window, only the last `window` of them).
        mask = causal_band_mask(S, T, past_len, window, q.device)
        if extra_mask is not None:
            mask = mask & extra_mask
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

    def _absorbed_attention(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        c_kv: torch.Tensor,
        k_pe: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Attention against latent cache entries without up-projecting them.

        q_nope [B, H, Q, nope], q_pe [B, H, Q, rope], c_kv [B, T, rank], k_pe [B, 1, T, rope],
        optional bool mask broadcastable to [B, H, Q, T] (True = may attend). Returns [B, H, Q, v].
        The key up-projection is folded into the query and the value up-projection is applied after the
        attention-weighted sum, so the cost per cached position scales with the rank, not H * head width.
        """
        H = self.num_heads
        # The whole computation runs in at least float32: the dot products with the latent cache are
        # accumulated in that precision, as SDPA does on the prefill path. Only the result is cast back.
        out_dtype = q_nope.dtype
        calc = torch.promote_types(out_dtype, torch.float32)
        q_nope, q_pe = q_nope.to(calc), q_pe.to(calc)
        c_kv, k_pe = c_kv.to(calc), k_pe.to(calc)
        w_kv = self.kv_b_proj.weight.to(calc).view(H, self.nope_dim + self.v_dim, self.kv_rank)
        w_uk, w_uv = w_kv.split([self.nope_dim, self.v_dim], dim=1)  # [H, nope, r], [H, v, r]
        if self.head_k_norm:
            # The per-head key norm is not linear, so the nope keys are materialised for the scores.
            k_nope = self.k_nope_norm(torch.einsum("btr,hnr->bhtn", c_kv, w_uk))   # [B, H, T, nope]
            nope_scores = torch.matmul(q_nope, k_nope.transpose(-1, -2))
        else:
            q_c = torch.einsum("bhsn,hnr->bhsr", q_nope, w_uk)        # [B, H, Q, r]
            nope_scores = torch.matmul(q_c, c_kv.unsqueeze(1).transpose(-1, -2))
        scores = (nope_scores + torch.matmul(q_pe, k_pe.transpose(-1, -2))) * self.scale  # [B, H, Q, T]
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        out_c = torch.matmul(probs, c_kv.unsqueeze(1))                 # [B, H, Q, r]
        out = torch.einsum("bhsr,hvr->bhsv", out_c, w_uv)              # [B, H, Q, v]
        return out.to(out_dtype)

    def _decode_absorbed(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        c_kv: torch.Tensor,
        k_pe: torch.Tensor,
    ) -> torch.Tensor:
        """One new token against the cache: every cached position is attended, so no mask is needed."""
        return self._absorbed_attention(q_nope, q_pe, c_kv, k_pe)

    def forward_subset(
        self,
        x_act: torch.Tensor,
        active: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        store: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Inference-only attention for the tokens still running (batch size 1).

        ``x_act`` is [1, K, D] for the positions in ``active`` (LongTensor [K]). Queries are computed
        only for those positions. Keys and values come from ``store``, a pair of full-length latent
        buffers ([1, S, rank] and [1, 1, S, rope]). Active positions overwrite their latents; the rest
        keep the latents from the last loop in which they were active (KV reuse for exited tokens).
        """
        _, K, _ = x_act.shape
        H = self.num_heads
        q = self.q_b_proj(self.q_a_norm(self.q_a_proj(x_act))).view(1, K, H, self.qk_dim).transpose(1, 2)
        q_nope, q_pe = self._split_q(q, cos[active], sin[active])

        c, k_pe = self.kv_a_proj(x_act).split([self.kv_rank, self.rope_dim], dim=-1)
        c = self.kv_a_norm(c)
        k_pe = self._prep_k_pe(k_pe.unsqueeze(1), cos[active], sin[active])  # [1, 1, K, rope]

        c_store, kpe_store = store
        c_store[:, active] = c.to(c_store.dtype)
        kpe_store[:, :, active] = k_pe.to(kpe_store.dtype)

        S = c_store.shape[1]
        # Query at absolute position active[i] may attend to keys j <= active[i]. The absorbed form
        # attends over the cached latents directly, so the whole store is never up-projected per loop.
        positions = torch.arange(S, device=x_act.device)
        mask = (positions.unsqueeze(0) <= active.unsqueeze(1)).view(1, 1, K, S)   # [1, 1, K, S]
        out = self._absorbed_attention(q_nope, q_pe, c_kv=c_store, k_pe=kpe_store, mask=mask)
        if self.out_gate is not None:
            out = out * torch.sigmoid(self.out_gate(x_act)).transpose(1, 2).unsqueeze(-1).to(out.dtype)
        out = out.transpose(1, 2).reshape(1, K, H * self.v_dim)
        return self.o_proj(out)

    def _attend_masked(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        c_kv: torch.Tensor,
        k_pe: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Up-projected attention with an explicit bool mask [B, 1, Q, T] (True = may attend)."""
        B, H, _, _ = q_nope.shape
        T = c_kv.shape[1]
        kv = self.kv_b_proj(c_kv).view(B, T, H, self.nope_dim + self.v_dim).transpose(1, 2)
        k_nope, v = kv.split([self.nope_dim, self.v_dim], dim=-1)
        if self.head_k_norm:
            k_nope = self.k_nope_norm(k_nope)
        k = torch.cat([k_nope, k_pe.expand(B, H, T, self.rope_dim)], dim=-1)
        q = torch.cat([q_nope, q_pe], dim=-1)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

    def forward_mor(
        self,
        x_rows: torch.Tensor,
        plan: "MoRPlan",
        c_store: Optional[torch.Tensor],
        k_store: Optional[torch.Tensor],
        past: Optional[KVEntry],
        B: int,
        S: int,
        want_present: bool,
    ):
        """Batched attention for the rows of one MoR loop (training and cached decode).

        ``x_rows`` [K, D] are the (normed) states of the rows listed in ``plan`` (flat indices into the
        B * S new-token positions). Queries, key/value latents, the output projection and everything before
        them are computed for those K rows only. ``c_store`` [B * S, rank] and ``k_store`` [B * S, rope] hold
        the latest latent of every new-token position (None before the first write): active rows overwrite
        theirs (out of place, so autograd and cache snapshots stay valid) and the other positions keep the
        latent from the last loop in which they were active (stale key/value reuse, as in ``forward_subset``).
        Keys and values for attention are ``past`` (the cache entry of earlier tokens, or None) followed by
        the store. Queries are padded per batch row to a [B, Kmax] layout only for the score computation.

        Returns ``(out_rows [K, D], c_store, k_store, present)``; ``present`` is the cache entry (past plus
        the store) when ``want_present`` is set, else None.
        """
        K = x_rows.shape[0]
        H = self.num_heads
        q = self.q_b_proj(self.q_a_norm(self.q_a_proj(x_rows))).view(1, K, H, self.qk_dim).transpose(1, 2)
        q_nope, q_pe = self._split_q(q, plan.cos, plan.sin)                     # [1, H, K, .]

        c, k_pe = self.kv_a_proj(x_rows).split([self.kv_rank, self.rope_dim], dim=-1)
        c = self.kv_a_norm(c)
        k_pe = self._prep_k_pe(k_pe.reshape(1, 1, K, self.rope_dim), plan.cos, plan.sin)[0, 0]   # [K, rope]

        if c_store is None:
            c_store = c.new_zeros(B * S, self.kv_rank)
            k_store = k_pe.new_zeros(B * S, self.rope_dim)
        c, k_pe = c.to(c_store.dtype), k_pe.to(k_store.dtype)
        if plan.write is not None:                    # masked reference path: only active rows write
            keep = plan.write.unsqueeze(-1)
            c = torch.where(keep, c, c_store[plan.rows])
            k_pe = torch.where(keep, k_pe, k_store[plan.rows])
        c_store = c_store.index_copy(0, plan.rows, c)
        k_store = k_store.index_copy(0, plan.rows, k_pe)

        c3 = c_store.view(B, S, self.kv_rank)
        k4 = k_store.view(B, S, self.rope_dim).unsqueeze(1)                    # [B, 1, S, rope]
        present = None
        if past is not None:
            present = self._append(past, c3, k4)
            c_kv, k_all = present
        else:
            c_kv, k_all = c3, k4
            if want_present:
                present = LatentKV(c3, k4, S)

        gidx = plan.gidx                                                        # [B, Kmax]
        qn = q_nope[0][:, gidx].permute(1, 0, 2, 3)                             # [B, H, Kmax, nope]
        qp = q_pe[0][:, gidx].permute(1, 0, 2, 3)
        if past is not None and plan.kmax <= 4:
            out = self._absorbed_attention(qn, qp, c_kv, k_all, mask=plan.mask)
        else:
            out = self._attend_masked(qn, qp, c_kv, k_all, plan.mask)           # [B, H, Kmax, v]
        out = out[plan.b_idx, :, plan.slot]                                     # [K, H, v], padding-free again
        if self.out_gate is not None:
            out = out * torch.sigmoid(self.out_gate(x_rows)).unsqueeze(-1).to(out.dtype)
        out = out.reshape(K, H * self.v_dim)
        return self.o_proj(out), c_store, k_store, present


class GroupedQueryAttention(nn.Module):
    """Plain causal attention with ``num_kv_heads`` shared key/value heads (``attention_type="gqa"``).

    It has the same call signature and cache protocol as ``MultiLatentAttention`` so a block can use either.
    Queries and keys have ``qk_nope_head_dim + qk_rope_head_dim`` dims, RoPE rotates the last
    ``qk_rope_head_dim`` of them, and values have ``v_head_dim``. ``qk_norm`` applies a per-head RMSNorm to the
    queries and keys before RoPE (``qk_norm_mode`` has no meaning here, there is no latent).

    Cache. The entry is a ``LatentKV`` whose two buffers hold the flattened keys (after RoPE and QK-norm)
    [B, T, Hkv * qk_dim] and the flattened values [B, 1, T, Hkv * v_dim], so the growable-buffer, fork and
    snapshot machinery of the latent cache works unchanged (``_append`` is shared with ``MultiLatentAttention``).
    It is larger than the latent cache (56 against 31.5 MiB in bf16 at 8k for the default shape with 2 KV heads),
    which does not matter at this size. Decoding runs standard attention over the whole cache, with no
    absorption. There is no ``forward_subset`` or ``forward_mor``: early exit and MoR routing need the latent
    layout and are rejected for this arm.
    """

    _grown_capacity = MultiLatentAttention._grown_capacity
    _append = MultiLatentAttention._append
    _extra_mask = staticmethod(MultiLatentAttention._extra_mask)

    def __init__(self, config: InkyConfig):
        super().__init__()
        d = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_kv_heads
        self.qk_dim = config.qk_head_dim
        self.rope_dim = config.qk_rope_head_dim
        self.v_dim = config.v_head_dim
        self.q_proj = nn.Linear(d, self.num_heads * self.qk_dim, bias=False)
        self.k_proj = nn.Linear(d, self.num_kv_heads * self.qk_dim, bias=False)
        self.v_proj = nn.Linear(d, self.num_kv_heads * self.v_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.v_dim, d, bias=False)
        self.max_len = int(config.max_position_embeddings)
        self.out_gate = nn.Linear(d, self.num_heads) if config.attention_gate else None
        self.qk_norm = bool(config.qk_norm)
        if self.qk_norm:
            self.q_norm = RMSNorm(self.qk_dim, config.rms_norm_eps)
            self.k_norm = RMSNorm(self.qk_dim, config.rms_norm_eps)

    def _rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Rotates the last ``rope_dim`` dims of x [B, h, S, qk_dim]."""
        keep, pe = x.split([self.qk_dim - self.rope_dim, self.rope_dim], dim=-1)
        return torch.cat([keep, apply_rope(pe, cos, sin)], dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past: Optional[KVEntry] = None,
        attn_mask: Optional[torch.Tensor] = None,
        segment_ids: Optional[torch.Tensor] = None,
        window: Optional[int] = None,
    ) -> Tuple[torch.Tensor, KVEntry]:
        B, S, _ = x.shape
        H, Hk = self.num_heads, self.num_kv_heads
        q = self.q_proj(x).view(B, S, H, self.qk_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, Hk, self.qk_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, Hk * self.v_dim)
        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        q, k = self._rope(q, cos, sin), self._rope(k, cos, sin)
        k_flat = k.transpose(1, 2).reshape(B, S, Hk * self.qk_dim)
        v_flat = v.unsqueeze(1)                                              # [B, 1, S, Hk * v]

        present = self._append(past, k_flat, v_flat) if past is not None else LatentKV(k_flat, v_flat, S)
        k_all, v_all = present
        T = k_all.shape[1]
        past_len = T - S
        k_all = k_all.reshape(B, T, Hk, self.qk_dim).transpose(1, 2)         # [B, Hk, T, qk]
        v_all = v_all[:, 0].reshape(B, T, Hk, self.v_dim).transpose(1, 2)    # [B, Hk, T, v]

        sdpa = F.scaled_dot_product_attention
        windowed = window is not None and T > window                         # a window that covers everything is global
        if attn_mask is None and segment_ids is None and (S == 1 or not windowed):
            if past_len == 0:
                out = sdpa(q, k_all, v_all, is_causal=True, enable_gqa=True)
            elif S == 1:
                if windowed:                                                  # only the last `window` keys
                    k_all, v_all = k_all[:, :, -window:], v_all[:, :, -window:]
                out = sdpa(q, k_all, v_all, enable_gqa=True)                  # the new token sees the whole cache
            else:
                mask = causal_band_mask(S, T, past_len, None, q.device)
                out = sdpa(q, k_all, v_all, attn_mask=mask, enable_gqa=True)
        else:
            extra = self._extra_mask(B, S, past_len, past is not None, attn_mask, segment_ids, x.device)
            mask = causal_band_mask(S, T, past_len, window if windowed else None, q.device)
            if extra is not None:
                mask = mask & extra
            out = sdpa(q, k_all, v_all, attn_mask=mask, enable_gqa=True)

        if self.out_gate is not None:
            out = out * torch.sigmoid(self.out_gate(x)).transpose(1, 2).unsqueeze(-1).to(out.dtype)
        out = out.transpose(1, 2).reshape(B, S, H * self.v_dim)
        return self.o_proj(out), present


class SelectiveSSMMixer(nn.Module):
    """Causal selective state-space mixer, an optional replacement for latent attention.

    Each channel keeps a diagonal state of size ``ssm_state_dim``. At each token the state decays
    by an input-dependent factor a_t and receives the token's value through an input-dependent
    gate g_t: h_t = a_t * h_{t-1} + g_t * u_t. The output reads each channel's state with a learned
    per-channel readout, adds a per-channel skip of u_t, and projects back to the hidden size.

    The cache entry is (state, empty), where state is [B, hidden, state_dim] and empty is [B, 1, 0],
    so it has the same tuple shape as a KVEntry. The mixer is causal by construction and does not
    use cos or sin (no RoPE).

    Packed documents are supported through ``segment_ids`` (prefill only): the state is reset at each
    segment boundary, so every document is processed as if it ran alone. The ids of a row must be contiguous
    (a document is one run of equal ids, as ``pack_sequences_with_segments`` produces); an id that comes back
    after another one is treated as the same document, as in the attention layers. Padding masks
    (``attn_mask``) are not supported.
    """

    scan_chunk = 64  # tokens per chunk in the chunked scan; the within-chunk work is O(chunk^2 * N)

    def __init__(self, config: InkyConfig):
        super().__init__()
        d, n = config.hidden_size, config.ssm_state_dim
        self.state_dim = n
        self.in_proj = nn.Linear(d, d, bias=False)
        self.decay_proj = nn.Linear(d, n)
        self.gate_proj = nn.Linear(d, n)
        self.readout = nn.Parameter(torch.empty(d, n))
        nn.init.normal_(self.readout, std=config.initializer_range)
        self.skip = nn.Parameter(torch.zeros(d))
        self.out_proj = nn.Linear(d, d, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        past: Optional[KVEntry] = None,
        attn_mask: Optional[torch.Tensor] = None,
        segment_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, KVEntry]:
        if attn_mask is not None:
            # Padding masks are not supported: a masked position would still feed later tokens through the state.
            raise NotImplementedError(
                "SelectiveSSMMixer (ssm_mixer_layers) does not support attn_mask (padding masks). Packed documents "
                "ARE supported through segment_ids; pad by running shorter rows separately instead."
            )
        B, S, _ = x.shape
        seg = None
        if segment_ids is not None:
            if past is not None:
                raise ValueError("segment_ids is only supported for prefill (no past cache)")
            if tuple(segment_ids.shape) != (B, S):
                raise ValueError(f"segment_ids must have shape {(B, S)}, got {tuple(segment_ids.shape)}")
            seg = segment_ids.to(x.device)
        u = self.in_proj(x)                              # [B, S, d]
        log_a = F.logsigmoid(self.decay_proj(x))         # [B, S, N] log of the input-dependent decay
        g = torch.sigmoid(self.gate_proj(x))             # [B, S, N] input-dependent input gate

        h0 = u.new_zeros(B, u.shape[-1], self.state_dim) if past is None else past[0]
        if S == 1 and seg is None:
            calc = torch.promote_types(u.dtype, torch.float32)
            a = torch.exp(log_a.squeeze(1).to(calc))                             # [B, N]
            ug = u.squeeze(1).to(calc).unsqueeze(-1) * g.squeeze(1).to(calc).unsqueeze(1)
            h = a.unsqueeze(1) * h0.to(calc) + ug                                # [B, d, N] in calc (fp32)
            y = torch.einsum("bdn,dn->bd", h.to(u.dtype), self.readout).unsqueeze(1)
        else:
            y, h = self._chunked_scan(log_a, g, u, h0, seg)
        y = y + self.skip * u                                                 # [B, S, d]

        present = (h, x.new_zeros(B, 1, 0))
        return self.out_proj(y), present

    def _chunked_scan(
        self, log_a: torch.Tensor, g: torch.Tensor, u: torch.Tensor, h0: torch.Tensor,
        seg: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluates h_t = a_t * h_{t-1} + g_t * u_t (per channel and state) and the readout, chunk by chunk.

        Inside a chunk, with L_t = sum_{r<=t} log a_r, the contribution of token s to h_t is
        exp(L_t - L_s) * g_s * u_s (exponent <= 0, so nothing overflows), and the carried-in state
        contributes exp(L_t) * h_0. The decay depends only on (token, state), so the chunk's
        pairwise weights are [B, C, C, N] and the channel dimension is handled by one matmul.
        Returns y [B, S, d] (readout only, no skip) and the final state [B, d, N].

        ``seg`` (LongTensor [B, S], optional) gives packed-document ids. Token t only receives contributions
        from earlier tokens of its own segment, and the carried-in state only reaches tokens of the segment
        that the previous token belonged to. Each segment is therefore scanned from a zero state.
        """
        B, S, _ = u.shape
        C_max = self.scan_chunk
        calc = torch.promote_types(u.dtype, torch.float32)  # cumulative sums & carry in at least float32
        h = h0.to(calc)
        ys = []
        for c0 in range(0, S, C_max):
            la = log_a[:, c0:c0 + C_max].to(calc)         # [B, C, N]
            gc = g[:, c0:c0 + C_max].to(u.dtype)           # [B, C, N]
            uc = u[:, c0:c0 + C_max]                       # [B, C, d]
            C = la.shape[1]
            L = torch.cumsum(la, dim=1)                    # [B, C, N], non-increasing
            diff = L.unsqueeze(2) - L.unsqueeze(1)         # [B, t, s, N] = L_t - L_s
            causal = torch.ones(C, C, dtype=torch.bool, device=u.device).tril().view(1, C, C, 1)
            W = torch.exp(diff.masked_fill(~causal, 0.0)).masked_fill(~causal, 0.0).to(u.dtype)
            Mg = W * gc.unsqueeze(1)                       # [B, t, s, N], weight with the gate g_s
            carry = torch.exp(L).unsqueeze(2) * h.unsqueeze(1)  # [B, C, d, N] in calc (fp32)
            if seg is not None:
                seg_c = seg[:, c0:c0 + C_max]                                # [B, C]
                same = seg_c.unsqueeze(2) == seg_c.unsqueeze(1)              # [B, t, s]: same segment
                Mg = Mg * same.unsqueeze(-1).to(Mg.dtype)
                if c0 > 0:
                    carry_mask = seg_c == seg[:, c0 - 1].unsqueeze(1)        # [B, C]: h belongs to token c0-1
                else:
                    carry_mask = torch.ones_like(seg_c, dtype=torch.bool)    # the initial state is zero anyway
                carry = carry * carry_mask[:, :, None, None].to(carry.dtype)
            intra = torch.matmul(Mg.permute(0, 3, 1, 2), uc.unsqueeze(1))   # [B, N, t, d]
            h_chunk = intra.permute(0, 2, 3, 1).to(calc) + carry            # [B, C, d, N] in calc (fp32)
            ys.append(torch.einsum("bcdn,dn->bcd", h_chunk.to(u.dtype), self.readout))
            h = h_chunk[:, -1]
        return torch.cat(ys, dim=1), h


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DenseFFN(SwiGLU):
    """Dense stand-in for ``SharedExpertMoE`` (``use_moe=False``): one SwiGLU, same call signature.

    Returns ``(output, aux)`` with a zero auxiliary loss, so blocks and the training loop treat it like
    the MoE. Its width is ``config.dense_ffn_size`` (the MoE's active parameter count per token by default).
    """

    def __init__(self, config: InkyConfig):
        super().__init__(config.hidden_size, config.dense_ffn_size)

    def forward(  # type: ignore[override]
        self, x: torch.Tensor, row_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return super().forward(x), x.new_zeros((), dtype=torch.float32)


class SharedExpertMoE(nn.Module):
    """Shared experts always run. Routed experts are chosen per token by a sigmoid router.

    Balancing is aux-loss-free: ``balance_bias`` is added to the scores only when picking
    experts, not when weighting them. Call ``update_balance_bias()`` once per optimizer step.
    """

    def __init__(self, config: InkyConfig):
        super().__init__()
        d, ff = config.hidden_size, config.intermediate_size
        self.num_routed = config.num_routed_experts
        self.top_k = min(config.num_active_routed, self.num_routed)
        self.balance_rate = config.balance_rate
        self.balance_update = getattr(config, "balance_update", "sign")
        self.z_loss_coef = config.router_z_loss_coef
        self.dense_max_tokens = int(config.moe_dense_max_tokens)

        self.shared_experts = nn.ModuleList(SwiGLU(d, config.shared_ffn_size) for _ in range(config.num_shared_experts))
        self.routed_experts = nn.ModuleList(SwiGLU(d, ff) for _ in range(self.num_routed))
        self.router = nn.Linear(d, self.num_routed, bias=False)
        self.register_buffer("balance_bias", torch.zeros(self.num_routed))
        self.register_buffer("expert_load", torch.full((self.num_routed,), 1.0 / self.num_routed))
        # Assignment counts summed over every forward call since the last update_balance_bias().
        # A looped core calls this MoE once per loop, so the counts of all loops are used. Not persisted.
        self.register_buffer("load_counts", torch.zeros(self.num_routed), persistent=False)

    def forward(
        self, x: torch.Tensor, row_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """``row_mask`` (optional bool, one entry per token row; None = every row counts) is used by the
        masked reference path of MoR routing: rows where it is False are still computed (their outputs are
        discarded by the caller) but do not enter the expert load counts or the router z-loss, so the
        auxiliary terms equal those of the gather path that never sees those rows."""
        B, S, D = x.shape
        flat = x.reshape(-1, D)

        logits = self.router(flat)
        scores = torch.sigmoid(logits.float())
        _, idx = torch.topk(scores + self.balance_bias, self.top_k, dim=-1)
        weights = scores.gather(-1, idx)
        weights = weights / (weights.sum(-1, keepdim=True) + 1e-9)

        out = torch.zeros_like(flat)
        for expert in self.shared_experts:
            out = out + expert(flat)

        with torch.no_grad():
            counted = idx if row_mask is None else idx[row_mask.reshape(-1)]
            self.load_counts.add_(torch.bincount(counted.reshape(-1), minlength=self.num_routed).float())
            total = self.load_counts.sum().clamp(min=1.0)
            self.expert_load.copy_(self.load_counts / total)

        out = self._dispatch_routed(flat, idx, weights, out)

        if row_mask is None:
            z_loss = torch.logsumexp(logits.float(), dim=-1).pow(2).mean() * self.z_loss_coef
        else:
            m = row_mask.reshape(-1).float()
            z_loss = (torch.logsumexp(logits.float(), dim=-1).pow(2) * m).sum() / m.sum().clamp(min=1.0) \
                * self.z_loss_coef
        return out.view(B, S, D), z_loss

    def _use_dense(self, n_tokens: int, device_type: str) -> bool:
        """True when the sync-free dense dispatch should be used: small batch on a non-CPU device."""
        return device_type != "cpu" and n_tokens <= self.dense_max_tokens

    def _dispatch_routed(
        self,
        flat: torch.Tensor,
        idx: torch.Tensor,
        weights: torch.Tensor,
        out: torch.Tensor,
    ) -> torch.Tensor:
        """Adds the weighted routed-expert outputs to ``out``.

        The (token, slot) assignments are sorted by expert once, so each expert sees one contiguous
        block of rows, and the outputs are scatter-added back in one index_add (grouped path). Its
        per-expert counts are read on the host, which is free on CPU but a device sync elsewhere. For
        small batches on an accelerator (decode) the dense form is used instead, see ``_use_dense``.
        """
        if self._use_dense(flat.shape[0], flat.device.type):
            return self._dispatch_routed_dense(flat, idx, weights, out)
        k = idx.shape[1]
        flat_idx = idx.reshape(-1)                                         # [N*k], assignment a = token*k + slot
        order = torch.argsort(flat_idx, stable=True)
        token_of = order // k                                              # token of each sorted assignment
        gate = weights.reshape(-1)[order].to(flat.dtype).unsqueeze(-1)     # [N*k, 1]
        counts = torch.bincount(flat_idx, minlength=self.num_routed).tolist()  # one host read per forward
        chunks = torch.split(flat[token_of], counts)
        ys = [expert(chunk) for expert, chunk in zip(self.routed_experts, chunks) if chunk.shape[0] > 0]
        if not ys:
            return out
        y = torch.cat(ys, dim=0) * gate
        return out.index_add(0, token_of, y)

    def _dispatch_routed_dense(
        self,
        flat: torch.Tensor,
        idx: torch.Tensor,
        weights: torch.Tensor,
        out: torch.Tensor,
    ) -> torch.Tensor:
        """Sync-free form of the routed dispatch: a dense [N, E] gate matrix, zero where not selected.

        Every routed expert runs on every token, so this only pays off for small N. The product is
        masked with ``where`` so a non-finite output of an unselected expert cannot turn into NaN
        through ``inf * 0``.
        """
        gate = torch.zeros(flat.shape[0], self.num_routed, device=flat.device, dtype=flat.dtype)
        gate = gate.scatter(1, idx, weights.to(flat.dtype))             # idx holds distinct experts per token
        for e, expert in enumerate(self.routed_experts):
            g = gate[:, e:e + 1]
            y = expert(flat)
            out = out + torch.where(g > 0, y * g, torch.zeros_like(y))
        return out

    @torch.no_grad()
    def update_balance_bias(self) -> None:
        """Moves the balance biases towards uniform load, using the counts since the last call, then resets them.

        ``balance_update="sign"`` moves every bias by exactly ``balance_rate`` (up for an under-used expert, down
        for an over-used one, no move at exactly the uniform load), as in DeepSeek-V3. ``"proportional"`` moves
        it by ``balance_rate * (uniform - load)``, which is at most ``balance_rate / num_routed`` and shrinks as
        the load evens out."""
        total = self.load_counts.sum()
        load = self.load_counts / total.clamp(min=1.0)
        target = 1.0 / self.num_routed
        if self.balance_update == "sign":
            step = self.balance_rate * torch.sign(target - load)
        else:
            step = self.balance_rate * (target - load)
        # No assignments since the last update means no information: leave the biases unchanged.
        self.balance_bias.add_(torch.where(total > 0, step, torch.zeros_like(step)))
        self.expert_load.copy_(torch.where(total > 0, load, self.expert_load))
        self.load_counts.zero_()

    def reset_expert_load(self) -> None:
        """Discards the load counts collected since the last balance update."""
        self.load_counts.zero_()


class InkyBlock(nn.Module):
    """Pre-norm block: latent-KV attention, then the MoE feed-forward.

    ``config.residual_dropout`` drops each branch output (attention and MoE) before it is added back to
    the residual stream. It is active only in train mode and is 0 by default, which changes nothing.
    """

    def __init__(self, config: InkyConfig, residual_scale: float = 1.0, use_ssm: bool = False):
        super().__init__()
        self.residual_scale = residual_scale
        self.use_checkpoint = config.gradient_checkpointing
        self.residual_dropout = float(getattr(config, "residual_dropout", 0.0))
        self.input_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        # The token mixer is named `attn` in both cases so the block's forward is the same.
        if use_ssm:
            self.attn = SelectiveSSMMixer(config)
        elif config.attention_type == "gqa":
            self.attn = GroupedQueryAttention(config)
        else:
            self.attn = MultiLatentAttention(config)
        self.post_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        # use_moe=False swaps in one dense SwiGLU of the MoE's active width (a baseline / ablation).
        self.mlp = SharedExpertMoE(config) if config.use_moe else DenseFFN(config)

    def _residual_drop(self, branch: torch.Tensor) -> torch.Tensor:
        if self.residual_dropout > 0.0 and self.training:
            return F.dropout(branch, p=self.residual_dropout, training=True)
        return branch

    def _forward(self, x, cos, sin, past, attn_mask=None, segment_ids=None, window=None):
        # The SSM mixer has no window (its state already decays); only the attention variants take one.
        extra = {} if window is None or isinstance(self.attn, SelectiveSSMMixer) else {"window": window}
        attn_out, present = self.attn(self.input_norm(x), cos, sin, past,
                                      attn_mask=attn_mask, segment_ids=segment_ids, **extra)
        x = x + self._residual_drop(attn_out) * self.residual_scale
        mlp_out, aux = self.mlp(self.post_norm(x))
        x = x + self._residual_drop(mlp_out) * self.residual_scale
        return x, present, aux

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past: Optional[KVEntry] = None,
        attn_mask: Optional[torch.Tensor] = None,
        segment_ids: Optional[torch.Tensor] = None,
        window: Optional[int] = None,
    ) -> Tuple[torch.Tensor, KVEntry, torch.Tensor]:
        """``attn_mask``, ``segment_ids`` and ``window`` are passed to the token mixer (see
        MultiLatentAttention.forward)."""
        if self.use_checkpoint and self.training and past is None and torch.is_grad_enabled():
            # Recompute activations in backward. Only used for full-sequence training, not decoding.
            def run(hs):
                out, _, aux = self._forward(hs, cos, sin, None, attn_mask, segment_ids, window)
                return out, aux

            out, aux = torch.utils.checkpoint.checkpoint(run, x, use_reentrant=False)
            return out, None, aux
        return self._forward(x, cos, sin, past, attn_mask, segment_ids, window)

    def forward_subset(
        self,
        x_act: torch.Tensor,
        active: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        store: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Inference-only block for the active tokens. See MultiLatentAttention.forward_subset."""
        if not isinstance(self.attn, MultiLatentAttention):
            raise NotImplementedError("early exit supports latent-attention blocks only")
        attn_out = self.attn.forward_subset(self.input_norm(x_act), active, cos, sin, store)
        x = x_act + attn_out * self.residual_scale
        mlp_out, _ = self.mlp(self.post_norm(x))
        return x + mlp_out * self.residual_scale

    def _forward_mor(self, x, c_store, k_store, plan, past, B, S, want_present):
        attn_out, c_store, k_store, present = self.attn.forward_mor(
            self.input_norm(x), plan, c_store, k_store, past, B, S, want_present
        )
        x = x + self._residual_drop(attn_out) * self.residual_scale
        mlp_out, aux = self.mlp(self.post_norm(x).unsqueeze(0), row_mask=plan.write)
        x = x + self._residual_drop(mlp_out.squeeze(0)) * self.residual_scale
        return x, c_store, k_store, present, aux

    def forward_mor(self, x_rows, plan, store, past, B, S, want_present):
        """Block for the rows of one MoR loop (see ``MultiLatentAttention.forward_mor``).

        ``x_rows`` is [K, D]; the attention projections, the MoE / dense feed-forward and the residual
        updates all run on those K rows only. ``store`` is the (c_store, k_store) pair of this layer, or
        None before its first write. Returns ``(x_rows, store, present, aux)``.
        """
        if not isinstance(self.attn, MultiLatentAttention):
            raise NotImplementedError(
                "routing_mode='mor' supports latent-attention blocks only (ssm_mixer_layers must be empty)"
            )
        c_store, k_store = store if store is not None else (None, None)
        if self.use_checkpoint and self.training and past is None and torch.is_grad_enabled():
            def run(hs, c_s, k_s):
                out, c_n, k_n, _, aux = self._forward_mor(hs, c_s, k_s, plan, None, B, S, False)
                return out, c_n, k_n, aux

            out, c_store, k_store, aux = torch.utils.checkpoint.checkpoint(
                run, x_rows, c_store, k_store, use_reentrant=False
            )
            return out, (c_store, k_store), None, aux
        out, c_store, k_store, present, aux = self._forward_mor(
            x_rows, c_store, k_store, plan, past, B, S, want_present
        )
        return out, (c_store, k_store), present, aux


class LoopStepModulation(nn.Module):
    """FiLM scale and shift that depend on the loop index. Starts as the identity."""

    def __init__(self, hidden_size: int, max_loops: int):
        super().__init__()
        self.step_embedding = nn.Parameter(torch.zeros(max_loops + 1, 2 * hidden_size))

    def forward(self, h: torch.Tensor, step: int) -> torch.Tensor:
        scale, shift = self.step_embedding[step].chunk(2, dim=-1)
        return h * (1.0 + scale) + shift


class StateSpaceMemoryHighway(nn.Module):
    """Carries a per-position memory across loop iterations.

    The causal running mean of the loop input is projected into a small state. That state is updated from
    the previous loop's state with a learned gate, and read back into the stream as
    ``h + out_gain * RMSNorm(out_proj(mem))``.

    ``out_gain`` is a learned scalar. The RMSNorm output has unit scale whatever the scale of ``h`` is, so a
    fixed multiplier (formerly 0.1) can be as large as the stream itself early on. ``gain_init`` (default 0.0)
    starts the highway switched off, so it only enters the stream as far as training makes it useful; 0.1
    reproduces the earlier fixed scale. While the gain is exactly 0 the highway's other weights receive no
    gradient, and the gain itself does.
    """

    def __init__(self, hidden_size: int, state_dim: int, eps: float, gain_init: float = 0.0):
        super().__init__()
        self.in_proj = nn.Linear(hidden_size, state_dim, bias=False)
        self.a_gate = nn.Linear(state_dim, state_dim)
        self.out_proj = nn.Linear(state_dim, hidden_size, bias=False)
        self.norm = RMSNorm(hidden_size, eps)
        self.out_gain = nn.Parameter(torch.full((1,), float(gain_init)))

    def forward(
        self,
        h: torch.Tensor,
        prev_mem: Optional[torch.Tensor],
        prefix_sum: Optional[torch.Tensor],
        past_len: int,
        segment_ids: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """``segment_ids`` (int [B, S], contiguous runs of equal ids = documents) makes the running mean
        restart at each document boundary, so packed documents do not see each other. It is a prefill-only
        option: it cannot be combined with a carried ``prefix_sum`` or a cache (``past_len`` > 0).
        With segments the returned prefix is None, since there is no single running sum to carry.

        ``valid_mask`` (bool or 0/1 [B, S], True = real token) excludes padding from the running mean: each
        position averages over the valid tokens up to and including itself. Pads contribute nothing to the
        sum or the count, whether they are left or right padding. None gives the default mean over all
        positions. Like ``segment_ids`` it is prefill-only, and the returned prefix is None."""
        S = h.shape[1]
        if valid_mask is not None:
            if prefix_sum is not None or past_len != 0:
                raise ValueError("valid_mask is only supported for prefill (no past cache or prefix_sum)")
            valid = self._check_valid_mask(valid_mask, h)
            if segment_ids is not None:
                h_mean = self._segment_mean(h, segment_ids, valid)
            else:
                h_mean = self._masked_prefix_mean(h, valid)
            new_prefix = None
        elif segment_ids is not None:
            if prefix_sum is not None or past_len != 0:
                raise ValueError("segment_ids is only supported for prefill (no past cache or prefix_sum)")
            h_mean = self._segment_mean(h, segment_ids)
            new_prefix = None
        else:
            counts = torch.arange(past_len + 1, past_len + S + 1, device=h.device, dtype=torch.float32)
            h_sum = h.float().cumsum(dim=1)
            if prefix_sum is not None:
                h_sum = h_sum + prefix_sum
            h_mean = (h_sum / counts.view(1, S, 1)).to(h.dtype)
            new_prefix = h_sum[:, -1:, :]
        x_m = self.in_proj(h_mean)

        if prev_mem is None:
            mem = torch.tanh(x_m)
        else:
            alpha = torch.sigmoid(self.a_gate(prev_mem))
            mem = alpha * prev_mem + (1.0 - alpha) * torch.tanh(x_m)

        out = h + self.out_gain.to(h.dtype) * self.norm(self.out_proj(mem))
        return out, mem, new_prefix

    @staticmethod
    def _check_valid_mask(valid_mask: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Returns the mask as float32 [B, S] (1 = valid), after checking its shape."""
        if tuple(valid_mask.shape) != tuple(h.shape[:2]):
            raise ValueError(f"valid_mask must have shape {tuple(h.shape[:2])}, got {tuple(valid_mask.shape)}")
        return valid_mask.to(device=h.device, dtype=torch.float32)

    @staticmethod
    def _masked_prefix_mean(h: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Causal mean over the valid positions up to and including each position. valid is float [B, S]."""
        h_sum = (h.float() * valid.unsqueeze(-1)).cumsum(dim=1)
        counts = valid.cumsum(dim=1).clamp(min=1.0).unsqueeze(-1)   # a pad before any real token gives 0
        return (h_sum / counts).to(h.dtype)

    @staticmethod
    def _segment_mean(
        h: torch.Tensor, segment_ids: torch.Tensor, valid: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Mean of h over the tokens of the same segment up to and including each position.

        Vectorised: with the cumulative sum C, the segment sum at t is C[t] minus the exclusive cumulative
        sum at the segment start s, and the count is t - s + 1. The start index is the running max of the
        boundary positions (cummax). No loop over tokens. With ``valid`` (float [B, S]) only valid tokens
        enter the sum and the count, so a pad inside a segment does not dilute the mean.
        """
        B, S, D = h.shape
        seg = segment_ids.to(h.device)
        if tuple(seg.shape) != (B, S):
            raise ValueError(f"segment_ids must have shape {(B, S)}, got {tuple(seg.shape)}")
        pos = torch.arange(S, device=h.device).expand(B, S)
        boundary = torch.ones(B, S, dtype=torch.bool, device=h.device)
        boundary[:, 1:] = seg[:, 1:] != seg[:, :-1]
        start = torch.where(boundary, pos, torch.zeros_like(pos)).cummax(dim=1).values   # [B, S]
        h_f = h.float()
        if valid is None:
            counts = (pos - start + 1).to(torch.float32)                                 # [B, S]
            csum = h_f.cumsum(dim=1)
        else:
            h_f = h_f * valid.unsqueeze(-1)
            vc = valid.cumsum(dim=1)                                                     # [B, S]
            vbase = (vc - valid).gather(1, start)
            counts = (vc - vbase).clamp(min=1.0)
            csum = h_f.cumsum(dim=1)
        excl = csum - h_f
        base = excl.gather(1, start.unsqueeze(-1).expand(B, S, D))                        # [B, S, D]
        seg_sum = csum - base
        return (seg_sum / counts.unsqueeze(-1)).to(h.dtype)


class MoRPlan:
    """Row bookkeeping for one loop of Mixture-of-Recursions (MoR) routing.

    ``rows`` are flat indices (``b * S + s``, ascending) of the new-token positions that are processed in
    this loop. ``write`` is None in the gather path (every listed row is active); in the masked reference
    path ``rows`` lists every position and ``write`` marks the active ones, whose results are the only ones
    kept. ``b_idx`` / ``s_idx`` / ``slot`` give each row's batch row, position and rank inside its batch
    row; ``gidx`` [B, Kmax] maps the padded query layout back to rows (padding points at row 0 and is
    masked). ``mask`` [B, 1, Kmax, T] (True = may attend) holds causality over absolute positions (cache
    plus new tokens) and any packing mask. Padded queries may attend to key 0, so no row is fully masked.
    """

    def __init__(
        self,
        rows: torch.Tensor,
        B: int,
        S: int,
        past_len: int,
        cos: torch.Tensor,
        sin: torch.Tensor,
        write: Optional[torch.Tensor] = None,
        extra_mask: Optional[torch.Tensor] = None,
    ):
        device = rows.device
        K = int(rows.numel())
        self.rows = rows
        self.write = write
        self.b_idx = torch.div(rows, S, rounding_mode="floor")
        self.s_idx = rows - self.b_idx * S
        counts = torch.bincount(self.b_idx, minlength=B)
        self.kmax = max(int(counts.max()), 1)
        starts = counts.cumsum(0) - counts
        self.slot = torch.arange(K, device=device) - starts[self.b_idx]
        self.gidx = torch.zeros(B, self.kmax, dtype=torch.long, device=device)
        self.gidx[self.b_idx, self.slot] = torch.arange(K, device=device)
        valid_q = torch.zeros(B, self.kmax, dtype=torch.bool, device=device)
        valid_q[self.b_idx, self.slot] = True
        qpos = torch.zeros(B, self.kmax, dtype=torch.long, device=device)
        qpos[self.b_idx, self.slot] = self.s_idx + past_len
        self.cos, self.sin = cos[self.s_idx], sin[self.s_idx]
        T = past_len + S
        allowed = torch.arange(T, device=device).view(1, 1, T) <= qpos.unsqueeze(-1)          # [B, Kmax, T]
        if extra_mask is not None:
            em = extra_mask.expand(B, 1, S, S)[:, 0]                                          # [B, S, S]
            allowed = allowed & em[self.b_idx, self.s_idx][self.gidx]
        allowed[:, :, 0] = allowed[:, :, 0] | ~valid_q
        self.mask = allowed.unsqueeze(1)


class RecurrentCore(nn.Module):
    """One loop iteration over the shared recurrent layers."""

    def __init__(self, config: InkyConfig):
        super().__init__()
        scale = 1.0 / math.sqrt(config.max_recurrent_loops)
        mixer_layers = set(config.ssm_mixer_layers)
        self.layers = nn.ModuleList(
            InkyBlock(config, residual_scale=scale, use_ssm=i in mixer_layers)
            for i in range(config.num_recurrent_layers)
        )
        self.step_mod = LoopStepModulation(config.hidden_size, config.max_recurrent_loops) \
            if config.use_step_modulation else None
        # use_highway=False (ablation) does not create the module, so parameter counts reflect it.
        self.ssm = StateSpaceMemoryHighway(
            config.hidden_size, config.ssm_state_dim, config.rms_norm_eps,
            gain_init=getattr(config, "highway_gain_init", 0.0),
        ) if config.use_highway else None
        self.damped_beta = nn.Parameter(torch.zeros(config.hidden_size))
        self.loop_dropout_prob = config.loop_dropout_prob
        # loop_input_injection: per-channel gain on the prelude output added to every loop's block input. It
        # starts at 0, so the model starts identical to one without it.
        self.inject_gain = nn.Parameter(torch.zeros(config.hidden_size)) if config.loop_input_injection else None
        # Local attention (config.local_window): which 1-based loops use the window.
        self.local_window = config.local_window
        self.local_loops = frozenset(config.local_window_loops) if config.local_window is not None else frozenset()

    def _inject(self, x: torch.Tensor, h0: Optional[torch.Tensor]) -> torch.Tensor:
        """``x + inject_gain * h0`` when injection is on (``h0`` is the prelude output), else ``x``."""
        if self.inject_gain is None:
            return x
        if h0 is None:
            raise ValueError("loop_input_injection needs the prelude output h0")
        return x + self.inject_gain.to(x.dtype) * h0

    def forward(
        self,
        h: torch.Tensor,
        step: int,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past: Optional[List[Optional[KVEntry]]] = None,
        prefix_sum: Optional[torch.Tensor] = None,
        prev_mem: Optional[torch.Tensor] = None,
        past_len: int = 0,
        attn_mask: Optional[torch.Tensor] = None,
        segment_ids: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
        h0: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], List[Optional[KVEntry]], torch.Tensor
    ]:
        """One loop. ``segment_ids`` [B, S] (packed documents) restarts the highway's running mean at each
        document and is passed to every block. ``attn_mask`` (see MultiLatentAttention.forward) is passed to
        every block. The highway has no attention, so it takes only ``segment_ids`` and ``valid_mask`` (the
        pads it must leave out of its running mean, see StateSpaceMemoryHighway.forward). The defaults of
        all three give the default behaviour. Packing needs ``past is None`` and ``prefix_sum is None``.
        ``h0`` is the prelude output, needed (and only used) when ``loop_input_injection`` is on. Loops listed in
        ``config.local_window_loops`` attend to the last ``config.local_window`` positions only."""
        h_in = h
        h = self._inject(h, h0)
        window = self.local_window if step in self.local_loops else None
        if self.step_mod is not None:
            h = self.step_mod(h, step)
        if self.ssm is not None:
            h, mem, new_prefix = self.ssm(
                h, prev_mem, prefix_sum, past_len, segment_ids=segment_ids, valid_mask=valid_mask
            )
        else:
            mem, new_prefix = None, None      # no highway: nothing is carried across loops or cached

        presents: List[Optional[KVEntry]] = []
        aux = h.new_zeros(())
        # Layer-drop decisions are drawn once per loop as a small CPU tensor and read as Python floats,
        # so the layer loop never waits on a device value.
        drop = (torch.rand(len(self.layers)).tolist()
                if self.training and self.loop_dropout_prob > 0.0 else None)
        for i, layer in enumerate(self.layers):
            if drop is not None and drop[i] < self.loop_dropout_prob:
                presents.append(None)
                continue
            past_i = past[i] if past is not None else None
            h, present, layer_aux = layer(h, cos, sin, past_i, attn_mask=attn_mask, segment_ids=segment_ids,
                                          window=window)
            presents.append(present)
            aux = aux + layer_aux

        beta = (0.05 + 0.90 * torch.sigmoid(self.damped_beta)).view(1, 1, -1)
        out = (1.0 - beta) * h_in + beta * h
        return out, mem, new_prefix, presents, aux

    def forward_mor(
        self,
        h: torch.Tensor,
        depth: torch.Tensor,
        gate: Optional[torch.Tensor],
        loops: int,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_core: Optional[List[Optional[list]]] = None,
        past_prefix: Optional[List[Optional[torch.Tensor]]] = None,
        past_len: int = 0,
        attn_mask: Optional[torch.Tensor] = None,
        segment_ids: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
        gather: bool = True,
        want_cache: bool = False,
    ):
        """Mixture-of-Recursions recurrence (token-level depth routing), for training and cached decode.

        ``h`` [B, S, D] is the state after the pre layers, ``depth`` [B, S] (int, 0 = never run) is each
        token's number of loops, ``gate`` [B, S] (float, or None) scales the update of an active token.
        Loop n runs the shared block stack on the tokens with ``depth >= n`` only; every other token keeps
        its state frozen (so a token's final state is the one after its last active loop).

        Per loop, for the active tokens (K rows, gathered into one flat [K, D] batch):
          * step modulation and the SSM highway run on the states of ALL new tokens, frozen ones included
            (they are cheap), so the highway's running mean at loop n averages the loop-n modulated state of
            every earlier token, active or exited. Only active tokens use the highway output.
          * queries, latents, attention output projection and the MoE run on the K active rows only.
          * keys and values come from the latent store of the layer: positions active in this loop hold the
            latents just computed, exited positions keep the latents from their last active loop.
          * the update is ``h + gate * (core_out - h)`` with ``core_out`` the usual damped residual output.
        Cached decode (``past_core`` / ``past_prefix``, ``want_cache``) gives each loop's cache the stale
        latents of tokens that exited earlier and adds their frozen modulated state to the highway prefix
        sum, so a token-by-token run equals the full-sequence pass.

        ``gather=False`` is the masked reference path: every layer runs on all B * S rows and only the
        active ones are kept. It gives the same numbers as the gather path and costs full compute.

        Returns ``(h_out [B, S, D], aux, rows_computed, core_entries, prefixes)``. ``aux`` is the MoE
        auxiliary loss averaged over the loops that ran. ``core_entries[n - 1]`` is the list of per-layer
        cache entries of loop n and ``prefixes[n - 1]`` its highway prefix sum (both only with
        ``want_cache``).
        """
        B, S, D = h.shape
        n_layers = len(self.layers)
        device = h.device
        depth_flat = depth.reshape(-1)
        gate_flat = gate.reshape(-1) if gate is not None else None
        state = h.reshape(B * S, D)
        extra = None
        if attn_mask is not None or segment_ids is not None:
            extra = MultiLatentAttention._extra_mask(B, S, past_len, past_len > 0, attn_mask, segment_ids, device)
        stores: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None] * n_layers
        beta = (0.05 + 0.90 * torch.sigmoid(self.damped_beta)).view(1, -1)
        all_rows = torch.arange(B * S, device=device)
        prev_mem = None
        aux_sum = h.new_zeros((), dtype=torch.float32)
        loops_run = 0
        rows_computed = 0
        core_entries: List[list] = []
        prefixes: List[Optional[torch.Tensor]] = []

        for n in range(1, loops + 1):
            idx = n - 1
            active = depth_flat >= n
            rows = active.nonzero(as_tuple=True)[0]
            K = int(rows.numel())
            if K == 0 and not want_cache:
                break                                  # depths only shrink: no later loop has an active token
            x_all = self._inject(state.view(B, S, D), h)       # ``h`` is the prelude output
            if self.step_mod is not None:
                x_all = self.step_mod(x_all, n)
            prefix_in = past_prefix[idx] if past_prefix is not None else None
            past_n = past_core[idx] if past_core is not None else None

            if K == 0:
                # Cache-fill only: nothing runs, the new tokens keep their frozen state. The loop's cache
                # still gets their stale latents and the highway prefix its frozen modulated state.
                if self.ssm is not None:
                    h_sum = x_all.float().cumsum(dim=1)
                    if prefix_in is not None:
                        h_sum = h_sum + prefix_in
                    prefixes.append(h_sum[:, -1:, :])
                else:
                    prefixes.append(None)
                entries = []
                for j, layer in enumerate(self.layers):
                    c_s, k_s = stores[j]
                    c3, k4 = c_s.view(B, S, -1), k_s.view(B, S, -1).unsqueeze(1)
                    p = past_n[j] if past_n is not None else None
                    entries.append(layer.attn._append(p, c3, k4) if p is not None else LatentKV(c3, k4, S))
                core_entries.append(entries)
                continue

            if self.ssm is not None:
                x_all, mem, new_prefix = self.ssm(
                    x_all, prev_mem, prefix_in, past_len, segment_ids=segment_ids, valid_mask=valid_mask
                )
                prev_mem = mem
            else:
                new_prefix = None
            prefixes.append(new_prefix)

            if gather:
                loop_rows, write = rows, None
            else:
                loop_rows, write = all_rows, active
            plan = MoRPlan(loop_rows, B, S, past_len, cos, sin, write=write, extra_mask=extra)
            rows_computed += int(loop_rows.numel())
            x_flat = x_all.reshape(B * S, D)
            h_in = state[loop_rows]
            x_rows = x_flat[loop_rows]
            drop = (torch.rand(n_layers).tolist()
                    if self.training and self.loop_dropout_prob > 0.0 and not want_cache else None)
            entries = []
            loop_aux = h.new_zeros((), dtype=torch.float32)
            for j, layer in enumerate(self.layers):
                if drop is not None and drop[j] < self.loop_dropout_prob:
                    entries.append(None)
                    continue
                p = past_n[j] if past_n is not None else None
                x_rows, stores[j], present, laux = layer.forward_mor(x_rows, plan, stores[j], p, B, S, want_cache)
                entries.append(present)
                loop_aux = loop_aux + laux
            core_entries.append(entries)
            aux_sum = aux_sum + loop_aux
            loops_run += 1

            core_out = (1.0 - beta) * h_in + beta * x_rows
            if gate_flat is not None:
                g = gate_flat[loop_rows].unsqueeze(-1).to(core_out.dtype)
                core_out = h_in + g * (core_out - h_in)
            if write is not None:
                core_out = torch.where(write.unsqueeze(-1), core_out, h_in)
            state = state.index_copy(0, loop_rows, core_out.to(state.dtype))

        aux = aux_sum / max(loops_run, 1)
        return state.view(B, S, D), aux, rows_computed, core_entries, prefixes

    def forward_subset(
        self,
        state: torch.Tensor,
        step: int,
        cos: torch.Tensor,
        sin: torch.Tensor,
        active: torch.Tensor,
        stores: List[Tuple[torch.Tensor, torch.Tensor]],
        prev_mem: Optional[torch.Tensor],
        h0: Optional[torch.Tensor] = None,
    ):
        """Inference-only loop step for the active tokens (batch size 1). ``h0`` [1, S, D] is the prelude output,
        needed only when ``loop_input_injection`` is on.

        ``state`` [1, S, D] holds every token's current loop state; exited tokens keep their final
        state. The SSM highway runs over all positions (it is cheap), and the attention blocks run
        only on ``active``. Returns the new states of the active tokens [1, K, D] and the SSM memory.
        """
        x = self._inject(state, h0)
        if self.step_mod is not None:
            x = self.step_mod(x, step)
        if self.ssm is not None:
            x, mem, _ = self.ssm(x, prev_mem, None, 0)
        else:
            mem = None
        x_act = x[:, active]
        h_in_act = state[:, active]
        for j, layer in enumerate(self.layers):
            x_act = layer.forward_subset(x_act, active, cos, sin, stores[j])
        beta = (0.05 + 0.90 * torch.sigmoid(self.damped_beta)).view(1, 1, -1)
        return (1.0 - beta) * h_in_act + beta * x_act, mem


class PonderHalting(nn.Module):
    """PonderNet halting: lambda_n(t) is the probability that token t stops at loop n, given the
    loop state h_n(t). The probability of stopping exactly at loop n is
    lambda_n * prod_{j<n} (1 - lambda_j); see ``geometric_halting_weights``.
    """

    def __init__(self, hidden_size: int, prior_steps: float):
        super().__init__()
        if prior_steps < 1.0:
            raise ValueError(f"ponder_prior_steps must be >= 1, got {prior_steps}")
        self.gate = nn.Linear(hidden_size, 1)
        self.prior_lambda = 1.0 / prior_steps

    def reset_prior(self) -> None:
        """Start the gate at the prior, so the initial halting distribution matches the prior.
        Call after the model's post_init, which would otherwise reset the bias."""
        p = self.prior_lambda
        nn.init.constant_(self.gate.bias, math.log(p / (1.0 - p)))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h [B, S, D] -> lambda [B, S] in (0, 1)."""
        return torch.sigmoid(self.gate(h).float()).squeeze(-1)


def geometric_halting_weights(lambdas: Sequence[torch.Tensor]) -> torch.Tensor:
    """Halting distribution over loops from per-loop stop probabilities.

    ``lambdas`` holds L tensors of shape [B, S] (lambda_1 .. lambda_L). Returns [L, B, S] with
    p_n = lambda_n * prod_{j<n}(1 - lambda_j) for n < L, and p_L = prod_{j<L}(1 - lambda_j), the
    remainder. The weights sum to 1 over n for every token. The last lambda is not used.
    """
    L = len(lambdas)
    surv = torch.ones_like(lambdas[0], dtype=torch.float32)
    weights = []
    for n, lam in enumerate(lambdas):
        if n < L - 1:
            weights.append(surv * lam)
            surv = surv * (1.0 - lam)
        else:
            weights.append(surv)
    return torch.stack(weights, dim=0)


def geometric_prior(num_loops: int, lam: float, device=None) -> torch.Tensor:
    """Truncated geometric prior over loops 1..L with stop probability ``lam``. Returns [L]."""
    probs = [lam * (1.0 - lam) ** n for n in range(num_loops - 1)]
    probs.append((1.0 - lam) ** (num_loops - 1))
    return torch.tensor(probs, dtype=torch.float32, device=device)


def pondernet_kl(weights: torch.Tensor, prior: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Mean over tokens of KL(p || prior) for weights [L, B, S] and prior [L].

    ``mask`` (optional bool or 0/1 [B, S], True = real token) restricts the mean to the real tokens, so
    padding does not enter the halting penalty (CORE-10). None averages over every position.
    """
    p = weights.clamp(min=1e-12)
    q = prior.clamp(min=1e-12).view(-1, 1, 1)
    per_token = (p * (p.log() - q.log())).sum(dim=0)            # [B, S]
    if mask is None:
        return per_token.mean()
    real = mask.to(device=per_token.device, dtype=per_token.dtype)
    return (per_token * real).sum() / real.sum().clamp(min=1.0)


class MoRRouter(nn.Module):
    """Token-level depth router for ``routing_mode='mor'`` (Mixture-of-Recursions, arXiv 2507.10524,
    token-choice variant).

    Maps a token's own state (after the pre layers) to logits over the depths
    ``min_recurrent_loops .. max_recurrent_loops``. It reads nothing but that one token's state, so a
    token's depth never depends on later tokens: routing is causal, and a cached decode step can pick the
    depth of the new token without looking at the future.

    Selection (done by ``InkyModel``): in training a depth is sampled with Gumbel-max from
    ``logits / mor_temperature`` (hard choice, exploration), in eval the argmax is taken. The router gets
    its gradient from the gate: an active token's update is scaled by the softmax probability of the depth
    it chose (``mor_gate_scale``), exactly as MoR multiplies the router weight into the recursion output,
    and from the depth-balance loss. The single ``router`` attribute keeps the parameter name segment that
    the optimiser routes to AdamW.
    """

    def __init__(self, config: InkyConfig):
        super().__init__()
        n = config.max_recurrent_loops - config.min_recurrent_loops + 1
        d, hidden = config.hidden_size, config.mor_router_hidden
        if hidden is None:
            self.router = nn.Linear(d, n)
        else:
            self.router = nn.Sequential(nn.Linear(d, hidden), nn.SiLU(), nn.Linear(hidden, n))
        self.min_depth = config.min_recurrent_loops
        self.num_depths = n

    def reset_prior(self, target_depth: float) -> None:
        """Start the depth distribution at a truncated geometric whose mean is about ``target_depth``
        (set the final bias to its log). Call after the model's post_init, which would reset the bias."""
        n = self.num_depths
        extra = min(max(float(target_depth) - self.min_depth, 0.0), n - 1)
        lam = 1.0 / (extra + 1.0)
        probs = [lam * (1.0 - lam) ** k for k in range(n - 1)] + [(1.0 - lam) ** (n - 1)]
        last = self.router if isinstance(self.router, nn.Linear) else self.router[-1]
        with torch.no_grad():
            last.bias.copy_(torch.log(torch.tensor(probs, dtype=last.bias.dtype).clamp(min=1e-6)))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h [B, S, D] -> depth logits [B, S, num_depths] (float32)."""
        return self.router(h).float()
