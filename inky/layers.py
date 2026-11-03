"""Building blocks for Inky: norms, RoPE, latent-KV attention, shared-expert MoE, recurrent core."""

import math
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from .config import InkyConfig

# Per-layer decoding cache. Latent attention: (c_kv, k_pe), the compressed KV latent [B, T, kv_lora_rank]
# and shared RoPE key [B, 1, T, rope_dim]. SSM mixer: (state, empty), state [B, hidden, state_dim].
KVEntry = Tuple[torch.Tensor, torch.Tensor]


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


class MultiLatentAttention(nn.Module):
    """Causal multi-latent attention.

    Queries go through a low-rank bottleneck. Keys and values share one compressed latent
    per token, and the RoPE part of the key is shared across heads. The cache stores only
    the latent and the RoPE key, which is much smaller than full per-head K and V.
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

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past: Optional[KVEntry] = None,
        attn_mask: Optional[torch.Tensor] = None,
        segment_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, KVEntry]:
        """``attn_mask`` (bool, broadcastable to [B, 1, S, past+S] or [B, S, past+S]; True = may attend)
        and ``segment_ids`` (int [B, S]; a query attends only to keys with the same id, prefill only)
        are optional and are combined with the causal mask. Both None gives the default behaviour."""
        B, S, _ = x.shape
        H = self.num_heads

        q = self.q_b_proj(self.q_a_norm(self.q_a_proj(x))).view(B, S, H, self.qk_dim).transpose(1, 2)
        q_nope, q_pe = q.split([self.nope_dim, self.rope_dim], dim=-1)
        q_pe = apply_rope(q_pe, cos, sin)

        c_kv, k_pe = self.kv_a_proj(x).split([self.kv_rank, self.rope_dim], dim=-1)
        c_kv = self.kv_a_norm(c_kv)
        k_pe = apply_rope(k_pe.unsqueeze(1), cos, sin)  # [B, 1, S, rope]

        if past is not None:
            c_kv = torch.cat([past[0], c_kv], dim=1)
            k_pe = torch.cat([past[1], k_pe], dim=2)
        present = (c_kv, k_pe)
        past_len = c_kv.shape[1] - S

        if attn_mask is None and segment_ids is None:
            if S == 1:
                out = self._decode_absorbed(q_nope, q_pe, c_kv, k_pe)
            else:
                out = self._full_attention(q_nope, q_pe, c_kv, k_pe, past_len)
        else:
            extra = self._extra_mask(B, S, past_len, past is not None, attn_mask, segment_ids, x.device)
            out = self._full_attention(q_nope, q_pe, c_kv, k_pe, past_len, extra_mask=extra)

        out = out.transpose(1, 2).reshape(B, S, H * self.v_dim)
        return self.o_proj(out), present

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
    ) -> torch.Tensor:
        B, H, S, _ = q_nope.shape
        T = c_kv.shape[1]
        kv = self.kv_b_proj(c_kv).view(B, T, H, self.nope_dim + self.v_dim).transpose(1, 2)
        k_nope, v = kv.split([self.nope_dim, self.v_dim], dim=-1)
        k = torch.cat([k_nope, k_pe.expand(B, H, T, self.rope_dim)], dim=-1)
        q = torch.cat([q_nope, q_pe], dim=-1)
        if extra_mask is None:
            if past_len == 0:
                return F.scaled_dot_product_attention(q, k, v, is_causal=True)
            # Query i sits at absolute position past_len + i and may attend to keys j <= past_len + i.
            mask = torch.ones(S, T, dtype=torch.bool, device=q.device).tril(diagonal=past_len)
        else:
            causal = torch.ones(S, T, dtype=torch.bool, device=q.device).tril(diagonal=past_len)
            mask = causal & extra_mask
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
        q_c = torch.einsum("bhsn,hnr->bhsr", q_nope, w_uk)            # [B, H, Q, r]
        scores = (
            torch.matmul(q_c, c_kv.unsqueeze(1).transpose(-1, -2))
            + torch.matmul(q_pe, k_pe.transpose(-1, -2))
        ) * self.scale                                                 # [B, H, Q, T]
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
        q_nope, q_pe = q.split([self.nope_dim, self.rope_dim], dim=-1)
        q_pe = apply_rope(q_pe, cos[active], sin[active])

        c, k_pe = self.kv_a_proj(x_act).split([self.kv_rank, self.rope_dim], dim=-1)
        c = self.kv_a_norm(c)
        k_pe = apply_rope(k_pe.unsqueeze(1), cos[active], sin[active])  # [1, 1, K, rope]

        c_store, kpe_store = store
        c_store[:, active] = c.to(c_store.dtype)
        kpe_store[:, :, active] = k_pe.to(kpe_store.dtype)

        S = c_store.shape[1]
        # Query at absolute position active[i] may attend to keys j <= active[i]. The absorbed form
        # attends over the cached latents directly, so the whole store is never up-projected per loop.
        positions = torch.arange(S, device=x_act.device)
        mask = (positions.unsqueeze(0) <= active.unsqueeze(1)).view(1, 1, K, S)   # [1, 1, K, S]
        out = self._absorbed_attention(q_nope, q_pe, c_kv=c_store, k_pe=kpe_store, mask=mask)
        out = out.transpose(1, 2).reshape(1, K, H * self.v_dim)
        return self.o_proj(out)


class SelectiveSSMMixer(nn.Module):
    """Causal selective state-space mixer, an optional replacement for latent attention.

    Each channel keeps a diagonal state of size ``ssm_state_dim``. At each token the state decays
    by an input-dependent factor a_t and receives the token's value through an input-dependent
    gate g_t: h_t = a_t * h_{t-1} + g_t * u_t. The output reads each channel's state with a learned
    per-channel readout, adds a per-channel skip of u_t, and projects back to the hidden size.

    The cache entry is (state, empty), where state is [B, hidden, state_dim] and empty is [B, 1, 0],
    so it has the same tuple shape as a KVEntry. The mixer is causal by construction and does not
    use cos or sin (no RoPE).
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
        if attn_mask is not None or segment_ids is not None:
            # The mixer's state runs along the whole sequence, so a mask cannot stop packed documents or
            # masked positions from influencing later tokens. Packed batches need latent-attention blocks.
            raise NotImplementedError(
                "SelectiveSSMMixer (ssm_mixer_layers) does not support attn_mask or segment_ids: its recurrent "
                "state runs across the whole sequence, so packed documents would leak into each other. "
                "Keep packed batches to attention-only recurrent layers, or run the SSM model unpacked."
            )
        B, S, _ = x.shape
        u = self.in_proj(x)                              # [B, S, d]
        log_a = F.logsigmoid(self.decay_proj(x))         # [B, S, N] log of the input-dependent decay
        g = torch.sigmoid(self.gate_proj(x))             # [B, S, N] input-dependent input gate

        h0 = u.new_zeros(B, u.shape[-1], self.state_dim) if past is None else past[0]
        y, h = self._chunked_scan(log_a, g, u, h0)
        y = y + self.skip * u                                                 # [B, S, d]

        present = (h, x.new_zeros(B, 1, 0))
        return self.out_proj(y), present

    def _chunked_scan(
        self, log_a: torch.Tensor, g: torch.Tensor, u: torch.Tensor, h0: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluates h_t = a_t * h_{t-1} + g_t * u_t (per channel and state) and the readout, chunk by chunk.

        Inside a chunk, with L_t = sum_{r<=t} log a_r, the contribution of token s to h_t is
        exp(L_t - L_s) * g_s * u_s (exponent <= 0, so nothing overflows), and the carried-in state
        contributes exp(L_t) * h_0. The decay depends only on (token, state), so the chunk's
        pairwise weights are [B, C, C, N] and the channel dimension is handled by one matmul.
        Returns y [B, S, d] (readout only, no skip) and the final state [B, d, N].
        """
        B, S, _ = u.shape
        C_max = self.scan_chunk
        calc = torch.promote_types(u.dtype, torch.float32)  # cumulative sums in at least float32
        h = h0
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
            intra = torch.matmul(Mg.permute(0, 3, 1, 2), uc.unsqueeze(1))   # [B, N, t, d]
            carry = torch.exp(L).to(u.dtype).unsqueeze(2) * h.unsqueeze(1)  # [B, C, d, N]
            h_chunk = intra.permute(0, 2, 3, 1) + carry                     # [B, C, d, N]
            ys.append(torch.einsum("bcdn,dn->bcd", h_chunk, self.readout))
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
        self.z_loss_coef = config.router_z_loss_coef

        self.shared_experts = nn.ModuleList(SwiGLU(d, ff) for _ in range(config.num_shared_experts))
        self.routed_experts = nn.ModuleList(SwiGLU(d, ff) for _ in range(self.num_routed))
        self.router = nn.Linear(d, self.num_routed, bias=False)
        self.register_buffer("balance_bias", torch.zeros(self.num_routed))
        self.register_buffer("expert_load", torch.full((self.num_routed,), 1.0 / self.num_routed))
        # Assignment counts summed over every forward call since the last update_balance_bias().
        # A looped core calls this MoE once per loop, so the counts of all loops are used. Not persisted.
        self.register_buffer("load_counts", torch.zeros(self.num_routed), persistent=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
            self.load_counts.add_(torch.bincount(idx.reshape(-1), minlength=self.num_routed).float())
            total = self.load_counts.sum().clamp(min=1.0)
            self.expert_load.copy_(self.load_counts / total)

        out = self._dispatch_routed(flat, idx, weights, out)

        z_loss = torch.logsumexp(logits.float(), dim=-1).pow(2).mean() * self.z_loss_coef
        return out.view(B, S, D), z_loss

    def _dispatch_routed(
        self,
        flat: torch.Tensor,
        idx: torch.Tensor,
        weights: torch.Tensor,
        out: torch.Tensor,
    ) -> torch.Tensor:
        """Adds the weighted routed-expert outputs to ``out``.

        The (token, slot) assignments are sorted by expert once, so each expert sees one contiguous
        block of rows. The outputs are scaled and scatter-added back to their tokens in one index_add.
        """
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

    @torch.no_grad()
    def update_balance_bias(self) -> None:
        """Moves the balance biases towards uniform load, using the counts since the last call, then resets them."""
        total = self.load_counts.sum()
        load = self.load_counts / total.clamp(min=1.0)
        target = 1.0 / self.num_routed
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
        self.attn = SelectiveSSMMixer(config) if use_ssm else MultiLatentAttention(config)
        self.post_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = SharedExpertMoE(config)

    def _residual_drop(self, branch: torch.Tensor) -> torch.Tensor:
        if self.residual_dropout > 0.0 and self.training:
            return F.dropout(branch, p=self.residual_dropout, training=True)
        return branch

    def _forward(self, x, cos, sin, past, attn_mask=None, segment_ids=None):
        attn_out, present = self.attn(self.input_norm(x), cos, sin, past,
                                      attn_mask=attn_mask, segment_ids=segment_ids)
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
    ) -> Tuple[torch.Tensor, KVEntry, torch.Tensor]:
        """``attn_mask`` and ``segment_ids`` are passed to the token mixer (see MultiLatentAttention.forward)."""
        if self.use_checkpoint and self.training and past is None and torch.is_grad_enabled():
            # Recompute activations in backward. Only used for full-sequence training, not decoding.
            def run(hs):
                out, _, aux = self._forward(hs, cos, sin, None, attn_mask, segment_ids)
                return out, aux

            out, aux = torch.utils.checkpoint.checkpoint(run, x, use_reentrant=False)
            return out, None, aux
        return self._forward(x, cos, sin, past, attn_mask, segment_ids)

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

    The input to each loop is replaced by its causal running mean, which is projected into
    a small state. That state is updated from the previous loop's state with a learned gate.
    """

    def __init__(self, hidden_size: int, state_dim: int, eps: float):
        super().__init__()
        self.in_proj = nn.Linear(hidden_size, state_dim, bias=False)
        self.a_gate = nn.Linear(state_dim, state_dim)
        self.out_proj = nn.Linear(state_dim, hidden_size, bias=False)
        self.norm = RMSNorm(hidden_size, eps)

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

        out = h + 0.1 * self.norm(self.out_proj(mem))
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
        self.ssm = StateSpaceMemoryHighway(config.hidden_size, config.ssm_state_dim, config.rms_norm_eps)
        self.damped_beta = nn.Parameter(torch.zeros(config.hidden_size))
        self.loop_dropout_prob = config.loop_dropout_prob

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
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], List[Optional[KVEntry]], torch.Tensor]:
        """One loop. ``segment_ids`` [B, S] (packed documents) restarts the highway's running mean at each
        document and is passed to every block. ``attn_mask`` (see MultiLatentAttention.forward) is passed to
        every block. The highway has no attention, so it takes only ``segment_ids`` and ``valid_mask`` (the
        pads it must leave out of its running mean, see StateSpaceMemoryHighway.forward). The defaults of
        all three give the default behaviour. Packing needs ``past is None`` and ``prefix_sum is None``."""
        h_in = h
        if self.step_mod is not None:
            h = self.step_mod(h, step)
        h, mem, new_prefix = self.ssm(
            h, prev_mem, prefix_sum, past_len, segment_ids=segment_ids, valid_mask=valid_mask
        )

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
            h, present, layer_aux = layer(h, cos, sin, past_i, attn_mask=attn_mask, segment_ids=segment_ids)
            presents.append(present)
            aux = aux + layer_aux

        beta = (0.05 + 0.90 * torch.sigmoid(self.damped_beta)).view(1, 1, -1)
        out = (1.0 - beta) * h_in + beta * h
        return out, mem, new_prefix, presents, aux

    def forward_subset(
        self,
        state: torch.Tensor,
        step: int,
        cos: torch.Tensor,
        sin: torch.Tensor,
        active: torch.Tensor,
        stores: List[Tuple[torch.Tensor, torch.Tensor]],
        prev_mem: Optional[torch.Tensor],
    ):
        """Inference-only loop step for the active tokens (batch size 1).

        ``state`` [1, S, D] holds every token's current loop state; exited tokens keep their final
        state. The SSM highway runs over all positions (it is cheap), and the attention blocks run
        only on ``active``. Returns the new states of the active tokens [1, K, D] and the SSM memory.
        """
        x = state
        if self.step_mod is not None:
            x = self.step_mod(x, step)
        x, mem, _ = self.ssm(x, prev_mem, None, 0)
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
    per_token = (weights * (p.log() - q.log())).sum(dim=0)            # [B, S]
    if mask is None:
        return per_token.mean()
    real = mask.to(device=per_token.device, dtype=per_token.dtype)
    return (per_token * real).sum() / real.sum().clamp(min=1.0)
