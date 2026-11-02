"""Building blocks for Inky: norms, RoPE, latent-KV attention, shared-expert MoE, recurrent core."""

import math
from typing import List, Optional, Tuple

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
        return out.to(x.dtype) * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_positions: int, base: float):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        freqs = torch.outer(torch.arange(max_positions, dtype=torch.float32), inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        # Persistent so that from_pretrained restores the exact tables instead of rebuilding them.
        self.register_buffer("cos", emb.cos())
        self.register_buffer("sin", emb.sin())

    def forward(self, start: int, length: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.cos[start:start + length], self.sin[start:start + length]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate the last dimension of x (broadcast over batch and heads); cos/sin are [S, D]."""
    return x * cos + _rotate_half(x) * sin


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
    ) -> Tuple[torch.Tensor, KVEntry]:
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

        if S == 1:
            out = self._decode_absorbed(q_nope, q_pe, c_kv, k_pe)
        else:
            out = self._full_attention(q_nope, q_pe, c_kv, k_pe, past_len)

        out = out.transpose(1, 2).reshape(B, S, H * self.v_dim)
        return self.o_proj(out), present

    def _full_attention(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        c_kv: torch.Tensor,
        k_pe: torch.Tensor,
        past_len: int,
    ) -> torch.Tensor:
        B, H, S, _ = q_nope.shape
        T = c_kv.shape[1]
        kv = self.kv_b_proj(c_kv).view(B, T, H, self.nope_dim + self.v_dim).transpose(1, 2)
        k_nope, v = kv.split([self.nope_dim, self.v_dim], dim=-1)
        k = torch.cat([k_nope, k_pe.expand(B, H, T, self.rope_dim)], dim=-1)
        q = torch.cat([q_nope, q_pe], dim=-1)
        if past_len == 0:
            return F.scaled_dot_product_attention(q, k, v, is_causal=True)
        # Query i sits at absolute position past_len + i and may attend to keys j <= past_len + i.
        mask = torch.ones(S, T, dtype=torch.bool, device=q.device).tril(diagonal=past_len)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

    def _decode_absorbed(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        c_kv: torch.Tensor,
        k_pe: torch.Tensor,
    ) -> torch.Tensor:
        """One new token against the cache, without up-projecting the cached latents.

        The key up-projection is folded into the query, and the value up-projection is applied
        after the attention-weighted sum. The result matches the full path, but the cost per
        cached position no longer scales with the head count times the per-head width.
        """
        H = self.num_heads
        w_kv = self.kv_b_proj.weight.view(H, self.nope_dim + self.v_dim, self.kv_rank)
        w_uk, w_uv = w_kv.split([self.nope_dim, self.v_dim], dim=1)  # [H, nope, r], [H, v, r]
        q_c = torch.einsum("bhsn,hnr->bhsr", q_nope, w_uk)            # [B, H, 1, r]
        scores = (
            torch.matmul(q_c, c_kv.unsqueeze(1).transpose(-1, -2))
            + torch.matmul(q_pe, k_pe.transpose(-1, -2))
        ) * self.scale                                                 # [B, H, 1, T]
        probs = torch.softmax(scores.float(), dim=-1).to(c_kv.dtype)
        out_c = torch.matmul(probs, c_kv.unsqueeze(1))                 # [B, H, 1, r]
        return torch.einsum("bhsr,hvr->bhsv", out_c, w_uv)             # [B, H, 1, v]


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
    ) -> Tuple[torch.Tensor, KVEntry]:
        B, S, _ = x.shape
        u = self.in_proj(x)                              # [B, S, d]
        a = torch.sigmoid(self.decay_proj(x))            # [B, S, N] input-dependent decay
        g = torch.sigmoid(self.gate_proj(x))             # [B, S, N] input-dependent input gate

        h = u.new_zeros(B, u.shape[-1], self.state_dim) if past is None else past[0]
        ys = []
        for t in range(S):
            h = a[:, t, None, :] * h + g[:, t, None, :] * u[:, t, :, None]   # [B, d, N]
            ys.append((h * self.readout).sum(-1))                             # [B, d]
        y = torch.stack(ys, dim=1) + self.skip * u                            # [B, S, d]

        present = (h, x.new_zeros(B, 1, 0))
        return self.out_proj(y), present


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
            counts = torch.bincount(idx.reshape(-1), minlength=self.num_routed).float()
            self.expert_load.copy_(counts / counts.sum().clamp(min=1.0))

        for e, expert in enumerate(self.routed_experts):
            token_idx, slot = torch.where(idx == e)
            if token_idx.numel() == 0:
                continue
            y = expert(flat[token_idx]) * weights[token_idx, slot].unsqueeze(-1).to(flat.dtype)
            out = out.index_add(0, token_idx, y)

        z_loss = torch.logsumexp(logits.float(), dim=-1).pow(2).mean() * self.z_loss_coef
        return out.view(B, S, D), z_loss

    @torch.no_grad()
    def update_balance_bias(self) -> None:
        target = 1.0 / self.num_routed
        self.balance_bias.add_(self.balance_rate * (target - self.expert_load))


class InkyBlock(nn.Module):
    """Pre-norm block: latent-KV attention, then the MoE feed-forward."""

    def __init__(self, config: InkyConfig, residual_scale: float = 1.0, use_ssm: bool = False):
        super().__init__()
        self.residual_scale = residual_scale
        self.use_checkpoint = config.gradient_checkpointing
        self.input_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        # The token mixer is named `attn` in both cases so the block's forward is the same.
        self.attn = SelectiveSSMMixer(config) if use_ssm else MultiLatentAttention(config)
        self.post_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = SharedExpertMoE(config)

    def _forward(self, x, cos, sin, past):
        attn_out, present = self.attn(self.input_norm(x), cos, sin, past)
        x = x + attn_out * self.residual_scale
        mlp_out, aux = self.mlp(self.post_norm(x))
        x = x + mlp_out * self.residual_scale
        return x, present, aux

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past: Optional[KVEntry] = None,
    ) -> Tuple[torch.Tensor, KVEntry, torch.Tensor]:
        if self.use_checkpoint and self.training and past is None and torch.is_grad_enabled():
            # Recompute activations in backward. Only used for full-sequence training, not decoding.
            def run(hs):
                out, _, aux = self._forward(hs, cos, sin, None)
                return out, aux

            out, aux = torch.utils.checkpoint.checkpoint(run, x, use_reentrant=False)
            return out, None, aux
        return self._forward(x, cos, sin, past)


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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        S = h.shape[1]
        counts = torch.arange(past_len + 1, past_len + S + 1, device=h.device, dtype=torch.float32)
        h_sum = h.float().cumsum(dim=1)
        if prefix_sum is not None:
            h_sum = h_sum + prefix_sum
        h_mean = (h_sum / counts.view(1, S, 1)).to(h.dtype)
        x_m = self.in_proj(h_mean)

        if prev_mem is None:
            mem = torch.tanh(x_m)
        else:
            alpha = torch.sigmoid(self.a_gate(prev_mem))
            mem = alpha * prev_mem + (1.0 - alpha) * torch.tanh(x_m)

        out = h + 0.1 * self.norm(self.out_proj(mem))
        return out, mem, h_sum[:, -1:, :]


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
    ):
        h_in = h
        if self.step_mod is not None:
            h = self.step_mod(h, step)
        h, mem, new_prefix = self.ssm(h, prev_mem, prefix_sum, past_len)

        presents: List[Optional[KVEntry]] = []
        aux = h.new_zeros(())
        for i, layer in enumerate(self.layers):
            if self.training and self.loop_dropout_prob > 0.0 and float(torch.rand(())) < self.loop_dropout_prob:
                presents.append(None)
                continue
            past_i = past[i] if past is not None else None
            h, present, layer_aux = layer(h, cos, sin, past_i)
            presents.append(present)
            aux = aux + layer_aux

        beta = (0.05 + 0.90 * torch.sigmoid(self.damped_beta)).view(1, 1, -1)
        out = (1.0 - beta) * h_in + beta * h
        return out, mem, new_prefix, presents, aux


class HaltingUnit(nn.Module):
    """Per-token probability of stopping the loop, from the current loop state."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.gate = nn.Linear(hidden_size, 1)

    def forward(self, h: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(self.gate(h).float()).squeeze(-1)
        # Tokens that barely moved in the last loop are more likely to stop.
        boost = torch.exp(-2.0 * velocity / math.sqrt(h.shape[-1]))
        return torch.clamp(p + 0.25 * boost, 0.0, 1.0)
