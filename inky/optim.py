"""Hybrid Muon + AdamW optimiser for Inky, ported from EOS (optimizers.py).

Parameter partitioning (see ``assign_parameter_groups``; deterministic, follows
``model.named_parameters()`` order):
  - Muon:   large 2D hidden matrices from Linear layers (attention projections, expert FFNs,
            state-space highway, SwiGLU gates). Updates are orthogonalised with a quintic
            Newton-Schulz iteration.
  - AdamW:  everything else: the token embedding (tied with the LM head), the per-loop FiLM
            table, RMSNorm gains, biases and the damping vector. Norms, biases and 1D scalars
            get no weight decay. Also routed here: tiny gating matrices, i.e. any 2D parameter
            with a dimension below ``min_muon_dim`` (MoE router, PonderNet halting gate, Arlo
            binary head), and any parameter with a ``router`` or ``halting`` name segment.
            Orthogonalising a one-row or few-row matrix makes the update size independent of
            the gradient, so these stay with AdamW.

Step order per training step (strict coupling, as in EOS):
  1. loss.backward()
  2. clip_grad_norm over all parameters (one global norm, both groups)
  3. muon.step()
  4. adamw.step()

Then call ``model.update_moe_balance()`` once per step, outside autograd.

Reference: Jordan, "Muon: MomentUm Orthogonalized by Newton-Schulz" (2024).
"""

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

# 2D parameters that are lookup tables or per-loop tables and must stay with AdamW.
ADAMW_ONLY_NAMES: Tuple[str, ...] = ("step_embedding", "slot_base")

# Dotted-name segments of tiny gating modules: the MoE router (layers.SharedExpertMoE) and the
# PonderNet halting gate (layers.PonderHalting, attribute ``halting``). Matched as whole
# segments, so ``gate_proj`` (SwiGLU, state-space gate) and ``a_gate`` are not caught.
ROUTING_NAME_SEGMENTS: Tuple[str, ...] = ("router", "halting")

# 2D matrices with a dimension below this are gating/readout-sized, not hidden matrices, and go
# to AdamW. The Inky router is (num_routed_experts, hidden) and the halting gate is (1, hidden).
MIN_MUON_DIM: int = 16

GROUP_MUON = "muon"
GROUP_ADAMW_DECAY = "adamw_decay"
GROUP_ADAMW_NO_DECAY = "adamw_no_decay"
GROUP_NAMES: Tuple[str, ...] = (GROUP_MUON, GROUP_ADAMW_DECAY, GROUP_ADAMW_NO_DECAY)


@torch.no_grad()
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Approximates the orthogonalisation UV^T of G with a quintic Newton-Schulz iteration.

    Singular values land near 1 after a few steps, so every direction of the update gets a
    similar step size. Runs in bf16 on CUDA for speed and in fp32 elsewhere.
    """
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)

    X = G.bfloat16() if G.is_cuda else G.float()
    transposed = G.size(-2) > G.size(-1)
    if transposed:
        X = X.mT

    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if transposed:
        X = X.mT
    return X


SCALE_MODES: Tuple[str, ...] = ("shape", "moonlight")


def muon_update_scale(rows: int, cols: int) -> float:
    """RMS scale for an orthogonalised update: tall matrices get sqrt(rows / cols), wide ones 1.0."""
    return max(1.0, rows / cols) ** 0.5


def _check_scale_mode(mode: str) -> str:
    if mode not in SCALE_MODES:
        raise ValueError(f"scale_mode must be one of {SCALE_MODES}, got {mode!r}")
    return mode


def muon_scale(rows: int, cols: int, mode: str = "shape") -> float:
    """Per-matrix scale for the orthogonalised update.

    "shape":     max(1, rows / cols) ** 0.5 (same as ``muon_update_scale``).
    "moonlight": 0.2 * sqrt(max(rows, cols)), which matches the update RMS to AdamW's
                (Moonlight, Liu et al. 2025).
    """
    _check_scale_mode(mode)
    if mode == "moonlight":
        return 0.2 * max(rows, cols) ** 0.5
    return muon_update_scale(rows, cols)


class Muon(torch.optim.Optimizer):
    """Momentum with Newton-Schulz orthogonalised updates, for 2D matrices only."""

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
        scale_mode: str = "shape",
    ):
        if lr <= 0.0:
            raise ValueError(f"Muon lr must be positive, got {lr}")
        self.scale_mode = _check_scale_mode(scale_mode)
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def state_dict(self) -> dict:
        sd = super().state_dict()
        sd["scale_mode"] = self.scale_mode
        return sd

    def load_state_dict(self, state_dict: dict) -> None:
        state_dict = dict(state_dict)
        mode = state_dict.pop("scale_mode", None)  # older state dicts have no key: keep current mode
        super().load_state_dict(state_dict)
        if mode is not None:
            self.scale_mode = _check_scale_mode(mode)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr, momentum = group["lr"], group["momentum"]
            nesterov, ns_steps, wd = group["nesterov"], group["ns_steps"], group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)

                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)                       # heavy-ball momentum
                update = g.add(buf, alpha=momentum) if nesterov else buf

                u = zeropower_via_newtonschulz5(update, steps=ns_steps)
                alpha = lr * muon_scale(p.size(-2), p.size(-1), self.scale_mode)

                if wd > 0.0:
                    p.mul_(1.0 - lr * wd)                        # decoupled weight decay
                p.add_(u.to(p.dtype).view_as(p), alpha=-alpha)

        return loss


def _check_min_muon_dim(min_muon_dim: int) -> int:
    if min_muon_dim < 1:
        raise ValueError(f"min_muon_dim must be >= 1, got {min_muon_dim}")
    return min_muon_dim


def _is_routing_name(name: str) -> bool:
    return any(segment in ROUTING_NAME_SEGMENTS for segment in name.split("."))


def _classify(
    name: str,
    p: nn.Parameter,
    embedding_ids: set,
    adamw_only_names: Sequence[str],
    min_muon_dim: int,
) -> str:
    is_adamw_only = any(key in name for key in adamw_only_names)
    is_tiny_matrix = p.ndim == 2 and min(p.shape) < min_muon_dim
    if (
        p.ndim == 2
        and id(p) not in embedding_ids
        and not is_adamw_only
        and not is_tiny_matrix
        and not _is_routing_name(name)
    ):
        return GROUP_MUON
    if p.ndim < 2 or name.endswith(".bias"):
        return GROUP_ADAMW_NO_DECAY
    return GROUP_ADAMW_DECAY


def assign_parameter_groups(
    model: nn.Module,
    adamw_only_names: Sequence[str] = ADAMW_ONLY_NAMES,
    min_muon_dim: int = MIN_MUON_DIM,
) -> List[Tuple[str, str, nn.Parameter]]:
    """Returns ``(name, group, param)`` for every trainable parameter.

    Deterministic: the order follows ``model.named_parameters()``, so the same model
    construction always yields the same assignment. Group labels are the ``GROUP_*`` constants.
    """
    _check_min_muon_dim(min_muon_dim)
    embedding_ids = set()
    for m in model.modules():
        if isinstance(m, nn.Embedding):
            embedding_ids.update(id(p) for p in m.parameters(recurse=False))

    out: List[Tuple[str, str, nn.Parameter]] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        group = _classify(name, p, embedding_ids, adamw_only_names, min_muon_dim)
        out.append((name, group, p))
    return out


def split_parameters(
    model: nn.Module,
    adamw_only_names: Sequence[str] = ADAMW_ONLY_NAMES,
    min_muon_dim: int = MIN_MUON_DIM,
) -> Tuple[List[nn.Parameter], List[nn.Parameter], List[nn.Parameter]]:
    """Partitions trainable parameters into (muon, adamw_decay, adamw_no_decay).

    See the module docstring and ``assign_parameter_groups`` for the rules.
    """
    buckets = {group: [] for group in GROUP_NAMES}
    for _, group, p in assign_parameter_groups(model, adamw_only_names, min_muon_dim):
        buckets[group].append(p)
    return buckets[GROUP_MUON], buckets[GROUP_ADAMW_DECAY], buckets[GROUP_ADAMW_NO_DECAY]


def _count_groups(groups: Sequence[dict]) -> Dict[str, int]:
    return {
        "tensors": sum(len(g["params"]) for g in groups),
        "params": sum(p.numel() for g in groups for p in g["params"]),
    }


def format_group_counts(counts: Dict[str, Dict[str, int]]) -> str:
    """One-line summary of ``HybridOptimizer.group_counts()`` for logs."""
    return "  ".join(
        f"{name}: {counts[name]['tensors']} tensors / {counts[name]['params']:,} params"
        for name in GROUP_NAMES
    )


class HybridOptimizer:
    """Muon for matrices, AdamW for the rest, stepped together with one global gradient clip.

    Config knobs:
      - ``muon_lr`` (default 0.02) is UNSWEPT: it was carried over from the EOS port and has not
        been tuned for Inky. Sweep it before relying on it, and pair any change with ``adamw_lr``
        (the two groups are scaled together by ``set_lr_scale``).
      - ``scale_mode``: "shape" (default) or "moonlight" update scaling (see ``muon_scale``).
      - ``min_muon_dim``: 2D matrices with a smaller dimension go to AdamW (see ``split_parameters``).
    """

    def __init__(
        self,
        model: nn.Module,
        adamw_lr: float = 6e-4,
        muon_lr: float = 0.02,  # UNSWEPT default; see class docstring
        muon_momentum: float = 0.95,
        muon_ns_steps: int = 5,
        muon_weight_decay: float = 0.1,  # decoupled decay, as in Moonlight; pass 0.0 to disable
        adamw_betas: Tuple[float, float] = (0.9, 0.95),
        adamw_weight_decay: float = 0.1,
        scale_mode: str = "shape",
        min_muon_dim: int = MIN_MUON_DIM,
    ):
        self.scale_mode = _check_scale_mode(scale_mode)
        muon_params, decay, no_decay = split_parameters(model, min_muon_dim=min_muon_dim)
        if not muon_params:
            raise ValueError("no 2D matrices found for Muon")
        self.muon = Muon(
            muon_params, lr=muon_lr, momentum=muon_momentum, nesterov=True,
            ns_steps=muon_ns_steps, weight_decay=muon_weight_decay, scale_mode=self.scale_mode,
        )
        self.adamw = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": adamw_weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=adamw_lr,
            betas=adamw_betas,
        )
        self._base_lrs = [g["lr"] for g in self.muon.param_groups + self.adamw.param_groups]
        self.stats = {
            "muon_tensors": len(muon_params),
            "muon_params": sum(p.numel() for p in muon_params),
            "adamw_tensors": len(decay) + len(no_decay),
            "adamw_params": sum(p.numel() for p in decay + no_decay),
        }

    @property
    def params(self) -> List[nn.Parameter]:
        return [p for g in self.muon.param_groups + self.adamw.param_groups for p in g["params"]]

    def group_counts(self) -> Dict[str, Dict[str, int]]:
        """Tensor and parameter counts per optimiser group, keyed by the ``GROUP_*`` labels.

        Read from the live optimiser, so it stays correct after ``load_state_dict``.
        Use ``format_group_counts`` to log it.
        """
        return {
            GROUP_MUON: _count_groups(self.muon.param_groups[:1]),
            GROUP_ADAMW_DECAY: _count_groups(self.adamw.param_groups[:1]),
            GROUP_ADAMW_NO_DECAY: _count_groups(self.adamw.param_groups[1:2]),
        }

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adamw.zero_grad(set_to_none=set_to_none)

    def set_lr_scale(self, scale: float) -> None:
        """Multiply both optimisers' base learning rates by ``scale`` (for warmup and decay schedules)."""
        for group, base in zip(self.muon.param_groups + self.adamw.param_groups, self._base_lrs):
            group["lr"] = base * scale

    def step(self, max_grad_norm: Optional[float] = None) -> Optional[float]:
        """One optimiser step. Returns the pre-clip gradient norm when clipping is requested."""
        grad_norm = None
        if max_grad_norm is not None:
            grad_norm = float(torch.nn.utils.clip_grad_norm_(self.params, max_grad_norm))
        self.muon.step()
        self.adamw.step()
        return grad_norm

    def state_dict(self) -> dict:
        return {
            "muon": self.muon.state_dict(),
            "adamw": self.adamw.state_dict(),
            "base_lrs": self._base_lrs,
            "scale_mode": self.scale_mode,
        }

    def load_state_dict(self, state: dict) -> None:
        self.muon.load_state_dict(state["muon"])
        self.adamw.load_state_dict(state["adamw"])
        self._base_lrs = list(state["base_lrs"])
        if "scale_mode" in state:  # older state dicts have no key: keep the mode this optimiser was built with
            self.scale_mode = _check_scale_mode(state["scale_mode"])
            self.muon.scale_mode = self.scale_mode


def build_hybrid_optimizer(model: nn.Module, **kwargs) -> HybridOptimizer:
    """Convenience constructor matching EOS's ``build_muon_adamw``."""
    return HybridOptimizer(model, **kwargs)
