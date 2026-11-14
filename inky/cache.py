"""Decoding state and the model output container: ``InkyCache`` and ``InkyOutput``."""

import weakref
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from .config import InkyConfig
from .layers import KVEntry, LatentKV


def _claim(entry, cache: "InkyCache"):
    """Marks a freshly built latent entry as owned by ``cache`` (see ``_writable_past``). Other entries pass through."""
    if isinstance(entry, LatentKV):
        entry.owner = weakref.ref(cache)
    return entry


def _writable_past(entry, cache: "InkyCache"):
    """The entry to hand to a layer as ``past``: an entry that ``cache`` itself produced may be extended in
    place (its buffer has spare capacity, so no per-token concatenation). An entry that came from another
    cache (a fork shares the lists of its parent), a snapshot or a plain tuple is not writable, so the layer
    copies it on the first append and the owner of the buffer is never overwritten."""
    if isinstance(entry, LatentKV) and entry.owner is not None and entry.owner() is cache:
        return entry.alias_writable()
    return entry


class InkyCache:
    """Per-sequence decoding state.

    One latent-KV entry per pre layer, per post layer, and per (loop, recurrent layer), plus the
    SSM highway's running prefix sum for each loop (None when ``use_highway`` is off). The loop count is
    fixed at the first call.

    Latent-attention entries are ``LatentKV``: a preallocated buffer that grows geometrically (up to
    ``max_position_embeddings``) plus a valid length. A decode step writes the new latents into the buffer
    instead of concatenating the whole cache. They unpack as ``(c_kv, k_pe)``, are never edited in place
    once built, and only the cache that produced an entry extends its buffer in place; a fork or snapshot
    shares the buffer safely because anything else copies on its first append.
    """

    def __init__(self, config: InkyConfig):
        self.pre: List[Optional[KVEntry]] = [None] * config.num_pre_layers
        self.post: List[Optional[KVEntry]] = [None] * config.num_post_layers
        self.num_loops: Optional[int] = None
        self.core: List[Optional[list]] = []
        self.prefix: List[Optional[torch.Tensor]] = []
        self.seq_len = 0

    def truncate(self, new_len: int) -> None:
        """Roll back sequence-indexed KV entries to ``new_len`` positions in place.

        Intended for speculative verification rollback when no recurrent state (SSM highway or
        SelectiveSSMMixer) prevents O(1) sequence slicing.
        """
        if new_len < 0 or new_len > self.seq_len:
            raise ValueError(f"cannot truncate InkyCache of length {self.seq_len} to {new_len}")
        if new_len == self.seq_len:
            return

        def _trim(entry):
            if entry is None:
                return None
            if isinstance(entry, LatentKV):
                return entry.truncate(new_len)
            k, v = entry
            if k.ndim == 4 and v.ndim == 4:
                return (k[:, :, :new_len], v[:, :, :new_len])
            return entry

        self.pre = [_trim(e) for e in self.pre]
        self.post = [_trim(e) for e in self.post]
        self.core = [[_trim(e) for e in layer_list] if layer_list is not None else None for layer_list in self.core]
        self.seq_len = new_len


@dataclass
class InkyOutput:
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    past: Optional[InkyCache] = None
    # MoE balance-free aux: pre and post layers summed, recurrent layers averaged over the loops run.
    moe_aux_loss: Optional[torch.Tensor] = None
    # Expected number of loops per token under the halting distribution, sum_n n * p_n(t), averaged over
    # tokens. Same definition in forward and forward_early_exit (in [1, L]). Under routing_mode='mor' it is
    # the mean of the routed depths (each token's realised loop count, capped by num_loops).
    mean_loops: Optional[torch.Tensor] = None
    hidden: Optional[torch.Tensor] = None      # final hidden states, set when return_hidden=True
    # Token-loop pairs actually computed, and tokens x loops for the same input. Set by forward_early_exit
    # and, under routing_mode='mor', by forward (token rows run through the recurrent block stack).
    loop_applications: Optional[int] = None
    full_loop_applications: Optional[int] = None
    # Realised exit loop per token, averaged (early exit only). Equals L when nothing exits early.
    # Not the same quantity as mean_loops: it is the loop a token stopped at, not the expected value.
    mean_exit_loop: Optional[torch.Tensor] = None
    # Set when labels are given: detached, unweighted loss terms. Total loss is
    # ce + z_loss_coef * z_loss + moe_aux + ponder_step_loss_coef * step_ce + ponder_kl_coef * ponder_kl.
    # Under routing_mode='mor': ce + z_loss_coef * z_loss + moe_aux + mor_capacity_aux_coef * mor_balance
    # (step_ce and ponder_kl are reported as 0).
    loss_parts: Optional[Dict[str, torch.Tensor]] = None
