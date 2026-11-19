"""Adapters that turn a small model into an *encoder*: token ids in, hidden states out.

An encoder is any callable ``encoder(input_ids, attention_mask) -> hidden`` where ``input_ids`` is a
[B, T] long tensor, ``attention_mask`` is a [B, T] bool tensor (True = real token) and ``hidden`` is
[B, T, hidden_size]. Arlo reads those states; it never changes the model.
"""

from typing import Callable

import torch


class FnEncoder:
    """Wrap a plain function ``fn(input_ids) -> [B, T, d]`` (an nn.Module works). The mask is dropped,
    which is correct for causal models: right padding cannot change any earlier position."""

    def __init__(self, fn: Callable[[torch.Tensor], torch.Tensor]):
        self.fn = fn

    def __call__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.fn(input_ids)


class HFEncoder:
    """Wrap a Hugging Face ``transformers`` model (``AutoModel`` or a model with a head). Uses the last
    hidden layer. Requires no import of transformers here; it only calls the model."""

    def __init__(self, model, layer: int = -1):
        self.model = model
        self.layer = layer

    def __call__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        out = self.model(
            input_ids=input_ids,
            attention_mask=None if attention_mask is None else attention_mask.long(),
            output_hidden_states=True,
        )
        return out.hidden_states[self.layer]


class AttrEncoder:
    """Wrap a model whose forward returns an object with a hidden-state attribute, e.g.
    ``model(ids, return_hidden=True).hidden``. Pass ``kwargs`` for the flags the model needs."""

    def __init__(self, model, attr: str = "hidden", **kwargs):
        self.model = model
        self.attr = attr
        self.kwargs = kwargs

    def __call__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        return getattr(self.model(input_ids, **self.kwargs), self.attr)


def as_encoder(obj) -> Callable:
    """Pass an encoder through, or wrap a bare callable/module in ``FnEncoder``."""
    if isinstance(obj, (FnEncoder, HFEncoder, AttrEncoder)):
        return obj
    if not callable(obj):
        raise TypeError("encoder must be callable: ids -> hidden states (see arlo.adapters)")
    try:
        import inspect

        params = [p for p in inspect.signature(obj).parameters.values()
                  if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        if len(params) >= 2 and params[1].name == "attention_mask":
            return obj
    except (TypeError, ValueError):
        pass
    return FnEncoder(obj)
