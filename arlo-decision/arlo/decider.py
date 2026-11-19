"""ArloDecider: attach an Arlo head to any small model and ask it decision questions."""

import warnings
from typing import Optional, Sequence

import torch

from .adapters import as_encoder
from .config import ArloConfig
from .engine import Arlo


def _ids(x) -> torch.Tensor:
    """A list / 1-D / [1, L] integer tensor -> [1, L] long."""
    return torch.as_tensor(x, dtype=torch.long).reshape(1, -1)


def _option_ids(option, index: int) -> torch.Tensor:
    ids = _ids(option)
    if ids.shape[1] == 0:
        raise ValueError(f"option {index} is empty; every option needs at least one token")
    return ids


def _mean_pool_options(h: torch.Tensor, prefix_len: int, lengths) -> torch.Tensor:
    """Mean of each option's own token states. h [N, T, d]; row i = prefix then option i (right-padded)."""
    N, T, _ = h.shape
    pos = torch.arange(T, device=h.device).unsqueeze(0)
    lens = torch.as_tensor(list(lengths), device=h.device).unsqueeze(1)
    mask = ((pos >= prefix_len) & (pos < lens)).unsqueeze(-1).to(h.dtype)
    return (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)


class ArloDecider(torch.nn.Module):
    """A decision maker for small models.

    ``encoder`` turns token ids into hidden states (an nn.Module, a function, or an adapter from
    ``arlo.adapters``). ``hidden_size`` is the width of those states. Arlo reads the states and answers:

      * ``choice(context, options)``  -> probabilities over the options
      * ``yes_no(context)``           -> probability of yes

    The encoder is never modified. Only the small Arlo head (about 0.4 M parameters at width 384)
    is trained, with ``arlo.train_arlo``.
    """

    def __init__(self, encoder, hidden_size: int | None = None, config: ArloConfig | None = None, **config_kwargs):
        super().__init__()
        if config is None:
            if hidden_size is None:
                raise ValueError("give hidden_size (the encoder's hidden width) or a config")
            config = ArloConfig(hidden_size=hidden_size, **config_kwargs)
        elif hidden_size is not None and hidden_size != config.hidden_size:
            raise ValueError(f"hidden_size {hidden_size} does not match config.hidden_size {config.hidden_size}")
        self.config = config
        self.arlo = Arlo(config)
        self.trained = False
        self.set_encoder(encoder)

    # -- encoder plumbing -------------------------------------------------------------------
    def set_encoder(self, encoder) -> None:
        """Attach or swap the model Arlo reads. The encoder is stored outside the head's parameters,
        so ``state_dict`` / ``save`` contain Arlo only."""
        object.__setattr__(self, "_encoder", as_encoder(encoder))

    @property
    def encoder(self):
        return self._encoder

    def _device(self) -> torch.device:
        return next(self.arlo.parameters()).device

    def _encode(self, ids: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self._encoder(ids.to(self._device()), mask)
        if h.ndim != 3 or h.shape[-1] != self.config.hidden_size:
            raise ValueError(
                f"encoder returned {tuple(h.shape)}; expected [B, T, {self.config.hidden_size}] "
                f"(config.hidden_size). Fix hidden_size or the encoder/adapter."
            )
        return h

    def _warn_if_untrained(self) -> None:
        if not self.trained:
            warnings.warn(
                "Arlo has not been trained: its answers are an untrained prior. Run arlo.train_arlo, or load "
                "a trained checkpoint with ArloDecider.load / mark_trained().",
                UserWarning,
                stacklevel=3,
            )

    def mark_trained(self) -> None:
        self.trained = True

    # -- features (shared by inference and training) ----------------------------------------
    def features(self, context, task: str, options: Optional[Sequence] = None, question=None):
        """Frozen-encoder features for one decision: (query [1, d], states [1, P, d], options [1, N, d] or None)."""
        context = _ids(context)
        prefix = context if question is None else torch.cat([context, _ids(question)], dim=1)
        P = prefix.shape[1]
        if task == "binary":
            h = self._encode(prefix)
            return h[:, -1], h[:, :P], None
        if task != "choice":
            raise ValueError(f"unknown task {task!r}; expected 'choice' or 'binary'")
        if options is None or len(options) == 0:
            raise ValueError("a choice needs at least one option")
        opts = [_option_ids(o, i) for i, o in enumerate(options)]
        seqs = [torch.cat([prefix, o], dim=1)[0] for o in opts]
        lengths = [s.shape[0] for s in seqs]
        batch = torch.zeros((len(seqs), max(lengths)), dtype=torch.long)
        mask = torch.zeros_like(batch, dtype=torch.bool)
        for i, s in enumerate(seqs):
            batch[i, :lengths[i]] = s
            mask[i, :lengths[i]] = True
        h = self._encode(batch, mask.to(self._device()))
        option_states = _mean_pool_options(h, P, lengths).unsqueeze(0)
        return h[:1, P - 1], h[:1, :P], option_states

    # -- decisions --------------------------------------------------------------------------
    @torch.no_grad()
    def choice(self, context, options: Sequence, question=None) -> torch.Tensor:
        """Probability of each option, shape [N], sums to 1.

        ``context`` and each option are token-id lists (or 1-D / [1, L] tensors). Each option is scored
        after the context and optional ``question`` and is represented by the mean of its own token states.
        """
        self._warn_if_untrained()
        query, states, option_states = self.features(context, "choice", options, question)
        return torch.softmax(self.arlo.choice_scores(query, option_states, states), dim=-1)[0]

    @torch.no_grad()
    def yes_no(self, context, question=None) -> torch.Tensor:
        """Probability that the answer is yes, a scalar tensor."""
        self._warn_if_untrained()
        query, states, _ = self.features(context, "binary", None, question)
        return torch.sigmoid(self.arlo.binary_logit(query, states))[0]

    def pick(self, context, options: Sequence, question=None) -> int:
        """Index of the most probable option."""
        return int(self.choice(context, options, question).argmax())

    def decide(self, context, question=None, threshold: float = 0.5) -> bool:
        """True if P(yes) >= threshold."""
        return bool(self.yes_no(context, question) >= threshold)

    # -- checkpoints ------------------------------------------------------------------------
    def save(self, path) -> None:
        """Save the Arlo head only (config + weights + trained flag). The encoder is not saved."""
        blob = {"config": self.config.to_dict(), "state_dict": self.arlo.state_dict(), "trained": self.trained}
        torch.save(blob, path)

    @classmethod
    def load(cls, path, encoder, map_location="cpu") -> "ArloDecider":
        """Load a head saved with ``save`` and attach it to ``encoder``."""
        blob = torch.load(path, map_location=map_location, weights_only=True)
        dec = cls(encoder, config=ArloConfig.from_dict(blob["config"]))
        dec.arlo.load_state_dict(blob["state_dict"])
        dec.trained = bool(blob.get("trained", True))
        return dec
