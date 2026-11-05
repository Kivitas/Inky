"""Writing decisions: the Arlo decision engine's entry points on the model (Rhen's Choice and Noul)."""

import warnings
from typing import Optional, Sequence

import torch

from .arlo import mean_pool_options, option_ids
from .metrics import binary_report, choice_report, print_report


class InkyDecisionMixin:
    """``decide_choice`` / ``decide_binary`` and the bookkeeping that says whether Arlo has been trained.

    Mixed into ``InkyModel``; it uses ``self.arlo``, ``self(...)`` and ``self._arlo_trained``, which the model's
    ``__init__`` sets up."""

    def _on_arlo_grad(self, param: torch.Tensor) -> None:
        self._arlo_trained = True

    @property
    def arlo_trained(self) -> bool:
        """True once the Arlo parameters have received a gradient in this process (train_arlo does this).

        Not saved with the weights: a model reloaded from a checkpoint reports False until its decision
        engine is trained again (or ``mark_arlo_trained`` is called after loading trained weights).
        """
        return bool(self._arlo_trained)

    def mark_arlo_trained(self) -> None:
        """Declare the decision engine trained, e.g. after loading weights trained elsewhere."""
        self._arlo_trained = True

    def _warn_if_arlo_untrained(self) -> None:
        if not self._arlo_trained:
            warnings.warn(
                "decision engine (Arlo) has not been trained in this process; its outputs are an untrained "
                "prior. Run inky.arlo.train_arlo first, or call mark_arlo_trained() if the weights were trained "
                "elsewhere (ARLO-01).",
                UserWarning,
                stacklevel=4,     # helper -> decide_* -> torch.no_grad wrapper -> caller
            )

    @torch.no_grad()
    def decide_choice(
        self,
        context_ids: torch.Tensor,
        options: Sequence[torch.Tensor],
        question_ids: Optional[torch.Tensor] = None,
        gold: Optional[int] = None,
        report: bool = False,
        display: bool = False,
    ):
        """Probability of each option, from the decision engine (Rhen Choice).

        ``context_ids`` is [1, S]. ``options`` is a list of [1, L_i] tensors, one per option; every option
        needs at least one token (ValueError otherwise). Each option is scored after the context and the
        optional question, and is represented by the mean of its own token states (padding excluded).
        Because the model is causal, the context states are the same for every option.
        Returns a tensor of shape [N] that sums to 1. The model's train/eval mode is restored afterwards.
        """
        if self.arlo is None:
            raise RuntimeError("decision engine is disabled (enable_arlo=False)")
        self._warn_if_arlo_untrained()
        if len(options) == 0:
            raise ValueError("decide_choice needs at least one option")
        opts = [option_ids(opt, i) for i, opt in enumerate(options)]
        was_training = self.training
        self.eval()
        try:
            prefix = context_ids if question_ids is None else torch.cat([context_ids, question_ids], dim=1)
            P = prefix.shape[1]
            seqs = [torch.cat([prefix, o.to(prefix.device)], dim=1)[0] for o in opts]
            lengths = [s.shape[0] for s in seqs]
            # One batched forward. Options are right-padded with token 0, which is always a valid id
            # (config.pad_token_id may lie outside a small vocabulary). The model is causal, so the padding
            # cannot change any position before it, and each option's mean covers only its own tokens.
            batch = torch.zeros((len(seqs), max(lengths)), dtype=torch.long, device=prefix.device)
            for i, s in enumerate(seqs):
                batch[i, :lengths[i]] = s
            h = self(batch, return_hidden=True).hidden
            query = h[:1, P - 1]          # the last prefix token is the same for every option
            states = h[:1, :P]
            option_states = mean_pool_options(h, P, lengths)                  # [N, D]
            scores = self.arlo.choice_scores(query, option_states.unsqueeze(0), states)
        finally:
            self.train(was_training)
        probs = torch.softmax(scores, dim=-1)[0]
        if not (report or display):
            return probs
        gold_t = None if gold is None else torch.tensor([gold])
        rep = choice_report(probs.unsqueeze(0), gold_t)
        if display:
            print_report(rep)
        return probs, rep

    @torch.no_grad()
    def decide_binary(
        self,
        context_ids: torch.Tensor,
        question_ids: Optional[torch.Tensor] = None,
        gold: Optional[int] = None,
        report: bool = False,
        display: bool = False,
    ):
        """Probability that the answer is yes, from the decision engine (Rhen Noul). Returns a scalar.

        With ``report=True`` or ``display=True`` the output is also summarised by ``binary_report``.
        Pass ``gold`` (0 or 1) to get the confusion metrics. The model's train/eval mode is restored afterwards.
        """
        if self.arlo is None:
            raise RuntimeError("decision engine is disabled (enable_arlo=False)")
        self._warn_if_arlo_untrained()
        was_training = self.training
        self.eval()
        try:
            prefix = context_ids if question_ids is None else torch.cat([context_ids, question_ids], dim=1)
            h = self(prefix, return_hidden=True).hidden
            p = torch.sigmoid(self.arlo.binary_logit(h[:, -1], h))[0]
        finally:
            self.train(was_training)
        if not (report or display):
            return p
        rep = binary_report(p.unsqueeze(0), None if gold is None else [gold])
        if display:
            print_report(rep)
        return p, rep
