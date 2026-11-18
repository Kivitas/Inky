"""Continual Reinforcement Learning with Group Relative Policy Optimization (GRPO) for Inky.

Enables Inky to continually learn from conversations and adapt to a user's evolving preferences
(lexical choices, phrasings, sentence/response length cadence, and formal vs. informal register)
without requiring a separate critic network.

Key components:
  - ``UserPreferenceProfile``: online preference model tracking liked/disliked vocabulary,
    n-grams, response length EMA, and style register preference across turns.
  - ``compute_group_relative_advantages``: group-normalized advantage estimator
    $A_i = (r_i - \\mu_r) / (\\sigma_r + \\epsilon)$.
  - ``sequence_log_probs``: per-token response log-probabilities $\\log \\pi(o_{i,t} \\mid q, o_{i,<t})$.
  - ``grpo_loss``: clipped surrogate objective with the unbiased non-negative Schulman KL penalty
    against a reference policy.
  - ``ContinualGRPOTrainer``: online GRPO trainer supporting group rollouts, user feedback
    integration, reference policy EMA anchoring, and MoE load-balance updates.
"""

import copy
import math
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

from .config import REASONING_MODES, STYLE_MODES
from .metrics import text_report
from .modeling import InkyModel
from .optim import HybridOptimizer


def _to_id_list(ids: Union[torch.Tensor, Sequence[int]]) -> List[int]:
    if isinstance(ids, torch.Tensor):
        return [int(x) for x in ids.detach().cpu().reshape(-1).tolist()]
    return [int(x) for x in ids]


def _extract_ngrams(ids: Sequence[int], n: int = 2) -> List[Tuple[int, ...]]:
    if len(ids) < n or n < 1:
        return []
    return [tuple(int(x) for x in ids[i : i + n]) for i in range(len(ids) - n + 1)]


class UserPreferenceProfile:
    """Tracks a user's evolving likes and dislikes across conversation turns.

    Maintains running counts of positively and negatively reinforced tokens and bigrams,
    an exponential moving average (EMA) of preferred response length, and a continuous
    style preference score in ``[-1.0, 1.0]`` (``> 0`` favors ``"formal"``, ``< 0`` favors
    ``"informal"``).
    """

    def __init__(
        self,
        preferred_length: float = 48.0,
        length_tolerance: float = 32.0,
        ema_alpha: float = 0.15,
        max_tracked_items: int = 4096,
    ):
        if preferred_length <= 0.0:
            raise ValueError(f"preferred_length must be > 0, got {preferred_length}")
        if length_tolerance <= 0.0:
            raise ValueError(f"length_tolerance must be > 0, got {length_tolerance}")
        if not 0.0 < ema_alpha <= 1.0:
            raise ValueError(f"ema_alpha must be in (0, 1], got {ema_alpha}")
        self.preferred_length = float(preferred_length)
        self.length_tolerance = float(length_tolerance)
        self.ema_alpha = float(ema_alpha)
        self.max_tracked_items = int(max_tracked_items)

        self.liked_token_ids: Counter = Counter()
        self.disliked_token_ids: Counter = Counter()
        self.liked_ngrams: Counter = Counter()
        self.disliked_ngrams: Counter = Counter()

        # Continuous style preference in [-1, 1]: +1 = formal, -1 = informal, 0 = neutral
        self.style_preference: float = 0.0
        self.total_interactions: int = 0
        self.positive_interactions: int = 0
        self.negative_interactions: int = 0

    @property
    def preferred_style_mode(self) -> Optional[str]:
        """Inferred dominant style mode ('formal', 'informal', or None if neutral)."""
        if self.style_preference >= 0.15:
            return "formal"
        if self.style_preference <= -0.15:
            return "informal"
        return None

    def _prune_counter(self, counter: Counter) -> None:
        if len(counter) > self.max_tracked_items:
            most_common = dict(counter.most_common(self.max_tracked_items))
            counter.clear()
            counter.update(most_common)

    def record_feedback(
        self,
        response_ids: Union[torch.Tensor, Sequence[int]],
        reward: float,
        style_mode: Optional[str] = None,
    ) -> None:
        """Update the preference profile from a conversation response and scalar feedback reward.

        Positive ``reward > 0`` reinforces the response's vocabulary, bigrams, length, and
        style register; negative ``reward < 0`` records them as disliked.
        """
        if style_mode is not None and style_mode not in STYLE_MODES:
            raise ValueError(f"style_mode must be one of {STYLE_MODES} or None, got {style_mode!r}")
        ids = _to_id_list(response_ids)
        if not ids or reward == 0.0:
            return

        weight = max(1, int(round(abs(float(reward)) * 2.0)))
        unique_tokens = set(ids)
        bigrams = set(_extract_ngrams(ids, n=2))

        self.total_interactions += 1
        if reward > 0.0:
            self.positive_interactions += 1
            for tok in unique_tokens:
                self.liked_token_ids[tok] += weight
            for bg in bigrams:
                self.liked_ngrams[bg] += weight
            # Pull preferred length toward positively rated responses
            eff_alpha = min(1.0, self.ema_alpha * min(abs(float(reward)), 2.0))
            self.preferred_length = (1.0 - eff_alpha) * self.preferred_length + eff_alpha * float(len(ids))
            if style_mode is not None:
                target = 1.0 if style_mode == "formal" else -1.0
                self.style_preference = (1.0 - eff_alpha) * self.style_preference + eff_alpha * target
        else:
            self.negative_interactions += 1
            for tok in unique_tokens:
                self.disliked_token_ids[tok] += weight
            for bg in bigrams:
                self.disliked_ngrams[bg] += weight
            if style_mode is not None:
                eff_alpha = min(1.0, self.ema_alpha * min(abs(float(reward)), 2.0))
                target = -1.0 if style_mode == "formal" else 1.0
                self.style_preference = (1.0 - eff_alpha) * self.style_preference + eff_alpha * target

        self.style_preference = max(-1.0, min(1.0, self.style_preference))
        self._prune_counter(self.liked_token_ids)
        self._prune_counter(self.disliked_token_ids)
        self._prune_counter(self.liked_ngrams)
        self._prune_counter(self.disliked_ngrams)

    def score_response(
        self,
        response_ids: Union[torch.Tensor, Sequence[int]],
        style_mode: Optional[str] = None,
        bad_word_ids: Optional[Sequence[Sequence[int]]] = None,
        soft_penalty_ids: Optional[Sequence[int]] = None,
    ) -> float:
        """Compute a composite scalar reward for ``response_ids`` under the user's preferences.

        Combines:
          1. User lexical & phrase affinity (liked vs. disliked tokens and bigrams).
          2. Length preference alignment (Gaussian proximity to ``preferred_length``).
          3. Human-like prose quality (distinct-2 diversity, repetition avoidance, zero banned
             stock phrases, and adherence to ``style_mode`` via ``soft_penalty_ids``).
        """
        eff_style = style_mode if style_mode is not None else self.preferred_style_mode
        if eff_style is not None and eff_style not in STYLE_MODES:
            raise ValueError(f"style_mode must be one of {STYLE_MODES} or None, got {eff_style!r}")

        ids = _to_id_list(response_ids)
        if not ids:
            return -1.0

        # 1. Lexical & phrase affinity in [-1, 1]
        token_score = 0.0
        for tok in set(ids):
            pos = float(self.liked_token_ids.get(tok, 0))
            neg = float(self.disliked_token_ids.get(tok, 0))
            if pos > 0.0 or neg > 0.0:
                token_score += (pos - neg) / (pos + neg + 1.0)
        token_affinity = token_score / max(1, len(set(ids)))

        bigrams = set(_extract_ngrams(ids, n=2))
        ngram_score = 0.0
        for bg in bigrams:
            pos = float(self.liked_ngrams.get(bg, 0))
            neg = float(self.disliked_ngrams.get(bg, 0))
            if pos > 0.0 or neg > 0.0:
                ngram_score += (pos - neg) / (pos + neg + 1.0)
        ngram_affinity = ngram_score / max(1, len(bigrams)) if bigrams else 0.0

        affinity_reward = 1.2 * token_affinity + 1.5 * ngram_affinity

        # 2. Length preference alignment in [0, 1]
        len_diff = (float(len(ids)) - self.preferred_length) / max(1.0, self.length_tolerance)
        length_reward = math.exp(-0.5 * len_diff * len_diff)

        # 3. Human-like prose quality & register adherence
        rep = text_report(ids, bad_word_ids=bad_word_ids)
        d2 = rep.get("distinct_2")
        distinct_bonus = float(d2) if d2 is not None else 0.5
        rep_4 = rep.get("repetition_rate_4")
        repetition_penalty = float(rep_4) if rep_4 is not None else 0.0
        blocked_hits = int(rep.get("blocked_phrase_hits", 0) or 0)

        soft_set = {int(t) for t in (soft_penalty_ids or ())}
        soft_hits = sum(1 for t in ids if t in soft_set) if soft_set else 0
        soft_rate = float(soft_hits) / float(len(ids))

        prose_reward = (
            0.6 * distinct_bonus
            - 0.8 * repetition_penalty
            - 0.5 * float(blocked_hits)
            - 0.6 * soft_rate
        )

        return float(affinity_reward + 0.4 * length_reward + prose_reward)

    def state_dict(self) -> Dict[str, Any]:
        """Serialize the user's preference profile to a plain dictionary."""
        return {
            "preferred_length": self.preferred_length,
            "length_tolerance": self.length_tolerance,
            "ema_alpha": self.ema_alpha,
            "max_tracked_items": self.max_tracked_items,
            "style_preference": self.style_preference,
            "total_interactions": self.total_interactions,
            "positive_interactions": self.positive_interactions,
            "negative_interactions": self.negative_interactions,
            "liked_token_ids": dict(self.liked_token_ids),
            "disliked_token_ids": dict(self.disliked_token_ids),
            "liked_ngrams": [[list(k), v] for k, v in self.liked_ngrams.items()],
            "disliked_ngrams": [[list(k), v] for k, v in self.disliked_ngrams.items()],
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore the user's preference profile from ``state_dict()``."""
        self.preferred_length = float(state.get("preferred_length", self.preferred_length))
        self.length_tolerance = float(state.get("length_tolerance", self.length_tolerance))
        self.ema_alpha = float(state.get("ema_alpha", self.ema_alpha))
        self.max_tracked_items = int(state.get("max_tracked_items", self.max_tracked_items))
        self.style_preference = float(state.get("style_preference", 0.0))
        self.total_interactions = int(state.get("total_interactions", 0))
        self.positive_interactions = int(state.get("positive_interactions", 0))
        self.negative_interactions = int(state.get("negative_interactions", 0))

        self.liked_token_ids = Counter({int(k): int(v) for k, v in state.get("liked_token_ids", {}).items()})
        self.disliked_token_ids = Counter({int(k): int(v) for k, v in state.get("disliked_token_ids", {}).items()})
        self.liked_ngrams = Counter({tuple(int(x) for x in k): int(v) for k, v in state.get("liked_ngrams", [])})
        self.disliked_ngrams = Counter(
            {tuple(int(x) for x in k): int(v) for k, v in state.get("disliked_ngrams", [])}
        )


# ------------------------------------------------------------------ GRPO math primitives


def compute_group_relative_advantages(
    rewards: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute group-normalized advantages $A_i = (r_i - \\text{mean}(r)) / (\\text{std}(r) + \\epsilon)$.

    Accepts a 1D tensor ``[G]`` or a 2D batch of groups ``[B, G]``. When ``G == 1`` or the group
    standard deviation is zero, returns finite centered/raw values without ``NaN``.
    """
    if eps <= 0.0:
        raise ValueError(f"eps must be > 0, got {eps}")
    r = rewards.float()
    if r.numel() == 0:
        return r
    group_size = r.shape[-1]
    if group_size <= 1:
        return r.clone()
    mean = r.mean(dim=-1, keepdim=True)
    centered = r - mean
    std = r.std(dim=-1, unbiased=False, keepdim=True)
    return centered / (std + eps)


def sequence_log_probs(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    prompt_len: int,
    pad_token_id: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute per-token log-probabilities $\\log \\pi(o_{i,t} \\mid q, o_{i,<t})$ on response tokens.

    Args:
        logits: Model logits ``[G, S, V]`` over the full sequence (or ``[G, R, V]`` already sliced).
        token_ids: Full sequence token IDs ``[G, S]`` where ``S = prompt_len + R``.
        prompt_len: Number of prompt tokens $P \\ge 1$ preceding the response.
        pad_token_id: Optional padding token ID to mask out in the response region.

    Returns:
        ``(log_probs, mask)`` each of shape ``[G, R]`` in ``float32``.
    """
    if logits.ndim != 3 or token_ids.ndim != 2:
        raise ValueError(
            f"expected logits [G, S, V] and token_ids [G, S], got {tuple(logits.shape)} and {tuple(token_ids.shape)}"
        )
    if prompt_len < 1 or prompt_len >= token_ids.shape[1]:
        raise ValueError(f"prompt_len must be in [1, {token_ids.shape[1] - 1}], got {prompt_len}")

    target_ids = token_ids[:, prompt_len:]
    resp_len = target_ids.shape[1]
    if logits.shape[1] == token_ids.shape[1]:
        pred_logits = logits[:, prompt_len - 1 : token_ids.shape[1] - 1, :]
    elif logits.shape[1] == resp_len:
        pred_logits = logits
    else:
        raise ValueError(
            f"logits sequence length {logits.shape[1]} does not match token_ids {token_ids.shape[1]} "
            f"or response length {resp_len}"
        )

    log_p = F.log_softmax(pred_logits.float(), dim=-1)
    token_log_p = log_p.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)

    if pad_token_id is None:
        mask = torch.ones_like(token_log_p, dtype=torch.float32)
    else:
        mask = (target_ids != int(pad_token_id)).to(dtype=torch.float32)

    return token_log_p * mask, mask


def grpo_loss(
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    ref_log_probs: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    clip_eps: float = 0.2,
    kl_beta: float = 0.04,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute the Group Relative Policy Optimization (GRPO) loss and diagnostic metrics.

    Objective per token (i, t):
      L_GRPO = -min(rho * A_i, clip(rho, 1 - eps, 1 + eps) * A_i) + beta * D_KL(pi || pi_ref)
    where rho = exp(log_pi - log_pi_old) and the unbiased non-negative Schulman KL estimator is
    D_KL(pi || pi_ref) = exp(log_pi_ref - log_pi) - (log_pi_ref - log_pi) - 1.
    """
    if clip_eps < 0.0:
        raise ValueError(f"clip_eps must be >= 0, got {clip_eps}")
    if kl_beta < 0.0:
        raise ValueError(f"kl_beta must be >= 0, got {kl_beta}")

    if mask is None:
        mask_f = torch.ones_like(policy_log_probs, dtype=torch.float32)
    else:
        mask_f = mask.to(dtype=torch.float32, device=policy_log_probs.device)

    adv = advantages.to(dtype=torch.float32, device=policy_log_probs.device)
    if adv.ndim == 1:
        adv = adv.unsqueeze(-1)

    log_ratio = (policy_log_probs.float() - old_log_probs.detach().float()).clamp(-20.0, 20.0)
    ratio = torch.exp(log_ratio)

    surr1 = ratio * adv
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    surr2 = clipped_ratio * adv
    surrogate = torch.minimum(surr1, surr2)

    denom = mask_f.sum().clamp(min=1.0)
    policy_loss = -(surrogate * mask_f).sum() / denom

    if ref_log_probs is not None and kl_beta > 0.0:
        log_ref_diff = (ref_log_probs.detach().float() - policy_log_probs.float()).clamp(-20.0, 20.0)
        per_token_kl = torch.exp(log_ref_diff) - log_ref_diff - 1.0
        kl_val = (per_token_kl * mask_f).sum() / denom
        total_loss = policy_loss + kl_beta * kl_val
    else:
        if ref_log_probs is not None:
            log_ref_diff = (ref_log_probs.detach().float() - policy_log_probs.float()).clamp(-20.0, 20.0)
            per_token_kl = torch.exp(log_ref_diff) - log_ref_diff - 1.0
            kl_val = (per_token_kl * mask_f).sum() / denom
        else:
            kl_val = policy_loss.new_zeros(())
        total_loss = policy_loss

    with torch.no_grad():
        clipped_mask = ((ratio - 1.0).abs() > clip_eps).float() * mask_f
        clip_frac = float((clipped_mask.sum() / denom).item())

    metrics = {
        "loss": float(total_loss.detach().item()),
        "policy_loss": float(policy_loss.detach().item()),
        "kl": float(kl_val.detach().item()),
        "mean_advantage": float(advantages.detach().float().mean().item()),
        "clip_fraction": clip_frac,
    }
    return total_loss, metrics


# ------------------------------------------------------------------ Continual GRPO Trainer


class ContinualGRPOTrainer:
    """Continual RL trainer using Group Relative Policy Optimization (GRPO) for Inky.

    Adapts ``model`` online from group rollouts and user conversation feedback while anchoring to a
    reference policy snapshot via the Schulman KL penalty.
    """

    def __init__(
        self,
        model: InkyModel,
        profile: Optional[UserPreferenceProfile] = None,
        optimizer: Optional[Union[torch.optim.Optimizer, HybridOptimizer]] = None,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        clip_eps: float = 0.2,
        kl_beta: float = 0.04,
        grpo_epochs: int = 2,
        max_grad_norm: float = 1.0,
        ref_ema_decay: Optional[float] = 0.995,
        bad_word_ids: Optional[Sequence[Sequence[int]]] = None,
        soft_penalty_ids: Optional[Sequence[int]] = None,
    ):
        if grpo_epochs < 1:
            raise ValueError(f"grpo_epochs must be >= 1, got {grpo_epochs}")
        if ref_ema_decay is not None and not 0.0 <= float(ref_ema_decay) <= 1.0:
            raise ValueError(f"ref_ema_decay must be in [0, 1] or None, got {ref_ema_decay}")

        self.model = model
        self.profile = profile if profile is not None else UserPreferenceProfile()
        if optimizer is None:
            trainable = [p for p in model.parameters() if p.requires_grad]
            self.optimizer: Union[torch.optim.Optimizer, HybridOptimizer] = torch.optim.AdamW(
                trainable, lr=lr, weight_decay=weight_decay
            )
        else:
            self.optimizer = optimizer

        self.clip_eps = float(clip_eps)
        self.kl_beta = float(kl_beta)
        self.grpo_epochs = int(grpo_epochs)
        self.max_grad_norm = float(max_grad_norm)
        self.ref_ema_decay = float(ref_ema_decay) if ref_ema_decay is not None else None
        self.bad_word_ids = list(bad_word_ids) if bad_word_ids is not None else None
        self.soft_penalty_ids = list(soft_penalty_ids) if soft_penalty_ids is not None else None

        # Frozen reference policy for KL regularization
        ref_cfg = copy.deepcopy(model.config)
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        self.ref_model = InkyModel(ref_cfg).to(device=device, dtype=dtype)
        self.ref_model.load_state_dict(model.state_dict())
        self.ref_model.eval()
        self.ref_model.requires_grad_(False)

        self.step_count: int = 0

    @torch.no_grad()
    def sync_reference_policy(self, tau: float = 1.0) -> None:
        """Update the reference policy toward the current policy: theta_ref <- (1 - tau) * theta_ref + tau * theta.

        Pass ``tau=1.0`` for a hard snapshot reset, or ``tau = 1 - ref_ema_decay`` for soft EMA anchoring.
        """
        if not 0.0 <= float(tau) <= 1.0:
            raise ValueError(f"tau must be in [0, 1], got {tau}")
        if tau == 1.0:
            self.ref_model.load_state_dict(self.model.state_dict())
            return
        model_params = dict(self.model.named_parameters())
        for name, ref_p in self.ref_model.named_parameters():
            if name in model_params:
                ref_p.data.mul_(1.0 - tau).add_(model_params[name].data, alpha=tau)
        model_buffers = dict(self.model.named_buffers())
        for name, ref_b in self.ref_model.named_buffers():
            if name in model_buffers and ref_b.dtype.is_floating_point:
                ref_b.data.mul_(1.0 - tau).add_(model_buffers[name].data, alpha=tau)

    def _collate_group(
        self,
        prompt_ids: torch.Tensor,
        responses: Sequence[Union[torch.Tensor, Sequence[int]]],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[int]]]:
        """Assemble a padded ``[G, prompt_len + max_resp_len]`` batch and response mask ``[G, max_resp_len]``."""
        device = next(self.model.parameters()).device
        prompt_1d = prompt_ids.detach().to(device=device, dtype=torch.long).reshape(-1)
        prompt_list = prompt_1d.tolist()
        prompt_len = len(prompt_list)
        if prompt_len < 1:
            raise ValueError("prompt_ids must contain at least 1 token")

        clean_responses: List[List[int]] = []
        for resp in responses:
            r_list = _to_id_list(resp)
            if not r_list:
                r_list = [int(self.model.config.eos_token_id)]
            clean_responses.append(r_list)

        group_size = len(clean_responses)
        max_resp_len = max(len(r) for r in clean_responses)
        pad_id = int(
            self.model.config.pad_token_id
            if self.model.config.pad_token_id is not None
            else self.model.config.eos_token_id
        )

        batch_ids = torch.full(
            (group_size, prompt_len + max_resp_len),
            fill_value=pad_id,
            dtype=torch.long,
            device=device,
        )
        resp_mask = torch.zeros((group_size, max_resp_len), dtype=torch.float32, device=device)

        for i, r_list in enumerate(clean_responses):
            seq = prompt_list + r_list
            batch_ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
            resp_mask[i, : len(r_list)] = 1.0

        return batch_ids, resp_mask, clean_responses

    def _step_optimizer(self) -> float:
        if isinstance(self.optimizer, HybridOptimizer):
            norm = self.optimizer.step(max_grad_norm=self.max_grad_norm)
            return float(norm) if norm is not None else 0.0
        trainable = [p for p in self.model.parameters() if p.requires_grad and p.grad is not None]
        grad_norm = (
            float(torch.nn.utils.clip_grad_norm_(trainable, self.max_grad_norm).item())
            if trainable and self.max_grad_norm > 0.0
            else 0.0
        )
        self.optimizer.step()
        return grad_norm

    def step_on_group(
        self,
        prompt_ids: torch.Tensor,
        responses: Sequence[Union[torch.Tensor, Sequence[int]]],
        explicit_rewards: Optional[Sequence[float]] = None,
        reasoning_mode: Optional[str] = None,
        style_mode: Optional[str] = None,
        record_preferences: bool = True,
    ) -> Dict[str, Any]:
        """Run one continual GRPO update on a group of $G$ candidate responses for a single prompt.

        Combines optional ``explicit_rewards`` from user feedback with ``profile.score_response``,
        records feedback into ``UserPreferenceProfile``, computes group-relative advantages, and
        performs ``grpo_epochs`` policy updates with KL regularization and MoE balance updates.
        """
        if not responses:
            raise ValueError("responses must contain at least 1 candidate sequence")
        if explicit_rewards is not None and len(explicit_rewards) != len(responses):
            raise ValueError(f"got {len(responses)} responses but {len(explicit_rewards)} explicit_rewards")

        eff_reasoning = (
            reasoning_mode if reasoning_mode is not None else getattr(self.model.config, "reasoning_mode", None)
        )
        eff_style = style_mode if style_mode is not None else getattr(self.model.config, "style_mode", None)
        if eff_style is None:
            eff_style = self.profile.preferred_style_mode
        if eff_reasoning is not None and eff_reasoning not in REASONING_MODES:
            raise ValueError(f"reasoning_mode must be one of {tuple(REASONING_MODES)} or None, got {eff_reasoning!r}")
        if eff_style is not None and eff_style not in STYLE_MODES:
            raise ValueError(f"style_mode must be one of {STYLE_MODES} or None, got {eff_style!r}")

        batch_ids, resp_mask, clean_responses = self._collate_group(prompt_ids, responses)
        prompt_len = prompt_ids.numel()
        device = batch_ids.device

        # 1. Score every candidate against the user's preference profile + optional explicit feedback
        profile_scores = [
            self.profile.score_response(
                r_ids,
                style_mode=eff_style,
                bad_word_ids=self.bad_word_ids,
                soft_penalty_ids=self.soft_penalty_ids,
            )
            for r_ids in clean_responses
        ]
        if explicit_rewards is not None:
            combined_rewards = [float(er) + 0.5 * float(ps) for er, ps in zip(explicit_rewards, profile_scores)]
        else:
            combined_rewards = [float(ps) for ps in profile_scores]

        if record_preferences:
            feedback_source = explicit_rewards if explicit_rewards is not None else combined_rewards
            mean_fb = sum(feedback_source) / len(feedback_source)
            for r_ids, fb in zip(clean_responses, feedback_source):
                centered_fb = float(fb) - (0.0 if explicit_rewards is not None else mean_fb)
                if centered_fb != 0.0:
                    self.profile.record_feedback(r_ids, reward=centered_fb, style_mode=eff_style)

        rewards_tensor = torch.tensor(combined_rewards, dtype=torch.float32, device=device)
        advantages = compute_group_relative_advantages(rewards_tensor)

        # Build attention mask so padded tail tokens do not pollute attention keys
        full_attn_mask = torch.ones_like(batch_ids, dtype=torch.bool)
        full_attn_mask[:, prompt_len:] = resp_mask.bool()

        # Use deterministic loop count across reference, old, and policy passes
        fixed_loops = (
            REASONING_MODES[eff_reasoning]
            if eff_reasoning is not None
            else self.model.config.max_recurrent_loops
        )

        # 2. Compute reference and old log-probabilities
        with torch.no_grad():
            self.ref_model.eval()
            ref_out = self.ref_model(
                batch_ids,
                attention_mask=full_attn_mask,
                num_loops=fixed_loops,
                reasoning_mode=eff_reasoning,
                style_mode=eff_style,
            )
            ref_log_probs, _ = sequence_log_probs(ref_out.logits, batch_ids, prompt_len=prompt_len)
            ref_log_probs = ref_log_probs * resp_mask

            was_training = self.model.training
            self.model.eval()
            old_out = self.model(
                batch_ids,
                attention_mask=full_attn_mask,
                num_loops=fixed_loops,
                reasoning_mode=eff_reasoning,
                style_mode=eff_style,
            )
            old_log_probs, _ = sequence_log_probs(old_out.logits, batch_ids, prompt_len=prompt_len)
            old_log_probs = (old_log_probs * resp_mask).detach()
            self.model.reset_moe_load()

        # 3. Run GRPO policy optimization epochs
        self.model.train()
        last_metrics: Dict[str, float] = {}
        grad_norm = 0.0
        for _ in range(self.grpo_epochs):
            self.optimizer.zero_grad(set_to_none=True)
            out = self.model(
                batch_ids,
                attention_mask=full_attn_mask,
                num_loops=fixed_loops,
                reasoning_mode=eff_reasoning,
                style_mode=eff_style,
            )
            policy_log_probs, _ = sequence_log_probs(out.logits, batch_ids, prompt_len=prompt_len)
            policy_log_probs = policy_log_probs * resp_mask

            rl_loss, last_metrics = grpo_loss(
                policy_log_probs=policy_log_probs,
                old_log_probs=old_log_probs,
                advantages=advantages,
                ref_log_probs=ref_log_probs,
                mask=resp_mask,
                clip_eps=self.clip_eps,
                kl_beta=self.kl_beta,
            )
            total_loss = rl_loss + out.moe_aux_loss
            total_loss.backward()
            grad_norm = self._step_optimizer()
            self.model.update_moe_balance()

        if not was_training:
            self.model.eval()

        if self.ref_ema_decay is not None and self.ref_ema_decay < 1.0:
            self.sync_reference_policy(tau=1.0 - self.ref_ema_decay)

        self.step_count += 1
        best_idx = int(torch.argmax(rewards_tensor).item())
        return {
            "step": self.step_count,
            "loss": last_metrics["loss"],
            "policy_loss": last_metrics["policy_loss"],
            "kl": last_metrics["kl"],
            "clip_fraction": last_metrics["clip_fraction"],
            "grad_norm": grad_norm,
            "rewards": combined_rewards,
            "advantages": advantages.detach().cpu().tolist(),
            "mean_reward": float(rewards_tensor.mean().item()),
            "max_reward": float(rewards_tensor.max().item()),
            "best_index": best_idx,
            "best_response": clean_responses[best_idx],
        }

    def learn_from_conversation(
        self,
        prompt_ids: torch.Tensor,
        user_feedback: float = 1.0,
        chosen_response: Optional[Union[torch.Tensor, Sequence[int]]] = None,
        group_size: int = 4,
        max_new_tokens: int = 16,
        reasoning_mode: Optional[str] = None,
        style_mode: Optional[str] = None,
        temperature: float = 0.85,
    ) -> Dict[str, Any]:
        """Generate a group of candidate responses, score them with user feedback, and update the policy.

        If ``chosen_response`` is provided, it is included as the first candidate in the group and
        receives ``user_feedback`` directly (while the remaining $G - 1$ sampled candidates act as
        contrastive group baselines scored by ``UserPreferenceProfile``).
        """
        if group_size < 2:
            raise ValueError(f"group_size must be >= 2 for group-relative learning, got {group_size}")
        if max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be >= 1, got {max_new_tokens}")

        device = next(self.model.parameters()).device
        prompt_2d = prompt_ids.detach().to(device=device, dtype=torch.long).reshape(1, -1)
        prompt_len = prompt_2d.shape[1]

        eff_style = style_mode if style_mode is not None else self.profile.preferred_style_mode
        candidates: List[List[int]] = []
        explicit_rewards: Optional[List[float]] = None

        if chosen_response is not None:
            chosen_ids = _to_id_list(chosen_response)
            if chosen_ids:
                candidates.append(chosen_ids)
                # Record immediate user feedback on the chosen turn before scoring contrastive rollouts
                self.profile.record_feedback(chosen_ids, reward=float(user_feedback), style_mode=eff_style)

        num_to_sample = group_size - len(candidates)
        for i in range(num_to_sample):
            # Vary temperature slightly across the group to encourage diverse candidate rollouts
            sample_temp = max(0.3, float(temperature) + 0.1 * (i - (num_to_sample - 1) / 2.0))
            gen = self.model.generate(
                prompt_2d,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=sample_temp,
                reasoning_mode=reasoning_mode,
                style_mode=eff_style,
                bad_word_ids=self.bad_word_ids,
                soft_penalty_ids=self.soft_penalty_ids,
                stock_word_penalty=1.0 if self.soft_penalty_ids else 0.0,
            )
            resp = gen[0, prompt_len:].tolist()
            if not resp:
                resp = [int(self.model.config.eos_token_id)]
            candidates.append(resp)

        if chosen_response is not None and len(candidates) == group_size:
            explicit_rewards = [float(user_feedback)] + [0.0] * (group_size - 1)

        stats = self.step_on_group(
            prompt_ids=prompt_2d,
            responses=candidates,
            explicit_rewards=explicit_rewards,
            reasoning_mode=reasoning_mode,
            style_mode=eff_style,
            record_preferences=True,
        )
        stats["responses"] = candidates
        stats["preferred_style_mode"] = self.profile.preferred_style_mode
        stats["preferred_length"] = self.profile.preferred_length
        return stats
