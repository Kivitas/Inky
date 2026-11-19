"""Ultra-lightweight Joint Embedding Predictive Architecture (JEPA) Executive Controller for Inky.

This module implements a latent-space predictive world model and executive controller that
operates over compact abstract states ``z_t in R^{d_z}`` (``d_z <= 256``) rather than token space:

1. ``LatentStateEncoder`` (E_phi): projects token hidden states ``h_t`` into normalized abstract
   latent states ``z_t = RMSNorm(W_e h_t)`` and extracts a 100% causal goal state
   ``z_goal = RMSNorm(W_goal h_bar_0)`` from the causal running mean of the prelude states.
2. ``LatentWorldModelPredictor`` (P_psi): 2-layer MLP + GRU-style gated transition cell that rolls
   out ``num_actions`` candidate action plans over a lookahead horizon ``H`` without running the
   full Transformer block.
3. ``EnergyCostCritic`` (C_omega): scores predicted trajectories against ``z_goal`` using combined
   squared-L2 + cosine energy plus a linear compute penalty, selecting the minimum-energy action
   with straight-through softmin differentiability during training.
4. ``JEPAController``: unifies E_phi, P_psi, C_omega, and executive heads for MoE expert routing
   bias (``moe_bias``), external memory bank fetch gating (``fetch_gate``), and convergence-boosted
   adaptive halting (``p_halt``), alongside self-supervised multi-step prediction + VICReg
   anti-collapse regularization.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import InkyConfig
from .layers import RMSNorm

ACTION_PLAN_NAMES: Tuple[str, ...] = (
    "deductive-logic",
    "creative-prose",
    "memory-fetch",
    "fast-converge",
)


@dataclass
class JEPAControlSignals:
    """Executive control signals produced by ``JEPAController`` at recurrent step ``t``.

    Attributes:
        z_t: ``[B, S, d_z]`` current abstract latent state.
        z_pred: ``[B, S, d_z]`` 1-step predicted next latent state under optimal action ``a_t*``.
        trajectory: ``[H, B, S, d_z]`` ``H``-step simulated latent trajectory ``z_hat_{t+1:t+H}``.
        energy: ``[B, S]`` minimum trajectory energy ``E(z_hat, z_goal, t)``.
        fetch_gate: ``[B, S]`` memory bank fetch gate ``g_mem`` (hard ``{0, 1}`` in eval,
            straight-through sigmoid in training when uncertainty/energy exceeds ``tau_mem``).
        moe_bias: ``[B, S, num_routed_experts]`` additive expert routing bias ``b_moe``.
        p_halt: ``[B, S]`` adaptive halting probability combining ``sigma(w_h^T z_t)`` and
            latent convergence ``||z_hat_{t+1} - z_t||_2 < eps``.
        converged: ``[B, S]`` boolean mask where ``||z_hat_{t+1} - z_t||_2 < eps``.
        latent_delta: ``[B, S]`` L2 state velocity ``||z_hat_{t+1} - z_t||_2``.
    """

    z_t: torch.Tensor
    z_pred: torch.Tensor
    trajectory: torch.Tensor
    energy: torch.Tensor
    fetch_gate: torch.Tensor
    moe_bias: torch.Tensor
    p_halt: torch.Tensor
    converged: torch.Tensor
    latent_delta: torch.Tensor


class LatentStateEncoder(nn.Module):
    """Latent state and causal goal encoder ``E_phi``.

    Maps hidden states ``h_t in R^{d_model}`` into normalized abstract states
    ``z_t = RMSNorm(W_e h_t) in R^{d_z}`` (``d_z <= 256``), and computes a strictly causal
    goal embedding ``z_goal = RMSNorm(W_goal h_bar_0)`` from the prefix running mean of ``h_0``.
    """

    def __init__(self, hidden_size: int = 384, latent_dim: int = 128, eps: float = 1e-6):
        super().__init__()
        if not 1 <= int(latent_dim) <= 256:
            raise ValueError(f"latent_dim must be in [1, 256], got {latent_dim}")
        self.hidden_size = int(hidden_size)
        self.latent_dim = int(latent_dim)

        self.state_proj = nn.Linear(self.hidden_size, self.latent_dim, bias=False)
        self.state_norm = RMSNorm(self.latent_dim, eps=eps)

        self.goal_proj = nn.Linear(self.hidden_size, self.latent_dim, bias=False)
        self.goal_norm = RMSNorm(self.latent_dim, eps=eps)

    def encode_state(self, h_t: torch.Tensor) -> torch.Tensor:
        """Map ``h_t`` ``[B, S, d_model]`` -> ``z_t`` ``[B, S, d_z]``."""
        return self.state_norm(self.state_proj(h_t))

    def encode_goal(self, h_0: torch.Tensor) -> torch.Tensor:
        """Map prelude states ``h_0`` ``[B, S, d_model]`` -> causal ``z_goal`` ``[B, S, d_z]``."""
        if h_0.dim() != 3:
            raise ValueError(f"h_0 must have shape [B, S, d_model], got {tuple(h_0.shape)}")
        seq_len = h_0.shape[1]
        positions = torch.arange(1, seq_len + 1, device=h_0.device, dtype=torch.float32).view(1, seq_len, 1)
        causal_mean = (torch.cumsum(h_0.float(), dim=1) / positions).to(h_0.dtype)
        return self.goal_norm(self.goal_proj(causal_mean))

    def forward(
        self,
        h_t: torch.Tensor,
        h_0: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Encode ``h_t`` to ``z_t``, and optionally ``h_0`` to ``z_goal`` when ``h_0`` is provided."""
        z_t = self.encode_state(h_t)
        if h_0 is None:
            return z_t
        return z_t, self.encode_goal(h_0)


class LatentWorldModelPredictor(nn.Module):
    """Action-conditioned latent transition predictor ``P_psi``.

    Uses a 2-layer MLP + GRU-style gated transition cell over ``(z_t, a_k)`` to predict next
    abstract states ``z_hat_{t+1} = P_psi(z_t, a_k)`` and roll out multi-step trajectories
    for all candidate action plans without running the full Transformer block.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        action_dim: int = 32,
        num_actions: int = 4,
        mlp_hidden_dim: Optional[int] = None,
        eps: float = 1e-6,
    ):
        super().__init__()
        if int(num_actions) < 1:
            raise ValueError(f"num_actions must be >= 1, got {num_actions}")
        self.latent_dim = int(latent_dim)
        self.action_dim = int(action_dim)
        self.num_actions = int(num_actions)
        hidden = int(mlp_hidden_dim) if mlp_hidden_dim is not None else 2 * self.latent_dim

        # Learnable discrete action plan embeddings (e.g., deductive-logic, creative-prose,
        # memory-fetch, fast-converge).
        self.action_embeddings = nn.Parameter(torch.randn(self.num_actions, self.action_dim) * 0.02)

        in_dim = self.latent_dim + self.action_dim
        # GRU-style reset and update gates: [r_t, u_t] = sigmoid(W_g [z_t, a_k]).
        self.gate_proj = nn.Linear(in_dim, 2 * self.latent_dim)

        # 2-layer MLP transition over [r_t * z_t, a_k].
        self.mlp_fc1 = nn.Linear(in_dim, hidden)
        self.mlp_fc2 = nn.Linear(hidden, self.latent_dim)
        self.out_norm = RMSNorm(self.latent_dim, eps=eps)

    def step(self, z_t: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Single transition step ``P_psi(z_t, action) -> z_hat_{t+1}``."""
        if action.dim() < z_t.dim():
            view_shape = (action.shape[0],) + (1,) * (z_t.dim() - 2) + (action.shape[-1],)
            action = action.view(view_shape).expand(*z_t.shape[:-1], action.shape[-1])
        joint = torch.cat([z_t, action], dim=-1)
        reset_gate, update_gate = torch.sigmoid(self.gate_proj(joint)).chunk(2, dim=-1)
        gated_input = torch.cat([reset_gate * z_t, action], dim=-1)
        candidate = self.mlp_fc2(F.silu(self.mlp_fc1(gated_input)))
        z_next = (1.0 - update_gate) * z_t + update_gate * candidate
        return self.out_norm(z_next)

    def rollout(self, z_t: torch.Tensor, horizon: int = 3) -> torch.Tensor:
        """Roll out all ``num_actions`` candidate trajectories for ``horizon`` steps.

        Args:
            z_t: ``[B, S, d_z]`` current latent state.
            horizon: number of lookahead steps ``H >= 1``.

        Returns:
            ``[num_actions, H, B, S, d_z]`` simulated trajectories.
        """
        if int(horizon) < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        b, s, d_z = z_t.shape
        k = self.num_actions
        z_curr = z_t.unsqueeze(0).expand(k, b, s, d_z)
        actions = self.action_embeddings.view(k, 1, 1, self.action_dim).expand(k, b, s, self.action_dim)

        steps = []
        for _ in range(int(horizon)):
            z_curr = self.step(z_curr, actions)
            steps.append(z_curr)
        return torch.stack(steps, dim=1)

    def forward(
        self,
        z_t: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        horizon: int = 3,
    ) -> torch.Tensor:
        """Predict 1-step transition when ``action`` is provided, or ``H``-step rollout otherwise."""
        if action is not None:
            return self.step(z_t, action)
        return self.rollout(z_t, horizon=horizon)


class EnergyCostCritic(nn.Module):
    """Energy-based trajectory critic ``C_omega``.

    Scores predicted trajectories ``z_hat_{t+1:t+H}`` against ``z_goal`` at recurrent step ``t``:
        ``E(z_hat, z_goal, t) = ||z_hat_{t+H} - z_goal||_2^2
                                + alpha_cos * (1 - cos(z_hat_{t+H}, z_goal))
                                + lambda_compute * t``
    and selects the minimum-energy action plan ``a_t* = argmin_a E(z_hat, z_goal, t)`` with
    straight-through softmin weights in training mode so gradients propagate smoothly to all
    action embeddings and world-model weights.
    """

    def __init__(
        self,
        alpha_cos: float = 0.5,
        lambda_compute: float = 0.05,
        softmin_temp: float = 0.5,
        eps: float = 1e-6,
    ):
        super().__init__()
        if float(softmin_temp) <= 0.0:
            raise ValueError(f"softmin_temp must be > 0, got {softmin_temp}")
        self.alpha_cos = float(alpha_cos)
        self.lambda_compute = float(lambda_compute)
        self.softmin_temp = float(softmin_temp)
        self.eps = float(eps)

    def energy(
        self,
        z_terminal: torch.Tensor,
        z_goal: torch.Tensor,
        step: int = 0,
    ) -> torch.Tensor:
        """Compute ``E(z_hat_{t+H}, z_goal, t)`` for terminal states ``[..., B, S, d_z]``."""
        target = z_goal.unsqueeze(0) if z_terminal.dim() == 4 and z_goal.dim() == 3 else z_goal
        l2_sq = (z_terminal - target).pow(2).sum(dim=-1)
        cos_sim = F.cosine_similarity(z_terminal.float(), target.float(), dim=-1, eps=self.eps).to(z_terminal.dtype)
        cos_dist = 1.0 - cos_sim
        return l2_sq + self.alpha_cos * cos_dist + self.lambda_compute * float(step)

    def forward(
        self,
        trajectories: torch.Tensor,
        z_goal: torch.Tensor,
        step: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate candidate trajectories and select the optimal action trajectory.

        Args:
            trajectories: ``[K, H, B, S, d_z]`` candidate trajectories across ``K`` action plans.
            z_goal: ``[B, S, d_z]`` causal goal representation.
            step: recurrent loop step index ``t``.

        Returns:
            Tuple of:
              - ``selected_trajectory``: ``[H, B, S, d_z]`` optimal trajectory under ``a_t*``.
              - ``min_energy``: ``[B, S]`` minimum energy ``E(z_hat, z_goal, t)``.
              - ``action_weights``: ``[K, B, S]`` action selection weights (hard one-hot in eval,
                straight-through softmin in training).
              - ``all_energies``: ``[K, B, S]`` per-action energies.
        """
        if trajectories.dim() != 5:
            raise ValueError(f"trajectories must have shape [K, H, B, S, d_z], got {tuple(trajectories.shape)}")
        z_terminal = trajectories[:, -1]  # [K, B, S, d_z]
        all_energies = self.energy(z_terminal, z_goal, step=step)  # [K, B, S]

        k = trajectories.shape[0]
        best_action = all_energies.argmin(dim=0)  # [B, S]
        one_hot = F.one_hot(best_action, num_classes=k).permute(2, 0, 1).to(trajectories.dtype)
        hard_min_energy = all_energies.gather(0, best_action.unsqueeze(0)).squeeze(0)

        if self.training:
            soft_weights = torch.softmax(-all_energies.float() / self.softmin_temp, dim=0).to(trajectories.dtype)
            action_weights = one_hot - soft_weights.detach() + soft_weights
            soft_min_energy = (
                -self.softmin_temp * torch.logsumexp(-all_energies.float() / self.softmin_temp, dim=0)
            ).to(trajectories.dtype)
            min_energy = hard_min_energy - soft_min_energy.detach() + soft_min_energy
        else:
            action_weights = one_hot
            min_energy = hard_min_energy

        selected_trajectory = torch.einsum("kbs,khbsd->hbsd", action_weights, trajectories)
        return selected_trajectory, min_energy, action_weights, all_energies


class JEPAController(nn.Module):
    """Joint Embedding Predictive Architecture (JEPA) Executive Controller.

    Combines ``LatentStateEncoder`` (``E_phi``), ``LatentWorldModelPredictor`` (``P_psi``),
    ``EnergyCostCritic`` (``C_omega``), and executive routing heads:
      - ``action_to_moe``: maps optimal predicted latent ``z_hat_{t+1}`` to MoE routing bias ``b_moe``.
      - ``halt_head``: computes ``sigma(w_h^T z_t)`` and boosts halting probability toward ``1.0``
        when ``||z_hat_{t+1} - z_t||_2 < convergence_eps``.
      - ``uncertainty_head``: combines latent uncertainty with trajectory energy against
        ``mem_threshold`` (``tau_mem``) to produce ``fetch_gate`` (``g_mem``).
    """

    def __init__(
        self,
        config: Optional[Union[InkyConfig, int]] = None,
        *,
        hidden_size: int = 384,
        latent_dim: int = 128,
        action_dim: int = 32,
        num_actions: int = 4,
        horizon: int = 3,
        num_routed_experts: int = 8,
        convergence_eps: float = 0.05,
        mem_threshold: float = 0.5,
        alpha_cos: float = 0.5,
        lambda_compute: float = 0.05,
        softmin_temp: float = 0.5,
        vicreg_var_coef: float = 0.1,
        vicreg_cov_coef: float = 0.04,
        gate_temp: float = 0.1,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()
        if isinstance(config, InkyConfig):
            hidden_size = config.hidden_size
            num_routed_experts = config.num_routed_experts
            rms_norm_eps = config.rms_norm_eps
            latent_dim = int(getattr(config, "jepa_latent_dim", latent_dim))
            action_dim = int(getattr(config, "jepa_action_dim", action_dim))
            num_actions = int(getattr(config, "jepa_num_actions", num_actions))
            horizon = int(getattr(config, "jepa_horizon", horizon))
            convergence_eps = float(getattr(config, "jepa_convergence_eps", convergence_eps))
            mem_threshold = float(getattr(config, "jepa_mem_threshold", mem_threshold))
            lambda_compute = float(getattr(config, "jepa_compute_penalty", lambda_compute))
            alpha_cos = float(getattr(config, "jepa_alpha_cos", alpha_cos))
            softmin_temp = float(getattr(config, "jepa_softmin_temp", softmin_temp))
            vicreg_var_coef = float(getattr(config, "jepa_vicreg_var_coef", vicreg_var_coef))
            vicreg_cov_coef = float(getattr(config, "jepa_vicreg_cov_coef", vicreg_cov_coef))
            gate_temp = float(getattr(config, "jepa_gate_temp", gate_temp))
        elif isinstance(config, int):
            hidden_size = int(config)

        if not 1 <= int(latent_dim) <= 256:
            raise ValueError(f"latent_dim must be in [1, 256], got {latent_dim}")
        if int(horizon) < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        if float(convergence_eps) <= 0.0:
            raise ValueError(f"convergence_eps must be > 0, got {convergence_eps}")

        self.hidden_size = int(hidden_size)
        self.latent_dim = int(latent_dim)
        self.action_dim = int(action_dim)
        self.num_actions = int(num_actions)
        self.horizon = int(horizon)
        self.num_routed_experts = int(num_routed_experts)
        self.convergence_eps = float(convergence_eps)
        self.mem_threshold = float(mem_threshold)
        if float(gate_temp) <= 0.0:
            raise ValueError(f"gate_temp must be > 0, got {gate_temp}")
        self.gate_temp = float(gate_temp)
        self.vicreg_var_coef = float(vicreg_var_coef)
        self.vicreg_cov_coef = float(vicreg_cov_coef)

        self.encoder = LatentStateEncoder(
            hidden_size=self.hidden_size,
            latent_dim=self.latent_dim,
            eps=rms_norm_eps,
        )
        self.predictor = LatentWorldModelPredictor(
            latent_dim=self.latent_dim,
            action_dim=self.action_dim,
            num_actions=self.num_actions,
            eps=rms_norm_eps,
        )
        self.critic = EnergyCostCritic(
            alpha_cos=alpha_cos,
            lambda_compute=lambda_compute,
            softmin_temp=softmin_temp,
            eps=rms_norm_eps,
        )

        # Executive heads.
        self.action_to_moe = nn.Linear(self.latent_dim, self.num_routed_experts)
        self.halt_head = nn.Linear(self.latent_dim, 1)
        self.uncertainty_head = nn.Linear(self.latent_dim, 1)

    def forward(
        self,
        h_t: torch.Tensor,
        step: int = 0,
        h_goal_source: Optional[torch.Tensor] = None,
    ) -> JEPAControlSignals:
        """Run 1 executive planning cycle over hidden states ``h_t`` ``[B, S, d_model]``."""
        if h_t.dim() != 3:
            raise ValueError(f"h_t must have shape [B, S, d_model], got {tuple(h_t.shape)}")

        z_t = self.encoder.encode_state(h_t)  # [B, S, d_z]
        goal_source = h_t if h_goal_source is None else h_goal_source
        z_goal = self.encoder.encode_goal(goal_source)  # [B, S_goal, d_z]
        if z_goal.shape[1] != z_t.shape[1]:
            if z_t.shape[1] == 1 and z_goal.shape[1] > 1:
                z_goal = z_goal[:, -1:, :]
            elif z_goal.shape[1] == 1 and z_t.shape[1] > 1:
                z_goal = z_goal.expand(-1, z_t.shape[1], -1)
            else:
                z_goal = z_goal[:, : z_t.shape[1], :]

        candidate_trajectories = self.predictor.rollout(z_t, horizon=self.horizon)  # [K, H, B, S, d_z]
        trajectory, energy, _, _ = self.critic(candidate_trajectories, z_goal, step=step)
        z_pred = trajectory[0]  # [B, S, d_z]

        latent_delta = torch.linalg.vector_norm(z_pred - z_t, ord=2, dim=-1)  # [B, S]
        converged = latent_delta < self.convergence_eps  # [B, S]

        # Expert routing bias from the optimal action-conditioned predicted latent state.
        moe_bias = self.action_to_moe(z_pred)  # [B, S, num_routed_experts]

        # Adaptive halting probability: sigma(w_h^T z_t), boosted toward 1.0 upon convergence.
        base_halt = torch.sigmoid(self.halt_head(z_t).squeeze(-1))
        conv_ratio = torch.clamp(1.0 - latent_delta / self.convergence_eps, min=0.0, max=1.0)
        conv_boost = torch.where(converged, torch.clamp(conv_ratio, min=0.5), torch.zeros_like(base_halt))
        p_halt = base_halt + (1.0 - base_halt) * conv_boost

        # Memory bank fetch gate: triggers when combined latent uncertainty + energy exceeds tau_mem.
        uncertainty = torch.sigmoid(self.uncertainty_head(z_t).squeeze(-1))
        # Squash energy to [0, 1) so it is commensurate with tau_mem and the soft gate never saturates.
        energy_n = energy.clamp(min=0.0) / (1.0 + energy.clamp(min=0.0))
        mem_score = 0.5 * (energy_n + uncertainty)
        hard_gate = (mem_score > self.mem_threshold).to(z_t.dtype)
        if self.training:
            soft_gate = torch.sigmoid((mem_score - self.mem_threshold) / self.gate_temp)
            fetch_gate = hard_gate - soft_gate.detach() + soft_gate
        else:
            fetch_gate = hard_gate

        return JEPAControlSignals(
            z_t=z_t,
            z_pred=z_pred,
            trajectory=trajectory,
            energy=energy,
            fetch_gate=fetch_gate,
            moe_bias=moe_bias,
            p_halt=p_halt,
            converged=converged,
            latent_delta=latent_delta,
        )

    @staticmethod
    def _vicreg_terms(z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute VICReg variance and covariance anti-collapse terms on ``z`` ``[B, S, d_z]``."""
        flat = z.float().reshape(-1, z.shape[-1])
        n, d = flat.shape
        if n <= 1:
            zero = z.new_zeros(())
            return zero, zero
        std = torch.sqrt(flat.var(dim=0, unbiased=False) + 1e-4)
        var_loss = F.relu(1.0 - std).mean()
        centered = flat - flat.mean(dim=0, keepdim=True)
        cov = (centered.T @ centered) / float(n - 1)
        off_diag = cov - torch.diag(torch.diag(cov))
        cov_loss = off_diag.pow(2).sum() / float(d)
        return var_loss.to(z.dtype), cov_loss.to(z.dtype)

    def jepa_loss(
        self,
        z_preds: Sequence[torch.Tensor],
        h_targets: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """Self-supervised multi-step latent prediction + VICReg variance/covariance loss.

        Matches each predicted latent state ``z_hat_{t+1}`` against the stop-gradient target
        ``z_{t+1} = E_phi(h_{t+1}.detach())`` while applying VICReg variance and covariance
        regularization to prevent representation collapse.
        """
        if isinstance(z_preds, torch.Tensor) and z_preds.dim() == 3:
            z_pred_list = [z_preds]
        else:
            z_pred_list = list(z_preds)

        if isinstance(h_targets, torch.Tensor) and h_targets.dim() == 3:
            h_target_list = [h_targets]
        else:
            h_target_list = list(h_targets)

        if not z_pred_list or len(z_pred_list) != len(h_target_list):
            raise ValueError(
                f"z_preds and h_targets must be non-empty sequences of equal length, "
                f"got {len(z_pred_list)} and {len(h_target_list)}"
            )

        total_loss = z_pred_list[0].new_zeros(())
        for z_pred, h_target in zip(z_pred_list, h_target_list):
            z_target_live = self.encoder.encode_state(h_target.detach())
            z_target_sg = z_target_live.detach()
            pred_loss = F.mse_loss(z_pred, z_target_sg)

            var_pred, cov_pred = self._vicreg_terms(z_pred)
            var_tgt, cov_tgt = self._vicreg_terms(z_target_live)
            var_loss = 0.5 * (var_pred + var_tgt)
            cov_loss = 0.5 * (cov_pred + cov_tgt)

            step_loss = pred_loss + self.vicreg_var_coef * var_loss + self.vicreg_cov_coef * cov_loss
            total_loss = total_loss + step_loss

        return total_loss / float(len(z_pred_list))

    def memory_footprint_bytes(
        self, bits: int = 4, memory_bank: Optional[nn.Module] = None
    ) -> Dict[str, float]:
        """Memory budget estimates against the 1.3 GB mobile ceiling.

        The ceiling is a RAM budget, so ``within_mobile_ceiling`` checks the *active resident*
        footprint (controller + ~50M active core + any resident non-table bank parameters, i.e.
        the sub-key codebooks and projections of ``memory_bank``). The mmap-backed value table
        is paged from storage and is reported separately (``mmap_bank_bytes``); the stricter
        everything-resident total is ``within_total_ceiling``. When ``memory_bank`` is given its
        real ``value_table.storage_bytes`` replaces the nominal 1.8B-parameter bank estimate.
        """
        if int(bits) < 1:
            raise ValueError(f"bits must be >= 1, got {bits}")
        bytes_per_param = float(bits) / 8.0
        controller_params = float(sum(p.numel() for p in self.parameters()))
        controller_bytes = controller_params * bytes_per_param
        controller_budget_bytes = 15_000_000.0 * bytes_per_param
        active_core_bytes = 50_000_000.0 * bytes_per_param
        total_core_bytes = 300_000_000.0 * bytes_per_param
        mmap_bank_bytes = 1_800_000_000.0 * bytes_per_param
        resident_bank_bytes = 0.0
        if memory_bank is not None:
            vt = getattr(memory_bank, "value_table", None)
            resident_bank_bytes = float(
                sum(p.numel() for p in memory_bank.parameters())
            ) * bytes_per_param
            if vt is not None:
                mmap_bank_bytes = float(vt.storage_bytes)
        active_ram_bytes = controller_bytes + active_core_bytes + resident_bank_bytes
        total_bytes = controller_budget_bytes + total_core_bytes + mmap_bank_bytes + resident_bank_bytes
        mobile_ceiling_bytes = 1_300_000_000.0

        return {
            "bits": float(bits),
            "controller_params": controller_params,
            "controller_bytes": controller_bytes,
            "controller_budget_bytes": controller_budget_bytes,
            "active_core_bytes": active_core_bytes,
            "total_core_bytes": total_core_bytes,
            "mmap_bank_bytes": mmap_bank_bytes,
            "resident_bank_bytes": resident_bank_bytes,
            "active_ram_bytes": active_ram_bytes,
            "total_bytes": total_bytes,
            "mobile_ceiling_bytes": mobile_ceiling_bytes,
            "within_mobile_ceiling": 1.0 if active_ram_bytes <= mobile_ceiling_bytes else 0.0,
            "within_total_ceiling": 1.0 if total_bytes <= mobile_ceiling_bytes else 0.0,
        }
