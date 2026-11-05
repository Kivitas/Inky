"""Architecture diagnostics: stream scale across loops, loop-to-loop change, highway share, expert load.

``loop_diagnostics(model, input_ids)`` runs one eval-mode forward pass with hooks and reports plain numbers. It
needs no labels and no training, so it works on an untrained model (to check scales at initialisation) and on a
checkpoint (to watch them move). It changes nothing: the train/eval mode is restored and the MoE load counters that
training keeps are put back as they were.

What it reports, per recurrent loop n (``routing_mode='ponder'``; MoR has no per-loop pass to hook):

* ``in_rms`` / ``out_rms``: RMS of the residual stream entering and leaving the loop.
* ``rel_change``: ``||out - in|| / ||in||``, how far the loop moved the stream. A loop that stops changing the
  stream (values near 0) is a fixed point; values that stay large mean the recurrence is still rewriting it.
* ``highway_share``: RMS of what the highway added, divided by the RMS of its input (None without a highway).
* ``weight``: mean halting probability ``p_n`` of the loop. ``scale_share``: the loop's share of
  ``sum_m p_m * rms_m``, which is its share of the plain (unnormalised) mixture. With ``loop_mix_normalize`` the
  mixture uses ``weight`` itself, so a ``scale_share`` far from ``weight`` shows how skewed the plain mixture
  would have been.

and, for every MoE layer, the fraction of routed assignments each expert received in this pass (all loops counted).
"""

from typing import Any, Dict, List, Optional

import torch

from .layers import geometric_halting_weights


def _rms(x: torch.Tensor) -> float:
    return float(x.float().pow(2).mean().sqrt())


@torch.no_grad()
def loop_diagnostics(model, input_ids: torch.Tensor, num_loops: Optional[int] = None) -> Dict[str, Any]:
    """Scale, loop-change and expert-load numbers for one forward pass of ``input_ids`` [B, S].

    Returns a dict with ``embedding_rms``, ``pre_rms`` (after the pre layers), ``loops`` (a list, see the module
    docstring; empty under ``routing_mode='mor'``), ``mixed_rms`` (the loop mixture entering the post layers),
    ``highway_gain`` (None without a highway), ``mean_loops`` and ``expert_load`` (a list with one entry per MoE
    layer: ``name``, ``load``, ``min``, ``max``, ``cv``). All values are Python floats or lists of floats.
    """
    cfg = model.config
    was_training = model.training
    model.eval()

    rec: Dict[str, Any] = {"embedding": None, "pre": None, "mixed": None, "loop_in": [], "loop_out": [],
                           "hw_in": [], "hw_out": []}
    handles = []

    def keep(key):
        def hook(module, args, output):
            rec[key] = (output[0] if isinstance(output, tuple) else output).detach()
        return hook

    def keep_loop(module, args, output):
        rec["loop_in"].append(args[0].detach())
        rec["loop_out"].append(output[0].detach())

    def keep_highway(module, args, output):
        rec["hw_in"].append(args[0].detach())
        rec["hw_out"].append(output[0].detach())

    def keep_mixed(module, args):
        rec["mixed"] = args[0].detach()

    handles.append(model.embed_tokens.register_forward_hook(keep("embedding")))
    if len(model.pre_layers) > 0:
        handles.append(model.pre_layers[-1].register_forward_hook(keep("pre")))
    if cfg.routing_mode != "mor":
        handles.append(model.core.register_forward_hook(keep_loop))
        if model.core.ssm is not None:
            handles.append(model.core.ssm.register_forward_hook(keep_highway))
    if len(model.post_layers) > 0:
        handles.append(model.post_layers[0].register_forward_pre_hook(keep_mixed))

    # The MoE layers count routed assignments between optimizer steps. Save the training counters, count this
    # pass alone, and put the training counters back.
    moes = [(name, m) for name, m in model.named_modules() if hasattr(m, "load_counts")]
    saved = [(m.load_counts.clone(), m.expert_load.clone()) for _, m in moes]
    for _, m in moes:
        m.load_counts.zero_()

    try:
        out = model(input_ids, return_hidden=True, num_loops=num_loops)
        loads = [(name, (m.load_counts / m.load_counts.sum().clamp(min=1.0)).tolist()) for name, m in moes]
    finally:
        for h in handles:
            h.remove()
        for (_, m), (counts, load) in zip(moes, saved):
            m.load_counts.copy_(counts)
            m.expert_load.copy_(load)
        model.train(was_training)

    emb = rec["embedding"]
    pre = rec["pre"] if rec["pre"] is not None else emb
    result: Dict[str, Any] = {
        "routing_mode": cfg.routing_mode,
        "embedding_rms": _rms(emb),
        "pre_rms": _rms(pre),
        "mixed_rms": _rms(rec["mixed"]) if rec["mixed"] is not None else None,
        "highway_gain": float(model.core.ssm.out_gain) if model.core.ssm is not None else None,
        "mean_loops": float(out.mean_loops) if out.mean_loops is not None else None,
        "loops": [],
    }

    if rec["loop_out"]:
        if model.halting is None:       # routing_mode='last': the last loop is the output, all the weight is on it
            weights = torch.zeros(len(rec["loop_out"]), *rec["loop_out"][0].shape[:2], device=rec["loop_out"][0].device)
            weights[-1] = 1.0
        else:
            weights = geometric_halting_weights([model.halting(h) for h in rec["loop_out"]])   # [L, B, S]
        rms_per_token = torch.stack([h.float().pow(2).mean(-1).sqrt() for h in rec["loop_out"]])   # [L, B, S]
        mass = (weights * rms_per_token).mean(dim=(1, 2))
        mass = mass / mass.sum().clamp(min=1e-12)
        loops: List[Dict[str, Any]] = []
        for n, (h_in, h_out) in enumerate(zip(rec["loop_in"], rec["loop_out"])):
            rel = float((h_out - h_in).float().norm() / h_in.float().norm().clamp(min=1e-12))
            share = None
            if n < len(rec["hw_in"]):
                share = _rms(rec["hw_out"][n] - rec["hw_in"][n]) / max(_rms(rec["hw_in"][n]), 1e-12)
            loops.append({
                "loop": n + 1,
                "in_rms": _rms(h_in),
                "out_rms": _rms(h_out),
                "rel_change": rel,
                "highway_share": share,
                "weight": float(weights[n].mean()),
                "scale_share": float(mass[n]),
            })
        result["loops"] = loops

    expert_load = []
    for name, load in loads:
        t = torch.tensor(load)
        expert_load.append({
            "name": name,
            "load": load,
            "min": float(t.min()),
            "max": float(t.max()),
            "cv": float(t.std(unbiased=False) / t.mean().clamp(min=1e-12)),
        })
    result["expert_load"] = expert_load
    return result


def format_diagnostics_line(step: int, d: Dict[str, Any]) -> str:
    """One log line for the training loop: per-loop stream RMS and change, highway gain, expert-load spread."""
    parts = [f"step {step} diag:"]
    if d["loops"]:
        parts.append("out_rms [" + " ".join(f"{r['out_rms']:.3f}" for r in d["loops"]) + "]")
        parts.append("rel_change [" + " ".join(f"{r['rel_change']:.2f}" for r in d["loops"]) + "]")
    if d.get("highway_gain") is not None:
        parts.append(f"highway_gain {d['highway_gain']:.3f}")
    if d["expert_load"]:
        lo = min(e["min"] for e in d["expert_load"])
        hi = max(e["max"] for e in d["expert_load"])
        parts.append(f"expert_load min {lo:.3f} max {hi:.3f}")
    return " ".join(parts)


def format_diagnostics(d: Dict[str, Any]) -> str:
    """The ``loop_diagnostics`` result as a short text report."""
    lines = [
        f"stream RMS: embedding {d['embedding_rms']:.4f}, after pre layers {d['pre_rms']:.4f}"
        + (f", loop mixture {d['mixed_rms']:.4f}" if d.get("mixed_rms") is not None else ""),
    ]
    gain = d.get("highway_gain")
    lines.append("highway gain: " + ("none (no highway)" if gain is None else f"{gain:.4f}")
                 + (f"   mean loops {d['mean_loops']:.2f}" if d.get("mean_loops") is not None else ""))
    if d["loops"]:
        lines.append("loop   in_rms  out_rms  rel_change  highway_share  weight  scale_share")
        for r in d["loops"]:
            hs = "      -" if r["highway_share"] is None else f"{r['highway_share']:7.3f}"
            lines.append(
                f"{r['loop']:>4d}  {r['in_rms']:7.4f}  {r['out_rms']:7.4f}  {r['rel_change']:10.3f}  "
                f"      {hs}  {r['weight']:6.3f}  {r['scale_share']:11.3f}"
            )
    else:
        lines.append("(no per-loop numbers: routing_mode='mor' has no single loop pass to hook)")
    uniform = None
    for e in d["expert_load"]:
        uniform = 1.0 / len(e["load"])
        lines.append(f"experts {e['name']}: min {e['min']:.3f} max {e['max']:.3f} cv {e['cv']:.2f} "
                     f"(uniform {uniform:.3f})")
    return "\n".join(lines)
