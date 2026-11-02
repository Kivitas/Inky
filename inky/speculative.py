"""Self-speculative greedy decoding for Inky.

The same weights serve as two models. The drafter runs the recurrent core for ``draft_loops``
loops and proposes ``draft_len`` tokens greedily. The verifier runs the full
``max_recurrent_loops`` and checks those tokens in one forward pass. A drafted token is kept only
when it equals the verifier's argmax at that position, so the output is identical to plain greedy
decoding with the full loop count.

Caveat: this is a prototype. After every round both caches are rebuilt by re-prefilling the
accepted sequence, which costs O(L) work per round and so is slow. Proper cache truncation
(rolling the KV entries and SSM prefix sums back to the accepted length) is future work.
"""

from typing import Dict, Optional, Tuple

import torch

from .modeling import InkyCache, InkyModel


def _prefill(model: InkyModel, seq: torch.Tensor, num_loops: int) -> Tuple[InkyCache, torch.Tensor]:
    """Build a fresh cache over ``seq`` ([1, L]) with ``num_loops`` loops. Returns the cache and last logits."""
    cache = InkyCache(model.config)
    out = model(seq, past=cache, num_loops=num_loops, last_only=True)
    return cache, out.logits[:, -1, :]


def self_speculative_generate(
    model: InkyModel,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    draft_loops: int = 2,
    draft_len: int = 4,
    eos_token_id: Optional[int] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Greedy decoding with the same model as drafter (``draft_loops``) and verifier (full loops).

    ``input_ids`` is [1, S]. Returns ``(sequence, stats)`` where ``sequence`` is [1, S + new] and
    ``stats`` has the keys:

    - ``drafted``: number of drafted tokens checked by the verifier
    - ``accepted``: number of drafted tokens the verifier agreed with
    - ``acceptance_rate``: accepted / drafted (0.0 when nothing was drafted)
    - ``verifier_calls``: number of verifier rounds

    Output matches ``model.generate(input_ids, max_new_tokens, do_sample=False,
    repetition_penalty=1.0, no_repeat_ngram_size=0, num_loops=config.max_recurrent_loops)``
    token for token. Generation stops at ``max_new_tokens``, at ``eos_token_id`` (included in the
    output, as in ``generate``), or when the sequence reaches ``config.max_position_embeddings``.

    Slow prototype: both caches are rebuilt from the accepted sequence every round (see module
    docstring). Batch size must be 1.
    """
    if input_ids.shape[0] != 1:
        raise ValueError("self-speculative decoding needs batch size 1")
    if draft_loops < 1 or draft_len < 1:
        raise ValueError("draft_loops and draft_len must be at least 1")

    model.eval()
    cfg = model.config
    max_loops = cfg.max_recurrent_loops
    max_pos = cfg.max_position_embeddings

    seq = input_ids
    generated = 0
    stats = {"drafted": 0, "accepted": 0, "acceptance_rate": 0.0, "verifier_calls": 0}

    with torch.no_grad():
        while generated < max_new_tokens and seq.shape[1] < max_pos:
            L = seq.shape[1]
            # Drafts plus the verifier's bonus token must fit in the budget and the context window.
            k = min(draft_len, max_new_tokens - generated - 1, max_pos - L - 1)
            k = max(k, 0)

            # Drafter: greedy with draft_loops, using its own cache.
            drafts = []
            if k > 0:
                dcache, dlast = _prefill(model, seq, draft_loops)
                tok = dlast.argmax(dim=-1)
                drafts.append(int(tok[0]))
                for _ in range(k - 1):
                    out = model(tok[:, None], past=dcache, num_loops=draft_loops, last_only=True)
                    tok = out.logits[:, -1, :].argmax(dim=-1)
                    drafts.append(int(tok[0]))

            # Verifier: full loops. preds[i] is the verifier's greedy choice at the position of
            # drafts[i]; preds[k] is the bonus token after all drafts.
            vcache, vlast = _prefill(model, seq, max_loops)
            preds = [int(vlast.argmax(dim=-1)[0])]
            if k > 0:
                draft_t = torch.tensor([drafts], dtype=seq.dtype, device=seq.device)
                out = model(draft_t, past=vcache, num_loops=max_loops, last_only=False)
                preds += out.logits[0].argmax(dim=-1).tolist()

            n = 0
            while n < k and drafts[n] == preds[n]:
                n += 1
            new_tokens = drafts[:n] + [preds[n]]

            stats["drafted"] += k
            stats["accepted"] += n
            stats["verifier_calls"] += 1

            # Cut at the token budget and at the first eos (kept, as generate keeps it).
            new_tokens = new_tokens[: max_new_tokens - generated]
            if eos_token_id is not None and eos_token_id in new_tokens:
                new_tokens = new_tokens[: new_tokens.index(eos_token_id) + 1]
                done = True
            else:
                done = False

            seq = torch.cat([seq, torch.tensor([new_tokens], dtype=seq.dtype, device=seq.device)], dim=1)
            generated += len(new_tokens)
            if done:
                break

    if stats["drafted"] > 0:
        stats["acceptance_rate"] = stats["accepted"] / stats["drafted"]
    return seq, stats
