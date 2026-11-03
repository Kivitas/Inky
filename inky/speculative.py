"""Self-speculative greedy decoding for Inky.

The same weights serve as two models. The drafter runs the recurrent core for ``draft_loops``
loops and proposes ``draft_len`` tokens greedily. The verifier runs the full
``max_recurrent_loops`` and checks those tokens in one forward pass. A drafted token is kept only
when it equals the verifier's argmax at that position, so the output is identical to plain greedy
decoding with the full loop count.

Cache handling. Each model keeps one committed cache that covers the accepted sequence. Each round
only the tokens accepted since the last round are fed into the committed caches ("catch-up"). The
drafts and the verify block are fed into a fork of the committed cache that is thrown away after
the round. Rollback is therefore free: the committed cache never sees rejected drafts.

Why a fork is an exact snapshot: the forward pass never writes into a cache tensor in place. It
builds new tensors and replaces the list slots of the ``InkyCache`` object. Copying the container
lists (see ``_fork``) is enough to isolate a trial from the committed state. Model compute is
spent only on tokens that are new to a model, so the total work grows linearly with the number of
generated tokens. The fork itself copies a few list references, not tensor data.

Cost per round, with k drafts and n accepted: the verifier feeds (n + 1) catch-up tokens plus k
verify tokens, and the drafter feeds (n + 1) catch-up tokens plus (k - 1) draft steps. Rejected
drafts are paid for once and then discarded.
"""

import copy
from typing import Dict, Optional, Tuple

import torch

from .modeling import InkyCache, InkyModel


def _fork(cache: InkyCache) -> InkyCache:
    """Return a cache that shares tensors with ``cache`` but has its own container lists.

    Feeding tokens into the fork leaves ``cache`` unchanged. The forward pass only replaces list
    slots and never mutates tensors in place, so no tensor data is copied.
    """
    new = copy.copy(cache)
    new.pre = list(cache.pre)
    new.post = list(cache.post)
    new.core = list(cache.core)
    new.prefix = list(cache.prefix)
    return new


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
    - ``target_forward_passes``: forward calls on the full-loop model (catch-up and verify blocks)
    - ``draft_forward_passes``: forward calls on the drafter (catch-up and single-token steps)
    - ``target_tokens_fed``: input tokens summed over the full-loop forward calls
    - ``draft_tokens_fed``: input tokens summed over the drafter forward calls
    - ``tokens_fed_total``: ``target_tokens_fed + draft_tokens_fed``, the work measure to compare
      against plain greedy, which feeds one token per generated token after the prompt

    Output matches ``model.generate(input_ids, max_new_tokens, do_sample=False,
    repetition_penalty=1.0, no_repeat_ngram_size=0, num_loops=config.max_recurrent_loops)``
    token for token. Generation stops at ``max_new_tokens``, at ``eos_token_id`` (included in the
    output, as in ``generate``), or when the sequence reaches ``config.max_position_embeddings``.

    Batch size must be 1.
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
    stats = {
        "drafted": 0,
        "accepted": 0,
        "acceptance_rate": 0.0,
        "verifier_calls": 0,
        "target_forward_passes": 0,
        "draft_forward_passes": 0,
        "target_tokens_fed": 0,
        "draft_tokens_fed": 0,
        "tokens_fed_total": 0,
    }

    def count(target: bool, n_tokens: int) -> None:
        stats["target_forward_passes" if target else "draft_forward_passes"] += 1
        stats["target_tokens_fed" if target else "draft_tokens_fed"] += n_tokens
        stats["tokens_fed_total"] += n_tokens

    # Committed caches. ``*_fed`` is how many tokens of ``seq`` each cache has consumed.
    v_cache, v_fed = InkyCache(cfg), 0
    d_cache, d_fed = InkyCache(cfg), 0

    with torch.no_grad():
        while generated < max_new_tokens and seq.shape[1] < max_pos:
            L = seq.shape[1]
            # Drafts plus the verifier's bonus token must fit in the budget and the context window.
            k = min(draft_len, max_new_tokens - generated - 1, max_pos - L - 1)
            k = max(k, 0)

            drafts = []
            if k > 0:
                # Drafter catch-up: feed the accepted tokens it has not seen, then draft from a fork.
                out = model(seq[:, d_fed:], past=d_cache, num_loops=draft_loops, last_only=True)
                count(False, L - d_fed)
                d_fed = L
                tok = out.logits[:, -1, :].argmax(dim=-1)
                drafts.append(int(tok[0]))
                trial = _fork(d_cache)
                for _ in range(k - 1):
                    out = model(tok[:, None], past=trial, num_loops=draft_loops, last_only=True)
                    count(False, 1)
                    tok = out.logits[:, -1, :].argmax(dim=-1)
                    drafts.append(int(tok[0]))

            # Verifier catch-up: its last logits give preds[0], the greedy choice at position L.
            out = model(seq[:, v_fed:], past=v_cache, num_loops=max_loops, last_only=True)
            count(True, L - v_fed)
            v_fed = L
            preds = [int(out.logits[0, -1].argmax(dim=-1))]
            if k > 0:
                # Verify block on a fork. preds[i + 1] is the verifier's choice after drafts[i].
                draft_t = torch.tensor([drafts], dtype=seq.dtype, device=seq.device)
                trial = _fork(v_cache)
                out = model(draft_t, past=trial, num_loops=max_loops, last_only=False)
                count(True, k)
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
