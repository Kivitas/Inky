"""Generate text from an Inky checkpoint with the prose controls from the README.

    python scripts/sample.py --checkpoint runs/inky/checkpoints/step_500 --tokenizer-path tokenizer.json \
        --prompt "The lighthouse keeper said" --max-new-tokens 120 --temperature 0.8 --top-p 0.95 \
        --repetition-penalty 1.1 --no-repeat-ngram-size 4 --bad-words "tapestry,delve" --default-bad-words
    python scripts/sample.py --dry-run          # char-level encoder and a freshly built tiny model; no files

--checkpoint accepts a ``save_pretrained`` directory (holds config.json), a training step directory
(``step_N/``, whose weights are in ``step_N/model/``), a ``best.pt`` file written by train(), or a checkpoints
directory containing ``best.pt`` or ``step_N`` directories (``best.pt`` wins, else the newest step).

--tokenizer-path is a tokenizers JSON file, or a directory holding ``tokenizer.json`` (anything else falls back to
transformers.AutoTokenizer). An untrained or barely trained model writes noise; --dry-run only checks the plumbing.
The scratch ids (see the README) must exist in the tokenizer before --reduce-scratch-blocks can do anything.
"""

import argparse
import re
import sys
from pathlib import Path
from typing import List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from inky import InkyConfig, InkyModel, build_bad_word_ids  # noqa: E402
from inky.style import STOCK_PHRASES  # noqa: E402

TINY_VOCAB = 512
TINY_KWARGS = dict(
    vocab_size=TINY_VOCAB, scratch_open_id=TINY_VOCAB - 3, scratch_sep_id=TINY_VOCAB - 2,
    scratch_close_id=TINY_VOCAB - 1, hidden_size=64, num_attention_heads=4, qk_nope_head_dim=12,
    qk_rope_head_dim=4, v_head_dim=16, q_lora_rank=32, kv_lora_rank=24, intermediate_size=96,
    num_routed_experts=4, num_active_routed=2, ssm_state_dim=16, max_recurrent_loops=4, min_recurrent_loops=2,
    loop_dropout_prob=0.0, max_position_embeddings=256, arlo_dim=16, arlo_pointer_rank=8, arlo_slots=4,
    eos_token_id=1, pad_token_id=1,
)
_CHAR_OFFSET = 4
_CHAR_LIMIT = 400          # characters with ord < 400 round-trip; others encode as '?'


class CharCodec:
    """Reversible char-level encoder for --dry-run. Ids 0-3 and above 403 are special and decode to ''."""

    size = TINY_VOCAB

    def encode(self, text: str) -> List[int]:
        return [_CHAR_OFFSET + (ord(c) if ord(c) < _CHAR_LIMIT else ord("?")) for c in text]

    def decode(self, ids: Sequence[int]) -> str:
        return "".join(chr(i - _CHAR_OFFSET) for i in ids if _CHAR_OFFSET <= i < _CHAR_OFFSET + _CHAR_LIMIT)


class TokenizerCodec:
    """encode/decode over a ``tokenizers`` JSON file, or a transformers tokenizer as a fallback."""

    def __init__(self, path: str):
        p = Path(path)
        json_file = p if p.is_file() else (p / "tokenizer.json" if (p / "tokenizer.json").is_file() else None)
        if json_file is not None:
            from tokenizers import Tokenizer
            self._tok = Tokenizer.from_file(str(json_file))
            self.size = self._tok.get_vocab_size()
            self._encode = lambda text: self._tok.encode(text, add_special_tokens=False).ids
            self._decode = lambda ids: self._tok.decode(list(ids), skip_special_tokens=False)
            return
        try:
            from transformers import AutoTokenizer
            hf = AutoTokenizer.from_pretrained(path)
        except Exception as exc:
            raise SystemExit(f"error: cannot load tokenizer from {path!r}: {exc}") from exc
        self.size = len(hf)
        self._encode = lambda text: hf.encode(text, add_special_tokens=False)
        self._decode = lambda ids: hf.decode(list(ids), skip_special_tokens=False)

    def encode(self, text: str) -> List[int]:
        return list(self._encode(text))

    def decode(self, ids: Sequence[int]) -> str:
        return self._decode([i for i in ids if 0 <= i < self.size])   # ids the tokenizer lacks (scratch) are dropped


def load_model(path: str, device: str = "cpu") -> InkyModel:
    """Load a model from a save_pretrained dir, a training step dir, a best.pt file, or a checkpoints dir."""
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"error: checkpoint not found: {p}")
    if p.is_dir():
        if (p / "config.json").is_file():
            return InkyModel.from_pretrained(p).to(device).eval()
        if (p / "model" / "config.json").is_file():
            return InkyModel.from_pretrained(p / "model").to(device).eval()
        if (p / "best.pt").is_file():
            p = p / "best.pt"
        else:
            steps = sorted((int(m.group(1)), d) for d in p.iterdir()
                           if d.is_dir() and (m := re.fullmatch(r"step_(\d+)", d.name)))
            if not steps:
                raise SystemExit(f"error: {p} holds no config.json, model/, best.pt or step_N directory")
            return load_model(str(steps[-1][1]), device)
    state = torch.load(p, map_location="cpu", weights_only=True)
    if not (isinstance(state, dict) and "model" in state and "config" in state):
        raise SystemExit(f"error: {p} is not a best.pt written by inky.train (needs 'model' and 'config')")
    model = InkyModel(InkyConfig.from_dict(state["config"]))
    model.load_state_dict(state["model"])
    return model.to(device).eval()


def build_bad_words(codec, phrases: Optional[str], use_default: bool) -> Optional[List[List[int]]]:
    words = [w.strip() for w in (phrases or "").split(",") if w.strip()]
    if use_default:
        words = list(STOCK_PHRASES) + words
    return build_bad_word_ids(codec.encode, phrases=words) if words else None


def generate_text(model: InkyModel, codec, prompt: str, *, max_new_tokens: int = 64, temperature: float = 0.8,
                  top_p: float = 0.95, top_k: int = 50, repetition_penalty: float = 1.1,
                  no_repeat_ngram_size: int = 4, repeat_window: Optional[int] = None,
                  bad_word_ids: Optional[List[List[int]]] = None, reduce_scratch_blocks: bool = False,
                  num_loops: Optional[int] = None, greedy: bool = False, seed: Optional[int] = None) -> str:
    """Encode ``prompt``, sample with the given controls, and decode only the new tokens."""
    ids = codec.encode(prompt)
    if not ids:
        raise SystemExit("error: the prompt encodes to no tokens")
    cfg = model.config
    if max(ids) >= cfg.vocab_size:
        raise SystemExit(
            f"error: the tokenizer produced id {max(ids)}, outside the model vocabulary ({cfg.vocab_size})")
    if len(ids) >= cfg.max_position_embeddings:
        raise SystemExit(f"error: prompt is {len(ids)} tokens; the model context is {cfg.max_position_embeddings}")
    if reduce_scratch_blocks and codec.size <= cfg.scratch_open_id:
        print(f"note: the tokenizer has {codec.size} ids, so the scratch ids ({cfg.scratch_open_id}+) cannot be "
              "decoded; scratch reduction stays inert until the model is trained to use them", file=sys.stderr)
    if seed is not None:
        torch.manual_seed(seed)
    device = next(model.parameters()).device
    prompt_ids = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(prompt_ids, max_new_tokens=max_new_tokens, do_sample=not greedy, temperature=temperature,
                         top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty,
                         num_loops=num_loops, no_repeat_ngram_size=no_repeat_ngram_size,
                         repeat_window=repeat_window, bad_word_ids=bad_word_ids,
                         reduce_scratch_blocks=reduce_scratch_blocks)
    return codec.decode(out[0, len(ids):].tolist())


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", help="save_pretrained dir, step_N dir, best.pt, or a checkpoints dir")
    parser.add_argument("--tokenizer-path", help="tokenizers JSON file or a directory holding tokenizer.json")
    parser.add_argument("--dry-run", action="store_true", help="char-level encoder and a fresh tiny model; no files")
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=4, help="0 turns it off; 1 is rejected")
    parser.add_argument("--repeat-window", type=int, default=None, help="limit repetition rules to the last N tokens")
    parser.add_argument("--bad-words", default=None, help="comma-separated words or phrases to block")
    parser.add_argument("--default-bad-words", action="store_true", help="also block inky.style.STOCK_PHRASES")
    parser.add_argument("--reduce-scratch-blocks", action="store_true", help="PENCIL-style scratch reduction")
    parser.add_argument("--num-loops", type=int, default=None, help="recurrent loops (default: model maximum)")
    parser.add_argument("--greedy", action="store_true", help="argmax decoding instead of sampling")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    if not args.dry_run and not (args.checkpoint and args.tokenizer_path):
        parser.error("give --checkpoint and --tokenizer-path, or use --dry-run")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be >= 1")
    return args


def main(argv: Optional[List[str]] = None) -> str:
    args = parse_args(argv)
    if args.dry_run:
        torch.manual_seed(args.seed)
        model, codec = InkyModel(InkyConfig(**TINY_KWARGS)).eval(), CharCodec()
        print("dry run: char-level encoder, freshly built tiny model (output is noise)", file=sys.stderr)
    else:
        model, codec = load_model(args.checkpoint, args.device), TokenizerCodec(args.tokenizer_path)
        if codec.size > model.config.vocab_size:
            raise SystemExit(f"error: tokenizer has {codec.size} ids but the model vocabulary is "
                             f"{model.config.vocab_size}")
    try:
        text = generate_text(
            model, codec, args.prompt, max_new_tokens=args.max_new_tokens, temperature=args.temperature,
            top_p=args.top_p, top_k=args.top_k, repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size, repeat_window=args.repeat_window,
            bad_word_ids=build_bad_words(codec, args.bad_words, args.default_bad_words),
            reduce_scratch_blocks=args.reduce_scratch_blocks, num_loops=args.num_loops, greedy=args.greedy,
            seed=args.seed)
    except ValueError as exc:     # a control rejected by generate(), for example no_repeat_ngram_size=1
        raise SystemExit(f"error: {exc}") from exc
    out = args.prompt + text
    try:
        print(out)
    except UnicodeEncodeError:        # a console codec (e.g. cp1252 on Windows) cannot show every character
        enc = sys.stdout.encoding or "utf-8"
        print(out.encode(enc, errors="replace").decode(enc))
    return text


if __name__ == "__main__":
    main()
