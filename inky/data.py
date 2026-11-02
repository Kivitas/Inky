"""Text data pipeline for pretraining: clean, filter, dedupe, encode and pack documents.

Everything here works offline on in-memory strings or local JSONL files. Nothing downloads.
The pipeline order used by ``prepare_corpus`` is clean -> quality filter -> dedupe -> encode -> pack.
"""

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable, Iterator, List, Sequence, Set, Union

import torch

# Characters that count as ordinary prose punctuation in quality_filter.
BASIC_PUNCTUATION: Set[str] = set(".,;:!?'\"-()")

# Near-duplicate settings: word 5-gram Jaccard similarity at or above this value means duplicate.
SHINGLE_SIZE = 5
NEAR_DUPLICATE_THRESHOLD = 0.8


def clean_text(text: str) -> str:
    """Normalise a document for training.

    Applies Unicode NFC, converts CRLF to LF, removes control characters except newline and tab,
    collapses runs of spaces and tabs to one space, strips trailing spaces on each line, and
    collapses three or more blank lines to two.
    """
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n")
    text = "".join(ch for ch in text if ch in "\n\t" or unicodedata.category(ch) != "Cc")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" +$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text


def quality_filter(
    text: str,
    min_chars: int = 200,
    max_symbol_ratio: float = 0.10,
    max_repeated_line_ratio: float = 0.30,
) -> bool:
    """Return True to keep a document, False to reject it.

    Rejects text shorter than ``min_chars``. Rejects text where the share of characters that are
    not letters, digits, whitespace or basic punctuation exceeds ``max_symbol_ratio``. Rejects text
    where the share of non-empty lines that appear more than once exceeds ``max_repeated_line_ratio``
    (every copy of a repeated line counts, not only the extra copies).
    """
    if len(text) < min_chars:
        return False

    symbols = sum(1 for ch in text if not ch.isalnum() and not ch.isspace() and ch not in BASIC_PUNCTUATION)
    if symbols / max(len(text), 1) > max_symbol_ratio:
        return False

    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if lines:
        counts = Counter(lines)
        repeated = sum(count for count in counts.values() if count > 1)
        if repeated / len(lines) > max_repeated_line_ratio:
            return False
    return True


def _shingles(text: str) -> Set[tuple]:
    """Word n-grams of lowercased text. Texts shorter than SHINGLE_SIZE words become one shingle."""
    words = re.findall(r"\w+", text.lower())
    if not words:
        return set()
    if len(words) < SHINGLE_SIZE:
        return {tuple(words)}
    return {tuple(words[i:i + SHINGLE_SIZE]) for i in range(len(words) - SHINGLE_SIZE + 1)}


def _jaccard(a: Set[tuple], b: Set[tuple]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def dedupe_documents(docs: Sequence[str]) -> List[str]:
    """Remove duplicates, keeping the first occurrence of each. Returns the original strings.

    Exact duplicates are found after ``clean_text`` and lowercasing. Near-duplicates are documents
    whose word 5-gram Jaccard similarity is at least 0.8 with a document already kept. This is
    brute force (each candidate is compared with every kept document), which is fine for small
    inputs but is quadratic in the number of documents.
    """
    kept: List[str] = []
    kept_shingles: List[Set[tuple]] = []
    seen: Set[str] = set()
    for doc in docs:
        key = clean_text(doc).lower()
        if key in seen:
            continue
        seen.add(key)
        shingles = _shingles(key)
        if any(_jaccard(shingles, other) >= NEAR_DUPLICATE_THRESHOLD for other in kept_shingles):
            continue
        kept.append(doc)
        kept_shingles.append(shingles)
    return kept


def pack_sequences(token_id_lists: Iterable[Sequence[int]], seq_len: int, eos_id: int) -> torch.Tensor:
    """Concatenate documents with ``eos_id`` after each, then cut non-overlapping windows.

    Returns a LongTensor of shape [N, seq_len]. The final partial window is dropped, so N may be 0.
    Raises ValueError if ``seq_len`` is less than 2.
    """
    if seq_len < 2:
        raise ValueError(f"seq_len must be at least 2, got {seq_len}")
    flat: List[int] = []
    for ids in token_id_lists:
        flat.extend(int(t) for t in ids)
        flat.append(int(eos_id))
    rows = len(flat) // seq_len
    if rows == 0:
        return torch.empty((0, seq_len), dtype=torch.long)
    return torch.tensor(flat[: rows * seq_len], dtype=torch.long).view(rows, seq_len)


def iter_jsonl(path: Union[str, Path], field: str = "text") -> Iterator:
    """Yield ``field`` from each line of a UTF-8 JSONL file.

    Blank lines and lines whose JSON object lacks the field are skipped. Malformed JSON raises
    ``json.JSONDecodeError`` so bad input is not silently dropped.
    """
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            obj = json.loads(line)
            if isinstance(obj, dict) and field in obj:
                yield obj[field]


def prepare_corpus(
    docs: Iterable[str],
    encode: Callable[[str], List[int]],
    seq_len: int,
    eos_id: int,
    min_chars: int = 200,
) -> torch.Tensor:
    """Run the full pipeline: clean, quality filter, dedupe, encode each document, then pack.

    ``encode`` maps a string to a list of token ids. Returns a LongTensor of shape [N, seq_len].
    """
    cleaned = [clean_text(doc) for doc in docs]
    kept = [doc for doc in cleaned if quality_filter(doc, min_chars=min_chars)]
    unique = dedupe_documents(kept)
    token_lists = [list(encode(doc)) for doc in unique]
    return pack_sequences(token_lists, seq_len, eos_id)
