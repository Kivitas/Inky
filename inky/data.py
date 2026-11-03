"""Text data pipeline for pretraining: clean, filter, dedupe, encode and pack documents.

Everything here works offline on in-memory strings or local JSONL files. Nothing downloads.
The pipeline order used by ``prepare_corpus`` is clean -> quality filter -> dedupe -> encode -> pack.
"""

import functools
import hashlib
import json
import random
import re
import unicodedata
import warnings
import zlib
from array import array
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, FrozenSet, Iterable, Iterator, List, Optional, Sequence, Set, Tuple, Union

import torch

# Default minimum length for quality_filter and prepare_corpus, in length_units. Low on purpose: haiku,
# micro-fiction and short dialogue are legitimate creative forms. Fragments below it are still dropped.
DEFAULT_MIN_CHARS = 20

# Ids that encoded text must never use. EOS_Tokenizer has ids 0-49999, and 49974-49999 are its special
# tokens (endoftext, chat, tool, vision, fill-in-the-middle, thought). Inky's scratch ids are 50000-50002.
# Text that encodes to any of these would teach the model structural markup as prose, so prepare_corpus
# drops such documents by default.
FIRST_SPECIAL_ID = 49974
DEFAULT_RESERVED_IDS: FrozenSet[int] = frozenset(range(FIRST_SPECIAL_ID, 50003))

# Characters that count as ordinary prose punctuation in quality_filter.
BASIC_PUNCTUATION: Set[str] = set(".,;:!?'\"-()")

# Non-ASCII punctuation that is ordinary prose and is exempt from the symbol count in quality_filter:
# typographic quotes and dashes, CJK sentence punctuation, and Devanagari danda.
EXEMPT_PUNCTUATION: Set[str] = BASIC_PUNCTUATION | set(
    "“”‘’—–…"
    "。，、；：！？「」『』《》〈〉【】〔〕（）［］〖〗・～"
    "।॥"
)

# Near-duplicate settings: word 5-gram Jaccard similarity at or above this value means duplicate.
SHINGLE_SIZE = 5
NEAR_DUPLICATE_THRESHOLD = 0.8

# Candidate search for dedupe (MinHash + LSH banding). NUM_PERM hash functions are split into
# LSH_BANDS bands of NUM_PERM // LSH_BANDS rows. Two documents with shingle Jaccard s share a band with
# probability 1 - (1 - s**r) ** b (r rows, b bands). With r=2 and b=16 a pair at s=0.8 is missed with
# probability about 1e-7. Every candidate is then checked with exact Jaccard, so no false duplicates.
NUM_PERM = 32
LSH_BANDS = 16

# Script-aware length for quality_filter(min_chars). A Han or kana character carries several Latin
# letters' worth of text, and Thai (no word spaces, dense syllables) is weighted above Latin too.
CJK_CHAR_WEIGHT = 3
THAI_CHAR_WEIGHT = 2

# Repeated-line rule. Lines of at most SHORT_LINE_CHARS length (refrains, short verse lines) may repeat
# freely, up to MAX_SHORT_REPEATED_RATIO of all lines. Longer repeated lines are limited by
# quality_filter's max_repeated_line_ratio.
SHORT_LINE_CHARS = 40
MAX_SHORT_REPEATED_RATIO = 0.60

_HAN_KANA_CLASS = r"぀-ヿㇰ-ㇿ㐀-䶿一-鿿豈-﫿\U00020000-\U0002fa1f"
_HAN_KANA_RE = re.compile("[" + _HAN_KANA_CLASS + "]")
_THAI_RE = re.compile(r"[฀-๿]")
# Scripts written without word spaces. Each such character becomes its own token, so shingles are
# character n-grams there instead of one giant "word".
_NOSPACE_SRC = "[" + _HAN_KANA_CLASS + r"฀-๿]"

# Compact int store for packing. array('i') is 4 bytes per item on mainstream platforms.
_INT_CODE = next(code for code in ("i", "l") if array(code).itemsize == 4)
# Unsigned 64-bit store for shingle hashes in dedupe_stream.
_INT_CODE_U64 = "Q"


def clean_text(text: str) -> str:
    """Normalise a document for training while keeping its line layout.

    Applies Unicode NFC, converts CRLF and lone CR to LF, removes control characters except newline
    and tab, and strips trailing spaces on each line. Leading indentation is kept exactly, so verse
    and screenplay layout survive. Runs of spaces and tabs inside a line collapse to one space. Runs of
    three or more blank lines collapse to two, so single and double blank lines are preserved.
    """
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "".join(ch for ch in text if ch in "\n\t" or unicodedata.category(ch) != "Cc")
    lines = []
    for line in text.split("\n"):
        body = line.lstrip(" \t")
        indent = line[: len(line) - len(body)]
        body = re.sub(r"[ \t]+", " ", body).rstrip(" \t")
        lines.append(indent + body if body else "")
    text = "\n".join(lines)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text


def length_units(text: str) -> int:
    """Length of ``text`` in character units, used by the min_chars rule.

    Each character counts 1, except Han and kana characters (CJK_CHAR_WEIGHT) and Thai characters
    (THAI_CHAR_WEIGHT), which carry more text per character. So a short Chinese paragraph is not
    rejected for having few characters, while Latin text is measured exactly as before.
    """
    cjk = len(_HAN_KANA_RE.findall(text))
    thai = len(_THAI_RE.findall(text))
    return len(text) + (CJK_CHAR_WEIGHT - 1) * cjk + (THAI_CHAR_WEIGHT - 1) * thai


def _is_symbol(ch: str) -> bool:
    """True for Unicode punctuation (P*) and symbol (S*) characters that are not exempt prose punctuation.

    Letters in any script (L*), combining marks (M*), numbers (N*) and whitespace are never symbols.
    """
    if ch.isspace() or ch in EXEMPT_PUNCTUATION:
        return False
    return unicodedata.category(ch)[0] in "PS"


def quality_filter(
    text: str,
    min_chars: int = DEFAULT_MIN_CHARS,
    max_symbol_ratio: float = 0.10,
    max_repeated_line_ratio: float = 0.30,
    short_line_chars: int = SHORT_LINE_CHARS,
    max_short_repeated_ratio: float = MAX_SHORT_REPEATED_RATIO,
) -> bool:
    """Return True to keep a document, False to reject it.

    Rejects text with no non-whitespace characters. Rejects text whose length (see ``length_units``) is
    below ``min_chars`` (default DEFAULT_MIN_CHARS, 20, so haiku and short dialogue are kept). Han, kana and Thai
    characters count for more than Latin ones, so short CJK paragraphs are not rejected for length.
    Rejects text where the share of characters that are punctuation or symbols (Unicode categories
    P* and S*, such as emoji and currency signs) exceeds ``max_symbol_ratio``. Letters, combining
    marks, digits and whitespace in any script never count as symbols, and common ASCII, typographic,
    CJK and Devanagari punctuation is exempt.

    Repeated lines: a refrain or verse line of at most ``short_line_chars`` may repeat. Rejects when
    the share of non-empty lines that are longer and repeated exceeds ``max_repeated_line_ratio``
    (every copy counts), or when repeated lines of any length make up more than
    ``max_short_repeated_ratio`` of all non-empty lines (spam such as one line pasted many times).
    """
    if not text.strip():
        return False  # whitespace only: no text at all, whatever min_chars says
    if length_units(text) < min_chars:
        return False

    symbols = sum(1 for ch in text if _is_symbol(ch))
    if symbols / max(len(text), 1) > max_symbol_ratio:
        return False

    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if lines:
        counts = Counter(lines)
        total = len(lines)
        repeated_long = sum(
            count for line, count in counts.items()
            if count > 1 and length_units(line) > short_line_chars
        )
        repeated_any = sum(count for count in counts.values() if count > 1)
        if repeated_long / total > max_repeated_line_ratio:
            return False
        if repeated_any / total > max_short_repeated_ratio:
            return False
    return True


@functools.lru_cache(maxsize=1)
def _token_pattern() -> "re.Pattern[str]":
    """Token pattern: one token per no-space-script character, otherwise a run of word characters.

    Python's ``\\w`` does not match combining marks (category M*), so Devanagari and similar scripts
    would split mid-word at vowel signs. The mark ranges are added to the word class, built once from
    the Unicode database.
    """
    mark_ranges: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for cp in range(0x110000):
        is_mark = unicodedata.category(chr(cp)).startswith("M")
        if is_mark and start is None:
            start = cp
        elif not is_mark and start is not None:
            mark_ranges.append((start, cp - 1))
            start = None
    if start is not None:
        mark_ranges.append((start, 0x10FFFF))
    marks = "".join(f"\\U{a:08x}-\\U{b:08x}" for a, b in mark_ranges)
    word = f"[\\w{marks}]"
    return re.compile(f"{_NOSPACE_SRC}|(?:(?!{_NOSPACE_SRC}){word})+")


def word_tokens(text: str) -> List[str]:
    """Split text into tokens: words (keeping combining marks) and single characters for no-space scripts."""
    return _token_pattern().findall(text)


def _shingles(text: str) -> Set[tuple]:
    """Token n-grams of lowercased text. Texts shorter than SHINGLE_SIZE tokens become one shingle."""
    tokens = word_tokens(text.lower())
    if not tokens:
        return set()
    if len(tokens) < SHINGLE_SIZE:
        return {tuple(tokens)}
    return {tuple(tokens[i:i + SHINGLE_SIZE]) for i in range(len(tokens) - SHINGLE_SIZE + 1)}


def _jaccard(a: Set[tuple], b: Set[tuple]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


_HASH_PRIME = 2_147_483_647
_MINHASH_RNG = random.Random(20260601)
_MINHASH_PARAMS = [
    (_MINHASH_RNG.randrange(1, _HASH_PRIME), _MINHASH_RNG.randrange(0, _HASH_PRIME))
    for _ in range(NUM_PERM)
]
_ROWS_PER_BAND = NUM_PERM // LSH_BANDS


def _minhash(shingles: Set[tuple]) -> Tuple[int, ...]:
    """MinHash signature of a shingle set. Uses crc32 so signatures are the same in every process."""
    hashes = [
        zlib.crc32("\x1f".join(s).encode("utf-8", "surrogatepass")) % _HASH_PRIME
        for s in shingles
    ]
    return tuple(min((a * h + b) % _HASH_PRIME for h in hashes) for a, b in _MINHASH_PARAMS)


def _shingle_hash(shingle: tuple) -> int:
    """64-bit hash of one shingle. Collisions are negligible at corpus scale and cost only a wrong
    candidate pair, never a dropped document, because candidates are still compared exactly."""
    data = "\x1f".join(shingle).encode("utf-8", "surrogatepass")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "little")


def _hashed_jaccard(a: Sequence[int], b: Sequence[int]) -> float:
    """Exact Jaccard of two sorted, duplicate-free arrays of shingle hashes (merge walk)."""
    i = j = inter = 0
    na, nb = len(a), len(b)
    while i < na and j < nb:
        x, y = a[i], b[j]
        if x == y:
            inter += 1
            i += 1
            j += 1
        elif x < y:
            i += 1
        else:
            j += 1
    union = na + nb - inter
    return 1.0 if union == 0 else inter / union


def dedupe_documents(
    docs: Sequence[str],
    max_docs: Optional[int] = None,
    stats: Optional[dict] = None,
) -> List[str]:
    """Remove duplicates, keeping the first occurrence of each. Returns the original strings.

    Exact duplicates are found after ``clean_text``, stripping and lowercasing. Near-duplicates are
    documents whose word 5-gram Jaccard similarity is at least 0.8 with a document already kept.
    Characters of no-space scripts (CJK, Thai) are tokens of their own, so their shingles are
    character n-grams.

    Cost: candidates come from MinHash LSH buckets (see NUM_PERM and LSH_BANDS), and only candidates
    are compared with exact Jaccard. Time is roughly linear in the number of documents for typical
    corpora. The candidate search is probabilistic: a near-duplicate pair at similarity 0.8 is missed
    with probability about 1e-7, and no non-duplicate is ever dropped. Memory still holds a shingle
    set per kept document, so for very large corpora shard the input and dedupe each shard.

    If ``stats`` is a dict, it is filled with ``"comparisons"``, the number of exact Jaccard checks.

    If ``max_docs`` is given and ``docs`` has more documents than that, raises ValueError before any
    work is done. The check counts input documents, not kept ones.
    """
    if max_docs is not None and len(docs) > max_docs:
        raise ValueError(
            f"dedupe_documents got {len(docs)} documents, above max_docs={max_docs}; "
            "shard the corpus first and dedupe each shard"
        )
    kept: List[str] = []
    kept_shingles: List[Set[tuple]] = []
    seen: Set[str] = set()
    buckets: List[dict] = [{} for _ in range(LSH_BANDS)]
    kept_empty = False
    comparisons = 0
    for doc in docs:
        key = clean_text(doc).strip().lower()
        if key in seen:
            continue
        seen.add(key)
        shingles = _shingles(key)
        if not shingles:
            # Documents with no word tokens are duplicates of each other, as with empty Jaccard sets.
            if kept_empty:
                continue
            kept_empty = True
            kept.append(doc)
            kept_shingles.append(shingles)
            continue
        signature = _minhash(shingles)
        bands = [
            signature[b * _ROWS_PER_BAND:(b + 1) * _ROWS_PER_BAND] for b in range(LSH_BANDS)
        ]
        candidates: Set[int] = set()
        for b, band_key in enumerate(bands):
            candidates.update(buckets[b].get(band_key, ()))
        duplicate = False
        for idx in candidates:
            comparisons += 1
            if _jaccard(shingles, kept_shingles[idx]) >= NEAR_DUPLICATE_THRESHOLD:
                duplicate = True
                break
        if duplicate:
            continue
        idx = len(kept)
        kept.append(doc)
        kept_shingles.append(shingles)
        for b, band_key in enumerate(bands):
            buckets[b].setdefault(band_key, []).append(idx)
    if stats is not None:
        stats["comparisons"] = comparisons
    return kept


def dedupe_stream(docs: Iterable[str], stats: Optional[dict] = None) -> Iterator[str]:
    """Streaming version of ``dedupe_documents``: consumes ``docs`` lazily and yields kept originals.

    It makes the same keep or drop decision for every document as ``dedupe_documents`` on the same input
    (same LSH buckets, same exact-duplicate rule, same first-occurrence order), so it is a drop-in
    replacement for corpora that do not fit in memory as a list.

    What it holds between documents, and nothing else:
      - a 16-byte digest of each distinct cleaned, lowercased text (exact duplicates);
      - the LSH band keys of each kept document, as dictionary buckets (about 16 small tuples per document);
      - the sorted 64-bit shingle hashes of each kept document, 8 bytes per distinct shingle.
    The full shingle tuples and the document strings themselves are not kept, so memory per kept document
    is about 8 bytes per distinct shingle instead of about 170 bytes. Candidates are verified with exact
    Jaccard on those hashes. For very large corpora, still run it per shard, because the state grows with
    the number of kept documents.

    If ``stats`` is a dict, ``"comparisons"`` is kept up to date with the number of Jaccard checks.
    """
    seen: Set[bytes] = set()
    buckets: List[dict] = [{} for _ in range(LSH_BANDS)]
    kept_shingles: List[array] = []
    kept_empty = False
    comparisons = 0
    for doc in docs:
        key = clean_text(doc).strip().lower()
        digest = hashlib.blake2b(key.encode("utf-8", "surrogatepass"), digest_size=16).digest()
        if digest in seen:
            continue
        seen.add(digest)
        shingles = _shingles(key)
        if not shingles:
            # Same rule as dedupe_documents: token-less documents are duplicates of each other.
            if kept_empty:
                continue
            kept_empty = True
            if stats is not None:
                stats["comparisons"] = comparisons
            yield doc
            continue
        signature = _minhash(shingles)
        bands = [signature[b * _ROWS_PER_BAND:(b + 1) * _ROWS_PER_BAND] for b in range(LSH_BANDS)]
        candidates: Set[int] = set()
        for b, band_key in enumerate(bands):
            candidates.update(buckets[b].get(band_key, ()))
        hashed = array(_INT_CODE_U64, sorted({_shingle_hash(s) for s in shingles}))
        duplicate = False
        for idx in candidates:
            comparisons += 1
            if _hashed_jaccard(hashed, kept_shingles[idx]) >= NEAR_DUPLICATE_THRESHOLD:
                duplicate = True
                break
        if duplicate:
            continue
        idx = len(kept_shingles)
        kept_shingles.append(hashed)
        for b, band_key in enumerate(bands):
            buckets[b].setdefault(band_key, []).append(idx)
        if stats is not None:
            stats["comparisons"] = comparisons
        yield doc


def _pack(
    token_id_lists: Iterable[Sequence[int]],
    seq_len: int,
    eos_id: int,
    offset: int,
    rng: Optional[random.Random],
    with_segments: bool,
):
    """Shared packing core. Tokens (and optionally doc ids and in-document positions) go to int32 arrays."""
    if seq_len < 2:
        raise ValueError(f"seq_len must be at least 2, got {seq_len}")
    if rng is not None:
        offset = rng.randrange(seq_len)
    if offset < 0:
        raise ValueError(f"offset must be non-negative, got {offset}")
    eos = int(eos_id)
    tokens = array(_INT_CODE)
    doc_ids = array(_INT_CODE)
    positions = array(_INT_CODE)
    for doc_index, ids in enumerate(token_id_lists):
        start = len(tokens)
        tokens.extend(map(int, ids))
        length = len(tokens) - start
        tokens.append(eos)
        if with_segments:
            doc_ids.extend([doc_index] * (length + 1))
            positions.extend(range(length + 1))

    rows = (len(tokens) - offset) // seq_len if len(tokens) > offset else 0
    usable = rows * seq_len

    def window(buf: array) -> torch.Tensor:
        if rows == 0:
            return torch.empty((0, seq_len), dtype=torch.long)
        view = torch.frombuffer(buf, dtype=torch.int32)
        return view[offset:offset + usable].long().view(rows, seq_len)

    if with_segments:
        return window(tokens), window(doc_ids), window(positions)
    return window(tokens)


def pack_sequences(
    token_id_lists: Iterable[Sequence[int]],
    seq_len: int,
    eos_id: int,
    *,
    offset: int = 0,
    rng: Optional[random.Random] = None,
) -> torch.Tensor:
    """Concatenate documents with ``eos_id`` after each, then cut non-overlapping windows.

    Returns a LongTensor of shape [N, seq_len]. The final partial window is dropped, so N may be 0.
    Raises ValueError if ``seq_len`` is less than 2.

    ``offset`` drops that many tokens from the start of the stream before cutting windows, so window
    boundaries move. ``rng`` (for example ``random.Random(epoch_seed)``) picks the offset uniformly in
    [0, seq_len) and overrides ``offset``, so each epoch can use different boundaries. The default
    (offset 0, no rng) is unchanged.

    Tokens are stored in an int32 array while packing, not as a Python list, so memory is about
    4 bytes per token until the final tensor is built.
    """
    return _pack(token_id_lists, seq_len, eos_id, offset, rng, with_segments=False)


def pack_sequences_with_segments(
    token_id_lists: Iterable[Sequence[int]],
    seq_len: int,
    eos_id: int,
    *,
    offset: int = 0,
    rng: Optional[random.Random] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Like ``pack_sequences`` but also returns document boundary information for each window.

    Returns ``(tokens, doc_ids, positions)``, each a LongTensor of shape [N, seq_len]:
    - ``tokens``: identical to what ``pack_sequences`` returns for the same arguments.
    - ``doc_ids``: index of the source document of each token (0-based, in input order). The EOS that
      ends a document belongs to that document.
    - ``positions``: index of each token within its document, starting at 0 (the EOS takes the next
      position after the last token). Useful for resetting positional encodings per document.

    Use ``doc_boundary_mask(doc_ids)`` to find where a target token starts a new document.
    """
    return _pack(token_id_lists, seq_len, eos_id, offset, rng, with_segments=True)


def doc_boundary_mask(doc_ids: torch.Tensor) -> torch.Tensor:
    """Boolean mask of shape [N, seq_len - 1]. True where token t+1 belongs to a different document than t.

    Use it to drop next-token loss across document boundaries (the EOS to first-token transition).
    """
    return doc_ids[:, 1:] != doc_ids[:, :-1]


def iter_jsonl(path: Union[str, Path], field: str = "text") -> Iterator:
    """Yield ``field`` from each line of a JSONL file, read as UTF-8 (a leading byte-order mark is accepted).

    Blank lines and lines whose JSON object lacks the field are skipped, and so are records whose
    value is ``null``. A value that is not a string raises ``ValueError`` with the file and line number.
    Malformed JSON raises ``json.JSONDecodeError`` so bad input is not silently dropped.
    """
    with open(path, "r", encoding="utf-8-sig") as handle:
        for lineno, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            obj = json.loads(line)
            if isinstance(obj, dict) and field in obj:
                value = obj[field]
                if value is None:
                    continue
                if not isinstance(value, str):
                    raise ValueError(
                        f"{path}: line {lineno}: field {field!r} must be a string, "
                        f"got {type(value).__name__}"
                    )
                yield value


def prepare_corpus(
    docs: Iterable[str],
    encode: Callable[[str], List[int]],
    seq_len: int,
    eos_id: int,
    min_chars: int = DEFAULT_MIN_CHARS,
    *,
    reserved_ids: Iterable[int] = DEFAULT_RESERVED_IDS,
    stats: Optional[Dict[str, int]] = None,
) -> torch.Tensor:
    """Run the full pipeline: clean, quality filter, dedupe, encode each document, then pack.

    ``encode`` maps a string to a list of token ids. Returns a LongTensor of shape [N, seq_len].

    Streaming: every stage is a generator, so ``docs`` is read once and documents are cleaned, filtered,
    deduplicated (``dedupe_stream``), encoded and packed one at a time. Only the dedupe state and the
    int32 packing store are held, not the cleaned or kept strings.

    Reserved ids: a document whose encoding contains any id in ``reserved_ids`` is dropped whole, not
    stripped, so no id from the special range reaches training as ordinary text. The default covers
    EOS_Tokenizer's special tokens and Inky's scratch ids (``DEFAULT_RESERVED_IDS``). Pass an empty set to
    disable this check.

    If ``stats`` is a dict, it is filled with ``input`` (documents read), ``kept`` (documents packed) and
    ``dropped_reserved`` (documents dropped for a reserved id). ``warnings`` is emitted when no document
    survives, so an empty result is never silent.

    Raises ``TypeError`` for a document that is not a string.
    """
    reserved = frozenset(int(i) for i in reserved_ids)
    counts: Dict[str, int] = {"input": 0, "kept": 0, "dropped_reserved": 0}

    def _cleaned() -> Iterator[str]:
        for index, doc in enumerate(docs):
            if not isinstance(doc, str):
                raise TypeError(f"prepare_corpus needs string documents, got {type(doc).__name__} at index {index}")
            counts["input"] += 1
            text = clean_text(doc)
            if quality_filter(text, min_chars=min_chars):
                yield text

    def _encoded() -> Iterator[List[int]]:
        for text in dedupe_stream(_cleaned()):
            ids = [int(i) for i in encode(text)]
            if reserved and not reserved.isdisjoint(ids):
                counts["dropped_reserved"] += 1
                continue
            counts["kept"] += 1
            yield ids

    packed = pack_sequences(_encoded(), seq_len, eos_id)
    if stats is not None:
        stats.update(counts)
    if counts["kept"] == 0:
        warnings.warn(
            f"prepare_corpus kept no documents out of {counts['input']} input "
            f"({counts['dropped_reserved']} dropped for reserved ids); check the input, min_chars and encode",
            UserWarning,
            stacklevel=2,
        )
    return packed
