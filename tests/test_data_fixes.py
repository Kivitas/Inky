"""Tests for the data.py fixes: compact packing, LSH dedupe, script-aware length, refrains,
layout-preserving clean_text, mark-aware tokenisation and document-boundary packing.
Run with: python -m pytest tests/test_data_fixes.py -q
"""

import random
import sys
import tracemalloc
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inky.data import (  # noqa: E402
    clean_text,
    dedupe_documents,
    doc_boundary_mask,
    length_units,
    pack_sequences,
    pack_sequences_with_segments,
    quality_filter,
    word_tokens,
)

PROSE = (
    "The river bends slowly past the old mill, and farmers walk their cattle along the muddy bank "
    "each morning before the sun rises over the hills. Children gather stones near the bridge while "
    "their parents trade stories about harvests, weather, and the price of grain in the market town."
)
HINDI = (
    "नदी धीरे-धीरे पुराने मिल के पास से बहती है। सुबह होते ही किसान अपने मवेशियों को किनारे की ओर ले जाते हैं, "
    "और बच्चे पुल के पास पत्थर इकट्ठा करते हैं। माता-पिता बाज़ार में फसल, मौसम और अनाज के दामों पर बात करते हैं। "
    "सर्दियों में पहाड़ों से आने वाली हवा ठंडी होती है, इसलिए गाँव के लोग शाम को अलाव के पास बैठकर कहानियाँ सुनाते हैं। "
    "धीरे-धीरे रात गहरी होती है और पूरा गाँव शांत हो जाता है।"
)
THAI = (
    "แม่น้ำไหลอย่างช้าๆ ผ่านโรงสีเก่าริมทุ่งข้าว ทุกเช้าชาวนาจะพาวัวไปตามคันดินที่ชื้นแฉะ "
    "ก่อนดวงอาทิตย์จะขึ้นเหนือเนินเขา เด็ก ๆ เก็บก้อนหินใกล้สะพานขณะที่พ่อแม่คุยกันเรื่องฤดูกาล "
    "สภาพอากาศ และราคาข้าวในตลาดของอำเภอ ฤดูหนาวมาถึงเร็วกว่าปีก่อน ผู้เฝ้าบ้านจึงจุดเตาไฟ "
    "ในกระท่อมเล็ก ๆ และนับฟืนที่เหลืออยู่อีกครั้งหนึ่ง"
)
CHINESE = (
    "河水缓慢地流过老磨坊，农夫们每天清晨都会牵着牛沿着泥泞的河岸走过。孩子们在桥边捡石头，"
    "父母则谈论收成、天气以及集市上粮食的价格。冬天比往年来得更早，看守房屋的老人点燃小屋里的炉火，"
    "数了又数剩下的柴火，心里想着这些木柴能否撑到春天。村里的老人常讲《河边的故事》，"
    "说河神会在大雪之后醒来！你信吗？孩子们笑着点头：信，当然信。夜色渐渐深了，村子安静下来，"
    "只有远处的河水仍在不停地流动……第二天一早，磨坊的门又被推开，新的麦子运了进来，"
    "人们忙碌起来，开始了又一年的劳作。"
)
OTHER = (
    "Winter arrived early that year, covering the orchard in a thin white layer. The keeper lit the "
    "stove in the small cabin and counted the firewood twice, wondering whether it would be enough "
    "until the thaw came down from the northern valleys and the roads opened again for travellers."
)


def _brute_force_dedupe(docs):
    """Reference: the original exhaustive algorithm, exact Jaccard against every kept document."""
    from inky.data import NEAR_DUPLICATE_THRESHOLD, _jaccard, _shingles, clean_text as clean

    kept, kept_sh, seen = [], [], set()
    for doc in docs:
        key = clean(doc).strip().lower()
        if key in seen:
            continue
        seen.add(key)
        sh = _shingles(key)
        if any(_jaccard(sh, other) >= NEAR_DUPLICATE_THRESHOLD for other in kept_sh):
            continue
        kept.append(doc)
        kept_sh.append(sh)
    return kept


# Fix 1: compact store for packing

def test_pack_sequences_does_not_materialise_python_int_list():
    # 500k tokens as a Python list of ints would need well over 15 MB of Python objects.
    # The int32 array store keeps the Python-side peak to a few MB.
    def docs():
        for i in range(500):
            yield [(i * 1000 + j) % 50000 for j in range(1000)]

    tracemalloc.start()
    try:
        packed = pack_sequences(docs(), seq_len=1000, eos_id=0)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert packed.shape == (500, 1000)
    assert packed.dtype == torch.long
    assert peak < 10 * 1024 * 1024, f"peak Python allocation {peak} bytes is too large"
    # stream index 1000 is the EOS that ends the first document
    assert packed[1, 0].item() == 0


def test_pack_sequences_accepts_generators_and_tensors_of_ids():
    packed_lists = pack_sequences([[1, 2, 3], [4, 5]], seq_len=3, eos_id=0)
    packed_gen = pack_sequences(([1, 2, 3], iter([4, 5])), seq_len=3, eos_id=0)
    packed_tensor = pack_sequences([torch.tensor([1, 2, 3]), torch.tensor([4, 5])], seq_len=3, eos_id=0)
    assert packed_lists.tolist() == packed_gen.tolist() == packed_tensor.tolist() == [[1, 2, 3], [0, 4, 5]]


# Fix 2: near-linear dedupe

def test_dedupe_compares_no_candidates_for_distinct_documents():
    docs = [" ".join(f"w{i}_{j}" for j in range(60)) for i in range(1000)]
    stats = {}
    assert len(dedupe_documents(docs, stats=stats)) == 1000
    assert stats["comparisons"] == 0


def test_dedupe_near_linear_on_clustered_documents_and_matches_brute_force():
    # 300 clusters of 4 documents each, every variant changes one word of a 60-word base.
    docs = []
    for c in range(300):
        base = " ".join(f"c{c}t{j}" for j in range(60))
        docs.append(base)
        for k in range(3):
            docs.append(base.replace(f"c{c}t{10 + k}", "changed"))
    stats = {}
    result = dedupe_documents(docs, stats=stats)
    assert len(result) == 300
    assert result == _brute_force_dedupe(docs)
    assert stats["comparisons"] < 5 * len(docs)


def test_dedupe_keeps_result_identical_to_brute_force_on_mixed_inputs():
    rng = random.Random(7)
    words = [f"term{i}" for i in range(40)]
    docs = []
    for _ in range(80):
        length = rng.randint(3, 40)
        docs.append(" ".join(rng.choice(words) for _ in range(length)))
    assert dedupe_documents(docs) == _brute_force_dedupe(docs)


def test_dedupe_treats_documents_without_tokens_as_duplicates_of_each_other():
    assert dedupe_documents(["!!!", "...", PROSE]) == ["!!!", PROSE]


# Fix 3: script-aware length

def test_min_chars_counts_cjk_characters_with_weight():
    short_cjk = CHINESE[:90]          # 90 characters, roughly 270 units
    short_latin = PROSE[:90]          # 90 characters, 90 units
    assert length_units(short_cjk) > length_units(short_latin) * 2
    # explicit min_chars: the default is now 20 (DATA-03), so this boundary is tested at 200
    assert quality_filter(short_cjk, min_chars=200) is True
    assert quality_filter(short_latin, min_chars=200) is False
    assert quality_filter(short_cjk, min_chars=10_000) is False


def test_min_chars_latin_length_unchanged():
    assert length_units("a" * 50) == 50
    assert quality_filter("Too short.") is False


def test_thai_is_weighted_above_latin_length():
    assert length_units(THAI) > len(THAI)


# Fix 4: refrains and verse

def _poem_with_refrain(refrain_count):
    verse = [
        f"Verse line {i}: the rivers carry grain and song down toward the sea of hills."
        for i in range(12)
    ]
    refrain = ["Oh the river runs"] * refrain_count
    lines = []
    for i, line in enumerate(verse):
        lines.append(line)
        if i % 2 == 1 and refrain:
            lines.append(refrain.pop())
    lines.extend(refrain)
    return "\n".join(lines)


def test_quality_filter_keeps_poem_with_short_refrain():
    poem = _poem_with_refrain(8)   # 8 of 20 lines are the same 17-character refrain
    assert len(poem) >= 200
    assert quality_filter(poem) is True


def test_quality_filter_still_rejects_long_repeated_stanzas():
    stanza = "This entire long stanza line is pasted again and again into the document."
    text = "\n".join([f"Distinct line {i} about rivers, mills and farms in autumn." for i in range(5)]
                      + [stanza] * 5)
    assert quality_filter(text) is False


def test_quality_filter_rejects_spam_of_one_short_line():
    text = "\n".join(["Buy now"] * 30 + [f"Unique line number {i} with words." for i in range(3)])
    assert len(text) >= 200
    assert quality_filter(text) is False


# Fix 5: layout-preserving clean_text

def test_clean_text_preserves_indentation_and_single_and_double_blank_lines():
    raw = "Verse one\n    indented line\n\n\n\n\tTabbed  line  \n\n\nEnd"
    assert clean_text(raw) == "Verse one\n    indented line\n\n\n\tTabbed line\n\n\nEnd"
    assert clean_text("a\n\nb") == "a\n\nb"
    assert clean_text("a\n\n\nb") == "a\n\n\nb"


def test_clean_text_strips_trailing_spaces_and_whitespace_only_lines():
    assert clean_text("  stanza  \n   \nnext   ") == "  stanza\n\nnext"


def test_clean_text_converts_lone_carriage_returns():
    assert clean_text("one\rtwo") == "one\ntwo"


# Fix 6: mark-aware tokens and no-space scripts

def test_word_tokens_keep_devanagari_combining_marks_inside_words():
    assert word_tokens("बाज़ार की") == ["बाज़ार", "की"]
    assert word_tokens("पुराने मिल") == ["पुराने", "मिल"]


def test_word_tokens_split_no_space_scripts_into_characters():
    tokens = word_tokens(THAI)
    assert all(len(t) == 1 for t in tokens)
    assert "".join(tokens) == "".join(THAI.split())
    # Han characters are tokens of their own; the fullwidth comma is punctuation and is not a token.
    chinese_tokens = word_tokens(CHINESE[:10])
    assert chinese_tokens == [c for c in CHINESE[:10] if c != "，"]


def test_dedupe_detects_chinese_near_duplicate_with_one_character_changed():
    variant = CHINESE.replace("河", "湖", 1)
    assert dedupe_documents([CHINESE, variant]) == [CHINESE]
    assert dedupe_documents([CHINESE, OTHER]) == [CHINESE, OTHER]


def test_dedupe_detects_devanagari_near_duplicate():
    variant = HINDI.replace("नदी", "नहर", 1)
    assert dedupe_documents([HINDI, variant]) == [HINDI]


# Fix 7: document boundaries and per-epoch offsets

def test_pack_with_segments_gives_doc_ids_positions_and_same_tokens():
    docs = [[1, 2, 3], [4, 5]]
    tokens, doc_ids, positions = pack_sequences_with_segments(docs, seq_len=3, eos_id=0)
    assert tokens.tolist() == pack_sequences(docs, seq_len=3, eos_id=0).tolist() == [[1, 2, 3], [0, 4, 5]]
    # stream: 1 2 3 0 | 4 5 0 with document ids 0 0 0 0 | 1 1 1
    assert doc_ids.tolist() == [[0, 0, 0], [0, 1, 1]]
    # positions inside each document: 0 1 2 3 | 0 1 2
    assert positions.tolist() == [[0, 1, 2], [3, 0, 1]]


def test_doc_boundary_mask_marks_transitions_between_documents():
    _, doc_ids, _ = pack_sequences_with_segments([[1, 2, 3], [4, 5]], seq_len=3, eos_id=0)
    mask = doc_boundary_mask(doc_ids)
    assert mask.shape == (2, 2)
    assert mask.tolist() == [[False, False], [True, False]]


def test_pack_offset_moves_window_boundaries():
    docs = [[1, 2, 3], [4, 5]]
    assert pack_sequences(docs, seq_len=3, eos_id=0, offset=1).tolist() == [[2, 3, 0], [4, 5, 0]]


def test_pack_rng_gives_per_epoch_offsets_and_default_is_unchanged():
    docs = [list(range(1, 30))] * 3
    default = pack_sequences(docs, seq_len=8, eos_id=0)
    assert default.tolist() == pack_sequences(docs, seq_len=8, eos_id=0, offset=0).tolist()
    seen = set()
    for epoch in range(20):
        via_rng = pack_sequences(docs, seq_len=8, eos_id=0, rng=random.Random(epoch))
        expected_offset = random.Random(epoch).randrange(8)
        assert via_rng.tolist() == pack_sequences(docs, seq_len=8, eos_id=0, offset=expected_offset).tolist()
        seen.add(expected_offset)
    assert len(seen) > 1


def test_pack_rejects_negative_offset():
    try:
        pack_sequences([[1, 2]], seq_len=2, eos_id=0, offset=-1)
    except ValueError:
        return
    raise AssertionError("expected ValueError for negative offset")
