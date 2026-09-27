"""Tokenizer-bounded chunks with section, paragraph and sentence preferences."""

import re
from collections.abc import Callable, Iterator


CHUNKER_VERSION = "section_paragraph_sentence_v2"
SECTION = re.compile(r"^\s*(?:={2,}\s*(.*?)\s*={2,}|#{1,4}\s+(.*))\s*$")
SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[^\s])")


def sections(text: str) -> Iterator[tuple[str, str]]:
    current = ""
    lines = []
    for line in text.splitlines():
        match = SECTION.match(line)
        if match:
            if lines:
                yield current, "\n".join(lines).strip()
            current = (match.group(1) or match.group(2) or "").strip()
            lines = []
        else:
            lines.append(line)
    if lines:
        yield current, "\n".join(lines).strip()


def _split_oversize(value: str, count: Callable[[str], int], budget: int) -> Iterator[str]:
    """Split rare oversized units by maximal word prefix, then by characters."""
    remaining = value.strip()
    while remaining:
        if count(remaining) <= budget:
            yield remaining
            return
        ends = [match.end() for match in re.finditer(r"\S+", remaining)]
        low, high, best = 1, len(ends), 0
        while low <= high:
            mid = (low + high) // 2
            if count(remaining[:ends[mid - 1]]) <= budget:
                best = mid
                low = mid + 1
            else:
                high = mid - 1
        if best:
            cut = ends[best - 1]
            yield remaining[:cut]
            remaining = remaining[cut:].lstrip()
            continue
        word = remaining[:ends[0]]
        low, high, best = 1, len(word), 0
        while low <= high:
            mid = (low + high) // 2
            if count(word[:mid]) <= budget:
                best = mid
                low = mid + 1
            else:
                high = mid - 1
        if not best:
            raise ValueError("A single character exceeds the configured token budget")
        yield word[:best]
        remaining = remaining[best:].lstrip()


def _units(content: str, count: Callable[[str], int], budget: int) -> Iterator[tuple[str, int]]:
    for paragraph in re.split(r"\n\s*\n", content):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        size = count(paragraph)
        if size <= budget:
            yield paragraph, size
            continue
        for sentence in SENTENCE.split(paragraph):
            sentence = sentence.strip()
            if not sentence:
                continue
            size = count(sentence)
            if size <= budget:
                yield sentence, size
            else:
                for piece in _split_oversize(sentence, count, budget):
                    yield piece, count(piece)


def chunk_text(text: str, token_count: Callable[[str], int], budget: int = 384,
               overlap: int = 48) -> Iterator[tuple[str, str, int]]:
    if budget < 16 or overlap < 0 or overlap >= budget:
        raise ValueError("Require chunk_tokens >= 16 and 0 <= overlap < chunk_tokens")
    for section, content in sections(text):
        current: list[tuple[str, int]] = []
        estimate = 0
        for piece, piece_tokens in _units(content, token_count, budget):
            if current:
                # Most units are short. Tokenize the concatenation only near the limit;
                # the emitted chunk receives a final strict tokenizer check below.
                approximate = estimate + piece_tokens + len(current)
                if approximate <= budget - 8 or token_count(" ".join(x for x, _ in current) + " " + piece) <= budget:
                    current.append((piece, piece_tokens))
                    estimate += piece_tokens
                    continue
                value = " ".join(x for x, _ in current)
                for fitted in _split_oversize(value, token_count, budget):
                    yield section, fitted, token_count(fitted)
                tail: list[tuple[str, int]] = []
                for old in reversed(current):
                    if sum(size for _, size in tail) + old[1] > overlap:
                        break
                    tail.insert(0, old)
                while tail and token_count(" ".join(x for x, _ in tail) + " " + piece) > budget:
                    tail.pop(0)
                current = tail
                estimate = sum(size for _, size in tail)
            current.append((piece, piece_tokens))
            estimate += piece_tokens
        if current:
            value = " ".join(x for x, _ in current)
            for fitted in _split_oversize(value, token_count, budget):
                yield section, fitted, token_count(fitted)
