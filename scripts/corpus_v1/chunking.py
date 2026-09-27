"""Section/paragraph/sentence aware chunks measured by an actual tokenizer."""

import re
from collections.abc import Callable, Iterator


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


def chunk_text(text: str, token_count: Callable[[str], int], budget: int = 384,
               overlap: int = 48) -> Iterator[tuple[str, str, int]]:
    if budget < 16 or overlap < 0 or overlap >= budget:
        raise ValueError("Require chunk_tokens >= 16 and 0 <= overlap < chunk_tokens")
    for section, content in sections(text):
        units = [s.strip() for paragraph in re.split(r"\n\s*\n", content)
                 for s in SENTENCE.split(paragraph) if s.strip()]
        current: list[str] = []
        for unit in units:
            pieces = [unit]
            if token_count(unit) > budget:
                words = unit.split()
                pieces, part = [], []
                for word in words:
                    if part and token_count(" ".join(part + [word])) > budget:
                        pieces.append(" ".join(part))
                        part = []
                    part.append(word)
                if part:
                    pieces.append(" ".join(part))
            for piece in pieces:
                candidate = " ".join(current + [piece])
                if current and token_count(candidate) > budget:
                    result = " ".join(current)
                    yield section, result, token_count(result)
                    tail = []
                    for old in reversed(current):
                        if token_count(" ".join([old] + tail)) > overlap:
                            break
                        tail.insert(0, old)
                    while tail and token_count(" ".join(tail + [piece])) > budget:
                        tail.pop(0)
                    current = tail
                current.append(piece)
        if current:
            result = " ".join(current)
            yield section, result, token_count(result)
