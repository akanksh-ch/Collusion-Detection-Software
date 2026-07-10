"""Greedy String Tiling (GST) — classical local alignment for line-level
evidence generation.

Implements the Wise (1993) / Prechelt et al. (2002) algorithm used inside
JPlag.  Operates on language-agnostic token sequences with physical line
number tracking so that matched tile regions can be mapped back to exact
source ranges for the forensic diff sidebar.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Tuple

# ──────────────────────────────────────────────────────────────────────
# Token representation
# ──────────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class Token:
    """A single normalised token with provenance metadata."""
    value: str        # lowercased, identifier-normalised form
    line: int         # 1-indexed physical line number
    col: int          # 0-indexed column offset (for display)
    raw: str = ""     # original text before normalisation


@dataclass(slots=True)
class Tile:
    """One maximal contiguous match found by GST."""
    length: int
    tokens_a: List[Token] = field(default_factory=list)
    tokens_b: List[Token] = field(default_factory=list)
    a_lines: Tuple[int, int] = (0, 0)   # (start_line, end_line) inclusive
    b_lines: Tuple[int, int] = (0, 0)


# ──────────────────────────────────────────────────────────────────────
# Tokeniser
# ──────────────────────────────────────────────────────────────────────

# Match word-like tokens, operators, and punctuation individually
_TOKEN_RE = re.compile(r"[A-Za-z_]\w*|[0-9]+(?:\.[0-9]+)?|[^\s]")


def tokenize_source(source_text: str) -> List[Token]:
    """Convert raw source text into a flat list of normalised tokens.

    Normalisation rules (designed for Type-1/2/3 detection):
    * All identifiers are lowercased.
    * Numeric literals are replaced with the sentinel ``NUM``.
    * String / char literals are replaced with ``STR``.
    * All other tokens (keywords, operators, punctuation) are preserved.
    """
    tokens: list[Token] = []
    for line_no, line in enumerate(source_text.splitlines(), start=1):
        for m in _TOKEN_RE.finditer(line):
            raw = m.group()
            col = m.start()

            # normalise
            if raw.startswith('"') or raw.startswith("'"):
                val = "STR"
            elif re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", raw):
                val = "NUM"
            elif re.fullmatch(r"[A-Za-z_]\w*", raw):
                val = raw.lower()
            else:
                val = raw

            tokens.append(Token(value=val, line=line_no, col=col, raw=raw))

    return tokens


# ──────────────────────────────────────────────────────────────────────
# Greedy String Tiling core
# ──────────────────────────────────────────────────────────────────────

def greedy_string_tiling(
    tokens_a: List[Token],
    tokens_b: List[Token],
    min_match_length: int = 8,
) -> List[Tile]:
    """Run Greedy String Tiling on two token sequences.

    Returns a list of non-overlapping ``Tile`` objects sorted by descending
    length.  Each tile contains the matched token subsequences from both
    files and their inclusive physical line ranges.

    Algorithm (Wise 1993):
        Repeat until no match of length >= min_match_length is found:
            1. Scan all pairs (i, j) and record maximal match lengths.
            2. Among the longest matches, greedily select non-overlapping
               tiles from longest to shortest.
            3. Mark matched tokens so they cannot be reused.
    """
    n = len(tokens_a)
    m = len(tokens_b)

    marked_a = [False] * n
    marked_b = [False] * m

    tiles: list[Tile] = []

    while True:
        # Phase 1 — scanline: find all maximal matches
        max_match = min_match_length
        matches: list[tuple[int, int, int]] = []  # (i, j, length)

        for i in range(n):
            if marked_a[i]:
                continue
            for j in range(m):
                if marked_b[j]:
                    continue

                k = 0
                while (i + k < n
                       and j + k < m
                       and not marked_a[i + k]
                       and not marked_b[j + k]
                       and tokens_a[i + k].value == tokens_b[j + k].value):
                    k += 1

                if k >= max_match:
                    if k > max_match:
                        # New longest match — discard shorter candidates
                        max_match = k
                        matches = []
                    matches.append((i, j, k))

        if not matches:
            break

        # Phase 2 — mark tiles (greedy, longest first — ties broken by position)
        matches.sort(key=lambda t: (-t[2], t[0], t[1]))
        for i, j, length in matches:
            # Verify no overlap with already-marked tokens
            if any(marked_a[i + d] or marked_b[j + d] for d in range(length)):
                continue

            # Mark tokens
            for d in range(length):
                marked_a[i + d] = True
                marked_b[j + d] = True

            tile_tokens_a = tokens_a[i: i + length]
            tile_tokens_b = tokens_b[j: j + length]

            tiles.append(Tile(
                length=length,
                tokens_a=tile_tokens_a,
                tokens_b=tile_tokens_b,
                a_lines=(tile_tokens_a[0].line, tile_tokens_a[-1].line),
                b_lines=(tile_tokens_b[0].line, tile_tokens_b[-1].line),
            ))

    tiles.sort(key=lambda t: -t.length)
    return tiles


# ──────────────────────────────────────────────────────────────────────
# Aggregate coverage metric
# ──────────────────────────────────────────────────────────────────────

def coverage_score(tiles: List[Tile], len_a: int, len_b: int) -> float:
    """Symmetric coverage: fraction of tokens covered by tiles.

    Returns a value in [0, 1].  Uses the same formula as JPlag:
        2 * (sum of tile lengths) / (|A| + |B|)
    """
    total_matched = sum(t.length for t in tiles)
    denom = len_a + len_b
    if denom == 0:
        return 0.0
    return min(1.0, 2.0 * total_matched / denom)


def tiles_to_json(tiles: List[Tile]) -> list[dict]:
    """Serialise tiles into a JSON-friendly list for the report payload."""
    out = []
    for t in tiles:
        out.append({
            "length": t.length,
            "a_lines": list(t.a_lines),
            "b_lines": list(t.b_lines),
            "matched_text_a": " ".join(tok.raw for tok in t.tokens_a),
            "matched_text_b": " ".join(tok.raw for tok in t.tokens_b),
        })
    return out
