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
    filename: str = ""


@dataclass(slots=True)
class Tile:
    """One maximal contiguous match found by GST."""
    length: int
    tokens_a: List[Token] = field(default_factory=list)
    tokens_b: List[Token] = field(default_factory=list)
    a_lines: Tuple[int, int] = (0, 0)   # (start_line, end_line) inclusive
    b_lines: Tuple[int, int] = (0, 0)
    a_file: str = ""
    b_file: str = ""
    a_token_idx: int = 0
    b_token_idx: int = 0


# ──────────────────────────────────────────────────────────────────────
# Tokeniser
# ──────────────────────────────────────────────────────────────────────

# Match string/char literals, then word-like tokens, numbers, and punctuation
_TOKEN_RE = re.compile(r'"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'|[A-Za-z_]\w*|[0-9]+(?:\.[0-9]+)?|[^\s]')


def tokenize_source(sources: dict[str, str]) -> List[Token]:
    """Convert raw source files into a flat list of normalised tokens.
    Inserts ___EOF___ tokens between files to prevent cross-file matches."""
    tokens: list[Token] = []
    
    for filename, source_text in sources.items():
        for line_no, line in enumerate(source_text.splitlines(), start=1):
            for m in _TOKEN_RE.finditer(line):
                raw = m.group()
                col = m.start()

                if raw.startswith(("'", '"')):
                    value = "STR"
                elif raw.isdigit() or (raw.replace(".", "", 1).isdigit() and "." in raw):
                    value = "NUM"
                else:
                    value = raw.lower()

                tokens.append(Token(value=value, line=line_no, col=col, raw=raw, filename=filename))
        
        # Insert EOF token to act as a pivot and prevent matches crossing files
        tokens.append(Token(value="___EOF___", line=0, col=0, raw="___EOF___", filename=filename))
        
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
            
    Optimized with an n-gram dictionary to avoid O(N*M) scanning.
    """
    n = len(tokens_a)
    m = len(tokens_b)

    marked_a = [False] * n
    marked_b = [False] * m

    tiles: list[Tile] = []

    if n < min_match_length or m < min_match_length:
        return tiles

    vals_a = [t.value for t in tokens_a]
    vals_b = [t.value for t in tokens_b]

    # Prebuild index for B
    b_index = {}
    for j in range(m - min_match_length + 1):
        ngram = tuple(vals_b[j : j + min_match_length])
        if ngram not in b_index:
            b_index[ngram] = []
        b_index[ngram].append(j)

    while True:
        max_match = min_match_length
        matches = []

        i = 0
        while i < n - max_match + 1:
            if marked_a[i]:
                i += 1
                continue

            ngram = tuple(vals_a[i : i + min_match_length])
            if ngram in b_index:
                for j in b_index[ngram]:
                    if marked_b[j]:
                        continue
                    
                    k = 0
                    while (i + k < n and j + k < m 
                           and not marked_a[i + k] 
                           and not marked_b[j + k] 
                           and vals_a[i + k] == vals_b[j + k]):
                        k += 1
                    
                    if k >= max_match:
                        if k > max_match:
                            max_match = k
                            matches = []
                        matches.append((i, j, k))
            i += 1

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
                a_file=tile_tokens_a[0].filename,
                b_file=tile_tokens_b[0].filename,
                a_token_idx=i,
                b_token_idx=j
            ))

    tiles.sort(key=lambda t: -t.length)
    return tiles


# ──────────────────────────────────────────────────────────────────────
# Subsequence Match Merging (mirrors JPlag's --match-merging)
# ──────────────────────────────────────────────────────────────────────

def merge_neighboring_tiles(
    tiles: List[Tile],
    tokens_a: List[Token],
    tokens_b: List[Token],
    max_gap: int = 6,
    min_neighbor_length: int = 2,
) -> List[Tile]:
    """Merge tiles that are close neighbors in *both* token streams.

    Two tiles are neighbors if the gap between them (in both A and B)
    is ≤ ``max_gap`` tokens and both tiles are at least
    ``min_neighbor_length`` long.  Merged tiles absorb the gap tokens
    so the viewer renders one contiguous highlight instead of several
    fragmented ones.

    Mirrors JPlag's ``--match-merging`` / ``--gap-size`` / ``--neighbor-length``.
    """
    if len(tiles) < 2:
        return tiles

    # Sort by position in stream A (primary), then stream B
    tiles.sort(key=lambda t: (t.a_file, t.a_token_idx))

    merged: list[Tile] = []
    current = tiles[0]

    for nxt in tiles[1:]:
        # Only merge tiles within the same file pair
        if current.a_file != nxt.a_file or current.b_file != nxt.b_file:
            merged.append(current)
            current = nxt
            continue

        # Both tiles must meet minimum length
        if current.length < min_neighbor_length or nxt.length < min_neighbor_length:
            merged.append(current)
            current = nxt
            continue

        gap_a = nxt.a_token_idx - (current.a_token_idx + current.length)
        gap_b = nxt.b_token_idx - (current.b_token_idx + current.length)

        if 0 <= gap_a <= max_gap and 0 <= gap_b <= max_gap:
            # Merge: absorb gap + next tile
            new_end_a = nxt.a_token_idx + nxt.length
            new_end_b = nxt.b_token_idx + nxt.length
            new_length = new_end_a - current.a_token_idx

            new_tokens_a = tokens_a[current.a_token_idx:new_end_a]
            new_tokens_b = tokens_b[current.b_token_idx:new_end_b]

            current = Tile(
                length=new_length,
                tokens_a=new_tokens_a,
                tokens_b=new_tokens_b,
                a_lines=(new_tokens_a[0].line, new_tokens_a[-1].line),
                b_lines=(new_tokens_b[0].line, new_tokens_b[-1].line),
                a_file=current.a_file,
                b_file=current.b_file,
                a_token_idx=current.a_token_idx,
                b_token_idx=current.b_token_idx,
            )
        else:
            merged.append(current)
            current = nxt

    merged.append(current)
    merged.sort(key=lambda t: -t.length)
    return merged


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
            "a_file": t.a_file,
            "b_file": t.b_file,
            "a_token_idx": t.a_token_idx,
            "b_token_idx": t.b_token_idx,
            "matched_text_a": " ".join(tok.raw for tok in t.tokens_a),
            "matched_text_b": " ".join(tok.raw for tok in t.tokens_b),
        })
    return out
