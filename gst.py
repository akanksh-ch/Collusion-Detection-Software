"""Greedy String Tiling (GST) — classical local alignment for line-level
evidence generation.

Implements the Wise (1993) / Prechelt et al. (2002) algorithm used inside
JPlag.  Operates on language-agnostic token sequences with physical line
number tracking so that matched tile regions can be mapped back to exact
source ranges for the forensic diff sidebar.
"""

from __future__ import annotations

import logging
import re
import time
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

class _NextUnmarked:
    """Union-find with path compression, used to skip already-marked
    token positions in amortized ~O(1) instead of walking them one at a
    time on every outer-loop pass.

    ``find(i)`` returns the smallest unmarked index >= i (or n if none
    remain). ``mark(i)`` records that index i is now marked, so future
    ``find`` calls jump straight past it. Without this, a pair that
    needs H outer-loop passes to fully tile pays O(n) per pass just to
    walk past previously-marked tokens — O(n * H) total. With it, the
    total cost of all skipping across every pass combined is amortized
    O(n) for the whole match, regardless of H.
    """
    __slots__ = ("parent",)

    def __init__(self, n: int):
        # index n is a sentinel representing "past the end"
        self.parent = list(range(n + 1))

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def mark(self, i: int) -> None:
        self.parent[i] = self.find(i + 1)


def greedy_string_tiling(
    tokens_a: List[Token],
    tokens_b: List[Token],
    min_match_length: int = 8,
    max_bucket_size: int = 60,
    max_outer_iterations: int = 2000,
    max_seconds: float = 20.0,
    pair_label: str = "",
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

    Three safety valves against pathological worst cases (near-duplicate
    pairs and boilerplate-heavy corpora):

    - Skip pointers (``_NextUnmarked``): the outer loop used to rescan
      every position ``i`` from 0..n on *every* pass, stepping past
      already-marked tokens one at a time. For a pair needing H passes
      that's O(n * H) just to walk past marked tokens — on a
      near-duplicate pair (exactly what this tool exists to catch)
      that can mean hundreds of passes, each rescanning the full
      stream, and it's invisible in a per-N-pairs progress log because
      one slow pair blends into the averaged rate. A union-find
      "next unmarked" pointer with path compression makes the total
      cost of skipping marked tokens across all passes combined
      amortized O(n) for the whole match, regardless of H.
    - ``max_bucket_size``: an n-gram shared by more than this many
      positions in B is treated as boilerplate rather than distinctive
      evidence and is skipped entirely for matching. This is not just
      a perf hack — an n-gram common to dozens of unrelated submissions
      (e.g. shared imports, `public static void main`) isn't real
      similarity signal anyway, so dropping it also slightly improves
      precision, not just speed.
    - ``max_outer_iterations`` / ``max_seconds``: hard caps on how many
      times the find-longest-then-mark outer loop can run, and on
      total wall-clock time, for a single pair. Iteration count alone
      doesn't protect against a single huge ``n`` — a pair could still
      take a long time before hitting the iteration cap — so wall
      clock is checked too (every 25 iterations, to keep the check
      itself cheap). Hitting either cap logs a warning (including
      ``pair_label`` if provided) and returns whatever tiles have been
      found so far rather than continuing — this should only ever
      trigger on genuinely pathological pairs, and a slightly
      incomplete tile set for one outlier pair is far preferable to
      stalling the entire corpus run.
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

    # Prebuild index for B, dropping over-common ("boilerplate") n-grams
    b_index = {}
    dropped_ngrams = 0
    for j in range(m - min_match_length + 1):
        ngram = tuple(vals_b[j : j + min_match_length])
        if ngram not in b_index:
            b_index[ngram] = []
        b_index[ngram].append(j)

    for ngram, positions in list(b_index.items()):
        if len(positions) > max_bucket_size:
            dropped_ngrams += 1
            del b_index[ngram]

    if dropped_ngrams:
        logging.debug(
            f"[GST] dropped {dropped_ngrams} boilerplate n-gram bucket(s) "
            f"exceeding max_bucket_size={max_bucket_size}"
        )

    skip_a = _NextUnmarked(n)
    skip_b = _NextUnmarked(m)

    start_time = time.monotonic()
    outer_iterations = 0
    label_suffix = f" [{pair_label}]" if pair_label else ""
    while True:
        outer_iterations += 1
        if outer_iterations > max_outer_iterations:
            logging.warning(
                f"[GST] hit max_outer_iterations={max_outer_iterations} on a "
                f"pair (n={n}, m={m}){label_suffix} — likely a near-duplicate "
                f"or highly repetitive pair. Returning {len(tiles)} tile(s) "
                f"found so far rather than continuing indefinitely."
            )
            break

        # Wall-clock backstop, checked periodically (not every iteration)
        # so the time.monotonic() call itself doesn't add meaningful
        # overhead on pairs with many cheap passes.
        if outer_iterations % 25 == 0:
            elapsed = time.monotonic() - start_time
            if elapsed > max_seconds:
                logging.warning(
                    f"[GST] hit max_seconds={max_seconds:.0f}s on a pair "
                    f"(n={n}, m={m}){label_suffix} after {outer_iterations} "
                    f"passes — likely a near-duplicate or highly repetitive "
                    f"pair. Returning {len(tiles)} tile(s) found so far "
                    f"rather than continuing indefinitely."
                )
                break

        max_match = min_match_length
        matches = []

        i = skip_a.find(0)
        while i < n - max_match + 1:
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
            i = skip_a.find(i + 1)

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
                skip_a.mark(i + d)
                skip_b.mark(j + d)

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
        first_a, last_a = t.tokens_a[0], t.tokens_a[-1]
        first_b, last_b = t.tokens_b[0], t.tokens_b[-1]
        out.append({
            "length": t.length,
            "a_lines": list(t.a_lines),
            "b_lines": list(t.b_lines),
            "a_file": t.a_file,
            "b_file": t.b_file,
            "a_token_idx": t.a_token_idx,
            "b_token_idx": t.b_token_idx,
            # Precise column bounds of the *first* and *last* matched token
            # on each side, so highlighting can wrap the exact matched span
            # instead of defaulting to whole lines.
            "a_col_start": first_a.col,
            "a_col_end": last_a.col + len(last_a.raw),
            "b_col_start": first_b.col,
            "b_col_end": last_b.col + len(last_b.raw),
            "matched_text_a": " ".join(tok.raw for tok in t.tokens_a),
            "matched_text_b": " ".join(tok.raw for tok in t.tokens_b),
        })
    return out
