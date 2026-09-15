"""
Computes an NxN pairwise GST (Greedy String Tiling) coverage matrix for a set of code submissions, using the apluslms/greedy-string-tiling library, and can optionally also return the underlying tile positions as JPlag-style file/line/column match spans.
"""

import re
from pathlib import Path

import numpy as np
from gst import match

TOKEN_RE = re.compile(r"\w+|[^\w\s]")
PUA_START = 0xE100      # private-use-area codepoints, one per distinct token
BYTES_PER_TOKEN = 3     # match() indexes in UTF-8 bytes; PUA chars are always 3 bytes


def tokenize_with_positions(path_str: str) -> tuple[list[str], list[tuple[str, int, int]]]:
    # parsing: load every file under a submission and tokenize it, tracking each token's (filename, line, column) alongside its text so tile offsets can be mapped back to a real source location later
    path = Path(path_str)
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    tokens, positions = [], []
    for f in files:
        text = f.read_text(encoding="utf-8", errors="replace")
        for m in TOKEN_RE.finditer(text):
            tokens.append(m.group())
            line = text.count("\n", 0, m.start()) + 1
            last_newline = text.rfind("\n", 0, m.start())
            column = m.start() - last_newline
            positions.append((f.name, line, column))
    return tokens, positions


def compute_gst_coverage(submission_paths: list[str], min_match_length: int = 5, return_matches: bool = False):
    # parsing: load and tokenize every submission once, keeping position data alongside since it's needed for match spans and costs next to nothing on top of tokenizing alone
    token_lists = []
    position_lists = []
    for path_str in submission_paths:
        tokens, positions = tokenize_with_positions(path_str)
        token_lists.append(tokens)
        position_lists.append(positions)

    # encoding: map tokens to single-codepoint characters so GST matches whole tokens, not raw characters
    vocab = {}
    encoded = []
    for tokens in token_lists:
        chars = []
        for tok in tokens:
            if tok not in vocab:
                vocab[tok] = chr(PUA_START + len(vocab))
            chars.append(vocab[tok])
        encoded.append("".join(chars))

    # combining: run GST on every pair and convert matched byte-length back to matched token-count
    n = len(submission_paths)
    coverage = np.eye(n)
    matches_by_pair = {}

    for i in range(n):
        for j in range(i + 1, n):
            tiles = match(encoded[i], "", encoded[j], "", min_match_length * BYTES_PER_TOKEN)
            matched_tokens = sum(length for _, _, length in tiles) / BYTES_PER_TOKEN

            # normalizing: score based on the total token counts of both documents
            score = 2 * matched_tokens / (len(token_lists[i]) + len(token_lists[j]))
            coverage[i, j] = coverage[j, i] = min(1.0, max(0.0, score))

            # mapping: only pay for tile-to-span conversion when the caller actually wants match data, since coverage-only callers (e.g. pipeline.py's fusion step) don't need it
            if return_matches and tiles:
                pair_matches = []
                for start_i_bytes, start_j_bytes, length_bytes in tiles:
                    tok_i, tok_j = start_i_bytes // BYTES_PER_TOKEN, start_j_bytes // BYTES_PER_TOKEN
                    n_tok = length_bytes // BYTES_PER_TOKEN
                    first_file, first_line, first_col = position_lists[i][tok_i]
                    last_file, last_line, last_col = position_lists[i][tok_i + n_tok - 1]
                    second_file, second_line, second_col = position_lists[j][tok_j]
                    second_last_file, second_last_line, second_last_col = position_lists[j][tok_j + n_tok - 1]
                    pair_matches.append({
                        "firstFileName": first_file, "secondFileName": second_file,
                        "startInFirst": {"line": first_line, "column": first_col},
                        "endInFirst": {"line": last_line, "column": last_col},
                        "startInSecond": {"line": second_line, "column": second_col},
                        "endInSecond": {"line": second_last_line, "column": second_last_col},
                        "lengthOfFirst": n_tok, "lengthOfSecond": n_tok,
                    })
                matches_by_pair[(Path(submission_paths[i]).name, Path(submission_paths[j]).name)] = pair_matches

    if return_matches:
        return coverage, matches_by_pair
    return coverage


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compute pairwise GST coverage matrix")
    parser.add_argument("path", help="Directory containing one file or subdirectory per submission")
    parser.add_argument("--min-match-length", type=int, default=5,
                         help="Minimum matched token run to count as a tile (default: 5)")
    parser.add_argument("--show-matches", action="store_true", help="Also compute and print per-pair match spans, not just coverage")
    args = parser.parse_args()

    # Build absolute string paths to mimic the JPlag-style orchestrator input
    root_path = Path(args.path)
    submission_paths = sorted([str(p.absolute()) for p in root_path.iterdir() if p.name != ".DS_Store"])

    submission_ids = [Path(p).name for p in submission_paths]
    if args.show_matches:
        result, matches_by_pair = compute_gst_coverage(submission_paths, min_match_length=args.min_match_length, return_matches=True)
        print(submission_ids)
        print(result)
        for pair, matches in matches_by_pair.items():
            print(pair, f"{len(matches)} tiles")
    else:
        result = compute_gst_coverage(submission_paths, min_match_length=args.min_match_length)
        print(submission_ids)
        print(result)
