"""
Computes an NxN pairwise GST (Greedy String Tiling) coverage matrix for a
set of code submissions, using the apluslms/greedy-string-tiling library.
"""

import re
from pathlib import Path

import numpy as np
from gst import match

TOKEN_RE = re.compile(r"\w+|[^\w\s]")
PUA_START = 0xE100      # private-use-area codepoints, one per distinct token
BYTES_PER_TOKEN = 3     # match() indexes in UTF-8 bytes; PUA chars are always 3 bytes


def compute_gst_coverage(submission_ids, source_dir, min_match_length=5):
    base = Path(source_dir)

    # parsing: load each submission's source and tokenize it
    token_lists = []
    for sid in submission_ids:
        path = base / sid
        files = [path] if path.is_file() else sorted(path.rglob("*"))
        text = "\n".join(f.read_text(encoding="utf-8", errors="replace") for f in files if f.is_file())
        token_lists.append(TOKEN_RE.findall(text))

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
    n = len(submission_ids)
    coverage = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            tiles = match(encoded[i], "", encoded[j], "", min_match_length * BYTES_PER_TOKEN)
            matched_tokens = sum(length for _, _, length in tiles) / BYTES_PER_TOKEN
            score = 2 * matched_tokens / (len(token_lists[i]) + len(token_lists[j]))
            coverage[i, j] = coverage[j, i] = min(1.0, max(0.0, score))

    return coverage


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compute pairwise GST coverage matrix")
    parser.add_argument("path", help="Directory containing one file or subdirectory per submission")
    parser.add_argument("--min-match-length", type=int, default=5,
                         help="Minimum matched token run to count as a tile (default: 5)")
    args = parser.parse_args()
    submission_ids = sorted(p.name for p in Path(args.path).iterdir())
    result = compute_gst_coverage(submission_ids, args.path, min_match_length=args.min_match_length)
    print(submission_ids)
    print(result)
