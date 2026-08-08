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


def compute_gst_coverage(submission_paths: list[str], min_match_length=5) -> np.ndarray:
    # parsing: load each submission's source and tokenize it directly from the paths
    token_lists = []
    for path_str in submission_paths:
        path = Path(path_str)
        # Handle both single files and directories dynamically
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
    n = len(submission_paths)
    coverage = np.eye(n)
    
    # Calculate coverage between every pair of submissions
    for i in range(n):
        for j in range(i + 1, n):
            tiles = match(encoded[i], "", encoded[j], "", min_match_length * BYTES_PER_TOKEN)
            matched_tokens = sum(length for _, _, length in tiles) / BYTES_PER_TOKEN
            
            # Normalize the score based on the total token counts of both documents
            score = 2 * matched_tokens / (len(token_lists[i]) + len(token_lists[j]))
            
            # Fill matrix symmetrically
            coverage[i, j] = coverage[j, i] = min(1.0, max(0.0, score))

    return coverage


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compute pairwise GST coverage matrix")
    parser.add_argument("path", help="Directory containing one file or subdirectory per submission")
    parser.add_argument("--min-match-length", type=int, default=5,
                         help="Minimum matched token run to count as a tile (default: 5)")
    args = parser.parse_args()
    
    # Build absolute string paths to mimic the JPlag-style orchestrator input
    root_path = Path(args.path)
    submission_paths = sorted([str(p.absolute()) for p in root_path.iterdir() if p.name != ".DS_Store"])
    
    result = compute_gst_coverage(submission_paths, min_match_length=args.min_match_length)
    
    submission_ids = [Path(p).name for p in submission_paths]
    print(submission_ids)
    print(result)
