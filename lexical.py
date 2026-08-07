"""
Generates TF-IDF character n-gram (char_wb, 3-5, min_df>=2) embeddings
for a single file or all text files found recursively in a directory.
"""

import os
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer


def generate_embeddings(path, ignore_ext=None, min_df=2):

    # Step 1: resolve input path into a list of files
    ignore_ext = ignore_ext or {".txt", ".md", ".csv"}
    input_path = Path(path)
    if input_path.is_file():
        files = [input_path]
    else:
        files = [
            Path(root) / name
            for root, _dirs, names in os.walk(input_path)
            for name in names
            if (Path(root) / name).suffix.lower() not in ignore_ext
        ]

    # Step 2: read and combine file contents into documents
    docs = []
    for fp in sorted(files):
        text = fp.read_text(encoding="utf-8", errors="replace")
        if text.strip():
            docs.append(text)

    # Step 3: generate char n-gram TF-IDF embeddings
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=min_df,
        lowercase=False,
    )
    embeddings = vectorizer.fit_transform(docs)

    return embeddings


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate char n-gram TF-IDF embeddings")
    parser.add_argument("path", help="File or directory to embed")
    parser.add_argument("--ignore-ext", nargs="*", default=None,
                         help="Extensions to skip (default: .txt .md .csv)")
    parser.add_argument("--min-df", type=int, default=2,
                         help="Minimum document frequency for an n-gram (default: 2)")
    args = parser.parse_args()

    ignore_ext = set(args.ignore_ext) if args.ignore_ext else None
    result = generate_embeddings(args.path, ignore_ext=ignore_ext, min_df=args.min_df)
    print(result.shape)
