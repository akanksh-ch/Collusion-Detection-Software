"""
Generates one TF-IDF char n-gram embedding per submission, sharing a single vocabulary across the whole corpus so rows are directly comparable.
"""

import os
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer


def generate_embeddings(submission_paths: list[str], ignore_ext=None, min_df=2):
    # parsing: resolve each submission (file or directory) into one combined document, crawling directories and ingesting every non-ignored file underneath them
    ignore_ext = set(ignore_ext) if ignore_ext else {".txt", ".md", ".csv"}
    docs = []
    for path_str in submission_paths:
        path = Path(path_str)
        files = [path] if path.is_file() else [
            Path(root) / name
            for root, _dirs, names in os.walk(path)
            for name in names
            if (Path(root) / name).suffix.lower() not in ignore_ext
        ]
        text = "\n".join(f.read_text(encoding="utf-8", errors="replace") for f in sorted(files) if f.is_file())
        docs.append(text)

    # combining: fit one shared TF-IDF vocabulary across every submission in a single call, so all rows live in the same feature space and are comparable
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=min_df,
        lowercase=False,
    )
    sparse_embeddings = vectorizer.fit_transform(docs)

    # embedding: densify into a plain (N, D) numpy array to match the other signal matrices used downstream in SNF fusion
    embeddings = sparse_embeddings.toarray()

    return embeddings


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate char n-gram TF-IDF embeddings")
    parser.add_argument("path", help="Root directory containing one file or subdirectory per submission")
    parser.add_argument("--ignore-ext", nargs="*", default=None,
                         help="Extensions to skip (default: .txt .md .csv)")
    parser.add_argument("--min-df", type=int, default=2,
                         help="Minimum document frequency for an n-gram (default: 2)")
    args = parser.parse_args()

    # Build absolute string paths to mimic the JPlag-style orchestrator input
    root_path = Path(args.path)
    submission_paths = sorted([str(p.absolute()) for p in root_path.iterdir() if p.name != ".DS_Store"])

    result = generate_embeddings(submission_paths, ignore_ext=args.ignore_ext, min_df=args.min_df)

    submission_ids = [Path(p).name for p in submission_paths]
    print(submission_ids)
    print(result.shape)
