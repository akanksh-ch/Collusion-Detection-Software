import os
import re
import glob
import shutil
import subprocess
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed


# ---------------------------------------------------------------------------
# Comment-stripping regex patterns (Java / C / C++)
# ---------------------------------------------------------------------------
# Order matters: strings first to avoid stripping inside string literals.
_COMMENT_PATTERN = re.compile(
    r'(?P<string>"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\')'   # quoted strings
    r"|(?P<block>/\*[\s\S]*?\*/)"                            # block / Javadoc
    r"|(?P<line>//[^\n]*)",                                   # line comments
    re.MULTILINE,
)


def _strip_comments(source_text):
    """Return (stripped_code, extracted_comments) from raw source text.

    String literals are preserved verbatim.  Block and line comments are
    replaced with whitespace of equivalent line-count so that physical line
    numbers remain stable for downstream GST mapping.
    """
    comments = []

    def _replacer(m):
        if m.group("string"):
            return m.group("string")
        comment_text = m.group("block") or m.group("line")
        comments.append(comment_text)
        # Preserve line count so line numbers stay consistent
        newline_count = comment_text.count("\n")
        return "\n" * newline_count

    stripped = _COMMENT_PATTERN.sub(_replacer, source_text)
    return stripped, "\n".join(comments)


def _preprocess_single_source(args):
    """Strip comments from a single source file and write sidecar outputs."""
    src_path, stripped_dir, comments_dir = args
    basename = os.path.basename(src_path)
    stem, ext = os.path.splitext(basename)

    try:
        with open(src_path, "r", encoding="utf-8", errors="replace") as f:
            raw = f.read()

        stripped_code, comment_stream = _strip_comments(raw)

        stripped_path = os.path.join(stripped_dir, basename)
        with open(stripped_path, "w", encoding="utf-8") as f:
            f.write(stripped_code)

        comments_path = os.path.join(comments_dir, f"{stem}_comments.txt")
        with open(comments_path, "w", encoding="utf-8") as f:
            f.write(comment_stream)

        return stem, "OK"
    except Exception as e:
        return stem, f"PREPROCESS_FAIL: {e}"


def _parse_single_submission(args):
    """Isolated worker function executed inside an independent process core."""
    path, output_dir, parse_bin, export_bin = args

    # Extract file/folder stem dynamically to preserve native naming
    base_file_name = os.path.basename(path)
    student_id, _ = os.path.splitext(base_file_name)

    cpg_bin_path = os.path.join(output_dir, f"{student_id}_cpg.bin")
    temp_stage_dir = os.path.join(output_dir, f"{student_id}_tmp_stage")
    final_graphml_path = os.path.join(output_dir, f"{student_id}.graphml")

    if os.path.exists(final_graphml_path):
        return student_id, "CACHED"

    try:
        # Import NetworkX inside the worker process to avoid global context serialization locks
        import networkx as nx

        # 1. Compile source asset directly using Joern's native frontend auto-detection
        subprocess.run([parse_bin, path, "--output", cpg_bin_path],
                       check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # 2. Export CPG slice structures to raw XML datasets
        subprocess.run([
            export_bin, cpg_bin_path,
            "--repr", "cpg", "--format", "graphml", "--out", temp_stage_dir
        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # 3. Collapse XML slices into unified multi-relational graphs
        xml_files = glob.glob(os.path.join(temp_stage_dir, "**", "export.xml"), recursive=True)
        if xml_files:
            unified_graph = nx.MultiDiGraph()
            for xml_file in xml_files:
                try:
                    sub_graph = nx.read_graphml(xml_file)
                    unified_graph = nx.compose(unified_graph, sub_graph)
                except Exception:
                    continue

            nx.write_graphml(unified_graph, final_graphml_path)
            status = "SUCCESS"
        else:
            status = "EMPTY_EXPORT"

    except subprocess.CalledProcessError as e:
        status = f"FAILED_JOERN: {e.stderr.decode('utf-8', errors='ignore').strip()[:100]}"
    except Exception as e:
        status = f"FAILED_COMPOSITION: {str(e)}"
    finally:
        # Strict garbage collection cleanup of binary assets to protect disk I/O channels
        if os.path.exists(cpg_bin_path):
            os.remove(cpg_bin_path)
        if os.path.exists(temp_stage_dir):
            shutil.rmtree(temp_stage_dir)

    return student_id, status


class JoernAutomationParser:
    """Automates multi-processed asynchronous Joern compilation pipelines across
    all available hardware process cores.

    New in this revision: a ``preprocess_submissions`` pass strips natural-
    language comments into isolated sidecar files before Joern ingestion,
    keeping the CPG free of comment noise while preserving comment metadata
    for downstream stylometric analysis.
    """
    def __init__(self, joern_path=""):
        self.parse_bin = os.path.join(joern_path, "joern-parse") if joern_path else "joern-parse"
        self.export_bin = os.path.join(joern_path, "joern-export") if joern_path else "joern-export"

    # ------------------------------------------------------------------
    # Public helpers for downstream access to stripped / comment paths
    # ------------------------------------------------------------------
    @staticmethod
    def stripped_source_dir(workspace_dir):
        """Return the directory holding comment-stripped source files."""
        return os.path.join(workspace_dir, "_stripped")

    @staticmethod
    def comments_dir(workspace_dir):
        """Return the directory holding extracted comment-stream sidecars."""
        return os.path.join(workspace_dir, "_comments")

    @staticmethod
    def get_stripped_source_path(workspace_dir, student_id, ext=".java"):
        return os.path.join(workspace_dir, "_stripped", f"{student_id}{ext}")

    @staticmethod
    def get_comments_path(workspace_dir, student_id):
        return os.path.join(workspace_dir, "_comments", f"{student_id}_comments.txt")

    # ------------------------------------------------------------------
    # Step A: comment stripping preprocessor
    # ------------------------------------------------------------------
    def preprocess_submissions(self, source_dir, workspace_dir):
        """Strip comments from every source file in *source_dir*.

        Outputs:
            ``<workspace_dir>/_stripped/<filename>``  – code without comments
            ``<workspace_dir>/_comments/<stem>_comments.txt``  – isolated comments
        """
        stripped_dir = self.stripped_source_dir(workspace_dir)
        comments_dir = self.comments_dir(workspace_dir)
        os.makedirs(stripped_dir, exist_ok=True)
        os.makedirs(comments_dir, exist_ok=True)

        IGNORED_EXTENSIONS = ('.md', '.txt', '.json', '.yml', '.yaml', '.xml',
                              '.html', '.gitignore')

        tasks = []
        for item in os.listdir(source_dir):
            if item.startswith('.'):
                continue
            full = os.path.join(source_dir, item)
            if os.path.isfile(full) and not item.lower().endswith(IGNORED_EXTENSIONS):
                tasks.append((full, stripped_dir, comments_dir))

        if not tasks:
            logging.warning("[PREPROCESS] No source files found for comment stripping.")
            return False

        max_workers = max(1, os.cpu_count() - 1)
        ok = 0
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_preprocess_single_source, t): t for t in tasks}
            for future in as_completed(futures):
                stem, status = future.result()
                if status == "OK":
                    ok += 1
                else:
                    logging.error(f"[PREPROCESS][{stem}] {status}")

        logging.info(f"[PREPROCESS] Stripped comments from {ok}/{len(tasks)} files.")
        return ok > 0

    # ------------------------------------------------------------------
    # Step B: Joern compilation (feeds *stripped* source)
    # ------------------------------------------------------------------
    def process_submission_folder(self, source_dir, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        if not os.path.exists(source_dir):
            logging.error(f"Source directory '{source_dir}' does not exist.")
            return False

        items = [os.path.join(source_dir, x) for x in os.listdir(source_dir)]

        # Filter out common environmental, documentation, and metadata files
        IGNORED_EXTENSIONS = ('.md', '.txt', '.json', '.yml', '.yaml', '.xml',
                              '.html', '.gitignore')

        submissions = []
        for x in items:
            base = os.path.basename(x)

            # Skip hidden files/directories (like .DS_Store, .git, or .joern)
            if base.startswith('.'):
                continue

            # Skip documentation or non-code configurations if it's a plain file
            if os.path.isfile(x) and base.lower().endswith(IGNORED_EXTENSIONS):
                continue

            submissions.append(x)

        if not submissions:
            logging.warning(f"No valid source submissions discovered inside {source_dir}.")
            return False

        logging.info(f"[PARALLEL ORCHESTRATION] Submitting {len(submissions)} targets to CPU Process Pool...")

        # Safe Concurrency Ceiling: Spawns 1 less worker than total CPU cores to protect OS stability
        max_workers = max(1, os.cpu_count() - 1)
        worker_tasks = [
            (path, output_dir, self.parse_bin, self.export_bin)
            for path in submissions
        ]

        success_count = 0
        cached_count = 0

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_parse_single_submission, task): task for task in worker_tasks}

            for future in as_completed(futures):
                student_id, status = future.result()
                if status == "SUCCESS":
                    success_count += 1
                    logging.info(f"[{student_id}] Compilation successful.")
                elif status == "CACHED":
                    cached_count += 1
                else:
                    logging.error(f"[{student_id}] Processing pipeline crashed: {status}")

        logging.info(f"[EXECUTION COMPLETE] Compiled: {success_count}, Cached skips: {cached_count} across {max_workers} active worker cores.")
        return True
