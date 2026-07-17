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
_COMMENT_PATTERN = re.compile(
    r'(?P<string>"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\')'   # quoted strings
    r"|(?P<block>/\*[\s\S]*?\*/)"                            # block / Javadoc
    r"|(?P<line>//[^\n]*)",                                   # line comments
    re.MULTILINE,
)


def _strip_comments(source_text):
    """Return (stripped_code, extracted_comments) from raw source text."""
    comments = []

    def _replacer(m):
        if m.group("string"):
            return m.group("string")
        comment_text = m.group("block") or m.group("line")
        comments.append(comment_text)
        newline_count = comment_text.count("\n")
        return "\n" * newline_count

    stripped = _COMMENT_PATTERN.sub(_replacer, source_text)
    return stripped, "\n".join(comments)


def _preprocess_single_submission_worker(args):
    """Strip comments from a single submission and write structured sidecar outputs."""
    src_path, stripped_dir, comments_dir, rel_id = args
    
    stem = rel_id
    target_code_path = os.path.join(stripped_dir, rel_id)
    IGNORED_EXTENSIONS = ('.md', '.txt', '.json', '.yml', '.yaml', '.xml', '.html', '.gitignore')
    all_comments = []

    try:
        if os.path.isfile(src_path):
            os.makedirs(os.path.dirname(target_code_path), exist_ok=True)
            with open(src_path, "r", encoding="utf-8", errors="replace") as f:
                raw = f.read()

            stripped_code, comment_stream = _strip_comments(raw)
            with open(target_code_path, "w", encoding="utf-8") as f:
                f.write(stripped_code)
            all_comments.append(comment_stream)
            
        elif os.path.isdir(src_path):
            os.makedirs(target_code_path, exist_ok=True)
            for root, dirs, files in os.walk(src_path):
                rel_sub = os.path.relpath(root, src_path)
                curr_target_dir = os.path.join(target_code_path, rel_sub) if rel_sub != '.' else target_code_path
                os.makedirs(curr_target_dir, exist_ok=True)

                for f_name in files:
                    if f_name.startswith('.') or f_name.lower().endswith(IGNORED_EXTENSIONS):
                        continue
                    f_path = os.path.join(root, f_name)
                    with open(f_path, "r", encoding="utf-8", errors="replace") as f:
                        raw = f.read()

                    stripped_code, comment_stream = _strip_comments(raw)
                    if comment_stream.strip():
                        all_comments.append(comment_stream)

                    with open(os.path.join(curr_target_dir, f_name), "w", encoding="utf-8") as f:
                        f.write(stripped_code)

        comments_path = os.path.join(comments_dir, f"{rel_id}_comments.txt")
        os.makedirs(os.path.dirname(comments_path), exist_ok=True)
        with open(comments_path, "w", encoding="utf-8") as f:
            f.write("\n".join(all_comments))

        return stem, "OK"
    except Exception as e:
        return stem, f"PREPROCESS_FAIL: {e}"


def _parse_single_submission(args):
    """Isolated worker function executed inside an independent process core."""
    path, output_dir, parse_bin, export_bin, rel_id = args

    student_id = rel_id
    safe_id = rel_id.replace('/', '_').replace('\\', '_')
    
    cpg_bin_path = os.path.join(output_dir, f"{safe_id}_cpg.bin")
    temp_stage_dir = os.path.join(output_dir, f"{safe_id}_tmp_stage")
    final_graphml_path = os.path.join(output_dir, f"{student_id}.graphml")

    if os.path.exists(final_graphml_path):
        return student_id, "CACHED"

    try:
        import networkx as nx
        os.makedirs(os.path.dirname(final_graphml_path), exist_ok=True)

        subprocess.run([parse_bin, path, "--output", cpg_bin_path],
                       check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        subprocess.run([
            export_bin, cpg_bin_path,
            "--repr", "cpg", "--format", "graphml", "--out", temp_stage_dir
        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

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
        if os.path.exists(cpg_bin_path):
            os.remove(cpg_bin_path)
        if os.path.exists(temp_stage_dir):
            shutil.rmtree(temp_stage_dir)

    return student_id, status


class JoernAutomationParser:
    """Automates multi-processed asynchronous Joern compilation pipelines."""
    
    def __init__(self, joern_path=""):
        self.parse_bin = os.path.join(joern_path, "joern-parse") if joern_path else "joern-parse"
        self.export_bin = os.path.join(joern_path, "joern-export") if joern_path else "joern-export"

    @staticmethod
    def stripped_source_dir(workspace_dir):
        return os.path.join(workspace_dir, "_stripped")

    @staticmethod
    def comments_dir(workspace_dir):
        return os.path.join(workspace_dir, "_comments")

    @staticmethod
    def get_stripped_source_path(workspace_dir, student_id, ext=".java"):
        return os.path.join(workspace_dir, "_stripped", f"{student_id}{ext}")

    @staticmethod
    def get_comments_path(workspace_dir, student_id):
        return os.path.join(workspace_dir, "_comments", f"{student_id}_comments.txt")

    def _discover_submission_paths(self, source_dir: str) -> list[tuple[str, str]]:
        """Recursively parses directories to capture relative structural group paths."""
        CODE_EXTENSIONS = ('.java', '.c', '.cpp', '.py', '.h')
        submission_targets = []
        base_abs = os.path.abspath(source_dir)
        queue = [base_abs]
        
        while queue:
            current_dir = queue.pop(0)
            try:
                items = os.listdir(current_dir)
            except OSError:
                continue
                
            has_code_files = False
            sub_dirs = []
            
            for item in items:
                if item.startswith('.'):
                    continue
                full_path = os.path.join(current_dir, item)
                
                if os.path.isdir(full_path):
                    sub_dirs.append(full_path)
                elif os.path.isfile(full_path):
                    ext = os.path.splitext(item.lower())[1]
                    if ext in CODE_EXTENSIONS:
                        has_code_files = True
                        
            if has_code_files or not sub_dirs:
                if current_dir != base_abs or has_code_files:
                    rel_path = os.path.relpath(current_dir, base_abs)
                    rel_id = rel_path.replace(os.sep, "/")
                    submission_targets.append((current_dir, rel_id))
            else:
                queue.extend(sub_dirs)
                
        return submission_targets

    def preprocess_submissions(self, source_dir, workspace_dir):
        """Strip comments from deep true submission directories."""
        stripped_dir = self.stripped_source_dir(workspace_dir)
        comments_dir = self.comments_dir(workspace_dir)
        os.makedirs(stripped_dir, exist_ok=True)
        os.makedirs(comments_dir, exist_ok=True)

        discovered_targets = self._discover_submission_paths(source_dir)

        tasks = []
        for abs_path, rel_id in discovered_targets:
            tasks.append((abs_path, stripped_dir, comments_dir, rel_id))

        if not tasks:
            logging.warning("[PREPROCESS] No source files found for comment stripping.")
            return False

        max_workers = max(1, os.cpu_count() - 1)
        ok = 0
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_preprocess_single_submission_worker, t): t for t in tasks}
            for future in as_completed(futures):
                stem, status = future.result()
                if status == "OK":
                    ok += 1
                else:
                    logging.error(f"[PREPROCESS][{stem}] {status}")

        logging.info(f"[PREPROCESS] Stripped comments from {ok}/{len(tasks)} targets.")
        return ok > 0

    def process_submission_folder(self, source_dir, output_dir):
        """Processes structures inside the comment-stripped staging folder location."""
        os.makedirs(output_dir, exist_ok=True)
        if not os.path.exists(source_dir):
            logging.error(f"Source directory '{source_dir}' does not exist.")
            return False

        discovered_targets = self._discover_submission_paths(source_dir)

        if not discovered_targets:
            logging.warning(f"No valid source submissions discovered inside {source_dir}.")
            return False

        logging.info(f"[PARALLEL ORCHESTRATION] Submitting {len(discovered_targets)} targets to CPU Process Pool...")

        max_workers = max(1, os.cpu_count() - 1)
        worker_tasks = [
            (abs_path, output_dir, self.parse_bin, self.export_bin, rel_id)
            for abs_path, rel_id in discovered_targets
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
