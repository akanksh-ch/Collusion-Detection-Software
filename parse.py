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
    src_path, stripped_dir, comments_dir, stem = args
    
    IGNORED_EXTENSIONS = ('.md', '.txt', '.json', '.yml', '.yaml', '.xml', '.html', '.gitignore')
    all_comments = []

    try:
        if os.path.isfile(src_path):
            stripped_path = os.path.join(stripped_dir, stem)
            os.makedirs(os.path.dirname(stripped_path), exist_ok=True)
            with open(src_path, "r", encoding="utf-8", errors="replace") as f:
                raw = f.read()

            stripped_code, comment_stream = _strip_comments(raw)
            with open(stripped_path, "w", encoding="utf-8") as f:
                f.write(stripped_code)
            all_comments.append(comment_stream)
            
        elif os.path.isdir(src_path):
            target_dir = os.path.join(stripped_dir, stem)
            os.makedirs(target_dir, exist_ok=True)
            
            for root, dirs, files in os.walk(src_path):
                rel_path = os.path.relpath(root, src_path)
                curr_target_dir = os.path.join(target_dir, rel_path) if rel_path != '.' else target_dir
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

        comments_path = os.path.join(comments_dir, f"{stem}_comments.txt")
        os.makedirs(os.path.dirname(comments_path), exist_ok=True)
        with open(comments_path, "w", encoding="utf-8") as f:
            f.write("\n".join(all_comments))

        return stem, "OK"
    except Exception as e:
        return stem, f"PREPROCESS_FAIL: {e}"


def _parse_single_submission(args):
    """Isolated worker function executed inside an independent process core."""
    path, output_dir, parse_bin, export_bin = args

    base_file_name = os.path.basename(path)
    student_id, _ = os.path.splitext(base_file_name)
    if os.path.isdir(path):
        student_id = base_file_name

    cpg_bin_path = os.path.join(output_dir, f"{student_id}_cpg.bin")
    temp_stage_dir = os.path.join(output_dir, f"{student_id}_tmp_stage")
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

    def preprocess_submissions(self, source_dirs, workspace_dir):
        if isinstance(source_dirs, str):
            source_dirs = [source_dirs]

        stripped_dir = self.stripped_source_dir(workspace_dir)
        comments_dir = self.comments_dir(workspace_dir)
        os.makedirs(stripped_dir, exist_ok=True)
        os.makedirs(comments_dir, exist_ok=True)

        IGNORED_EXTENSIONS = ('.md', '.txt', '.json', '.yml', '.yaml', '.xml', '.html', '.gitignore')
        tasks = []
        
        # Loop through all provided root directories
        for src_dir in source_dirs:
            if not os.path.exists(src_dir):
                logging.warning(f"[PREPROCESS] Source directory '{src_dir}' does not exist.")
                continue
                
            root_name = os.path.basename(os.path.normpath(src_dir))
            
            # Every immediate child is treated as a submission exactly like the original code
            for item in os.listdir(src_dir):
                if item.startswith('.'):
                    continue
                    
                full = os.path.join(src_dir, item)
                
                if os.path.isdir(full) or (os.path.isfile(full) and not item.lower().endswith(IGNORED_EXTENSIONS)):
                    # Mimic JPlag: Prefix with root folder name if comparing multiple roots (e.g. orig_o1-wqfqn)
                    if len(source_dirs) > 1:
                        stem = f"{root_name}_{item}"
                    else:
                        stem = item
                        
                    tasks.append((full, stripped_dir, comments_dir, stem))

        if not tasks:
            logging.warning("[PREPROCESS] No valid submissions found to strip.")
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
        """Processes structures from the comment-stripped staging folder."""
        os.makedirs(output_dir, exist_ok=True)
        if not os.path.exists(source_dir):
            logging.error(f"Source directory '{source_dir}' does not exist.")
            return False

        # Read directly from the flattened _stripped directory
        items = [os.path.join(source_dir, x) for x in os.listdir(source_dir)]

        IGNORED_EXTENSIONS = ('.md', '.txt', '.json', '.yml', '.yaml', '.xml', '.html', '.gitignore')
        submissions = []
        for x in items:
            base = os.path.basename(x)
            if base.startswith('.'):
                continue
            if os.path.isfile(x) and base.lower().endswith(IGNORED_EXTENSIONS):
                continue
            submissions.append(x)

        if not submissions:
            logging.warning(f"No valid source submissions discovered inside {source_dir}.")
            return False

        logging.info(f"[PARALLEL ORCHESTRATION] Submitting {len(submissions)} targets to CPU Process Pool...")

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
