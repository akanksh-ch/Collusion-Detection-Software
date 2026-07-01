import os
import glob
import shutil
import subprocess
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

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
    all available hardware process cores."""
    def __init__(self, joern_path=""):
        self.parse_bin = os.path.join(joern_path, "joern-parse") if joern_path else "joern-parse"
        self.export_bin = os.path.join(joern_path, "joern-export") if joern_path else "joern-export"

    def process_submission_folder(self, source_dir, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        if not os.path.exists(source_dir):
            logging.error(f"Source directory '{source_dir}' does not exist.")
            return False

        items = [os.path.join(source_dir, x) for x in os.listdir(source_dir)]
        
        # Filter out common environmental, documentation, and metadata files
        IGNORED_EXTENSIONS = ('.md', '.txt', '.json', '.yml', '.yaml', '.xml', '.html', '.gitignore')
        
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
