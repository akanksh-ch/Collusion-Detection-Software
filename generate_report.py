"""Generates a standalone JPlag-compatible forensic archive.

Produces a `.jplag` zip file conforming to the official JPlag 5.1.0 format
so the results can be natively viewed in the official JPlag viewer.
"""

from __future__ import annotations

import json
import os
import zipfile

from gst import tokenize_source


class JPlagReportGenerator:
    """Generates a standalone `.jplag` zip archive."""

    @staticmethod
    def write_report(
        report_data: list[dict],
        families: dict[str, list[str]],
        source_texts: dict[str, str] | None = None,
        output_path: str = "result.jplag",
    ):
        if os.path.isdir(output_path) or output_path.endswith("/") or output_path.endswith("\\"):
            os.makedirs(output_path, exist_ok=True)
            output_path = os.path.join(output_path, "result.jplag")
        elif not output_path.endswith(".jplag"):
            output_path += ".jplag"

        # Handle directory vs file path
        parent_dir = os.path.dirname(output_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        if source_texts is None:
            source_texts = {}

        # ── Build data payload ───────────────────────────────────────
        all_students = set()
        for members in families.values():
            all_students.update(members)
        for row in report_data:
            all_students.add(row["student_a"])
            all_students.add(row["student_b"])
        for sid in source_texts.keys():
            all_students.add(sid)

        # JPlag options.json (mocked)
        options_json = {
            "language": "Javac based AST plugin",
            "minimumTokenMatch": 8,
            "submissionDirectories": ["submissions"],
            "oldSubmissionDirectories": [],
            "baseCodeSubmissionDirectory": None,
            "subdirectoryName": None,
            "fileSuffixes": [".java", ".c", ".cpp", ".py"],
            "exclusionFileName": None,
            "similarityMetric": "AVG",
            "similarityThreshold": 0.0,
            "maximumNumberOfComparisons": 0,
            "clusteringOptions": {
                "similarityMetric": "AVG",
                "spectralKernelBandwidth": 20.0,
                "spectralGaussianProcessVariance": 0.0025,
                "spectralMinRuns": 5,
                "spectralMaxRuns": 50,
                "spectralMaxKMeansIterationPerRun": 200,
                "agglomerativeThreshold": 0.2,
                "preprocessor": "CUMULATIVE_DISTRIBUTION_FUNCTION",
                "enabled": True,
                "algorithm": "SPECTRAL",
                "agglomerativeInterClusterSimilarity": "AVERAGE",
                "preprocessorThreshold": 0.2,
                "preprocessorPercentile": 0.5
            },
            "debugParser": False,
            "mergingOptions": {
                "enabled": False,
                "minimumNeighborLength": 2,
                "maximumGapSize": 6,
                "minimumRequiredMerges": 6
            },
            "normalize": False,
            "analyzeComments": False,
            "frequencyAnalysisOptions": {
                "enabled": False,
                "analysisStrategy": "COMPLETE_MATCHES_STRATEGY",
                "weightingFunction": "SIGMOID_WEIGHTING",
                "weightingFactor": 0.25
            }
        }

        submission_id_to_display_name = {sid: sid for sid in all_students}
        submission_ids_to_comparison_file_name = {}
        for sid in all_students:
            submission_ids_to_comparison_file_name[sid] = {}

        file_indexes = {}
        for sid in all_students:
            src_dict = source_texts.get(sid, {})
            if isinstance(src_dict, str):
                src_dict = {f"{sid}.txt": src_dict}
            
            file_index = {}
            for filename, src in src_dict.items():
                tokens = tokenize_source({filename: src})
                file_index[f"{sid}/{filename}"] = {"tokenCount": len(tokens)}
            file_indexes[sid] = file_index

        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
            # Write files inside zip
            for sid in all_students:
                src_dict = source_texts.get(sid, {})
                if isinstance(src_dict, str):
                    src_dict = {f"{sid}.txt": src_dict}
                for filename, src in src_dict.items():
                    zf.writestr(f"files/{sid}/{filename}", src)
            
            # Write basecode stubs
            for sid in all_students:
                zf.writestr(f"basecode/{sid}.json", "[]")

            top_comparisons = []
            distribution_avg = [0] * 100
            distribution_max = [0] * 100

            # Write comparisons
            for row in report_data:
                u = row["student_a"]
                v = row["student_b"]
                comp_filename = f"{u}-{v}.json"
                
                submission_ids_to_comparison_file_name[u][v] = comp_filename
                submission_ids_to_comparison_file_name[v][u] = comp_filename

                similarity = row["similarity"] # 0 to 1
                
                # Update distributions
                bin_idx = min(int(similarity * 100), 99)
                distribution_avg[bin_idx] += 1
                distribution_max[bin_idx] += 1


                longest_match = 0
                max_len = 0
                matches = []
                for tile in row.get("gst_tiles", []):
                    tl = tile["length"]
                    longest_match = max(longest_match, tl)
                    max_len += tl
                    
                    start_u = tile["a_token_idx"]
                    end_u = start_u + tl
                        
                    start_v = tile["b_token_idx"]
                    end_v = start_v + tl

                    matches.append({
                        "firstFileName": f"{u}/{tile['a_file']}",
                        "secondFileName": f"{v}/{tile['b_file']}",
                        "startInFirst": {"line": tile["a_lines"][0], "column": tile.get("a_col_start", 0), "tokenListIndex": start_u},
                        "endInFirst": {"line": tile["a_lines"][1], "column": tile.get("a_col_end", 0), "tokenListIndex": end_u},
                        "startInSecond": {"line": tile["b_lines"][0], "column": tile.get("b_col_start", 0), "tokenListIndex": start_v},
                        "endInSecond": {"line": tile["b_lines"][1], "column": tile.get("b_col_end", 0), "tokenListIndex": end_v},
                        "lengthOfFirst": tl,
                        "lengthOfSecond": tl,
                        "tokens": tl
                    })
                
                similarities = {
                    "AVG": similarity,
                    "MAX": similarity,
                    "MAXIMUM_LENGTH": float(max_len),
                    "LONGEST_MATCH": float(longest_match)
                }

                comp_payload = {
                    "firstSubmissionId": u,
                    "secondSubmissionId": v,
                    "similarities": similarities,
                    "matches": matches,
                    "firstSimilarity": similarity,
                    "secondSimilarity": similarity
                }
                zf.writestr(f"comparisons/{comp_filename}", json.dumps(comp_payload, indent=2))
                
                top_comparisons.append({
                    "firstSubmission": u,
                    "secondSubmission": v,
                    "similarities": similarities
                })

            top_comparisons.sort(key=lambda x: x["similarities"]["AVG"], reverse=True)
            top_comparisons = top_comparisons[:500] # Cap top comparisons

            # ── Root JSONs for v6.3.0 ──────────────────────────────────────────
            
            run_information = {
                "version": {"major": 6, "minor": 3, "patch": 0},
                "failedSubmissions": [],
                "dateOfExecution": "01/01/70",
                "executionTime": 100,
                "totalComparisons": len(report_data)
            }

            cluster_json = []
            for fam_id, members in families.items():
                if len(members) > 1:
                    cluster_json.append({
                        "averageSimilarity": 0.5,
                        "strength": 0.1,
                        "members": members
                    })

            submission_mappings = {
                "submissionIds": submission_id_to_display_name,
                "submissionIdsToComparisonFileName": submission_ids_to_comparison_file_name
            }

            distribution_json = {
                "AVG": distribution_avg,
                "MAX": distribution_max
            }

            zf.writestr("options.json", json.dumps(options_json, indent=2))
            zf.writestr("runInformation.json", json.dumps(run_information, indent=2))
            zf.writestr("cluster.json", json.dumps(cluster_json, indent=2))
            zf.writestr("submissionMappings.json", json.dumps(submission_mappings, indent=2))
            zf.writestr("distribution.json", json.dumps(distribution_json, indent=2))
            zf.writestr("topComparisons.json", json.dumps(top_comparisons, indent=2))
            zf.writestr("submissionFileIndex.json", json.dumps({"fileIndexes": file_indexes}, indent=2))
            zf.writestr("results.json", json.dumps(report_data, indent=2))
