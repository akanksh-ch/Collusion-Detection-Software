import numpy as np
from sklearn.cluster import HDBSCAN
from gnn import StructuralEncoder

class CohortAnalyzer:
    """Ingests per-submission graph elements, processes structural representations, 
    and clusters profiles using accelerated spatial indexing trees."""
    def __init__(self, cohort_data, node_vocab_size, edge_vocab_size):
        self.cohort_data = cohort_data
        self.student_ids = list(cohort_data.keys())
        self.encoder = StructuralEncoder(node_vocab_size=node_vocab_size, edge_vocab_size=edge_vocab_size)
        self.embeddings = None

    def generate_all_embeddings(self):
        """Encodes structural metrics across the cohort. Completely omits the O(N^2) pairwise similarity matrix."""
        if not self.student_ids:
            print("\n[CRITICAL ERROR] Cohort Analyzer contains zero valid student submissions.")
            raise ValueError("Aborting pipeline execution: Empty inputs.")

        full_cohort_corpus = []
        for sid in self.student_ids:
            pyg_data = self.cohort_data[sid]
            full_cohort_corpus.append(getattr(pyg_data, 'text_document', ''))
        
        self.encoder.fit_cohort_text(full_cohort_corpus)

        embeddings_list = []
        for student_id in self.student_ids:
            pyg_data = self.cohort_data[student_id]
            student_vector = self.encoder.encode_submission(pyg_data)
            embeddings_list.append(student_vector)
        
        # This is now our master dense feature matrix: Shape (N, Dimensions)
        self.embeddings = np.array(embeddings_list)
        print(f"\n[DIAGNOSTIC] Generated {self.embeddings.shape[0]} unit-normalized fingerprint fingerprints. Matrix step omitted.")

    def extract_solution_families(self, min_cluster_size=2):
        """Clusters structural profiles in O(N log N) using accelerated internal space trees."""
        if len(self.embeddings) < min_cluster_size:
            return {f"Unique_Solution_{sid}": [sid] for sid in self.student_ids}

        # FIX: Switched metric to 'euclidean'. Since vectors are globally unit-normalized,
        # HDBSCAN will internally utilize high-performance spatial partitioning trees (KD-Tree/Ball-Tree).
        # It completely circumvents computing a quadratic matrix.
        clustering = HDBSCAN(
            min_cluster_size=min_cluster_size, 
            min_samples=min_cluster_size, 
            metric='euclidean'
        ).fit(self.embeddings)
        
        labels = clustering.labels_

        families = {}
        for idx, label in enumerate(labels):
            student_id = self.student_ids[idx]
            if label == -1:
                families[f"Unique_Solution_{student_id}"] = [student_id]
            else:
                family_name = f"Solution_Family_{label + 1}"
                if family_name not in families:
                    families[family_name] = []
                families[family_name].append(student_id)
                
        return families

    def compute_suspicion_scores(self, families):
        """Localized Risk Profiler: Computes exact cosine similarities ONLY for localized family members,
        preserving end-to-end efficiency by ignoring unrelated pairs completely."""
        report_data = []

        for family_name, members in families.items():
            if len(members) <= 1 or "Unique_Solution" in family_name:
                continue  
                
            id_to_idx = {sid: self.student_ids.index(sid) for sid in members}
            
            # Localized Scope: Only compute vector dot products for active cluster elements
            for i, student_a in enumerate(members):
                for j in range(i + 1, len(members)):
                    student_b = members[j]
                    vec_a = self.embeddings[id_to_idx[student_a]]
                    vec_b = self.embeddings[id_to_idx[student_b]]
                    
                    # Manual dot product of globally unit-normalized vectors equals pure Cosine Similarity
                    global_score = float(np.dot(vec_a, vec_b))
                    
                    # Bounded safety clipping
                    global_score = max(0.0, min(1.0, global_score))

                    report_data.append({
                        "family": family_name,
                        "student_a": student_a,
                        "student_b": student_b,
                        "similarity": global_score,
                        "family_density": 0.0,  # Legacy structure placeholder to preserve template layout compatibility
                        "risk_level": "CRITICAL" if global_score > 0.85 else "HIGH"
                    })
                    
        report_data.sort(key=lambda x: x['similarity'], reverse=True)
        return report_data
