import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import HDBSCAN
from gnn import StructuralEncoder

class CohortAnalyzer:
    """Ingests per-submission graph vectors, processes textual structures, and groups lineages via HDBSCAN."""
    def __init__(self, cohort_data, node_vocab_size, edge_vocab_size):
        self.cohort_data = cohort_data
        self.student_ids = list(cohort_data.keys())
        # --- FIX: Encoder initialized dynamically using explicit discovery matrix scales ---
        self.encoder = StructuralEncoder(node_vocab_size=node_vocab_size, edge_vocab_size=edge_vocab_size)
        self.embeddings = None
        self.similarity_matrix = None

    def generate_all_embeddings(self):
        """Extracts text vocabularies and encodes structural metrics across the entire student population."""
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
        
        self.embeddings = np.array(embeddings_list)
        self.similarity_matrix = cosine_similarity(self.embeddings)
        
        print(f"\n[DIAGNOSTIC] Relational Metrics -> Min Sim: {self.similarity_matrix.min():.4f}, Max Sim: {self.similarity_matrix.max():.4f}, Mean: {self.similarity_matrix.mean():.4f}")

    def extract_solution_families(self, min_cluster_size=2):
        """Clusters structural profiles, assigning copy lineages to groups and honest code to noise (-1)."""
        if len(self.embeddings) < min_cluster_size:
            return {f"Unique_Solution_{sid}": [sid] for sid in self.student_ids}

        clustering = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=1).fit(self.embeddings)
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
        """Computes localized risk scores relative to isolated family structural baselines."""
        report_data = []
        id_to_idx = {sid: i for i, sid in enumerate(self.student_ids)}

        for family_name, members in families.items():
            if len(members) <= 1 or "Unique_Solution" in family_name:
                continue 
                
            for i, student_a in enumerate(members):
                for j in range(i + 1, len(members)):
                    student_b = members[j]
                    idx_a, idx_b = id_to_idx[student_a], id_to_idx[student_b]
                    
                    global_score = float(self.similarity_matrix[idx_a, idx_b])
                    family_scores = [self.similarity_matrix[id_to_idx[m1], id_to_idx[m2]] 
                                     for m1 in members for m2 in members if m1 != m2]
                    family_mean = float(np.mean(family_scores)) if family_scores else global_score

                    report_data.append({
                        "family": family_name,
                        "student_a": student_a,
                        "student_b": student_b,
                        "similarity": global_score,
                        "family_density": family_mean,
                        "risk_level": "CRITICAL" if global_score > 0.85 else "HIGH"
                    })
                    
        report_data.sort(key=lambda x: x['similarity'], reverse=True)
        return report_data
