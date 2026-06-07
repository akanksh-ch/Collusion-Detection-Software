import numpy as np
from sklearn.cluster import HDBSCAN
from gnn import StructuralEncoder

class CohortAnalyzer:
    """Manages full vector embedding orchestration and space tree clustering calls."""
    def __init__(self, cohort_data, node_vocab_size, edge_vocab_size):
        self.cohort_data = cohort_data
        self.student_ids = list(cohort_data.keys())
        self.encoder = StructuralEncoder(node_vocab_size=node_vocab_size, edge_vocab_size=edge_vocab_size)
        self.embeddings = None

    def generate_all_embeddings(self, alpha=0.5):
        """Builds structural vector mappings across all active student files."""
        if not self.student_ids:
            print("\n[CRITICAL ERROR] Submissions dataset empty.")
            raise ValueError("Aborting profile execution.")

        full_cohort_corpus = []
        for sid in self.student_ids:
            pyg_data = self.cohort_data[sid]
            full_cohort_corpus.append(getattr(pyg_data, 'text_document', ''))
        
        self.encoder.fit_cohort_text(full_cohort_corpus)

        embeddings_list = []
        for student_id in self.student_ids:
            pyg_data = self.cohort_data[student_id]
            student_vector = self.encoder.encode_submission(pyg_data, alpha=alpha)
            embeddings_list.append(student_vector)
        
        self.embeddings = np.array(embeddings_list)
        print(f"[DIAGNOSTIC] Array allocation complete. Master Shape: {self.embeddings.shape}")

    def extract_solution_families(self, min_cluster_size=2):
        """Groups close vector shapes using internal space tree partitioning frameworks."""
        if len(self.embeddings) < min_cluster_size:
            return {f"Unique_Solution_{sid}": [sid] for sid in self.student_ids}

        # Utilizing native Euclidean distance on unit-normalized vectors mirrors Cosine behavior
        # while dropping the algorithmic overhead from O(N^2) to O(N log N) via internal space trees.
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
        """Isolates and computes exact metric similarity properties within grouped families only."""
        report_data = []

        for family_name, members in families.items():
            if len(members) <= 1 or "Unique_Solution" in family_name:
                continue  
                
            id_to_idx = {sid: self.student_ids.index(sid) for sid in members}
            
            # Compute the cluster cohesion metric (average similarity within the family)
            pairwise_scores = []
            for i, student_a in enumerate(members):
                for j in range(i + 1, len(members)):
                    student_b = members[j]
                    vec_a = self.embeddings[id_to_idx[student_a]]
                    vec_b = self.embeddings[id_to_idx[student_b]]
                    score = float(np.dot(vec_a, vec_b))
                    pairwise_scores.append(max(0.0, min(1.0, score)))
            
            family_density = float(np.mean(pairwise_scores)) if pairwise_scores else 0.0
            
            # Map structural components into distinct pairs using the evaluated density metric
            for i, student_a in enumerate(members):
                for j in range(i + 1, len(members)):
                    student_b = members[j]
                    vec_a = self.embeddings[id_to_idx[student_a]]
                    vec_b = self.embeddings[id_to_idx[student_b]]
                    
                    # Direct dot product calculation yields pure cosine metric score on unit vectors
                    global_score = float(np.dot(vec_a, vec_b))
                    global_score = max(0.0, min(1.0, global_score))

                    report_data.append({
                        "family": family_name,
                        "student_a": student_a,
                        "student_b": student_b,
                        "similarity": global_score,
                        "family_density": family_density,
                        "risk_level": "CRITICAL" if global_score > 0.85 else "HIGH"
                    })
                    
        report_data.sort(key=lambda x: x['similarity'], reverse=True)
        return report_data
