import numpy as np
import networkx as nx
import faiss
from gnn import StructuralEncoder

class CohortAnalyzer:
    """
    Orchestrates cohort embedding generation, constructs a sparse K-Nearest Neighbor
    similarity graph using an approximate HNSW index, and partitions the network into 
    communities representing distinct algorithmic approaches using the Louvain algorithm.
    """
    
    def __init__(self, cohort_data, node_vocab_size, edge_vocab_size):
        self.cohort_data = cohort_data
        self.student_ids = list(cohort_data.keys())
        self.encoder = StructuralEncoder(node_vocab_size=node_vocab_size, edge_vocab_size=edge_vocab_size)
        self.embeddings = None
        self.knn_graph = None
        self.node_to_community = {}

    def generate_all_embeddings(self, alpha=0.5):
        """
        Extracts textual and structural properties across all submissions to generate
        unit-normalized hybrid embedding vectors.
        """
        if not self.student_ids:
            raise ValueError("Cannot generate embeddings: Cohort dataset is empty.")

        full_cohort_corpus = [
            getattr(self.cohort_data[sid], 'text_document', '')
            for sid in self.student_ids
        ]
        
        self.encoder.fit_cohort_text(full_cohort_corpus)

        embeddings_list = []
        for student_id in self.student_ids:
            pyg_data = self.cohort_data[student_id]
            student_vector = self.encoder.encode_submission(pyg_data, alpha=alpha)
            embeddings_list.append(student_vector)
        
        self.embeddings = np.array(embeddings_list)

    def extract_solution_families(self, knn_k=3, **kwargs):
        """
        Constructs a sparse similarity graph using a Hierarchical Navigable Small World 
        (HNSW) approximate index and identifies structural families via Louvain community detection.
        """
        num_students = len(self.student_ids)
        self.knn_graph = nx.Graph()
        self.node_to_community = {}

        if num_students < 2:
            return {f"Community_{sid}": [sid] for sid in self.student_ids}

        effective_k = min(knn_k, num_students - 1)

        for sid in self.student_ids:
            self.knn_graph.add_node(sid)

        # FAISS requires contiguous float32 arrays for optimized lower-level C++ execution
        embeddings_f32 = np.ascontiguousarray(self.embeddings.astype('float32'))
        dimension = embeddings_f32.shape[1]

        # Initialize an uncompressed HNSW index mapping L2 Euclidean distance.
        # 32 represents the number of bi-directional link connections constructed per node.
        index = faiss.IndexHNSWFlat(dimension, 32)
        index.add(embeddings_f32)

        # Query the hierarchical graph index for approximate nearest neighborhoods in O(log N) time.
        # Minimizing Euclidean distance on unit-normalized vectors directly optimizes cosine similarity.
        _, indices = index.search(embeddings_f32, effective_k + 1)

        for idx, student_id in enumerate(self.student_ids):
            neighbor_indices = indices[idx]
            
            for nb_idx in neighbor_indices:
                # FAISS can return -1 pads if a query requests more neighbors than existing nodes
                if nb_idx == -1:
                    continue
                    
                neighbor_id = self.student_ids[nb_idx]
                
                # Prevent self-loop generation caused by perfect spatial distance ties at 0.0
                if student_id == neighbor_id:
                    continue
                
                vec_a = self.embeddings[idx]
                vec_b = self.embeddings[nb_idx]
                similarity = float(np.dot(vec_a, vec_b))
                similarity = max(0.0, min(1.0, similarity))

                if self.knn_graph.has_edge(student_id, neighbor_id):
                    current_w = self.knn_graph[student_id][neighbor_id]['weight']
                    self.knn_graph[student_id][neighbor_id]['weight'] = max(current_w, similarity)
                else:
                    self.knn_graph.add_edge(student_id, neighbor_id, weight=similarity)

        try:
            communities_list = nx.community.louvain_communities(self.knn_graph, weight='weight', seed=42)
        except Exception:
            communities_list = [set(self.student_ids)]

        families = {}
        for comm_idx, community_nodes in enumerate(communities_list):
            family_name = f"Solution_Family_{comm_idx + 1}"
            families[family_name] = sorted(list(community_nodes))
            
            for node in community_nodes:
                self.node_to_community[node] = family_name

        return families

    def compute_suspicion_scores(self, families=None):
        """
        Parses the sparse network graph edges to evaluate pairwise similarity and macro-community
        density, returning prioritized anomalies exceeding the suspicion threshold.
        """
        report_data = []
        seen_pairs = set()

        if self.knn_graph is None or len(self.knn_graph.edges) == 0:
            return report_data

        # Calculate macro-density for each community based on its internal active edges
        community_densities = {}
        if families:
            for family_name, members in families.items():
                internal_weights = []
                for i, node_a in enumerate(members):
                    for j in range(i + 1, len(members)):
                        node_b = members[j]
                        if self.knn_graph.has_edge(node_a, node_b):
                            internal_weights.append(self.knn_graph[node_a][node_b]['weight'])
                community_densities[family_name] = float(np.mean(internal_weights)) if internal_weights else 0.0

        for u, v, data in self.knn_graph.edges(data=True):
            if u == v:
                continue
                
            similarity = data['weight']
            
            # Filter out low-level structural baselines to prevent reporting noise
            if similarity >= 0.85:
                pair = tuple(sorted([u, v]))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)

                comm_u = self.node_to_community.get(u, "Isolated_Node")
                comm_v = self.node_to_community.get(v, "Isolated_Node")
                
                if comm_u == comm_v:
                    family_label = comm_u
                    family_density = community_densities.get(family_label, similarity)
                else:
                    family_label = "Cross_Template_Overlap"
                    family_density = similarity

                if similarity >= 0.9995:
                    risk_level = "CRITICAL"
                elif similarity >= 0.9980:
                    risk_level = "HIGH"
                else:
                    risk_level = "SUSPICIOUS"

                report_data.append({
                    "family": family_label,
                    "student_a": pair[0],
                    "student_b": pair[1],
                    "similarity": similarity,
                    "family_density": family_density,
                    "risk_level": risk_level
                })

        report_data.sort(key=lambda x: x['similarity'], reverse=True)
        return report_data
