import numpy as np
import networkx as nx
import faiss
import logging
from gnn import StructuralEncoder

class CohortAnalyzer:
    """
    Orchestrates spatial profiling workflows, maps neighborhoods using accelerated
    approximate HNSW structures, and segments clusters via Leiden or Louvain partitioning.
    """
    def __init__(self, cohort_data, node_vocab_size, edge_vocab_size):
        self.cohort_data = cohort_data
        self.student_ids = list(cohort_data.keys())
        self.encoder = StructuralEncoder(node_vocab_size=node_vocab_size, edge_vocab_size=edge_vocab_size)
        self.embeddings = None
        self.knn_graph = None
        self.node_to_community = {}

    def generate_all_embeddings(self, alpha=0.5):
        """Transforms structural layout states into L2 unit-normalized hybrid embeddings."""
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

    def extract_solution_families(self, knn_k=4, use_leiden=True, **kwargs):
        """
        Groups cohort submissions into distinct structural solution groups by building 
        an HNSW approximate graph index and applying edge-pruned community detection.
        """
        num_students = len(self.student_ids)
        self.knn_graph = nx.Graph()
        self.node_to_community = {}

        if num_students < 2:
            return {f"Solution_Family_{idx + 1}": [sid] for idx, sid in enumerate(self.student_ids)}

        effective_k = min(knn_k, num_students - 1)

        for sid in self.student_ids:
            self.knn_graph.add_node(sid)

        embeddings_f32 = np.ascontiguousarray(self.embeddings.astype('float32'))
        dimension = embeddings_f32.shape[1]

        index = faiss.IndexHNSWFlat(dimension, 32)
        index.add(embeddings_f32)

        # Search nearest neighborhoods in logarithmic O(log N) runtime
        _, indices = index.search(embeddings_f32, effective_k + 1)

        for idx, student_id in enumerate(self.student_ids):
            neighbor_indices = indices[idx]
            
            for nb_idx in neighbor_indices:
                if nb_idx == -1:
                    continue
                    
                neighbor_id = self.student_ids[nb_idx]
                if student_id == neighbor_id:
                    continue
                
                vec_a = self.embeddings[idx]
                vec_b = self.embeddings[nb_idx]
                similarity = float(np.dot(vec_a, vec_b))
                similarity = max(0.0, min(1.0, similarity))

                # Discard connections below the background noise threshold to separate independent submissions
                if similarity < 0.92:
                    continue

                if self.knn_graph.has_edge(student_id, neighbor_id):
                    current_w = self.knn_graph[student_id][neighbor_id]['weight']
                    self.knn_graph[student_id][neighbor_id]['weight'] = max(current_w, similarity)
                else:
                    self.knn_graph.add_edge(student_id, neighbor_id, weight=similarity)

        families = {}
        
        if use_leiden:
            try:
                import igraph as ig
                import leidenalg as la
                
                # Unpack active NetworkX nodes into an isolated igraph data footprint structure
                ig_graph = ig.Graph(directed=False)
                ig_graph.vs["name"] = self.student_ids
                name_to_vidx = {name: i for i, name in enumerate(self.student_ids)}
                
                edges_to_add = []
                weights_to_add = []
                for u, v, d in self.knn_graph.edges(data=True):
                    edges_to_add.append((name_to_vidx[u], name_to_vidx[v]))
                    weights_to_add.append(d['weight'])
                
                if edges_to_add:
                    ig_graph.add_edges(edges_to_add)
                    ig_graph.es["weight"] = weights_to_add
                
                # Execute Leiden optimization using an enriched Modularity partition metric.
                # A resolution parameter of 1.4 forces dense background baseline cliques to split.
                partition = la.find_partition(
                    ig_graph, 
                    la.ModularityVertexPartition, 
                    weights='weight' if edges_to_add else None,
                    resolution_parameter=1.4
                )
                
                for comm_idx, community_nodes in enumerate(partition):
                    family_name = f"Solution_Family_{comm_idx + 1}"
                    members = [ig_graph.vs[n_idx]['name'] for n_idx in community_nodes]
                    families[family_name] = sorted(members)
                    
            except Exception as e:
                logging.warning(f"[LEIDEN ERROR] Reverting to Louvain community fallback engine: {e}")
                use_leiden = False

        if not use_leiden:
            try:
                # Apply high-resolution parameters to the Louvain algorithm pass
                communities_list = nx.community.louvain_communities(
                    self.knn_graph, weight='weight', seed=42, resolution=1.4
                )
            except Exception:
                communities_list = [set(self.student_ids)]

            for comm_idx, community_nodes in enumerate(communities_list):
                family_name = f"Solution_Family_{comm_idx + 1}"
                families[family_name] = sorted(list(community_nodes))

        for family_name, members in families.items():
            for node in members:
                self.node_to_community[node] = family_name

        return families

    def compute_suspicion_scores(self, families=None):
        """Evaluates pairwise relationships and outputs prioritized indicators exceeding suspicion limits."""
        report_data = []
        seen_pairs = set()

        if self.knn_graph is None or len(self.knn_graph.edges) == 0:
            return report_data

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
