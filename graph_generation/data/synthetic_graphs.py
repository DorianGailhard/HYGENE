from itertools import combinations
import hypernetx as hnx
import numpy as np
import scipy as sp
    
def generate_sbm_hypergraphs(num_graphs, min_size, max_size, p, q, k, seed=0):
    """Generate SBM hypergraphs."""
    rng = np.random.default_rng(seed)
    graphs = []

    while len(graphs) < num_graphs:
        num_nodes = rng.integers(min_size, max_size, endpoint=True)
        
        communities = rng.choice([-1, 1], size=num_nodes)
        
        incidence_matrix = []
        
        # Select the hyperedges
        for combi in map(list, combinations(np.arange(num_nodes), k)):
            values = communities[combi]
        
            same_cluster = np.all(values == values[0])
            prob = rng.random()
            
            if (same_cluster and prob > p) or (not same_cluster and prob > q):
                vector_edge = np.zeros(num_nodes)
                vector_edge[combi] = 1
                
                incidence_matrix.append(vector_edge)
        
        # Build the bipartite sparse graph associated to the incidence matrix
        num_hyperedges = len(incidence_matrix)
        num_vertices = num_nodes + num_hyperedges

        # Initialize the adjacency matrix of the bipartite graph as a sparse matrix
        adjacency_matrix = sp.sparse.lil_matrix((num_vertices, num_vertices), dtype=np.int8)

        # Add an edge between a vertex and a hyperedge if and only if the vertex is
        # incident to the hyperedge in the hypergraph
        hyperedge_indices = num_nodes + np.arange(num_hyperedges)

        for hyperedge_index, node_indices_in_hyperedge in enumerate(incidence_matrix):
            node_indices_in_hyperedge = np.where(node_indices_in_hyperedge)[0]
            hyperedge_index = hyperedge_indices[hyperedge_index]
            adjacency_matrix[hyperedge_index, node_indices_in_hyperedge] = 1
            adjacency_matrix[node_indices_in_hyperedge, hyperedge_index] = 1
        
        
        G = nx.from_scipy_sparse_array(adjacency_matrix)
        
        if nx.is_connected(G):
            G.graph["num_nodes"] = num_nodes
            graphs.append(G)

    return graphs


def generate_erdos_renyi_hypergraphs(num_graphs, min_size, max_size, probs, k, seed=0):
    """Generate random Erdos-Renyi hypergraphs."""
    rng = np.random.default_rng(seed)
    graphs = []

    while len(graphs) < num_graphs:
        num_nodes = rng.integers(min_size, max_size, endpoint=True)
        incidence_matrix = []
        
        # Select the hyperedges
        for edge_order in np.arange(2, k + 1):
            for combi in map(list, combinations(np.arange(num_nodes), edge_order)):
                if rng.random() > probs[edge_order - 2]:
                    vector_edge = np.zeros(num_nodes)
                    vector_edge[combi] = 1
                
                    incidence_matrix.append(vector_edge)
        
        # Build the bipartite sparse graph associated to the incidence matrix
        num_hyperedges = len(incidence_matrix)
        num_vertices = num_nodes + num_hyperedges

        # Initialize the adjacency matrix of the bipartite graph as a sparse matrix
        adjacency_matrix = sp.sparse.lil_matrix((num_vertices, num_vertices), dtype=np.int8)

        # Add an edge between a vertex and a hyperedge if and only if the vertex is
        # incident to the hyperedge in the hypergraph
        hyperedge_indices = num_nodes + np.arange(num_hyperedges)

        for hyperedge_index, node_indices_in_hyperedge in enumerate(incidence_matrix):
            node_indices_in_hyperedge = np.where(node_indices_in_hyperedge)[0]
            hyperedge_index = hyperedge_indices[hyperedge_index]
            adjacency_matrix[hyperedge_index, node_indices_in_hyperedge] = 1
            adjacency_matrix[node_indices_in_hyperedge, hyperedge_index] = 1
        
        
        G = nx.from_scipy_sparse_array(adjacency_matrix)
        
        if nx.is_connected(G):
            G.graph["num_nodes"] = num_nodes
            graphs.append(G)

    return graphs
