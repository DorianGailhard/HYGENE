from itertools import combinations
import hypernetx as hnx
import numpy as np
    
def generate_sbm_hypergraphs(num_hypergraphs, min_size, max_size, p, q, k, seed=0):
    """Generate SBM hypergraphs."""
    rng = np.random.default_rng(seed)
    hypergraphs = []

    while len(hypergraphs) < num_hypergraphs:
        num_nodes = rng.integers(min_size, max_size, endpoint=True)
        
        communities = rng.choice([-1, 1], size=num_nodes)
        
        incidence_matrix = []
        
        # Select the hyperedges
        for combi in map(list, combinations(np.arange(num_nodes), k)):
            values = communities[combi]
        
            same_cluster = np.all(values == values[0])
            prob = rng.random()
            
            if (same_cluster and prob < p) or (not same_cluster and prob < q):
                vector_edge = np.zeros(num_nodes)
                vector_edge[combi] = 1
                
                incidence_matrix.append(vector_edge)
        
        if (len(incidence_matrix) > 0):
            incidence_matrix = np.array(incidence_matrix).T
            H = hnx.Hypergraph.from_incidence_matrix(incidence_matrix)

            if H.is_connected():
                hypergraphs.append(H)

    return hypergraphs


def generate_erdos_renyi_hypergraphs(num_hypergraphs, min_size, max_size, probs, k, seed=0):
    """Generate random Erdos-Renyi hypergraphs."""
    rng = np.random.default_rng(seed)
    hypergraphs = []

    while len(hypergraphs) < num_hypergraphs:
        num_nodes = rng.integers(min_size, max_size, endpoint=True)
        incidence_matrix = []
        
        # Select the hyperedges
        for edge_order in np.arange(2, k + 1):
            for combi in map(list, combinations(np.arange(num_nodes), edge_order)):
                if rng.random() < probs[edge_order - 2]:
                    vector_edge = np.zeros(num_nodes)
                    vector_edge[combi] = 1
                
                    incidence_matrix.append(vector_edge)
        
        if (len(incidence_matrix) > 0):
            incidence_matrix = np.array(incidence_matrix).T
            H = hnx.Hypergraph.from_incidence_matrix(incidence_matrix)

            if H.is_connected():
                hypergraphs.append(H)

    return hypergraphs
