from itertools import combinations
import networkx as nx
import hypernetx as hnx
import numpy as np
import random
    
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


def generate_ego_hypergraph(num_hypergraphs, min_size, max_size, num_edges, max_edge_size, seed=0):
    """Generate random Ego hypergraphs."""
    rng = np.random.default_rng(seed)
    hypergraphs = []
    
    while len(hypergraphs) < num_hypergraphs:
        num_nodes = rng.integers(min_size, max_size, endpoint=True)
        # Generate a random graph
        nodes = range(num_nodes)
        edges = {}
        for i in range(num_edges):
            edge_size = random.randint(2, min(max_edge_size, num_nodes))
            edge_nodes = random.sample(nodes, edge_size)
            edges[i] = edge_nodes

        H = hnx.Hypergraph(edges)

        # Generate the ego hypergraph
        ego_node = random.choice(list(H.nodes))

        ego_edges = {}
        for edge in H.edges:
            if ego_node in H.edges[edge]:
                ego_edges[edge] = list(H.edges[edge])
                
        if H.is_connected():
            H = hnx.Hypergraph(ego_edges)
            hypergraphs.append(H)
                
    return hypergraphs


def generate_hypertrees(num_hypergraphs, min_size, max_size, p, k, seed=0):
    """Generate hypertrees by merging connected edges."""
    rng = np.random.default_rng(seed)
    hypergraphs = []
    
    while len(hypergraphs) < num_hypergraphs:
        # Generate a random tree
        num_nodes = rng.integers(min_size, max_size, endpoint=True)
        T = nx.random_labeled_tree(n=num_nodes, seed=rng)
        
        # Initialize the hyperedges list
        hyperedges = []
        
        # Start with all edges as potential hyperedges
        potential_edges = list(T.edges())
        
        while potential_edges:
            # Start with a random edge
            current_edge = potential_edges.pop(rng.integers(len(potential_edges)))
            hyperedge = set(current_edge)
            
            # Grow the hyperedge
            while len(hyperedge) < k and potential_edges:
                # Find edges connected to the current hyperedge
                connected_edges = [e for e in potential_edges if set(e) & hyperedge]
                
                if not connected_edges:
                    break
                
                # Randomly choose a connected edge to add
                if rng.random() < p:
                    new_edge = connected_edges[rng.integers(len(connected_edges))]
                    hyperedge.update(new_edge)
                    potential_edges.remove(new_edge)
                else:
                    break
            
            hyperedges.append(hyperedge)
        
        # Add any remaining edges as hyperedges
        hyperedges.extend([set(e) for e in potential_edges])
        
        # Create a Hypergraph object
        H = hnx.Hypergraph(hyperedges)
        hypergraphs.append(H)
    
    return hypergraphs