from abc import ABC, abstractmethod
from dataclasses import dataclass

import hypernetx as hnx

import numpy as np

from collections import Counter
from scipy.stats import wasserstein_distance

from util.eval_helper import (
    spectral_stats,
    eval_fraction_unique,
    eval_fraction_isomorphic,
)


import math
from scipy.stats import chi2
import graph_tool.all as gt

import concurrent.futures
import networkx as nx


@dataclass
class Metric(ABC):
    @abstractmethod
    def __str__(self) -> str:
        pass

    @abstractmethod
    def __call__(
        self,
        reference_hypergraphs: list[hnx.Hypergraph],
        predicted_hypergraphs: list[hnx.Hypergraph],
        train_hypergraphs: list[hnx.Hypergraph],
    ) -> float:
        pass


class NodeNumDiff(Metric):
    def __str__(self):
        return "NodeNumDiff"

    def __call__(self, reference_hypergraphs, predicted_hypergraphs, train_hypergraphs):
        ref_node_num = np.array([len(H.nodes)
                                for H in reference_hypergraphs])
        pred_node_num = np.array([len(H.nodes)
                                 for H in predicted_hypergraphs])
        return np.mean(np.abs(ref_node_num - pred_node_num))


class NodeDegreeDistrWasserstein(Metric):
    def __str__(self):
        return "NodeDegreeDistrWasserstein"

    def __call__(self, reference_hypergraphs, predicted_hypergraphs, train_hypergraphs):
        reference_dist = []
        for H in reference_hypergraphs:
            reference_dist += hnx.reports.descriptive_stats.degree_dist(H)
    
        pred_dist = []
        for H in predicted_hypergraphs:
            pred_dist += hnx.reports.descriptive_stats.degree_dist(H)

        # Convert to counters
        degree_dist1 = Counter(reference_dist)
        degree_dist2 = Counter(pred_dist)
        
        # Extract keys (degree values) and values (frequencies) from the counters
        degree_dist1_keys = list(degree_dist1.keys())
        degree_dist1_values = list(degree_dist1.values())
        degree_dist2_keys = list(degree_dist2.keys())
        degree_dist2_values = list(degree_dist2.values())
        
        # Compute the Wasserstein distance
        return wasserstein_distance(
            np.array(degree_dist1_keys),
            np.array(degree_dist2_keys),
            np.array(degree_dist1_values),
            np.array(degree_dist2_values)
        )


class EdgeSizeDistrWasserstein(Metric):
    def __str__(self):
        return "EdgeSizeDistrWasserstein"

    def __call__(self, reference_hypergraphs, predicted_hypergraphs, train_hypergraphs):
        reference_dist = []
        for H in reference_hypergraphs:
            reference_dist += hnx.reports.descriptive_stats.edge_size_dist(H)
    
        pred_dist = []
        for H in predicted_hypergraphs:
            pred_dist += hnx.reports.descriptive_stats.edge_size_dist(H)

        # Convert to counters
        degree_dist1 = Counter(reference_dist)
        degree_dist2 = Counter(pred_dist)
        
        # Extract keys (degree values) and values (frequencies) from the counters
        degree_dist1_keys = list(degree_dist1.keys())
        degree_dist1_values = list(degree_dist1.values())
        degree_dist2_keys = list(degree_dist2.keys())
        degree_dist2_values = list(degree_dist2.values())
        
        # Compute the Wasserstein distance
        return wasserstein_distance(
            np.array(degree_dist1_keys),
            np.array(degree_dist2_keys),
            np.array(degree_dist1_values),
            np.array(degree_dist2_values)
        )
    

class Spectral(Metric):
    # Eigenvalues of normalized Laplacian
    def __str__(self):
        return "Spectral"

    def __call__(self, reference_hypergraphs, predicted_hypergraphs, train_hypergraphs):
        return spectral_stats(reference_hypergraphs, predicted_hypergraphs)
    

class Uniqueness(Metric):
    def __str__(self):
        return "Uniqueness"

    def __call__(self, reference_hypergraphs, predicted_hypergraphs, train_hypergraphs):
        return eval_fraction_unique(predicted_hypergraphs, precise=True)


class Novelty(Metric):
    def __str__(self):
        return "Novelty"

    def __call__(self, reference_hypergraphs, predicted_hypergraphs, train_hypergraphs):
        return 1 - eval_fraction_isomorphic(
            fake_hypergraphs=predicted_hypergraphs, train_hypergraphs=train_hypergraphs
        )
    
    
class CentralityCloseness(Metric):
    def __str__(self):
        return "CentralityCloseness"

    def compute_centrality_distribution(self, hypergraphs):
        all_centralities = []
        for H in hypergraphs:
            centralities = hnx.algorithms.s_centrality_measures.s_closeness_centrality(H)
            all_centralities.extend(centralities.values())
        return np.array(all_centralities)

    def __call__(self, reference_hypergraphs, predicted_hypergraphs, train_hypergraphs):
        centrality_pred = self.compute_centrality_distribution(predicted_hypergraphs)
        centrality_reference = self.compute_centrality_distribution(reference_hypergraphs)
        
        return wasserstein_distance(centrality_pred, centrality_reference)


class CentralityBetweenness(Metric):
    def __str__(self):
        return "CentralityBetweenness"

    def compute_centrality_distribution(self, hypergraphs):
        all_centralities = []
        for H in hypergraphs:
            centralities = hnx.algorithms.s_centrality_measures.s_betweenness_centrality(H)
            all_centralities.extend(centralities.values())
        return np.array(all_centralities)

    def __call__(self, reference_hypergraphs, predicted_hypergraphs, train_hypergraphs):
        centrality_pred = self.compute_centrality_distribution(predicted_hypergraphs)
        centrality_reference = self.compute_centrality_distribution(reference_hypergraphs)
        
        return wasserstein_distance(centrality_pred, centrality_reference)


class CentralityHarmonic(Metric):
    def __str__(self):
        return "CentralityHarmonic"

    def compute_centrality_distribution(self, hypergraphs):
        all_centralities = []
        for H in hypergraphs:
            centralities = hnx.algorithms.s_centrality_measures.s_harmonic_centrality(H)
            all_centralities.extend(centralities.values())
        return np.array(all_centralities)

    def __call__(self, reference_hypergraphs, predicted_hypergraphs, train_hypergraphs):
        centrality_pred = self.compute_centrality_distribution(predicted_hypergraphs)
        centrality_reference = self.compute_centrality_distribution(reference_hypergraphs)
        
        return wasserstein_distance(centrality_pred, centrality_reference)


class ValidEgo(Metric):
    def is_ego_hypergraph(self, H):
        if len(H.nodes) == 0 or len(H.edges) == 0:
            return False
    
        # Find the node that appears in the most hyperedges
        node_frequencies = {node: 0 for node in H.nodes}
        for edge in H.edges:
            for node in H.edges[edge]:
                node_frequencies[node] += 1
        
        potential_ego = max(node_frequencies, key=node_frequencies.get)
        
        # Check if the potential ego is in all or most hyperedges
        edges_with_ego = sum(1 for edge in H.edges if potential_ego in H.edges[edge])
        if edges_with_ego < len(H.edges):
            return False
    
        # Check if all other nodes are directly connected to the ego
        nodes_connected_to_ego = set()
        for edge in H.edges:
            if potential_ego in H.edges[edge]:
                nodes_connected_to_ego.update(H.edges[edge])
        
        if nodes_connected_to_ego != set(H.nodes):
            return False
    
        # Check if there are any hyperedges not including the ego or its direct connections
        for edge in H.edges:
            if potential_ego not in H.edges[edge] and not any(node in nodes_connected_to_ego for node in H.edges[edge]):
                return False
    
        return True
    
    def __str__(self):
        return "ValidEgo"
    
    def __call__(self, reference_hypergraphs, predicted_hypergraphs, train_hypergraphs):
        valid = 0
        
        for H in predicted_hypergraphs:
            if self.is_ego_hypergraph(H):
                valid += 1
        
        return valid/len(predicted_hypergraphs)
    
    
def is_sbm_hypergraph(H, p_intra=0.05, p_inter=0.001, k=3, strict=True, refinement_steps=1000):
    """
    Check how closely a given hypernetx hypergraph matches an SBM
    by computing the mean probability of the Wald test statistic for each recovered parameter,
    comparing to the real distribution in the hypergraph.
    """
    # Convert hypernetx hypergraph to graph-tool graph
    g = gt.Graph(directed=False)
    edge_list = []
    for e in H.edges:
        nodes = list(H.edges[e])
        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                edge_list.append((nodes[i], nodes[j]))
    
    # Add unique vertices and edges
    vertices = {v: g.add_vertex() for v in H.nodes}
    for e in set(edge_list):  # Remove duplicates
        g.add_edge(vertices[e[0]], vertices[e[1]])

    try:
        state = gt.minimize_blockmodel_dl(g)
    except ValueError:
        return False if strict else 0.0

    # Refine using merge-split MCMC
    for _ in range(refinement_steps): 
        state.multiflip_mcmc_sweep(beta=np.inf, niter=10)
    
    b = gt.contiguous_map(state.get_blocks())
    state = state.copy(b=b)
    e = state.get_matrix()
    n_blocks = state.get_nonempty_B()
    
    # Calculate real probabilities from the hypergraph
    est_p_intra = []
    est_p_inter = []

    if strict and n_blocks != 2:
        return False

    est_p_inter = np.zeros((n_blocks, n_blocks))
    
    for i in range(n_blocks):
        block_nodes = [v for v, block in enumerate(b) if block == i]
        intra_edges = sum(1 for edge in H.edges if len(set(H.edges[edge]) & set(block_nodes)) == len(set(H.edges[edge])))
        possible_intra_edges = math.comb(len(block_nodes), k)
        est_p_intra.append(intra_edges / possible_intra_edges)
        
        for j in range(i+1, n_blocks):
            other_block_nodes = [v for v, block in enumerate(b) if block == j]
            inter_edges = sum(1 for edge in H.edges() if 
                              len(set(H.edges[edge]) & set(block_nodes)) >= 1 and 
                              len(set(H.edges[edge]) & set(other_block_nodes)) >= 1)
            possible_inter_edges = math.comb(len(block_nodes) + len(other_block_nodes), k) - math.comb(len(block_nodes), k) - math.comb(len(other_block_nodes), k)
            
            est_p_inter[i, j] = est_p_inter[j, i] =  inter_edges / possible_inter_edges
    
    est_p_intra = np.array(est_p_intra)

    W_p_intra = (est_p_intra - p_intra)**2 / (est_p_intra * (1-est_p_intra) + 1e-6)
    W_p_inter = (est_p_inter - p_inter)**2 / (est_p_inter * (1-est_p_inter) + 1e-6)
    
    W = W_p_inter.copy()
    np.fill_diagonal(W, W_p_intra)
    p = 1 - chi2.cdf(abs(W), 1)
    p = p.mean()
    if strict:
        return p > 0.9 # p value < 10 %
    else:
        return p


def eval_acc_sbm_hypergraph(H_list, p_intra=0.05, p_inter=0.001, k=3, strict=True, refinement_steps=1000, is_parallel=False):
    count = 0.0
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
                for prob in executor.map(is_sbm_hypergraph,
                        [H for H in H_list], [p_intra for i in range(len(H_list))], [p_inter for i in range(len(H_list))],
                        [k for i in range(len(H_list))], [strict for i in range(len(H_list))],
                        [refinement_steps for i in range(len(H_list))]):
                    count += prob
    else:
        for H in H_list:
            count += is_sbm_hypergraph(H, p_intra=p_intra, p_inter=p_inter, k=k, strict=strict, refinement_steps=refinement_steps)
    return count / float(len(H_list))



def is_hypertree(H):
    if H.is_connected():
        # Step 1: Create the line graph of the hypergraph
        line_graph = nx.Graph()
        dic = {}

        # Add a node for each hyperedge
        for i, hyperedge in enumerate(H.edges):
            line_graph.add_node(i)
            dic[i] = hyperedge

        # Add edges between nodes (hyperedges) that share at least one vertex
        for i, edge1 in enumerate(H.edges):
            for j, edge2 in enumerate(H.edges):
                if i < j and set(H.edges[edge1]).intersection(H.edges[edge2]):
                    line_graph.add_edge(i, j)

        # Step 2: Check for cycles in the line graph        
        for cycle in nx.simple_cycles(line_graph):
            involved_edges = [set(H.edges[dic[line_node]]) for line_node in cycle]
            intersection = set.intersection(*involved_edges)
            if not intersection:
                return False

        return True
    

def eval_acc_tree_hypergraph(H_list):
    count = 0
    for H in H_list:
        if is_hypertree(H):
            count += 1
    return count / float(len(H_list))



class ValidHypertree(Metric):
    def __str__(self):
        return "ValidHypertree"

    def __call__(self, reference_hypergraphs, predicted_hypergraphs, train_hypergraphs):
        return eval_acc_tree_hypergraph(predicted_hypergraphs)
    

class ValidSBM(Metric):
    def __str__(self):
        return "ValidSBM"

    def __call__(self, reference_hypergraphs, predicted_hypergraphs, train_hypergraphs):
        return eval_acc_sbm_hypergraph(predicted_hypergraphs)
