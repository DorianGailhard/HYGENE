###############################################################################
#
# Adapted from https://github.com/AndreasBergmeister/graph-generation/ which in turn is adapted
# from https://github.com/lrjconan/GRAN/ which in turn is adapted from https://github.com/JiaxuanYou/graph-generation
#
###############################################################################


import numpy as np
import concurrent.futures
import pygsp as pg
from datetime import datetime
from scipy.linalg import eigvalsh
from util.dist_helper import compute_mmd, gaussian_emd, emd, gaussian_tv
import networkx as nx
import math
from scipy.stats import chi2
import graph_tool.all as gt
from functools import lru_cache

PRINT_TIME = False
__all__ = [
        'spectral_stats',
]

def normalized_laplacian_matrix(H):
    # Compute the incidence matrix
    incidence_matrix = H.incidence_matrix().toarray()
    
    # Compute the degree of nodes and hyperedges
    node_degree = np.sum(incidence_matrix, axis=1)
    hyperedge_degree = np.sum(incidence_matrix, axis=0)
    
    # Compute the diagonal matrices Dv and De
    Dv = np.diag(node_degree)
    De = np.diag(hyperedge_degree)
    
    # Compute the inverse square root of Dv
    Dv_inv_sqrt = np.linalg.inv(np.sqrt(Dv))
    
    # Compute the inverse of De
    De_inv = np.linalg.inv(De)
    
    # Compute the normalized Laplacian
    normalized_laplacian = np.eye(Dv.shape[0]) - Dv_inv_sqrt @ incidence_matrix @ De_inv @ incidence_matrix.T @ Dv_inv_sqrt

    return normalized_laplacian

def spectral_worker(H, n_eigvals=-1):
    # eigs = nx.laplacian_spectrum(G)
    try:
        eigs = eigvalsh(normalized_laplacian_matrix(H))  
    except:
        eigs = np.zeros(len(H))
    if n_eigvals > 0:
        eigs = eigs[1:n_eigvals+1]
    spectral_pmf, _ = np.histogram(eigs, bins=200, range=(-1e-5, 2), density=False)
    spectral_pmf = spectral_pmf / spectral_pmf.sum()
    return spectral_pmf

def get_spectral_pmf(eigs, max_eig):
    spectral_pmf, _ = np.histogram(np.clip(eigs, 0, max_eig), bins=200, range=(-1e-5, max_eig), density=False)
    spectral_pmf = spectral_pmf / spectral_pmf.sum()
    return spectral_pmf

def eigval_stats(eig_ref_list, eig_pred_list, max_eig=20, is_parallel=True, compute_emd=False):
    ''' Compute the distance between the degree distributions of two unordered sets of hypergraphs.
        Args:
            hypergraph_ref_list, hypergraph_target_list: two lists of hypernetx hypergraphs to be evaluated
        '''
    sample_ref = []
    sample_pred = []

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for spectral_density in executor.map(get_spectral_pmf, eig_ref_list, [max_eig for i in range(len(eig_ref_list))]):
                sample_ref.append(spectral_density)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for spectral_density in executor.map(get_spectral_pmf, eig_pred_list, [max_eig for i in range(len(eig_ref_list))]):
                sample_pred.append(spectral_density)
    else:
        for i in range(len(eig_ref_list)):
            spectral_temp = get_spectral_pmf(eig_ref_list[i])
            sample_ref.append(spectral_temp)
        for i in range(len(eig_pred_list)):
            spectral_temp = get_spectral_pmf(eig_pred_list[i])
            sample_pred.append(spectral_temp)

    # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_emd)
    if compute_emd:
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=emd)
    else:
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_tv)
    # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian)

    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print('Time computing eig mmd: ', elapsed)
    return mmd_dist

def eigh_worker(H):
    L = normalized_laplacian_matrix(H)
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
    except:
        eigvals = np.zeros(L[0,:].shape)
        eigvecs = np.zeros(L.shape)
    return (eigvals, eigvecs)

def compute_list_eigh(hypergraph_list, is_parallel=False):
    eigval_list = []
    eigvec_list = []
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for e_U in executor.map(eigh_worker, hypergraph_list):
                eigval_list.append(e_U[0])
                eigvec_list.append(e_U[1])
    else:
        for i in range(len(hypergraph_list)):
            e_U = eigh_worker(hypergraph_list[i])
            eigval_list.append(e_U[0])
            eigvec_list.append(e_U[1])
    return eigval_list, eigvec_list

def get_spectral_filter_worker(eigvec, eigval, filters, bound=1.4):
    ges = filters.evaluate(eigval)
    linop = []
    for ge in ges:
        linop.append(eigvec @ np.diag(ge) @ eigvec.T)
    linop = np.array(linop)
    norm_filt = np.sum(linop**2, axis=2)
    hist_range = [0, bound]
    hist = np.array([np.histogram(x, range=hist_range, bins=100)[0] for x in norm_filt]) #NOTE: change number of bins
    return hist.flatten()

def spectral_filter_stats(eigvec_ref_list, eigval_ref_list, eigvec_pred_list, eigval_pred_list, is_parallel=False, compute_emd=False):
    ''' Compute the distance between the eigvector sets.
        Args:
            hypergraph_ref_list, hypergraph_target_list: two lists of hypernetx hypergraphs to be evaluated
        '''
    prev = datetime.now()
    class DMG(object):
        """Dummy Normalized hypergraph"""
        lmax = 2
    n_filters = 12
    filters = pg.filters.Abspline(DMG,n_filters)
    bound = np.max(filters.evaluate(np.arange(0, 2, 0.01)))
    sample_ref = []
    sample_pred = []
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for spectral_density in executor.map(get_spectral_filter_worker, eigvec_ref_list, eigval_ref_list, [filters for i in range(len(eigval_ref_list))], [bound for i in range(len(eigval_ref_list))]):
                sample_ref.append(spectral_density)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for spectral_density in executor.map(get_spectral_filter_worker, eigvec_pred_list, eigval_pred_list, [filters for i in range(len(eigval_pred_list))], [bound for i in range(len(eigval_pred_list))]):
                sample_pred.append(spectral_density)
    else:
        for i in range(len(eigval_ref_list)):
            try:
                spectral_temp = get_spectral_filter_worker(eigvec_ref_list[i], eigval_ref_list[i], filters, bound)
                sample_ref.append(spectral_temp)
            except:
                pass
        for i in range(len(eigval_pred_list)):
            try:
                spectral_temp = get_spectral_filter_worker(eigvec_pred_list[i], eigval_pred_list[i], filters, bound)
                sample_pred.append(spectral_temp)
            except:
                pass
    
    if compute_emd:
        # EMD option uses the same computation as hypergraphRNN, the alternative is MMD as computed by GRAN
        # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=emd)
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_emd)
    else:
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_tv)

    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print('Time computing spectral filter stats: ', elapsed)
    return mmd_dist



def spectral_stats(hypergraph_ref_list, hypergraph_pred_list, is_parallel=True, n_eigvals=-1, compute_emd=False):
    ''' Compute the distance between the degree distributions of two unordered sets of hypergraphs.
        Args:
            hypergraph_ref_list, hypergraph_target_list: two lists of hypernetx hypergraphs to be evaluated
        '''
    sample_ref = []
    sample_pred = []
    # in case an empty hypergraph is generated
    hypergraph_pred_list_remove_empty = [
            H for H in hypergraph_pred_list if not len(H) == 0
    ]

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for spectral_density in executor.map(spectral_worker, hypergraph_ref_list, [n_eigvals for i in hypergraph_ref_list]):
                sample_ref.append(spectral_density)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for spectral_density in executor.map(spectral_worker, hypergraph_pred_list_remove_empty, [n_eigvals for i in hypergraph_pred_list_remove_empty]):
                sample_pred.append(spectral_density)
    else:
        for i in range(len(hypergraph_ref_list)):
            spectral_temp = spectral_worker(hypergraph_ref_list[i], n_eigvals)
            sample_ref.append(spectral_temp)
        for i in range(len(hypergraph_pred_list_remove_empty)):
            spectral_temp = spectral_worker(hypergraph_pred_list_remove_empty[i], n_eigvals)
            sample_pred.append(spectral_temp)

    # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_emd)
    # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=emd)
    if compute_emd:
        # EMD option uses the same computation as hypergraphRNN, the alternative is MMD as computed by GRAN
        # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=emd)
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_emd)
    else:
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_tv)
    # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian)

    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print('Time computing degree mmd: ', elapsed)
    return mmd_dist

###############################################################################

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

###############################################################################

@lru_cache(maxsize=None)
def compute_sorted_spectrum(hg):
    """Compute and sort the singular values of a hypergraph's incidence matrix."""
    inc = hg.incidence_matrix().toarray()
    sv = np.linalg.svd(inc, compute_uv=False)
    return tuple(sorted(sv))

def could_be_isomorphic(hg1, hg2):
    """Simple heuristic to check if two hypergraphs are not isomorphic by comparing the singular values of their incidence matrices."""
    # Compute the sorted spectra using the cached function
    sv1 = compute_sorted_spectrum(hg1)
    sv2 = compute_sorted_spectrum(hg2)
    
    # Compare the singular values
    return len(sv1) == len(sv2) and np.allclose(sv1, sv2)


def eval_fraction_isomorphic(fake_hypergraphs, train_hypergraphs):
    count = 0
    for fake_g in fake_hypergraphs:
        for train_g in train_hypergraphs:
            if could_be_isomorphic(fake_g, train_g):
                    count += 1
                    break
    return count / float(len(fake_hypergraphs))

def eval_fraction_unique(fake_hypergraphs, precise=False):
    count_non_unique = 0
    fake_evaluated = []
    for fake_g in fake_hypergraphs:
        unique = True
        if not len(fake_g.nodes) == 0:
            for fake_old in fake_evaluated:
                if could_be_isomorphic(fake_g, fake_old):
                    count_non_unique += 1
                    unique = False
                    break
                
            if unique:
                fake_evaluated.append(fake_g)

    frac_unique = (float(len(fake_hypergraphs)) - count_non_unique) / float(len(fake_hypergraphs)) # Fraction of distinct isomorphism classes in the fake graphs

    return frac_unique