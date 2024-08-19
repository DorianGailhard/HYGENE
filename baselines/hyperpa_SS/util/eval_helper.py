###############################################################################
#
# Adapted from https://github.com/AndreasBergmeister/hypergraph-generation/ which in turn is adapted
# from https://github.com/lrjconan/GRAN/ which in turn is adapted from https://github.com/JiaxuanYou/hypergraph-generation
#
###############################################################################


import numpy as np
import concurrent.futures
import pygsp as pg
from datetime import datetime
from scipy.linalg import eigvalsh
from util.dist_helper import compute_mmd, gaussian_emd, emd, gaussian_tv

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

def could_be_isomorphic(hg1, hg2):
    """ Simple heuristic to check if two hypergraphs are not isomorphic by comparing the singular values of their incidence matrices."""
    # Compute the incidence matrices
    inc1 = hg1.incidence_matrix().toarray()
    inc2 = hg2.incidence_matrix().toarray()

    # Compute the singular values and sort them
    sv1 = np.linalg.svd(inc1, compute_uv=False)
    sv1.sort()
    sv2 = np.linalg.svd(inc2, compute_uv=False)
    sv2.sort()

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
