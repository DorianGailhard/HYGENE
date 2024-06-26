from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np
import scipy as sp
from numpy.typing import NDArray
from scipy.sparse import coo_array, csr_array, eye, hstack, vstack, diags, csc_matrix, bmat

import hypernetx as hnx

real = np.floating | float


class Reduction(ABC):
    """Abstract class for graph reduction."""

    preserved_eig_size: int
    local_variation_cost: bool
    sqrt_partition_size: bool
    weighted_reduction: bool
    min_red_frac: real
    max_red_frac: real
    red_threshold: int
    rand_lambda: int

    def __init__(self, bipartite_adj, clique_adj, lap=None, B=None, expansion_matrix=None, level=0):
        self.bipartite_adj = bipartite_adj
        self.clique_adj = clique_adj
        self.n = self.clique_adj.shape[0]
        self.node_degree = self.clique_adj.sum(0)
        self.lap = sp.sparse.diags(self.node_degree) - self.clique_adj if lap is None else lap
        if B is None:
            self.B = self.get_B0()
            self.A = self.B
        else:
            self.B = B
            self.A = self.get_A()

        self.expansion_matrix = expansion_matrix
        self.node_expansion = (
            np.ones(self.n, dtype=np.int32)
            if expansion_matrix is None
            else expansion_matrix.sum(0)[:self.n].astype(np.int32)
        )
        self.edge_expansion = (
            np.ones(self.bipartite_adj.shape[0] - self.n, dtype=np.int32)
            if expansion_matrix is None
            else expansion_matrix.sum(0)[self.n:].astype(np.int32)
        )
        self.level = level

    def get_reduced_hypergraph(self, rng=np.random.default_rng()):
        # Compute the coarsened clique representation
        C = self.get_coarsening_matrix(rng)
        
        # P_inv = C.T with all non-zero entries set to 1
        P_inv = C.T.astype(bool).astype(C.dtype)

        if self.weighted_reduction:
            lap_reduced = P_inv.T @ self.lap @ P_inv
            adj_clique_reduced = -lap_reduced + sp.sparse.diags(lap_reduced.diagonal())
        else:
            lap_reduced = None
            adj_clique_reduced = (P_inv.T @ self.clique_adj @ P_inv).tocoo()
            # remove self-loops and edge weights
            row, col = adj_clique_reduced.row, adj_clique_reduced.col
            mask = row != col
            row, col = row[mask], col[mask]
            adj_clique_reduced = sp.sparse.coo_array(
                (np.ones(len(row), dtype=adj_clique_reduced.dtype), (row, col)),
                shape=adj_clique_reduced.shape,
            )
            
        # Compute the coarsened bipartite representation
        incidence_matrix = self.bipartite_adj [:self.n, self.n:]
        
        coarsened_incidence = (P_inv.T@incidence_matrix)
        coarsened_incidence.data[:] = 1
        
        collapsed_coarsened_incidence, edge_lifting_matrix = self.collapse_duplicate_edges(coarsened_incidence)

        # Reconstruct the bipartite adjacency matrix
        collapsed_coarsened_incidence = csr_array(collapsed_coarsened_incidence)
        
        num_nodes, num_hyperedges = collapsed_coarsened_incidence.shape
        
        zero_nodes = csr_array((num_nodes, num_nodes))
        zero_hyperedges = csr_array((num_hyperedges, num_hyperedges))
        
        top = hstack([zero_nodes, collapsed_coarsened_incidence])
        bottom = hstack([collapsed_coarsened_incidence.T, zero_hyperedges])
        bipartite_adjacency = csr_array(vstack([top, bottom]))
        
        # Add the edge_lifting_matrix to the expansion matrix
        expansion_matrix = bmat([[P_inv, None], [None, edge_lifting_matrix]])
        
        return self.__class__(
            bipartite_adj=bipartite_adjacency,
            clique_adj=adj_clique_reduced,
            lap=lap_reduced,
            B=C @ self.B,
            expansion_matrix=expansion_matrix,
            level=self.level + 1,
        )

    def get_B0(self) -> NDArray:
        offset = 2 * np.max(self.node_degree)
        T = offset * sp.sparse.eye(self.n, format="csc") - self.lap
        lk, Uk = sp.sparse.linalg.eigsh(
            T, k=self.preserved_eig_size, which="LM", tol=1e-5
        )
        lk = (offset - lk)[::-1]
        Uk = Uk[:, ::-1]

        # compute L^-1/2
        mask = lk < 1e-5
        lk[mask] = 1
        lk_inv = 1 / np.sqrt(lk)
        lk_inv[mask] = 0
        return Uk * lk_inv[np.newaxis, :]  # = Uk @ np.diag(lk_inv)

    def get_A(self) -> NDArray:
        # A = B @ (B.T @ L @ B)^-1/2
        d, V = np.linalg.eig(self.B.T @ self.lap @ self.B)
        mask = d < 1e-8
        d[mask] = 1
        d_inv_sqrt = 1 / np.sqrt(d)
        d_inv_sqrt[mask] = 0
        return self.B @ np.diag(d_inv_sqrt) @ V
    
    def collapse_duplicate_edges(self, coarsened_incidence: csr_array):
        # Convert the matrix to CSC format for efficient column operations
        csc_matrix_form = csc_matrix(coarsened_incidence)
    
        # Create a dictionary to store unique columns and their counts
        unique_columns = {}
        column_map = {}
        next_col_index = 0
        mergings = {}
    
        # Iterate through each column
        for col_index in range(csc_matrix_form.shape[1]):
            # Extract the column as a tuple of (row_index, data)
            col_data = tuple(zip(csc_matrix_form[:, col_index].indices, csc_matrix_form[:, col_index].data))
    
            if col_data in unique_columns:
                mergings[unique_columns[col_data]].append(col_index)
            else:
                unique_columns[col_data] = next_col_index
                column_map[col_index] = next_col_index
                mergings[next_col_index] = [col_index]
                next_col_index += 1
    
        # Construct the collapsed matrix
        collapsed_data = []
        collapsed_rows = []
        collapsed_cols = []
        for original_col_index, new_col_index in column_map.items():
            col = csc_matrix_form[:, original_col_index]
            collapsed_data.extend(col.data)
            collapsed_rows.extend(col.indices)
            collapsed_cols.extend([new_col_index] * len(col.data))
    
        # Create the collapsed matrix
        collapsed_matrix = csr_array((collapsed_data, (collapsed_rows, collapsed_cols)), shape=(coarsened_incidence.shape[0], next_col_index))
    
        # Create the edge lifting matrix, which is an equivalent to the lifting matrix but for the edges node in the bipartite graph
        rows = []
        cols = []
    
        for col, col_idx in unique_columns.items():
            for node_idx in mergings[col_idx]:
                cols.append(col_idx)
                rows.append(node_idx)
    
        data = np.ones(len(rows))
    
        lifting_matrix = coo_array((data, (rows, cols)), shape=(coarsened_incidence.shape[1], len(unique_columns)))
    
        return collapsed_matrix, lifting_matrix
    

    def get_coarsening_matrix(self, rng) -> coo_array:
        # get the contraction sets and their costs
        contraction_sets = self.get_contraction_sets()
        costs = (
            np.apply_along_axis(self.get_cost, 1, contraction_sets)
            if len(contraction_sets) > 0
            else np.array([])
        )

        # compute reduction fraction
        if self.n <= self.red_threshold:
            reduction_fraction = self.max_red_frac
        else:
            reduction_fraction = rng.uniform(self.min_red_frac, self.max_red_frac)

        # get partitioning minimizing the cost in a randomized fashion
        perm = costs.argsort()
        contraction_sets = contraction_sets[perm]
        partitions = []
        marked = np.zeros(self.n, dtype=bool)
        for contraction_set in contraction_sets:
            if (
                not marked[contraction_set].any()
                and rng.uniform() >= self.rand_lambda  # randomize
            ):
                partitions.append(contraction_set)
                marked[contraction_set] = True
                break
        
        # construct projection matrix
        P = eye(self.n, format="lil")
        mask = np.ones(self.n, dtype=bool)
        for partition in partitions:
            size = len(partition)
            size = np.sqrt(size) if self.sqrt_partition_size else size
            P[partition[0], partition] = 1 / size
            mask[partition[1:]] = False
        P = P[mask, :]
        return coo_array(P, dtype=np.float64)


    @abstractmethod
    def get_contraction_sets(self) -> Sequence[NDArray]:
        pass

    def get_cost(self, nodes: NDArray) -> real:
        if self.local_variation_cost:
            return self.get_local_variation_cost(nodes)
        else:
            return np.random.rand()

    def get_local_variation_cost(self, nodes: NDArray) -> real:
        """Compute the local variation cost for a set of nodes"""
        nc = len(nodes)
        if nc == 1:
            return np.inf

        ones = np.ones(nc)
        W = self.clique_adj[nodes, :][:, nodes]
        L = np.diag(2 * self.node_degree[nodes] - W @ ones) - W
        B = (np.eye(nc) - np.outer(ones, ones) / nc) @ self.A[nodes, :]
        return np.linalg.norm(B.T @ L @ B) / (nc - 1)


class NeighborhoodReduction(Reduction):
    """Graph reduction by contracting neighborhoods."""

    def get_contraction_sets(self) -> Sequence[NDArray]:
        """Returns neighborhood contraction sets"""
        adj_with_self_loops = self.clique_adj.copy().tolil()
        adj_with_self_loops.setdiag(1)
        return [np.array(nbrs) for nbrs in adj_with_self_loops.rows]


class EdgeReduction(Reduction):
    """Graph reduction by contracting edges.

    Class implements optimized routines for local variation cost computation.
    """

    def get_contraction_sets(self) -> Sequence[NDArray]:
        us, vs, _ = sp.sparse.find(sp.sparse.triu(self.clique_adj))
        return np.stack([us, vs], axis=1)

    def get_local_variation_cost(self, edge: NDArray) -> real:
        """Compute the local variation cost for an edge"""
        u, v = edge
        w = self.clique_adj[u, v]
        L = np.array(
            [[2 * self.node_degree[u] - w, -w], [-w, 2 * self.node_degree[v] - w]]
        )
        B = self.A[edge, :]
        return np.linalg.norm(B.T @ L @ B)


class ReductionFactory:
    def getBipartiteAdj(self, H) -> csr_array:
        incidence_matrix = H.incidence_matrix()
        
        # Ensure the incidence matrix is in CSR format
        incidence_matrix = csr_array(incidence_matrix)
        
        # Number of nodes and hyperedges
        num_nodes, num_hyperedges = incidence_matrix.shape
        
        # Create zero matrices for the non-connected parts
        zero_nodes = csr_array((num_nodes, num_nodes))
        zero_hyperedges = csr_array((num_hyperedges, num_hyperedges))
        
        # Stack and concatenate the matrices to form the bipartite adjacency matrix
        top = hstack([zero_nodes, incidence_matrix])
        bottom = hstack([incidence_matrix.T, zero_hyperedges])
        bipartite_adjacency = vstack([top, bottom])
        
        return csr_array(bipartite_adjacency)

    def getCliqueAdj(self, H) -> csr_array:
        incidence_matrix = csr_array(H.incidence_matrix())
        
        # Compute the diagonal matrix D_E
        hyperedge_degrees = np.array(incidence_matrix.sum(axis=0)).flatten()
        D_E_inv = diags(1 / hyperedge_degrees)
        
        # Compute the adjacency matrix of the weighted clique expansion
        clique_adjacency = incidence_matrix @ D_E_inv @ incidence_matrix.T
        
        # Subtract the diagonal
        diagonal = clique_adjacency.diagonal()
        diag_matrix = sp.sparse.diags(diagonal, offsets=0, shape=clique_adjacency.shape, format='csr')

        clique_adjacency = clique_adjacency - diag_matrix

        return csr_array(clique_adjacency)
    
    
    def __init__(
        self,
        contraction_family,
        cost_type,
        preserved_eig_size,
        sqrt_partition_size,
        weighted_reduction,
        min_red_frac,
        max_red_frac,
        red_threshold,
        rand_lambda,
    ):
        self.contraction_family = contraction_family
        self.cost_type = cost_type
        self.preserved_eig_size = preserved_eig_size
        self.sqrt_partition_size = sqrt_partition_size
        self.weighted_reduction = weighted_reduction
        self.min_red_frac = min_red_frac
        self.max_red_frac = max_red_frac
        self.red_threshold = red_threshold
        self.rand_lambda = rand_lambda

    def __call__(self, H: hnx.Hypergraph) -> Reduction:
        if self.contraction_family == "neighborhoods":
            reduction = NeighborhoodReduction
        elif self.contraction_family == "edges":
            reduction = EdgeReduction
        else:
            raise ValueError("Unknown contraction family.")

        if self.cost_type == "local_variation":
            reduction.local_variation_cost = True
        elif self.cost_type == "random":
            reduction.local_variation_cost = False
        else:
            raise ValueError("Unknown reduction cost type.")

        reduction.preserved_eig_size = self.preserved_eig_size
        reduction.sqrt_partition_size = self.sqrt_partition_size
        reduction.weighted_reduction = self.weighted_reduction
        reduction.min_red_frac = self.min_red_frac
        reduction.max_red_frac = self.max_red_frac
        reduction.red_threshold = self.red_threshold
        reduction.rand_lambda = self.rand_lambda

        return reduction(self.getBipartiteAdj(H), self.getCliqueAdj(H))
