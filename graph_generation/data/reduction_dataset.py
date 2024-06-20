from abc import ABC

import numpy as np
import scipy as sp
import torch as th
from torch.utils.data import IterableDataset
from torch_geometric.data import Data
from torch_geometric.typing import SparseTensor

from ..reduction import ReductionFactory


class RandRedDataset(IterableDataset, ABC):
    def __init__(self, hypergraphs, red_factory: ReductionFactory, spectrum_extractor):
        super().__init__()

        self.red_factory = red_factory
        self.hypergraphs = hypergraphs
        self.spectrum_extractor = spectrum_extractor

    def get_random_reduction_sequence(self, hypergraph, rng):
        data = []
        while True:
            reduced_hypergraph = hypergraph.get_reduced_graph(rng)
            data.append(
                ReducedGraphData(
                    target_size=hypergraph.n,
                    reduction_level=hypergraph.level,
                    adj=hypergraph.adj.astype(bool).astype(np.float32),
                    node_expansion=hypergraph.node_expansion,
                    edge_expansion=hypergraph.edge_expansion,
                    adj_reduced=reduced_hypergraph.adj.astype(bool).astype(np.float32),
                    expansion_matrix=reduced_hypergraph.expansion_matrix,
                    spectral_features_reduced=self.spectrum_extractor(reduced_hypergraph.adj)
                    if self.spectrum_extractor is not None
                    else None,
                )
            )
            if hypergraph.n <= 1:
                break
            hypergraph = reduced_hypergraph

        return data


class FiniteRandRedDataset(RandRedDataset):
    def __init__(
        self, hypergraphs, red_factory: ReductionFactory, spectrum_extractor, num_red_seqs
    ):
        super().__init__(hypergraphs, red_factory, spectrum_extractor)
        self.num_red_seqs = num_red_seqs

        self.rng = np.random.default_rng(seed=0)
        self.hypergraphs_reduced_data = {i: [] for i in range(len(hypergraphs))}
        for i, hypergraph in enumerate(hypergraphs):
            hypergraph = red_factory(hypergraph)
            for _ in range(num_red_seqs):
                self.hypergraph_reduced_data[i] += self.get_random_reduction_sequence(
                    hypergraph, self.rng
                )

    def __iter__(self):
        while True:
            i = self.rng.integers(len(self.hypergraphs))
            j = self.rng.integers(len(self.hypergraph_reduced_data[i]))
            yield self.hypergraph_reduced_data[i][j]

    @property
    def max_node_expansion(self):
        return max(
            [
                rgd.node_expansion.max().item()
                for seq in self.hypergraph_reduced_data
                for rgd in seq
            ]
        )


class InfiniteRandRedDataset(RandRedDataset):
    def __iter__(self):
        # get process id
        worker_info = th.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        rng = np.random.default_rng(worker_id)

        # initialize graph_reduced_data
        hypergraph_reduced_data = {
            i: self.get_random_reduction_sequence(hypergraph, rng)
            for i, hypergraph in enumerate(self.hypergraphs)
        }

        # yield random reduced graph data
        while True:
            i = rng.integers(len(self.hypergraphs))
            if len(hypergraph_reduced_data[i]) == 0:
                hypergraph_reduced_data[i] = self.get_random_reduction_sequence(
                    self.hypergraphs[i], rng
                )
                rng.shuffle(hypergraph_reduced_data[i])

            yield hypergraph_reduced_data[i].pop()

    @property
    def max_node_expansion(self):
        raise NotImplementedError


class ReducedGraphData(Data):
    def __init__(self, **kwargs):
        if not kwargs:
            super().__init__()
            return

        super().__init__(x=th.zeros(kwargs["adj"].shape[0]))
        for key, value in kwargs.items():
            if value is None:
                continue
            elif isinstance(value, int):
                value = th.tensor(value).type(th.long)
            elif isinstance(value, np.ndarray):
                value = th.from_numpy(value).type(
                    th.float32 if np.issubdtype(value.dtype, np.floating) else th.long
                )
            elif isinstance(value, sp.sparse.sparray):
                value = SparseTensor.from_scipy(value).type(
                    th.float32 if np.issubdtype(value.dtype, np.floating) else th.long
                )
            else:
                raise ValueError(f"Unsupported type {type(value)} for key {key}")

            setattr(self, key, value)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if isinstance(value, SparseTensor):
            return (0, 1)  # concatenate along diagonal
        return super().__cat_dim__(key, value, *args, **kwargs)
