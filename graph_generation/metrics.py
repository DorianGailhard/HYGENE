from util.eval_helper import (
    clustering_stats,
    compute_list_eigh,
    degree_stats,
    eval_acc_planar_hypergraph,
    eval_acc_sbm_hypergraph,
    eval_acc_tree_hypergraph,
    eval_fraction_isomorphic,
    eval_fraction_unique,
    eval_fraction_unique_non_isomorphic_valid,
    is_planar_hypergraph,
    is_sbm_hypergraph,
    orbit_stats_all,
    spectral_filter_stats,
    spectral_stats,
)

from abc import ABC, abstractmethod
from dataclasses import dataclass

import hypernetx as hnx

import numpy as np

@dataclass
class Metric(ABC):
    @abstractmethod
    def __str__(self) -> str:
        pass

    @abstractmethod
    def __call__(
        self,
        reference_hypergraphs: list[hnx.Hyperhypergraph],
        predicted_hypergraphs: list[hnx.Hyperhypergraph],
        train_hypergraphs: list[hnx.Hyperhypergraph],
    ) -> float:
        pass


class NodeNumDiff(Metric):
    def __str__(self):
        return "NodeNumDiff"

    def __call__(self, reference_hypergraphs, predicted_hypergraphs, train_hypergraphs):
        ref_node_num = np.array([len(H)
                                for H in reference_hypergraphs])
        pred_node_num = np.array([len(H)
                                 for H in predicted_hypergraphs])
        return np.mean(np.abs(ref_node_num - pred_node_num))


class NodeNumDiff(Metric):
    def __str__(self):
        return "NodeNumDiff"

    def __call__(self, reference_hypergraphs, predicted_hypergraphs, train_hypergraphs):
        ref_node_num = np.array([len(H)
                                for H in reference_hypergraphs])
        pred_node_num = np.array([len(H)
                                 for H in predicted_hypergraphs])
        return np.mean(np.abs(ref_node_num - pred_node_num))
    

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