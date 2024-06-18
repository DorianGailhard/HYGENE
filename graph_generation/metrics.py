from abc import ABC, abstractmethod
from dataclasses import dataclass

import hypernetx as hnx

import numpy as np

from collections import Counter
from scipy.spatial.distance import wasserstein_distance

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


class DegreeDistrWasserstein(Metric):
    def __str__(self):
        return "DegreeDistrWasserstein"

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