import hypernetx as hnx
import torch as th
from torch.nn import Module
from torch_geometric.utils import to_edge_index
from torch_scatter import scatter
from torch_sparse import SparseTensor
import numpy as np

from .method import Method


class Expansion(Method):
    """Hypergraph generation method generating graphs by local expansion."""

    def __init__(
        self,
        diffusion,
        spectrum_extractor,
        emb_dim,
        augmented_radius=1,
        augmented_dropout=0.0,
        deterministic_expansion=False,
        min_red_frac=0.0,
        max_red_frac=0.5,
        red_threshold=0,
    ):
        self.diffusion = diffusion
        self.spectrum_extractor = spectrum_extractor
        self.emb_dim = emb_dim
        self.augmented_radius = augmented_radius
        self.augmented_dropout = augmented_dropout
        self.deterministic_expansion = deterministic_expansion
        self.min_red_frac = min_red_frac
        self.max_red_frac = max_red_frac
        self.red_threshold = red_threshold

    def sample_hypergraphs(self, target_size, model: Module, sign_net: Module):
        """Samples a batch of hypergraphs."""
        num_hypergraphs = len(target_size)
        
        dense_tensor = th.zeros(num_hypergraphs*2, num_hypergraphs*2, device=self.device)
        dense_tensor[th.arange(num_hypergraphs, device=self.device)*2, 1 + th.arange(num_hypergraphs, device=self.device)*2] = 1
        dense_tensor[1 + th.arange(num_hypergraphs, device=self.device)*2, th.arange(num_hypergraphs, device=self.device)*2] = 1
        
        adj = SparseTensor.from_dense(dense_tensor)

        batch = th.repeat_interleave(th.arange(0, num_hypergraphs, device=self.device), 2*th.ones(num_hypergraphs, dtype = th.int32, device=self.device))
        node_expansion = th.ones(num_hypergraphs*2, dtype=th.long, device=self.device)
        
        # Node_type == 0 correspond to hyperedges (right side nodes), == 1 corresponds to nodes (left side nodes)
        node_type = th.ones(num_hypergraphs*2, dtype=th.int, device=self.device)
        node_type[1::2] = 0
        
        while node_type.sum() < target_size.sum():
            adj, batch, node_expansion, node_type = self.expand(
                adj,
                batch,
                node_expansion,
                node_type,
                target_size,
                model=model,
                sign_net=sign_net,
            )
            if node_expansion.max() <= 1:
                break
                
        # return hypergraphs
        adjs, num_nodes = unbatch_adj(adj, batch, node_type)
        hypergraphs = []
        
        for (adj, n) in zip(adjs, num_nodes):
            # remind that we have :
            # Adj_bipartite = ( 0  H)
            #                 (H^T 0)
            adj = adj.to_dense().cpu().numpy()
            
            if np.max(adj) > 0.:
                num_hyperedges = adj.shape[0] - n
    
                incidence_matrix = adj[:n, n:n + num_hyperedges]
                
                H = hnx.Hypergraph.from_incidence_matrix(incidence_matrix)
                hypergraphs.append(H)
            else:
                H = hnx.Hypergraph(np.ones((2,2)))
                hypergraphs.append(H)
        
        return hypergraphs

    @th.no_grad()
    def expand(
        self,
        adj_reduced,
        batch_reduced,
        node_expansion,
        node_type,
        target_size,
        model: Module,
        sign_net: Module,
    ):
        """Expands a hypergraph by a single level."""
        reduced_size = scatter(node_type, batch_reduced)

        # expand
        # don't expand hypergraphs reached their target size
        node_expansion[(reduced_size >= target_size)[batch_reduced]] = 1
        node_map = th.repeat_interleave(
            th.arange(0, adj_reduced.size(0), device=self.device), node_expansion
        )
        
        expanded_node_type = node_type[node_map]
        batch = batch_reduced[node_map]
        size = scatter(expanded_node_type, batch)
        expansion_matrix = SparseTensor(
            row=th.arange(node_map.size(0), device=self.device),
            col=node_map,
            value=th.ones(node_map.size(0), device=self.device),
        )
        adj_augmented = self.get_augmented_hypergraph(adj_reduced, expansion_matrix)
        augmented_incidence_index = th.stack(adj_augmented.coo()[:2], dim=0)
        
        # get node embeddings
        if self.spectrum_extractor is not None:
            spectral_features = th.cat(
                [
                    th.tensor(
                        self.spectrum_extractor(adj.to("cpu").to_scipy(layout="coo")),
                        dtype=th.float32,
                        device=self.device,
                    )
                    for adj in unbatch_adj(adj_reduced, batch_reduced, node_type)[0]
                ]
            )
            both_type_node_emb_reduced = sign_net(
                spectral_features=spectral_features, edge_index=adj_reduced
            )
        else:
            both_type_node_emb_reduced = th.randn(
                node_type.size(0), self.emb_dim, device=self.device
            )
            
        both_type_node_emb = both_type_node_emb_reduced[node_map]
        node_emb = both_type_node_emb[expanded_node_type == 1]
        hyperedge_emb = both_type_node_emb[expanded_node_type == 0]
        
        # compute number of nodes in expanded hypergraph
        random_reduction_fraction = (
            th.rand(len(target_size), device=self.device)
            * (self.max_red_frac - self.min_red_frac)
            + self.min_red_frac
        )

        # if expanded number of nodes is less than threshold, use max_red_frac
        max_reduction_mask = (
            th.ceil(size / (1 - self.max_red_frac)) <= self.red_threshold
        ).float()
        random_reduction_fraction = (
            1 - max_reduction_mask
        ) * random_reduction_fraction + max_reduction_mask * self.max_red_frac

        # expanded number of nodes is ⌈n / (1-r)⌉ and at least n+1 and at most target_size
        expanded_size = th.minimum(
            th.maximum(
                th.ceil(size / (1 - random_reduction_fraction)).long(),
                size + 1,
            ),
            target_size,
        )

        # make predictions
        node_pred, hyperedge_pred, augmented_incidence_pred = self.diffusion.sample(
            incidence_index=augmented_incidence_index,
            batch=batch,
            node_type=expanded_node_type,
            model=model,
            model_kwargs={
                "node_emb": node_emb,
                "hyperedge_emb": hyperedge_emb,
                "red_frac": 1 - size / expanded_size,
                "target_size": target_size.float(),
            },
        )

        # get node attributes
        if self.deterministic_expansion:
            node_attr = th.zeros_like(node_pred, dtype=th.long)
            num_new_nodes = expanded_size - size
            node_range_end = size.cumsum(0)
            node_range_start = node_range_end - size
            # get top-k nodes per graph
            for i in range(len(target_size)):
                new_node_idx = (
                    th.topk(
                        node_pred[node_range_start[i] : node_range_end[i]],
                        num_new_nodes[i],
                        largest=True,
                    )[1]
                    + node_range_start[i]
                )
                node_attr[new_node_idx] = 1
        else:
            node_attr = (node_pred > 0.5).long()
        
        hyperedge_attr = (hyperedge_pred > 0.66).long() + (hyperedge_pred > 1.33).long()
            
        # construct new hypergraph
        adj = SparseTensor.from_edge_index(
            augmented_incidence_index[:, augmented_incidence_pred > 0.5],
            sparse_sizes=adj_augmented.sizes(),
            edge_attr = th.ones(th.sum(augmented_incidence_pred > 0.5), device=augmented_incidence_pred.device)
        )
        
        all_node_attr = th.zeros(batch.size(0), dtype=th.long, device = node_attr.device)
        all_node_attr[expanded_node_type == 0] = hyperedge_attr
        all_node_attr[expanded_node_type == 1] = node_attr
        
        return adj, batch, all_node_attr + 1, expanded_node_type

    def get_loss(self, batch, model: Module, sign_net: Module):
        """Returns a weighted sum of the node and incidence expansion loss and the augmented incidence loss."""
        # get augmented hypergraph
        adj_augmented = self.get_augmented_hypergraph(
            batch.adj_reduced, batch.expansion_matrix
        )

        # construct labels
        node_attr = batch.node_expansion - 1
        hyperedge_attr = batch.hyperedge_expansion - 1
        augmented_incidence_index, incidence_val = to_edge_index(adj_augmented + batch.adj)
        augmented_incidence_attr = incidence_val.long() - 1
        
        # get node embeddings
        if sign_net is not None:
            both_type_node_emb_reduced = sign_net(
                spectral_features=batch.spectral_features_reduced,
                edge_index=batch.adj_reduced,
            )
        else:
            both_type_node_emb_reduced = th.randn(
                batch.node_type.size(0), self.emb_dim, device=self.device
            )
                
        both_type_node_emb = batch.expansion_matrix @ both_type_node_emb_reduced
        node_emb = both_type_node_emb[batch.node_type == 1]
        hyperedge_emb = both_type_node_emb[batch.node_type == 0]
            
        # reduction fraction
        size = scatter(batch.node_type, batch.batch)
        expanded_size = scatter(batch.node_expansion, batch.batch[batch.node_type == 1])
        red_frac = 1 - size / expanded_size

        # loss
        node_loss, hyperedge_loss, incidence_loss = self.diffusion.get_loss(
            incidence_index=augmented_incidence_index,
            batch=batch.batch,
            node_type=batch.node_type,
            node_attr=node_attr,
            hyperedge_attr=hyperedge_attr,
            incidence_attr=augmented_incidence_attr,
            model=model,
            model_kwargs={
                "node_emb": node_emb,
                "hyperedge_emb": hyperedge_emb,
                "red_frac": red_frac,
                "target_size": batch.target_size.float(),
            },
        )

        # ignore node_loss for first level
        node_loss = node_loss[batch.reduction_level[batch.batch[batch.node_type == 1]] > 0].mean()
        hyperedge_loss = hyperedge_loss[batch.reduction_level[batch.batch[batch.node_type == 0]] > 0].mean()
        incidence_loss = incidence_loss.mean()
        loss = node_loss + hyperedge_loss + incidence_loss
            
        return loss, {
            "node_expansion_loss": node_loss.item(),
            "hyperedge_expansion_loss": hyperedge_loss.item(),
            "augmented_incidence_loss": incidence_loss.item(),
            "loss": loss.item(),
        }

    def get_augmented_hypergraph(self, adj_reduced, expansion_matrix):
        """Returns the expanded bipartite adjacency matrix with additional augmented incidences.

        All incidence weights are set to 1.
        """
        # construct augmented adjacency matrix
        adj_reduced_augmented = adj_reduced.copy()
        
        if self.augmented_radius > 1:
            adj_reduced_square = (adj_reduced @ adj_reduced).set_diag(1)
            for _ in range(1, self.augmented_radius):
                adj_reduced_augmented = adj_reduced_augmented @ adj_reduced_square

            adj_reduced_augmented = adj_reduced_augmented.set_value(
                th.ones(adj_reduced_augmented.nnz(), device=self.device), layout="coo"
            )
            
            adj_reduced_augmented = adj_reduced_augmented + adj_reduced
            
        adj_augmented = (
            expansion_matrix @ adj_reduced_augmented @ expansion_matrix.t()
        )
        
        # drop out incidences
        if self.augmented_radius > 1:
            if self.augmented_dropout > 0.0:
                row, col, val = adj_augmented.coo()
                incidence_mask = th.rand_like(val) >= self.augmented_dropout
                incidence_mask = incidence_mask | (val > 1)  # keep required incidences
                
                # make undirected
                incidence_mask = incidence_mask & (row < col)
                incidence_index = th.stack([row[incidence_mask], col[incidence_mask]], dim=0)
                incidence_index = th.cat([incidence_index, incidence_index.flip(0)], dim=1)
                adj_augmented = SparseTensor.from_edge_index(
                    incidence_index,
                    edge_attr=th.ones(incidence_index.shape[1], device=self.device),
                    sparse_sizes=adj_augmented.sizes(),
                )
            else:
                adj_augmented.values().fill_(1)
                
        return adj_augmented


def unbatch_adj(adj, batch, node_type) -> tuple:
    size = scatter(th.ones_like(batch), batch)
    hypergraph_end_idx = size.cumsum(0)
    hypergraph_start_idx = hypergraph_end_idx - size
    return [ adj[hypergraph_start_idx[i] : hypergraph_end_idx[i], :][:, hypergraph_start_idx[i] : hypergraph_end_idx[i] ]
    for i in range(len(size)) ], [th.sum(node_type[hypergraph_start_idx[i] : hypergraph_end_idx[i]]).item() for i in range(len(size))]
