import os
import pickle
import random
from collections import defaultdict
from pathlib import Path

import hydra
import hypernetx as hnx
import numpy as np
import torch as th
import torch.multiprocessing as mp
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

import graph_generation as gg


def get_expansion_items(cfg: DictConfig, train_hypergraphs):
    # Spectral Features
    spectrum_extractor = (
        gg.spectral.SpectrumExtractor(
            num_features=cfg.spectral.num_features,
            normalized=cfg.spectral.normalized_laplacian,
        )
        if cfg.spectral.num_features > 0
        else None
    )

    # Train Dataset
    red_factory = gg.reduction.ReductionFactory(
        contraction_family=cfg.reduction.contraction_family,
        cost_type=cfg.reduction.cost_type,
        preserved_eig_size=cfg.reduction.preserved_eig_size,
        sqrt_partition_size=cfg.reduction.sqrt_partition_size,
        weighted_reduction=cfg.reduction.weighted_reduction,
        min_red_frac=cfg.reduction.min_red_frac,
        max_red_frac=cfg.reduction.max_red_frac,
        red_threshold=cfg.reduction.red_threshold,
        rand_lambda=cfg.reduction.rand_lambda,
    )

    if cfg.reduction.num_red_seqs > 0:
        train_dataset = gg.data.FiniteRandRedDataset(
            hypergraphs=train_hypergraphs,
            red_factory=red_factory,
            spectrum_extractor=spectrum_extractor,
            num_red_seqs=cfg.reduction.num_red_seqs,
        )
    else:
        train_dataset = gg.data.InfiniteRandRedDataset(
            hypergraphs=train_hypergraphs,
            red_factory=red_factory,
            spectrum_extractor=spectrum_extractor,
        )

    # Dataloader
    is_mp = cfg.reduction.num_red_seqs < 0  # if infinite dataset
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        pin_memory=True,
        collate_fn=Batch.from_data_list,
        num_workers=min(mp.cpu_count(), cfg.training.max_num_workers) * is_mp,
        multiprocessing_context="spawn" if is_mp else None,
    )

    # Model
    if cfg.spectral.num_features > 0:
        sign_net = gg.model.SignNet(
            num_eigenvectors=cfg.spectral.num_features,
            hidden_features=cfg.sign_net.hidden_features,
            out_features=cfg.model.emb_features,
            num_layers=cfg.sign_net.num_layers,
        )
    else:
        sign_net = None

    features = 2 if cfg.diffusion.name == "discrete" else 1
    if cfg.model.name == "ppgn":
        model = gg.model.SparsePPGN(
            node_in_features=features * (1 + cfg.diffusion.self_conditioning),
            edge_in_features=features * (1 + cfg.diffusion.self_conditioning),
            node_out_features=features,
            edge_out_features=features,
            emb_features=cfg.model.emb_features,
            hidden_features=cfg.model.hidden_features,
            ppgn_features=cfg.model.ppgn_features,
            num_layers=cfg.model.num_layers,
            dropout=cfg.model.dropout,
        )
    elif cfg.model.name == "gine":
        model = gg.model.GINE(
            node_in_features=features * (1 + cfg.diffusion.self_conditioning),
            edge_in_features=features * (1 + cfg.diffusion.self_conditioning),
            node_out_features=features,
            edge_out_features=features,
            emb_features=cfg.model.emb_features,
            hidden_features=cfg.model.hidden_features,
            num_layers=cfg.model.num_layers,
            dropout=cfg.model.dropout,
        )
    else:
        raise ValueError(f"Unknown model name: {cfg.model.name}")

    # Diffusion
    if cfg.diffusion.name == "discrete":
        diffusion = gg.diffusion.sparse.DiscretehypergraphDiffusion(
            self_conditioning=cfg.diffusion.self_conditioning,
            num_steps=cfg.diffusion.num_steps,
        )
    elif cfg.diffusion.name == "edm":
        diffusion = gg.diffusion.sparse.EDM(
            self_conditioning=cfg.diffusion.self_conditioning,
            num_steps=cfg.diffusion.num_steps,
        )
    else:
        raise ValueError(f"Unknown diffusion name: {cfg.diffusion.name}")

    # Method
    method = gg.method.Expansion(
        diffusion=diffusion,
        spectrum_extractor=spectrum_extractor,
        emb_features=cfg.model.emb_features,
        augmented_radius=cfg.method.augmented_radius,
        augmented_dropout=cfg.method.augmented_dropout,
        deterministic_expansion=cfg.method.deterministic_expansion,
        min_red_frac=cfg.reduction.min_red_frac,
        max_red_frac=cfg.reduction.max_red_frac,
        red_threshold=cfg.reduction.red_threshold,
    )

    return {
        "train_dataloader": train_dataloader,
        "method": method,
        "model": model,
        "sign_net": sign_net,
    }

@hydra.main(config_path="config", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    if cfg.debugging:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # Fix random seeds
    random.seed(0)
    np.random.seed(0)
    th.manual_seed(0)
       
    # hypergraphs
    if cfg.dataset.load:
        with open(Path("./data") / f"{cfg.dataset.name}.pkl", "rb") as f:
            dataset = pickle.load(f)

        train_hypergraphs = dataset["train"]
        validation_hypergraphs = dataset["val"]
        test_hypergraphs = dataset["test"]
    elif cfg.dataset.name == "hypergraphErdosRenyi":
        train_hypergraphs = gg.data.generate_erdos_renyi_hypergraphs(
            num_hypergraphs=cfg.dataset.train_size,
            min_size=cfg.dataset.min_size,
            max_size=cfg.dataset.max_size,
            probs=cfg.dataset.probs,
            k=cfg.dataset.k,
            seed=0,
        )
        validation_hypergraphs = gg.data.generate_erdos_renyi_hypergraphs(
            num_hypergraphs=cfg.dataset.val_size,
            min_size=cfg.dataset.min_size,
            max_size=cfg.dataset.max_size,
            probs=cfg.dataset.probs,
            k=cfg.dataset.k,
            seed=1,
        )
        test_hypergraphs = gg.data.generate_erdos_renyi_hypergraphs(
            num_hypergraphs=cfg.dataset.test_size,
            min_size=cfg.dataset.min_size,
            max_size=cfg.dataset.max_size,
            probs=cfg.dataset.probs,
            k=cfg.dataset.k,
            seed=2,
        )

    else:
        raise ValueError(f"Unknown dataset name: {cfg.dataset.name}")

    # Metrics
    validation_metrics = [
        gg.metrics.NodeNumDiff(),
        gg.metrics.DegreeDistrWasserstein(),
        gg.metrics.Spectral(),
    ]

    if "hypergraphSBM" in cfg.dataset.name:
        validation_metrics += [
        ]
    elif "hypergraphErdosRenyi" in cfg.dataset.name:
        validation_metrics += [
        ]

    # Method
    if cfg.method.name == "expansion":
        method_items = get_expansion_items(cfg, train_hypergraphs)
    else:
        raise ValueError(f"Unknown method name: {cfg.method.name}")
    method_items = defaultdict(lambda: None, method_items)

    # Trainer
    th.set_float32_matmul_precision("high")
    trainer = gg.training.Trainer(
        sign_net=method_items["sign_net"],
        model=method_items["model"],
        method=method_items["method"],
        train_dataloader=method_items["train_dataloader"],
        train_hypergraphs=train_hypergraphs,
        validation_hypergraphs=validation_hypergraphs,
        test_hypergraphs=test_hypergraphs,
        metrics=validation_metrics,
        cfg=cfg,
    )
    if cfg.testing:
        trainer.test()
    else:
        trainer.train()


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
