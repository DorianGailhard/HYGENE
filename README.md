# HYGENE: A Diffusion-based Hypergraph Generation Method

This repository contains the reference implementation of the paper [HYGENE: A Diffusion-based Hypergraph Generation Method](https://arxiv.org/abs/2408.16457). Our work is based on [Efficient and Scalable Graph Generation through Iterative Local Expansion](https://openreview.net/forum?id=2XkTz7gdpc), many thanks for their great code!


## Setup

To get started, follow these steps:

+ Clone this repository

+ Create the specified [conda](https://docs.conda.io/en/latest/) environment named `hypergraph-generation` by running the following command:

    ```bash
    conda env create -f environment.yaml
    ```

    Note that the `graph-tool` library is not available on Windows. This library is used for SBM graph evaluation, which will consequently not work on Windows. Everything else should work regardless of the operating system.


## Usage

The main entry point is `main.py` with parameters managed by the [Hydra](https://hydra.cc/) framework.
To reproduce the results from the paper, run:

```bash
python main.py +experiment=XXX
```

where `XXX` is one of the following experiments:
`hypergraphErdosRenyi`, `hypergraphSBM`, `hypergraphEgo`, `hypergraphTree`, `meshBookshelf`, `meshPlant`, `meshPiano`

New experiments can be added by adding a pickle file in `data/` and creating a new config file in `config/experiment/` or passing the parameters directly through the command line. Please refer to the [Hydra documentation](https://hydra.cc/docs/intro/) for more information.


### Checkpoints

When `training.save_checkpoint` in the configuration is set to `True`, checkpoints are saved. To resume training from a checkpoint, set `training.resume` to the step number of the checkpoint, or to `True` to resume from the latest checkpoint.


### Wandb

To log the results to [Wandb](https://wandb.ai/), set `wandb.logging` to `True` in the configuration.


## Citation
When using this code, please cite our paper:
```
@misc{gailhard2024hygenediffusionbasedhypergraphgeneration,
      title={HYGENE: A Diffusion-based Hypergraph Generation Method}, 
      author={Dorian Gailhard and Enzo Tartaglione and Lirida Naviner De Barros and Jhony H. Giraldo},
      year={2024},
      eprint={2408.16457},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2408.16457}, 
}
```

