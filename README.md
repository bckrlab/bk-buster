# bk-buster

Code for the **synthetic node-classification benchmark** from:

> K. Coşkun, F. Blanke, V. S. Cangalovic, B. C. Hiller, A. Mirzaei, T. Siegl, S. Lüdtke, M. Becker.
> *Informed, but Not Always Improved: Challenging the Benefit of Background Knowledge in GNNs.*
> IEEE Transactions on Computational Biology and Bioinformatics.

This benchmark (Section V.D / Fig. 11 in the paper) is a controlled, synthetic node-classification
task used as a positive control: node features are drawn from overlapping distributions, and a
background-knowledge (BK) graph connects nodes of the same class. It lets us test how BK-informed
GNNs (GCN/GAT) respond to graph perturbations (edge removal, edge addition, node isolation,
detach-and-rewire) compared to uninformed baselines (LogReg, SVM, MLP), in a setting where the BK
graph is guaranteed to be informative by construction.

## Installation

The dependencies (PyTorch, PyG, RAPIDS, etc.) are pinned in [dev/env.yml](dev/env.yml).

**Option A — Docker (recommended, includes GPU support):**

```bash
cd dev
docker compose up -d --build
docker compose exec app bash
```

This builds an image with a `dev` micromamba environment and installs the package into it in
editable mode (see the `command` in [dev/compose.yml](dev/compose.yml)).

**Option B — conda/micromamba directly:**

```bash
micromamba env create -f dev/env.yml -n dev
micromamba activate dev
pip install -e .
```

## Notebooks

All notebooks live under [exps/syn_node/](exps/syn_node/).

- **[data_generation.ipynb](exps/syn_node/data_generation.ipynb)** — builds the synthetic
  node-classification dataset (overlapping node-feature distributions, same-class BK graph) and
  visualizes it, along with each perturbation type (edge removal/addition, node isolation,
  detach-and-rewire) applied to an example graph.
- **[binary_test.ipynb](exps/syn_node/binary_test.ipynb)** — runs the actual benchmark: trains
  GCN/GAT (informed) against LogReg/SVM/MLP (uninformed baselines) over 10 runs, for each
  perturbation type at increasing severity, and reproduces Fig. 11 of the paper.

Running `binary_test.ipynb` writes results to `exps/syn_node/syn_node_{rem_edges,add_edges,
iso_nodes,dr_nodes}.csv` and the corresponding `.pdf` figures. These outputs are gitignored —
rerun the notebook to regenerate them.

## Package layout

- [src/bk_buster/generator/node_binary.py](src/bk_buster/generator/node_binary.py) — synthetic
  data/graph generation (`generate_data`, `make_graph`).
- [src/bk_buster/model/synthetic.py](src/bk_buster/model/synthetic.py) — GCN/GAT/MLP model
  definitions and train/test loops used in the benchmark.
- [src/bk_buster/perturb.py](src/bk_buster/perturb.py) — graph perturbation utilities (edge
  removal/addition, node isolation, rewiring, and related graph metrics).
- [src/bk_buster/utils/](src/bk_buster/utils/) — data loading (`loader.py`) and figure styling
  (`plot.py`, using [conf/pub_ltx.mplstyle](src/bk_buster/conf/pub_ltx.mplstyle)) helpers shared
  across notebooks.

## License

MIT — see [LICENSE.txt](LICENSE.txt).

---

This project was set up using [PyScaffold](https://pyscaffold.org/) 4.6.
