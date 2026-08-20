# Financial Fraud Detection

Graph Neural Network pipeline for detecting money-laundering transactions in the IBM AML dataset (GIN, GAT, PNA, RGCN architectures), with support for focal loss, Bayesian hyperparameter search, and GNNExplainer-based explanations.

## 1. Setup

```bash
conda env create -f env.yml
conda activate multignn
```

Requires a CUDA 11.8-capable GPU (PyTorch + PyG with CUDA are pinned in `env.yml`).

## 2. Prepare the data

1. Download a raw IBM AML Kaggle dataset (e.g. `Small_HI`, `Small_LI`, etc.).
2. Format it into the expected edge-list CSV:

```bash
python format_kaggle_files.py /path/to/raw_transactions.csv
```

This writes `formatted_transactions.csv` next to the input file.

3. Point `data_config.json` at your local paths:

```json
{
  "paths": {
    "aml_data": "/path/to/AML/dataset",
    "model_to_load": "/path/to/AML/model",
    "model_to_save": "/path/to/AML/model"
  }
}
```

Data is expected at `<aml_data>/<--data>/formatted_transactions.csv`.

## 3. Train a model

```bash
python main.py --data Small_HI --model gin --focal_loss --seed 1 --alpha 0.75 --gamma 1.3
```

Key arguments (see `util.py` for the full list):

| Flag | Description |
|---|---|
| `--data` | Dataset folder name under `aml_data` (required) |
| `--model` | `gin`, `gat`, `pna`, or `rgcn` (required) |
| `--reverse_mp` | Enable reverse message passing (heterogeneous graph) |
| `--ports`, `--tds`, `--ego` | Optional edge/node feature augmentations |
| `--focal_loss --alpha --gamma` | Use focal loss instead of cross-entropy |
| `--over_sample` / `--under_sample` / `--hybrid_sample` | Class-imbalance handling |
| `--weighted_loader` | Use weighted sampling in the training loader |
| `--n_epochs`, `--batch_size`, `--num_neighs` | Standard training/sampling params |
| `--save_model --unique_name NAME` | Save checkpoint as `checkpoint_NAME.tar` |
| `--testing` | Disable Weights & Biases logging |

Per-model default hyperparameters (lr, hidden size, dropout, etc.) live in `model_settings.json` and are loaded automatically via `--model`.

An example SLURM job is in `script.sh`.

## 4. Run inference

```bash
python main.py --data Small_HI --model gin --inference --unique_name NAME
```

Loads `checkpoint_NAME.tar` from `model_to_load` and reports the test F1 score.

## 5. Hyperparameter search (focal loss)

```bash
python bayes_opt_focal.py --data Small_HI --model gin --n_trials 30 --n_epochs 20 --batch_size 8192 --seed 1
```

Uses Optuna (TPE sampler) to search `alpha`/`gamma`. Results are saved to `./focal_opt_results/focal_opt_<model>_<data>.json`; `bayes_opt_focal_fig.py` can plot a contour of a completed search.

## 6. Explainability

```bash
python main.py --data Small_HI --model gin --explain --unique_name NAME
```

Runs GNNExplainer (or CaptumExplainer for `--reverse_mp` models) on one example each of TP/FP/FN/TN from the test set — or a specific edge via `--explain_edge_idx` — and saves annotated subgraph plots to `--explain_plot_dir` (default `explanations/`).

## Project structure

```
main.py                     # entry point (train / inference / explain)
util.py                     # CLI args, seeding, logging
data_loading.py             # dataset load, split, over/under-sampling
data_util.py                # PyG Data/HeteroData wrappers, feature engineering
train_util.py               # loaders, FocalLoss, evaluation, checkpointing
training.py                 # training loops, model construction
models.py                   # GINe, GATe, PNA, RGCN architectures
inference.py                # test-set evaluation from a checkpoint
explain.py                  # GNNExplainer / CaptumExplainer plots
bayes_opt_focal.py          # Optuna search over focal-loss alpha/gamma
bayes_opt_focal_fig.py      # contour plot of a finished search
format_kaggle_files.py      # raw Kaggle CSV -> formatted_transactions.csv
data_config.json            # data/model paths
model_settings.json         # per-model default hyperparameters
script.sh                   # example SLURM submission script
```
