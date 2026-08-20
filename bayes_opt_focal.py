"""
Bayesian Optimization for FocalLoss hyperparameters (alpha, gamma).

Usage:
    python bayes_opt_focal.py --data Small_HI --model gin --n_trials 30 --n_epochs 20 --batch_size 8192 --seed 1

Dependencies:
    pip install optuna
"""

import optuna
import torch
import logging
import json
import argparse
import time
import os

from util import set_seed, logger_setup
from data_loading import get_data
from training import get_model
from train_util import (
    AddEgoIds, FocalLoss, extract_param, add_arange_ids,
    get_loaders, evaluate_homo, evaluate_hetero, save_model
)
from torch_geometric.data import HeteroData
from torch_geometric.nn import to_hetero
from torch_geometric.utils import degree
import tqdm
import wandb
from sklearn.metrics import f1_score


# ─────────────────────────────────────────────────────────────────────────────
# Single training run (one epoch loop) returning best val F1
# ─────────────────────────────────────────────────────────────────────────────

def run_training(
    tr_data, val_data, te_data,
    tr_inds, val_inds, te_inds,
    args, data_config,
    alpha: float, gamma: float
) -> float:
    """
    Trains the model using FocalLoss(alpha, gamma) and returns
    the best validation F1 score observed across all epochs.

    Args:
        tr_data, val_data, te_data : PyG graph data objects
        tr_inds, val_inds, te_inds : Edge index tensors for each split
        args                        : Parsed CLI arguments (augmented below)
        data_config                 : Dict loaded from data_config.json
        alpha                       : FocalLoss alpha to evaluate
        gamma                       : FocalLoss gamma to evaluate

    Returns:
        best_val_f1 (float)
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ── Build a minimal wandb-compatible config from model_settings.json ──
    # We disable wandb during optimization to avoid cluttering the dashboard.
    wandb.init(mode="disabled")
    config = wandb.config
    config.update({
        "epochs":       args.n_epochs,
        "batch_size":   args.batch_size,
        "model":        args.model,
        "data":         args.data,
        "num_neighbors": args.num_neighs,
        "lr":           extract_param("lr",           args),
        "n_hidden":     extract_param("n_hidden",      args),
        "n_gnn_layers": extract_param("n_gnn_layers",  args),
        "loss":         "focal",
        "w_ce1":        extract_param("w_ce1",         args),
        "w_ce2":        extract_param("w_ce2",         args),
        "dropout":      extract_param("dropout",       args),
        "final_dropout":extract_param("final_dropout", args),
        "n_heads":      extract_param("n_heads",       args) if args.model == "gat" else None,
    })

    # ── Transforms & loaders ──
    transform = AddEgoIds() if args.ego else None

    # Deep-copy edge_attr so arange ids don't accumulate across trials
    import copy
    tr_data_c  = copy.deepcopy(tr_data)
    val_data_c = copy.deepcopy(val_data)
    te_data_c  = copy.deepcopy(te_data)

    add_arange_ids([tr_data_c, val_data_c, te_data_c])
    tr_loader, val_loader, te_loader = get_loaders(
        tr_data_c, val_data_c, te_data_c,
        tr_inds, val_inds, te_inds,
        transform, args
    )

    # ── Model ──
    sample_batch = next(iter(tr_loader))
    model = get_model(sample_batch, config, args)

    if args.reverse_mp:
        model = to_hetero(model, te_data_c.metadata(), aggr="mean")

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    loss_fn   = FocalLoss(alpha=alpha, gamma=gamma)

    is_hetero = args.reverse_mp

    # ── Training loop ──
    best_val_f1 = 0.0
    for epoch in range(config.epochs):
        model.train()
        preds, ground_truths = [], []

        for batch in tr_loader:
            optimizer.zero_grad()

            # Identify seed edges
            if args.under_sample or args.hybrid_sample:
                seed_edges_local_inds = batch.input_id.detach().cpu() if not is_hetero \
                    else batch["node", "to", "node"].input_id.detach().cpu()
                if not is_hetero:
                    batch_edge_ids = tr_loader.data.edge_attr.detach().cpu()[seed_edges_local_inds, 0]
                else:
                    batch_edge_ids = tr_loader.data["node", "to", "node"].edge_attr.detach().cpu()[seed_edges_local_inds, 0]
            else:
                inds_cpu = tr_inds.detach().cpu()
                if not is_hetero:
                    batch_edge_inds = inds_cpu[batch.input_id.detach().cpu()]
                    batch_edge_ids  = tr_loader.data.edge_attr.detach().cpu()[batch_edge_inds, 0]
                else:
                    batch_edge_inds = inds_cpu[batch["node", "to", "node"].input_id.detach().cpu()]
                    batch_edge_ids  = tr_loader.data["node", "to", "node"].edge_attr.detach().cpu()[batch_edge_inds, 0]

            if not is_hetero:
                mask = torch.isin(batch.edge_attr[:, 0].detach().cpu(), batch_edge_ids)
                batch.edge_attr = batch.edge_attr[:, 1:]
                batch.to(device)
                out   = model(batch.x, batch.edge_index, batch.edge_attr)
                pred  = out[mask]
                gt    = batch.y[mask]
            else:
                mask = torch.isin(
                    batch["node", "to", "node"].edge_attr[:, 0].detach().cpu(),
                    batch_edge_ids
                )
                batch["node", "to", "node"].edge_attr    = batch["node", "to", "node"].edge_attr[:, 1:]
                batch["node", "rev_to", "node"].edge_attr = batch["node", "rev_to", "node"].edge_attr[:, 1:]
                batch.to(device)
                out  = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)
                out  = out[("node", "to", "node")]
                pred = out[mask]
                gt   = batch["node", "to", "node"].y[mask]

            loss = loss_fn(pred, gt)
            loss.backward()
            optimizer.step()

            preds.append(pred.argmax(dim=-1))
            ground_truths.append(gt)

        # ── Validation ──
        if not is_hetero:
            val_f1 = evaluate_homo(val_loader, val_inds, model, val_data_c, device, args)
        else:
            val_f1 = evaluate_hetero(val_loader, val_inds, model, val_data_c, device, args)

        logging.info(
            f"  [alpha={alpha:.4f}, gamma={gamma:.4f}] "
            f"Epoch {epoch:03d}: Val F1={val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1

    wandb.finish()
    return best_val_f1


# ─────────────────────────────────────────────────────────────────────────────
# Optuna objective
# ─────────────────────────────────────────────────────────────────────────────

def make_objective(tr_data, val_data, te_data,
                   tr_inds, val_inds, te_inds,
                   args, data_config):
    """Returns a closure that Optuna calls for each trial."""

    def objective(trial: optuna.Trial) -> float:
        alpha = trial.suggest_float("alpha", 0.1,  0.9,  step=0.05)
        gamma = trial.suggest_float("gamma", 0.5,  5.0,  step=0.25)

        logging.info(
            f"\n{'='*60}\n"
            f"Trial {trial.number:03d}  |  alpha={alpha:.4f}, gamma={gamma:.4f}\n"
            f"{'='*60}"
        )

        # Re-seed for reproducibility within each trial
        set_seed(args.seed)

        val_f1 = run_training(
            tr_data, val_data, te_data,
            tr_inds, val_inds, te_inds,
            args, data_config,
            alpha=alpha, gamma=gamma
        )

        logging.info(
            f"Trial {trial.number:03d} finished  |  "
            f"alpha={alpha:.4f}, gamma={gamma:.4f}  →  Val F1={val_f1:.4f}"
        )
        return val_f1

    return objective


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def create_parser():
    """Extends the base parser with Bayesian-opt specific arguments."""
    # Import the base parser from util and add extra args
    from util import create_parser as base_parser
    parser = base_parser()

    parser.add_argument(
        "--n_trials", default=30, type=int,
        help="Number of Optuna trials for Bayesian optimisation"
    )
    parser.add_argument(
        "--alpha_min",  default=0.10, type=float,
        help="Lower bound for FocalLoss alpha search space"
    )
    parser.add_argument(
        "--alpha_max",  default=0.90, type=float,
        help="Upper bound for FocalLoss alpha search space"
    )
    parser.add_argument(
        "--gamma_min",  default=0.50, type=float,
        help="Lower bound for FocalLoss gamma search space"
    )
    parser.add_argument(
        "--gamma_max",  default=5.00, type=float,
        help="Upper bound for FocalLoss gamma search space"
    )
    parser.add_argument(
        "--results_dir", default="./focal_opt_results", type=str,
        help="Directory where optimisation results will be saved"
    )
    parser.add_argument(
        "--study_name", default="focal_loss_bayes_opt", type=str,
        help="Optuna study name (also used as SQLite DB filename)"
    )
    return parser


def main():
    parser = create_parser()
    args   = parser.parse_args()

    # Force focal_loss=True for this script
    args.focal_loss = True

    logger_setup()
    set_seed(args.seed)

    with open("data_config.json", "r") as f:
        data_config = json.load(f)

    os.makedirs(args.results_dir, exist_ok=True)

    # Load data once
    logging.info("Loading dataset …")
    t0 = time.perf_counter()
    tr_data, val_data, te_data, tr_inds, val_inds, te_inds = get_data(args, data_config)
    logging.info(f"Dataset loaded in {time.perf_counter()-t0:.1f}s")

    # ── Build study with Tree-structured Parzen Estimator (TPE) sampler ──
    # TPE is the standard Bayesian method in Optuna (replaces Gaussian Process
    # for flexibility with mixed/discrete spaces).
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    # storage = f"sqlite:///{args.results_dir}/{args.study_name}.db"

    study = optuna.create_study(
        study_name   = args.study_name,
        direction    = "maximize",       # maximise Val F1
        sampler      = sampler,
        # storage      = storage,
        # load_if_exists = True,           # resume if interrupted
    )

    # Warm-start: seed with your current default values so the first
    # trial is never wasted on a random guess.
    study.enqueue_trial({"alpha": args.alpha, "gamma": args.gamma})

    objective = make_objective(
        tr_data, val_data, te_data,
        tr_inds, val_inds, te_inds,
        args, data_config
    )

    logging.info(
        f"Starting Bayesian optimisation: {args.n_trials} trials, "
        f"alpha ∈ [{args.alpha_min}, {args.alpha_max}], "
        f"gamma ∈ [{args.gamma_min}, {args.gamma_max}]"
    )

    study.optimize(
        objective,
        n_trials   = args.n_trials,
        gc_after_trial = True,           # free GPU memory between trials
    )

    # ── Report results ──
    best = study.best_trial
    logging.info("\n" + "="*60)
    logging.info("BAYESIAN OPTIMISATION COMPLETE")
    logging.info(f"  Best trial   : #{best.number}")
    logging.info(f"  Best Val F1  : {best.value:.4f}")
    logging.info(f"  Best alpha   : {best.params['alpha']:.4f}")
    logging.info(f"  Best gamma   : {best.params['gamma']:.4f}")
    logging.info("\n" + "="*60)

    # ── Persist best params to JSON (easy to read back in main.py) ──
    result = {
        "best_trial":  best.number,
        "best_val_f1": best.value,
        "best_alpha":  best.params["alpha"],
        "best_gamma":  best.params["gamma"],
        "all_trials": [
            {
                "trial":   t.number,
                "alpha":   t.params["alpha"],
                "gamma":   t.params["gamma"],
                "val_f1":  t.value,
            }
            for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]
    }

    out_path = os.path.join(args.results_dir, f"focal_opt_{args.model}_{args.data}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    logging.info(f"Results saved to {out_path}")

    # ── Print sorted trial table ──
    trials_df = study.trials_dataframe()
    trials_df = trials_df.sort_values("value", ascending=False)
    logging.info("\nTop-10 trials:\n" + trials_df[["number","value","params_alpha","params_gamma"]].head(10).to_string(index=False))

    # ── Suggest: update model_settings.json with best focal params ──
    logging.info(
        "\nTo use these params, run main.py with:\n"
        f"  --focal_loss --alpha {best.params['alpha']:.4f} --gamma {best.params['gamma']:.4f}"
    )


if __name__ == "__main__":
    main()
