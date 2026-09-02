"""
Runs the GNN + GNNExplainer/CaptumExplainer + LLM-reasoning pipeline
(from llm_reasoning.py) over a set of test-set edges, and compares:

  - GNN-only F1        : model's raw argmax prediction vs ground truth
  - LLM-reasoning F1   : LLM's parsed Suspicious/Not-Suspicious conclusion
                          vs ground truth

...to give a sense of whether the LLM's reasoning step is adding value on
top of the raw GNN classifier, or degrading it.

Requires a fewshot_examples.json (see mine_fewshot_candidates.py) and a
trained checkpoint (see main.py --save_model).

Usage:
    pip install transformers accelerate torch --break-system-packages
    python evaluate_llm_reasoning.py --data Small_HI --model gin --unique_name run1 \
        --n_samples 200 --stratify \
        --llm_provider local --llm_model Qwen/Qwen3-14B \
        --out_path llm_eval_results.csv

    # To (expensively) run over the *entire* test set instead of a sample:
    python evaluate_llm_reasoning.py --data Small_HI --model gin --unique_name run1 \
        --n_samples -1 --out_path llm_eval_full.csv
"""

import json
import logging
import random
import time

import torch
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

from util import create_parser as base_parser, set_seed, logger_setup
from data_loading import get_data
from train_util import AddEgoIds, add_arange_ids, get_loaders
from training import get_model
from torch_geometric.nn import to_hetero

from llm_reasoning import (
    build_edge_metadata_lookup, explain_with_llm, LLMClient, _build_config,
)


def conclusion_to_label(conclusion):
    """Maps the LLM's free-text 'Conclusion' field to a 0/1 label.
    Returns None if it can't be confidently parsed (such edges are excluded
    from the LLM F1 calculation, and counted/reported separately)."""
    if conclusion is None:
        return None
    c = conclusion.strip().lower()
    # check the negative phrasing first since "suspicious" is a substring of it
    if 'not suspicious' in c or 'non-suspicious' in c or 'non suspicious' in c:
        return 0
    if 'suspicious' in c:
        return 1
    return None


def select_eval_indices(te_inds, edge_y, args):
    all_inds = te_inds.tolist()

    if args.n_samples == -1:
        eval_inds = all_inds
    elif args.stratify:
        fraud = [i for i in all_inds if edge_y[i].item() == 1]
        non_fraud = [i for i in all_inds if edge_y[i].item() == 0]
        random.shuffle(fraud)
        random.shuffle(non_fraud)
        half = args.n_samples // 2
        eval_inds = fraud[:half] + non_fraud[:args.n_samples - half]
        random.shuffle(eval_inds)
    else:
        eval_inds = list(all_inds)
        random.shuffle(eval_inds)
        eval_inds = eval_inds[:args.n_samples]

    return eval_inds


def dump_results(results, out_path):
    import pandas as pd
    df = pd.DataFrame([
        {
            'edge_id': r['edge_id'],
            'actual_label': r['actual_label'],
            'gnn_pred': r['gnn_pred'],
            'llm_conclusion': r['llm_conclusion'],
            'llm_pred': r['llm_pred'],
            'llm_pattern': r['llm_pattern'],
        }
        for r in results
    ])
    df.to_csv(out_path, index=False)


def score_and_report(results):
    if not results:
        logging.warning("No results collected -- nothing to score.")
        return

    y_true = [r['actual_label'] for r in results]
    y_gnn = [r['gnn_pred'] for r in results]

    llm_pairs = [(r['actual_label'], r['llm_pred']) for r in results if r['llm_pred'] is not None]
    n_unparsed = len(results) - len(llm_pairs)

    logging.info("\n" + "=" * 60)
    logging.info("RESULTS")
    logging.info("=" * 60)
    logging.info(f"Total edges evaluated                    : {len(results)}")
    logging.info(f"Fraud / non-fraud in evaluated set        : "
                 f"{sum(y_true)} / {len(y_true) - sum(y_true)}")
    logging.info(f"LLM conclusions unparseable (excluded)    : {n_unparsed}")

    gnn_f1 = f1_score(y_true, y_gnn, zero_division=0)
    gnn_prec = precision_score(y_true, y_gnn, zero_division=0)
    gnn_rec = recall_score(y_true, y_gnn, zero_division=0)
    gnn_acc = accuracy_score(y_true, y_gnn)
    logging.info("\nGNN-only (all evaluated edges):")
    logging.info(f"  F1        = {gnn_f1:.4f}")
    logging.info(f"  Precision = {gnn_prec:.4f}")
    logging.info(f"  Recall    = {gnn_rec:.4f}")
    logging.info(f"  Accuracy  = {gnn_acc:.4f}")

    if llm_pairs:
        y_true_llm = [p[0] for p in llm_pairs]
        y_llm = [p[1] for p in llm_pairs]
        llm_f1 = f1_score(y_true_llm, y_llm, zero_division=0)
        llm_prec = precision_score(y_true_llm, y_llm, zero_division=0)
        llm_rec = recall_score(y_true_llm, y_llm, zero_division=0)
        llm_acc = accuracy_score(y_true_llm, y_llm)
        logging.info(f"\nLLM reasoning (on {len(llm_pairs)} parseable examples):")
        logging.info(f"  F1        = {llm_f1:.4f}")
        logging.info(f"  Precision = {llm_prec:.4f}")
        logging.info(f"  Recall    = {llm_rec:.4f}")
        logging.info(f"  Accuracy  = {llm_acc:.4f}")

        # fair, apples-to-apples comparison: GNN F1 restricted to the same
        # subset the LLM's answer could actually be scored on
        gnn_pred_matched = [r['gnn_pred'] for r in results if r['llm_pred'] is not None]
        gnn_f1_matched = f1_score(y_true_llm, gnn_pred_matched, zero_division=0)
        logging.info(f"\nGNN-only F1 on the SAME subset (fair comparison) = {gnn_f1_matched:.4f}")
        logging.info(f"Delta (LLM F1 - GNN F1, matched subset)          = {llm_f1 - gnn_f1_matched:+.4f}")
    else:
        logging.info("No parseable LLM conclusions -- can't compute LLM F1.")

    logging.info("=" * 60)


def main():
    parser = base_parser()
    parser.add_argument("--fewshot_path", default="fewshot_examples.json")
    parser.add_argument("--llm_provider", default="local", choices=["local"])
    parser.add_argument("--llm_model", default="Qwen/Qwen3-14B")
    parser.add_argument("--llm_max_new_tokens", type=int, default=512, help="Max tokens to generate")
    parser.add_argument("--max_subgraph_edges", type=int, default=20)
    parser.add_argument("--n_samples", type=int, default=200,
                         help="Number of test-set edges to evaluate. Use -1 to run on "
                              "the entire te_inds (can be very slow/expensive).")
    parser.add_argument("--stratify", action='store_true',
                         help="Sample evenly from fraud / non-fraud edges instead of "
                              "uniformly at random. Recommended -- fraud edges are rare, "
                              "so a plain random sample can end up with too few positives "
                              "to compute a meaningful F1.")
    parser.add_argument("--sleep_between_calls", type=float, default=0.0,
                         help="Seconds to sleep between LLM calls (rate-limit safety).")
    parser.add_argument("--checkpoint_every", type=int, default=20,
                         help="Write partial results to disk every N edges.")
    parser.add_argument("--out_path", default="llm_eval_results.csv")
    args = parser.parse_args()

    logger_setup()
    set_seed(args.seed)
    random.seed(args.seed)

    with open("data_config.json") as f:
        data_config = json.load(f)

    logging.info("Loading dataset ...")
    tr_data, val_data, te_data, tr_inds, val_inds, te_inds = get_data(args, data_config)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    transform = AddEgoIds() if args.ego else None
    add_arange_ids([tr_data, val_data, te_data])
    tr_loader, val_loader, te_loader = get_loaders(
        tr_data, val_data, te_data, tr_inds, val_inds, te_inds, transform, args
    )

    config = _build_config(args)
    sample_batch = next(iter(tr_loader))
    model = get_model(sample_batch, config, args)
    if args.reverse_mp:
        model = to_hetero(model, te_data.metadata(), aggr='mean')

    logging.info("Loading model checkpoint ...")
    ckpt_path = f'{data_config["paths"]["model_to_load"]}/checkpoint_{args.unique_name}.tar'
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    logging.info("Building raw-CSV metadata lookup ...")
    edge_metadata_lookup = build_edge_metadata_lookup(args, data_config)

    with open(args.fewshot_path) as f:
        fewshot = json.load(f)

    llm_client = LLMClient(
        provider=args.llm_provider, model=args.llm_model,
        max_new_tokens=args.llm_max_new_tokens
    )

    edge_y = te_data['node', 'to', 'node'].y if args.reverse_mp else te_data.y
    eval_inds = select_eval_indices(te_inds, edge_y, args)

    n_fraud = sum(edge_y[i].item() for i in eval_inds)
    logging.info(f"Evaluating {len(eval_inds)} test edges "
                 f"({n_fraud} fraud / {len(eval_inds) - n_fraud} non-fraud).")

    results = []
    n_failed = 0
    for i, edge_idx in enumerate(eval_inds):
        try:
            r = explain_with_llm(
                edge_idx, te_data, model, device, args, transform, args.reverse_mp,
                edge_metadata_lookup, fewshot, llm_client, max_edges=args.max_subgraph_edges,
            )
            r['llm_pred'] = conclusion_to_label(r['llm_conclusion'])
            results.append(r)
            logging.info(
                f"[{i + 1}/{len(eval_inds)}] edge={edge_idx} actual={r['actual_label']} "
                f"gnn_pred={r['gnn_pred']} llm_conclusion={r['llm_conclusion']!r} "
                f"llm_pred={r['llm_pred']}"
            )
        except RuntimeError as e:
            # e.g. sample_predict_explain couldn't locate exactly one seed edge
            logging.warning(f"Skipping edge {edge_idx} (sampling error): {e}")
            n_failed += 1
        except Exception as e:
            # covers explainer failures, LLM API errors, malformed responses, etc.
            logging.warning(f"Skipping edge {edge_idx} (error): {e}")
            n_failed += 1

        if args.sleep_between_calls > 0:
            time.sleep(args.sleep_between_calls)

        if args.checkpoint_every and (i + 1) % args.checkpoint_every == 0:
            dump_results(results, args.out_path)
            logging.info(f"Checkpoint written to {args.out_path} ({len(results)} rows so far)")

    dump_results(results, args.out_path)
    logging.info(f"Done. {len(results)} scored, {n_failed} skipped/failed. Results -> {args.out_path}")

    score_and_report(results)

if __name__ == "__main__":
    main()