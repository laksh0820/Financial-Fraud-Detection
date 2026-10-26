"""Visualize explanations of GNN predictions.

    # one TP / FP / FN / TN example from the test set
    python -m explanation_model.visualizer --data Small_HI --model gin --unique_name run1

    # a specific transaction
    python -m explanation_model.visualizer --data Small_HI --model gin --unique_name run1 --explain_edge_idx 12345

The architecture flags (--emlps, --reverse_mp, --ports, ...) must match the trained checkpoint.
"""
import logging

import torch

from common.config import load_data_config
from explanation_model.visualizer.cases import CASE_DEFINITIONS, CASE_LABELS, find_case_examples
from explanation_model.visualizer.plotting import add_visualizer_args
from explanation_model.visualizer.visualize import explain_and_plot
from prediction_model.predictor import TransactionPredictor, sample_and_predict
from prediction_model.util import create_parser, logger_setup, set_seed


def run_explain(predictor, args):
    model, device, te_data, te_inds = predictor.model, predictor.device, predictor.te_data, predictor.te_inds
    transform, is_hetero = predictor.transform, predictor.is_hetero

    # explicit edge requested: explain just that one, skip the 4-case search
    if args.explain_edge_idx != -1:
        edge_idx = args.explain_edge_idx
        logging.info(f"Explaining user-specified edge {edge_idx}")
        try:
            sample = predictor.predict(edge_idx)
        except RuntimeError as e:
            logging.error(f"Could not sample/predict edge {edge_idx}: {e}")
            return
        explain_and_plot(model, sample, edge_idx, args, 'USER', 'User-specified edge')
        return

    # find one example each of TP / FP / FN / TN. Fraud is rare, so we split
    # te_inds by actual label up front and scan each group only for the
    # cases it can produce, instead of one flat scan over all of te_inds.
    found = {key: None for key in CASE_DEFINITIONS}

    labels = predictor.edge_labels[te_inds]
    fraud_inds = te_inds[labels == 1]
    non_fraud_inds = te_inds[labels == 0]

    logging.info(
        f"Test set: {len(fraud_inds)} fraud edges, {len(non_fraud_inds)} non-fraud edges "
        f"(of {len(te_inds)} total)."
    )

    seed_gen = torch.Generator().manual_seed(getattr(args, 'seed', 42))
    fraud_inds = fraud_inds[torch.randperm(len(fraud_inds), generator=seed_gen)]
    non_fraud_inds = non_fraud_inds[torch.randperm(len(non_fraud_inds), generator=seed_gen)]

    found_ids = {}
    found_ids.update(find_case_examples(fraud_inds, ['TP', 'FN'], te_data, model, device, args, transform, is_hetero))
    found_ids.update(find_case_examples(non_fraud_inds, ['FP', 'TN'], te_data, model, device, args, transform, is_hetero))

    # only resample individually for the (up to 4) edges we're actually going to explain
    for case_key, edge_idx in found_ids.items():
        try:
            found[case_key] = (edge_idx, sample_and_predict(edge_idx, te_data, model, device, args, transform, is_hetero))
        except RuntimeError as e:
            logging.warning(f"Could not resample edge {edge_idx} for explanation: {e}")

    for case_key, label in CASE_LABELS.items():
        entry = found[case_key]
        if entry is None:
            logging.info(f"No example found for case: '{label}'. Skipping.")
            continue
        edge_idx, sample = entry
        logging.info(f"Explaining case '{label}' using edge {edge_idx}")
        explain_and_plot(model, sample, edge_idx, args, case_key, label)


def main():
    parser = create_parser()
    parser.add_argument("--explain_edge_idx", type=int, default=-1,
                        help="Global edge index to explain. If -1, pick one TP/FP/FN/TN example from the test set.")
    add_visualizer_args(parser)
    args = parser.parse_args()

    logger_setup()
    set_seed(args.seed)

    predictor = TransactionPredictor(args, load_data_config(args.data_config))
    run_explain(predictor, args)


if __name__ == "__main__":
    main()
