"""End-to-end AML detection: GNN prediction -> (only if fraud) pattern classification + explanation.

    python detect.py --data Small_HI --model gin --unique_name run1 --edge_ids 12345 67890 \
        --classifier llm --llm_model Qwen/Qwen3-14B --adapter_path ./pattern_clf/adapter --load_in_4bit

    # try it on random fraud transactions of the test set, with explanation plots
    python detect.py --data Small_HI --model gin --unique_name run1 --n_random_edges 5 --sample_label fraud \
        --adapter_path ./pattern_clf/adapter --load_in_4bit --visualize

The GNN flags (--model, --emlps, --reverse_mp, --ports, --tds, --ego, --num_neighs, ...) must match the
ones the checkpoint (--unique_name) was trained with. Results are printed as JSON (and written to --out_path).
"""
import argparse
import json
import logging
import random

from common.config import load_data_config
from explanation_model.classifier import available_classifiers, get_classifier_class
from explanation_model.explainer import available_explainers, get_explainer_class
from explanation_model.metadata import EdgeMetadataStore
from explanation_model.pipeline import ExplanationPipeline
from explanation_model.visualizer.plotting import add_visualizer_args
from prediction_model.util import create_parser, logger_setup, set_seed


def build_parser(argv=None):
    parser = create_parser()  # GNN flags (+ --data, --model, --unique_name, ...)
    parser.add_argument("--edge_ids", nargs="+", type=int, default=None, help="Global transaction ids to check")
    parser.add_argument("--n_random_edges", type=int, default=0, help="Or: check N random test-set transactions")
    parser.add_argument("--sample_label", choices=["any", "fraud", "non-fraud"], default="any",
                        help="With --n_random_edges: only draw transactions with this ground-truth label")
    parser.add_argument("--classifier", default="llm", choices=available_classifiers())
    parser.add_argument("--explainer", default="template", choices=available_explainers())
    parser.add_argument("--max_subgraph_edges", type=int, default=40,
                        help="Max edges of the subgraph shown to the classifier (keeps the most important ones)")
    parser.add_argument("--visualize", action="store_true", help="Also save an explanation plot for flagged transactions")
    parser.add_argument("--out_path", default=None, help="Write the JSON results here")
    add_visualizer_args(parser)

    # classifier / explainer specific flags: peek at --classifier / --explainer first (a separate mini
    # parser, so --help and missing-argument errors still come from the full parser below)
    peek = argparse.ArgumentParser(add_help=False)
    peek.add_argument("--classifier", default="llm")
    peek.add_argument("--explainer", default="template")
    known, _ = peek.parse_known_args(argv)
    get_classifier_class(known.classifier).add_cli_args(parser)
    get_explainer_class(known.explainer).add_cli_args(parser)
    return parser


def select_edges(args, predictor):
    if args.edge_ids:
        return list(args.edge_ids)
    inds = predictor.te_inds
    if args.sample_label != "any":
        wanted = 1 if args.sample_label == "fraud" else 0
        inds = inds[predictor.edge_labels[inds] == wanted]
    return random.Random(args.seed).sample(inds.tolist(), min(args.n_random_edges, len(inds)))


def main():
    parser = build_parser()
    args = parser.parse_args()
    if not args.edge_ids and args.n_random_edges <= 0:
        parser.error("give --edge_ids ... or --n_random_edges N")

    logger_setup()
    set_seed(args.seed)
    data_config = load_data_config(args.data_config)

    from prediction_model.predictor import TransactionPredictor  # heavy import (torch_geometric)
    predictor = TransactionPredictor(args, data_config)

    pipeline = ExplanationPipeline(
        classifier=get_classifier_class(args.classifier).from_args(args),   # models load lazily
        explainer=get_explainer_class(args.explainer).from_args(args),
        metadata=EdgeMetadataStore.from_data_config(data_config, args.data),
        max_subgraph_edges=args.max_subgraph_edges,
        visualize=args.visualize,
    )

    results, n_flagged = [], 0
    for edge_idx in select_edges(args, predictor):
        record = {"edge_id": edge_idx}
        try:
            sample = predictor.predict(edge_idx)
            record.update(gnn_pred=sample["pred"], fraud_prob=round(sample["fraud_prob"], 4),
                          actual_label=sample["actual"])
            # explanation models run if and only if the prediction model says "fraud"
            if sample["pred"] == 1:
                n_flagged += 1
                record["explanation"] = pipeline.run(predictor.model, sample, edge_idx, args)
        except Exception as e:  # keep going: one bad transaction should not stop a batch
            logging.warning(f"Edge {edge_idx} failed: {e}")
            record["error"] = str(e)
        results.append(record)

    logging.info(f"Checked {len(results)} transactions, {n_flagged} flagged as fraud and explained.")
    print(json.dumps(results, indent=2))
    if args.out_path:
        with open(args.out_path, "w") as f:
            json.dump(results, f, indent=2)
        logging.info(f"Results written to {args.out_path}")


if __name__ == "__main__":
    main()
