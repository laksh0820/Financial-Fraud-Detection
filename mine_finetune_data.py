"""
Builds a supervised fine-tuning (SFT) dataset for the LLM-reasoning stage.

Requires a trained GNN checkpoint (see main.py --save_model) and the raw
`*_Patterns.txt` file for --data (the ground-truth laundering-attempt log
that ships alongside the IBM AML Kaggle datasets).

Usage:
    python mine_finetune_data.py --data Small_HI --model gin --unique_name run1 \
        --patterns_file HI-Small_Patterns.txt \
        --n_samples 2000 --stratify --max_subgraph_edges 20 \
        --out finetune_train.jsonl

    # Mine every training edge instead of a sample (slow -- GNNExplainer
    # runs once per edge):
    python mine_finetune_data.py --data Small_HI --model gin --unique_name run1 \
        --patterns_file HI-Small_Patterns.txt --n_samples -1 --out finetune_train_full.jsonl
"""

import json
import logging
import random

import torch

from util import create_parser as base_parser, set_seed, logger_setup
from data_loading import get_data
from train_util import AddEgoIds, add_arange_ids, get_loaders
from training import get_model
from torch_geometric.nn import to_hetero

from llm_reasoning import (
    build_edge_metadata_lookup, serialize_subgraph,
    sample_predict_explain, build_prompt_finetuned, _build_config,
)
from mine_fewshot_candidates import (
    parse_patterns_file, match_pattern_rows_to_edge_ids,
    PATTERN_EXPLANATIONS, NON_SUSPICIOUS_EXPLANATION_TEMPLATE,
)

FALLBACK_PATTERN = "layering"
FALLBACK_SUSPICIOUS_EXPLANATION = (
    "This transaction is part of a confirmed laundering attempt in the source "
    "data. Its surrounding subgraph shows layering behaviour -- funds routed "
    "through a small set of accounts in a short time window in a way that is "
    "inconsistent with routine, one-off commerce."
)


def build_completion(actual_label, pattern, explanation):
    """The target assistant turn the LLM is fine-tuned to produce, in the
    exact format `parse_llm_response` (llm_reasoning.py) expects to parse
    back out at eval time."""
    conclusion = "Suspicious" if actual_label == 1 else "Not Suspicious"
    return (
        f"- Conclusion: {conclusion}\n"
        f"- Explanation: {explanation}\n"
        f"- Observed Pattern: {pattern}\n"
    )


def select_train_indices(tr_inds, edge_y, args):
    all_inds = tr_inds.tolist()
    if args.n_samples == -1:
        mine_inds = all_inds
    elif args.stratify:
        fraud = [i for i in all_inds if edge_y[i].item() == 1]
        non_fraud = [i for i in all_inds if edge_y[i].item() == 0]
        random.shuffle(fraud)
        random.shuffle(non_fraud)
        half = args.n_samples // 2
        mine_inds = fraud[:half] + non_fraud[:args.n_samples - half]
        random.shuffle(mine_inds)
    else:
        mine_inds = list(all_inds)
        random.shuffle(mine_inds)
        mine_inds = mine_inds[:args.n_samples]
    return mine_inds


def main():
    parser = base_parser()
    parser.add_argument("--patterns_file", required=True,
                        help="Path to the raw IBM AML *_Patterns.txt file.")
    parser.add_argument("--max_subgraph_edges", type=int, default=20)
    parser.add_argument("--n_samples", type=int, default=2000,
                        help="Number of training edges to mine. Use -1 for all of tr_inds.")
    parser.add_argument("--stratify", action='store_true',
                        help="Sample evenly from fraud / non-fraud training edges instead of uniformly at random.")
    parser.add_argument("--out", default="finetune_train.jsonl")
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
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    logging.info("Building raw-CSV metadata lookup ...")
    edge_metadata_lookup = build_edge_metadata_lookup(args, data_config)

    logging.info("Parsing + matching laundering-attempt patterns for training edges ...")
    blocks = parse_patterns_file(args.patterns_file)
    blocks = match_pattern_rows_to_edge_ids(blocks, edge_metadata_lookup)
    
    pattern_by_edge_id = {}
    for block in blocks:
        for row in block["rows"]:
            eid = row.get("edge_id")
            if eid is not None:
                pattern_by_edge_id[eid] = block["pattern"]

    edge_y = te_data['node', 'to', 'node'].y if args.reverse_mp else te_data.y
    mine_inds = select_train_indices(tr_inds, edge_y, args)

    n_fraud = sum(edge_y[i].item() for i in mine_inds)
    logging.info(f"Mining {len(mine_inds)} training edges "
                 f"({n_fraud} fraud / {len(mine_inds) - n_fraud} non-fraud) for fine-tuning data.")

    n_ok, n_failed, n_fallback = 0, 0, 0
    with open(args.out, "w") as f_out:
        for i, edge_idx in enumerate(mine_inds):
            try:
                sampled = sample_predict_explain(
                    edge_idx, te_data, model, device, args, transform, args.reverse_mp
                )
                subgraph_text = serialize_subgraph(
                    edge_metadata_lookup, sampled['global_edge_ids'], sampled['target_edge_id'],
                    imp_lookup_by_id=sampled['imp_by_id'], max_edges=args.max_subgraph_edges,
                )
                prompt = build_prompt_finetuned(subgraph_text, edge_idx, gnn_pred=sampled['pred'])

                actual_label = sampled['actual']
                if actual_label == 1:
                    pattern = pattern_by_edge_id.get(edge_idx)
                    if pattern is not None:
                        explanation = PATTERN_EXPLANATIONS.get(pattern, FALLBACK_SUSPICIOUS_EXPLANATION)
                    else:
                        pattern = FALLBACK_PATTERN
                        explanation = FALLBACK_SUSPICIOUS_EXPLANATION
                        n_fallback += 1
                else:
                    pattern = "routine"
                    explanation = NON_SUSPICIOUS_EXPLANATION_TEMPLATE

                completion = build_completion(actual_label, pattern, explanation)

                f_out.write(json.dumps({
                    "edge_id": edge_idx,
                    "actual_label": actual_label,
                    "gnn_pred": sampled['pred'],
                    "pattern": pattern,
                    "prompt": prompt,
                    "completion": completion,
                }) + "\n")
                f_out.flush()
                n_ok += 1

                if (i + 1) % 50 == 0:
                    logging.info(f"[{i + 1}/{len(mine_inds)}] mined (edge={edge_idx}, "
                                 f"label={actual_label}, pattern={pattern})")

            except RuntimeError as e:
                # e.g. sample_predict_explain couldn't locate exactly one seed edge
                logging.warning(f"Skipping edge {edge_idx} (sampling error): {e}")
                n_failed += 1
            except Exception as e:
                logging.warning(f"Skipping edge {edge_idx} (error): {e}")
                n_failed += 1

    logging.info(
        f"Done. {n_ok} examples written to {args.out} "
        f"({n_fallback} suspicious edges used the generic fallback explanation because they "
        f"didn't match a block in {args.patterns_file}), {n_failed} failed/skipped."
    )


if __name__ == "__main__":
    main()