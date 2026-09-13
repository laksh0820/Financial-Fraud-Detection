"""
Usage:
    python mine_fewshot_candidates.py --data Small_HI --model gin \
        --unique_name run1 --patterns_file HI-Small_Patterns.txt \
        --n_non_suspicious 6 --out fewshot_examples_auto.json
"""

import csv
import json
import logging
import random
from collections import defaultdict
from datetime import datetime

import torch

from util import create_parser as base_parser, set_seed, logger_setup
from data_loading import get_data
from train_util import AddEgoIds, add_arange_ids, get_loaders
from training import get_model
from torch_geometric.nn import to_hetero

from llm_reasoning import (
    build_edge_metadata_lookup, serialize_subgraph,
    sample_predict_explain, _build_config,
)

PATTERN_NAME_MAP = {
    "FAN-OUT": "fan-out",
    "FAN-IN": "fan-in",
    "GATHER-SCATTER": "gather-scatter",
    "SCATTER-GATHER": "scatter-gather",
    "CYCLE": "simple cycle",
    "RANDOM": "random",
    "BIPARTITE": "bipartite",
    "STACK": "stack",
}

# Short analyst-style rationale per pattern
PATTERN_EXPLANATIONS = {
    "fan-out": (
        "The account rapidly disperses funds to several distinct counterparties "
        "in a short time window. This fan-out structure is often used to break a "
        "large sum into smaller transfers and obscure its final destination."
    ),
    "fan-in": (
        "Several distinct accounts route funds into a single destination account "
        "within a short window. This fan-in structure is often used to consolidate "
        "funds from multiple sources before a final transfer out."
    ),
    "gather-scatter": (
        "Funds converge into an intermediary account from several sources and are "
        "then quickly redistributed onward. This gather-then-scatter sequence is "
        "characteristic of layering."
    ),
    "scatter-gather": (
        "Funds fan out from a single account to several intermediaries and are "
        "then reconsolidated into one account shortly after. This scatter-then-"
        "gather sequence obscures the funds' path through parallel branches."
    ),
    "simple cycle": (
        "Funds move through a short chain of accounts and return to (or near) the "
        "originating account. This closed loop has no obvious legitimate economic "
        "purpose and is a classic layering technique."
    ),
    "random": (
        "The transfers do not follow one clean canonical shape, but the irregular "
        "sequence of counterparties, payment formats, and short time gaps between "
        "hops is consistent with deliberate obfuscation rather than routine commerce."
    ),
    "bipartite": (
        "A dense many-to-many web of transfers connects two clusters of accounts, "
        "consistent with a mule network or a coordinated burst of transfers between "
        "two groups."
    ),
    "stack": (
        "Funds move through a linear chain of intermediary accounts in quick "
        "succession (A to B to C to ...), increasing the number of hops between "
        "origin and destination in a layering pattern."
    ),
}

NON_SUSPICIOUS_EXPLANATION_TEMPLATE = (
    "Sampled from confirmed non-laundering training data. The transaction and its "
    "surrounding subgraph show no fan-out/fan-in/cycle/stack structure, and "
    "GNNExplainer does not concentrate elevated importance on any narrow set of "
    "adjacent edges -- consistent with routine activity rather than a layering "
    "pattern."
)


def parse_patterns_file(path):
    """
    Parses the IBM AML `*_Patterns.txt` log into blocks:
        [{"pattern": "fan-out", "rows": [ {timestamp, from_bank, from_account,
          to_bank, to_account, amount_received, receiving_currency,
          amount_paid, payment_currency, payment_format, is_laundering}, ... ]}]

    Each block corresponds to one `BEGIN LAUNDERING ATTEMPT - X ... END
    LAUNDERING ATTEMPT - X` section. Handles the file's CRLF line endings.
    """
    fields = [
        "timestamp", "from_bank", "from_account", "to_bank", "to_account",
        "amount_received", "receiving_currency", "amount_paid",
        "payment_currency", "payment_format", "is_laundering",
    ]

    blocks = []
    current = None
    with open(path, "r") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n").rstrip("\r")
            if not line:
                continue
            if line.startswith("BEGIN LAUNDERING ATTEMPT"):
                header = line[len("BEGIN LAUNDERING ATTEMPT - "):]
                pattern_raw = header.split(":", 1)[0].strip()
                pattern = PATTERN_NAME_MAP.get(pattern_raw, pattern_raw.lower())
                current = {"pattern": pattern, "pattern_raw": pattern_raw, "rows": []}
            elif line.startswith("END LAUNDERING ATTEMPT"):
                if current is not None:
                    blocks.append(current)
                current = None
            else:
                if current is None:
                    continue  # stray line outside a block; ignore
                parsed = next(csv.reader([line]))
                if len(parsed) != len(fields):
                    logging.warning(
                        f"Skipping malformed pattern line (expected {len(fields)} "
                        f"fields, got {len(parsed)}): {line}"
                    )
                    continue
                row = dict(zip(fields, parsed))
                row["amount_received"] = float(row["amount_received"])
                row["amount_paid"] = float(row["amount_paid"])
                row["is_laundering"] = int(row["is_laundering"])
                current["rows"].append(row)

    logging.info(f"Parsed {len(blocks)} laundering-attempt blocks from {path}")
    counts = defaultdict(int)
    for b in blocks:
        counts[b["pattern"]] += 1
    logging.info(f"Pattern counts: {dict(counts)}")
    return blocks


# Matching pattern-file transactions to global edge ids in formatted_transactions.csv

def match_pattern_rows_to_edge_ids(blocks, edge_metadata_lookup):
    fields_used = ["src", "dst","amount","currency","payment_format","timestamp"]
    logging.info(f"Matching pattern rows to edge ids using fields: {fields_used}")

    def make_meta_key(meta):
        key = []
        for field in fields_used:
            key.append(meta[field])
        return tuple(key)

    def make_row_key(row):
        key = []
        key.append(row["from_bank"] + row["from_account"])
        key.append(row["to_bank"] + row["to_account"])
        key.append(row["amount_received"])
        key.append(row["receiving_currency"])
        key.append(row["payment_format"])
        key.append(row["timestamp"])
        return tuple(key)

    key_index = defaultdict(list)
    for eid, meta in edge_metadata_lookup.items():
        if meta["label"] == 1:
            key_index[make_meta_key(meta)].append(eid)
    for k in key_index:
        key_index[k].sort()

    n_matched, n_unmatched, n_ambiguous = 0, 0, 0
    for block in blocks:
        for row in block["rows"]:
            key = make_row_key(row)
            candidates = key_index.get(key, [])

            if len(candidates) == 0:
                row["edge_id"] = None
                n_unmatched += 1
                continue
            if len(candidates) == 1:
                row["edge_id"] = candidates[0]
                n_matched += 1
                continue
            n_ambiguous += 1
            row["edge_id"] = None

    logging.info(
        f"Matched {n_matched} pattern-file transactions to edge ids "
        f"({n_ambiguous} were ambiguous by key, "
        f"{n_unmatched} could not be matched at all)"
    )
    return blocks


def pick_target_row(block):
    matched_rows = [r for r in block["rows"] if r.get("edge_id") is not None]
    if not matched_rows:
        return None
    return matched_rows[0]


def main():
    parser = base_parser()
    parser.add_argument("--patterns_file", required=True, help="Path to patterns files")
    parser.add_argument("--n_examples_per_pattern", type=int, default=1,
                        help="How many blocks to sample per pattern type (default )")
    parser.add_argument("--n_non_suspicious", type=int, default=4,
                        help="How many non-suspicious examples to auto-sample")
    parser.add_argument("--max_subgraph_edges", type=int, default=10)
    parser.add_argument("--out", default="fewshot_examples.json")
    args = parser.parse_args()

    logger_setup()
    set_seed(args.seed)
    random.seed(args.seed)

    with open("data_config.json") as f:
        data_config = json.load(f)

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

    ckpt_path = f'{data_config["paths"]["model_to_load"]}/checkpoint_{args.unique_name}.tar'
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device).eval()

    edge_metadata_lookup = build_edge_metadata_lookup(args, data_config)

    # Suspicious examples: parse + match + explain
    blocks = parse_patterns_file(args.patterns_file)
    blocks = match_pattern_rows_to_edge_ids(blocks, edge_metadata_lookup)

    blocks_by_pattern = defaultdict(list)
    for b in blocks:
        blocks_by_pattern[b["pattern"]].append(b)

    suspicious_examples = []
    for pattern in PATTERN_NAME_MAP.values():
        candidate_blocks = list(blocks_by_pattern.get(pattern, []))
        random.shuffle(candidate_blocks)
        n_added = 0
        for block in candidate_blocks:
            if n_added >= args.n_examples_per_pattern:
                break
            target_row = pick_target_row(block)
            if target_row is None:
                continue
            edge_idx = target_row["edge_id"]
            try:
                sampled = sample_predict_explain(edge_idx, te_data, model, device, args, transform, args.reverse_mp)
            except RuntimeError as e:
                logging.warning(f"Skipping edge {edge_idx} ({pattern}): {e}")
                continue
            text = serialize_subgraph(
                edge_metadata_lookup, sampled['global_edge_ids'], sampled['target_edge_id'],
                imp_lookup_by_id=sampled['imp_by_id'], max_edges=args.max_subgraph_edges,
            )
            suspicious_examples.append({
                "pattern": pattern,
                "subgraph_text": text,
                "explanation": PATTERN_EXPLANATIONS.get(pattern, "TODO: write a rationale for this pattern."),
            })
            n_added += 1
            logging.info(f"Added suspicious example: pattern={pattern}, edge_id={edge_idx}")
        if n_added == 0:
            logging.warning(f"No matched examples found for pattern '{pattern}' -- leaving it out of the output.")

    # Non-suspicious examples: random sample
    if not args.reverse_mp:
        non_fraud = te_inds[te_data.y[te_inds] == 0].tolist()
    else:
        non_fraud = te_inds[te_data['node', 'to', 'node'].y[te_inds] == 0].tolist()
    random.shuffle(non_fraud)

    non_suspicious_examples = []
    for edge_idx in non_fraud:
        if len(non_suspicious_examples) >= args.n_non_suspicious:
            break
        try:
            sampled = sample_predict_explain(edge_idx, te_data, model, device, args, transform, args.reverse_mp)
        except RuntimeError as e:
            logging.warning(f"Skipping non-fraud edge {edge_idx}: {e}")
            continue
        text = serialize_subgraph(
            edge_metadata_lookup, sampled['global_edge_ids'], sampled['target_edge_id'],
            imp_lookup_by_id=sampled['imp_by_id'], max_edges=args.max_subgraph_edges,
        )
        non_suspicious_examples.append({
            "pattern": "routine",
            "subgraph_text": text,
            "explanation": NON_SUSPICIOUS_EXPLANATION_TEMPLATE,
        })
        logging.info(f"Added non-suspicious example: edge_id={edge_idx}")

    result = {
        "suspicious": suspicious_examples,
        "non_suspicious": non_suspicious_examples,
    }

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    logging.info(
        f"Wrote {len(suspicious_examples)} suspicious and "
        f"{len(non_suspicious_examples)} non-suspicious examples to {args.out}."
    )

if __name__ == "__main__":
    main()