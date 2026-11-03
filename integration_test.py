"""End-to-end test of the detect.py path (GNN -> GNN-sampled subgraph -> LLM pattern classifier).

The standalone classifier (explanation_model/classifier/llm) is trained / evaluated on COMPLETE
laundering-attempt blocks of the patterns file. At detection time, however, the LLM only sees the
GNN-sampled neighbourhood of one transaction (trimmed to --max_subgraph_edges), which may be incomplete
and may contain unrelated transactions. This script measures how well the pattern is still recognised:

  1. Take the patterns-file blocks and split them train/test exactly like the classifier CLI does
     (same seed + same build_splits -> the SAME test edges as the classifier's test.jsonl, none of them
     from a block the adapter was trained on). Pick the target edges from the TEST blocks only.
  2. The ground-truth label of an edge is the pattern of its block. Every row of a block is a laundering
     transaction, so the GNN is also expected to flag it.
  3. The patterns file has no transaction ids -> every picked row is located in
     formatted_transactions.csv (accounts + timestamp + currency + payment format + amount) to get its
     global edge id, which is what detect.py takes.
  4. Run the real pipeline once:  python detect.py <your GNN / LLM flags> --edge_ids <ids> --out_path ...
     (GNN prediction -> if fraud: GNNExplainer importance -> sampled subgraph -> LLM -> pattern)
  5. Compare the predicted pattern with the block pattern and report
        * GNN recall on these edges (an edge the GNN does not flag is never explained),
        * pattern accuracy / macro-F1 of the edges that were flagged and explained,
        * end-to-end accuracy over ALL picked edges (a GNN miss counts as wrong),
        * how much the sampled subgraph overlaps the true block (missing / foreign edges).

Everything this script does not know is forwarded untouched to detect.py, so use the flags you would use
for detect.py (the GNN flags must match the checkpoint):

    python integration_test.py --patterns_file HI-Small_Patterns.txt --work_dir ./e2e_test \\
        --n_test_edges 100 \\
        --data Small_HI --model gin --unique_name run1 \\
        --adapter_path ./pattern_clf/adapter --load_in_4bit

Outputs (in --work_dir): selected_edges.csv, detections.json (raw detect.py output), detect.log,
e2e_results.csv (one row per edge), e2e_metrics.json.

Use the same --seed / --n_samples / --train_frac as when the classifier was trained (the defaults match the
classifier CLI), or pass --test_jsonl <work_dir of the classifier>/test.jsonl to use exactly its test edges.
"""
import argparse
import calendar
import json
import logging
import os
import random
import re
import shlex
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from common.config import DEFAULT_DATA_CONFIG, dataset_dir, load_data_config, load_json
from common.utils import read_jsonl, setup_logging
from explanation_model.classifier.dataset import build_splits, parse_patterns_file
from explanation_model.patterns import PATTERN_LABELS, UNPARSED

REPO_ROOT = Path(__file__).resolve().parent
NOT_FLAGGED = "<not_flagged>"   # "predicted pattern" of an edge for which no pattern was produced
AMOUNT_TOLERANCE = 0.005        # patterns file prints 2 decimals, the csv 6 -> compare within half a cent
# flags that detect.py would interpret differently / that this script sets itself
RESERVED_DETECT_FLAGS = {"--edge_ids", "--n_random_edges", "--sample_label", "--out_path"}


@dataclass
class Target:
    """One test edge: row `target_row` of laundering-attempt block `block_id`."""
    block_id: int
    target_row: int
    pattern: str                      # ground truth = pattern of the block
    n_block_edges: int
    edge_id: Optional[int] = None     # global transaction id (row in formatted_transactions.csv)
    csv_label: Optional[int] = None   # 'Is Laundering' of that csv row (should be 1)
    n_candidates: int = 0             # how many csv rows matched the pattern-file row
    unmapped_reason: str = ""


# ----------------------------------------------------------------------------
# 1. Pick the test edges (same procedure as llm_pattern_classifier.py / classifier CLI)
# ----------------------------------------------------------------------------

def pick_test_targets(args, blocks) -> List[Target]:
    if args.test_jsonl:
        targets = []
        for r in read_jsonl(args.test_jsonl):
            bid = r["block_id"]
            if bid >= len(blocks) or blocks[bid]["pattern"] != r["pattern"]:
                raise ValueError(f"{args.test_jsonl} does not belong to {args.patterns_file} "
                                 f"(block {bid} is missing or has another pattern)")
            targets.append(Target(bid, r["target_row"], r["pattern"], len(blocks[bid]["rows"])))
        logging.info(f"Read {len(targets)} test edges from {args.test_jsonl}")
        return targets

    # identical call order to classifier/llm/cli.py (random.seed -> parse -> build_splits), so with the same
    # seed / n_samples / train_frac this reproduces the classifier's train/test block split and test edges
    random.seed(args.split_seed)
    train_samples, test_samples = build_splits(blocks, args.n_samples, args.train_frac, max_edges=40)
    train_blocks = {s.block_id for s in train_samples}
    test_blocks = {s.block_id for s in test_samples}
    assert not train_blocks & test_blocks, "block-level split is broken: a block is in train and test"
    logging.info(f"Test split: {len(test_samples)} edges from {len(test_blocks)} blocks "
                 f"(per pattern: {dict(Counter(s.pattern for s in test_samples))})")
    return [Target(s.block_id, s.target_row, s.pattern, s.n_block_edges) for s in test_samples]


def stratified_cap(targets: List[Target], n: int) -> List[Target]:
    """At most n targets, taken round-robin over the patterns so that every pattern is represented."""
    if n < 0 or len(targets) <= n:
        return targets
    by_pattern: Dict[str, List[Target]] = {}
    for t in targets:
        by_pattern.setdefault(t.pattern, []).append(t)
    picked, queues = [], list(by_pattern.values())
    while len(picked) < n and queues:
        for q in list(queues):
            if len(picked) >= n:
                break
            picked.append(q.pop(0))
            if not q:
                queues.remove(q)
    return picked


# ----------------------------------------------------------------------------
# 2. Patterns-file row -> global transaction id in formatted_transactions.csv
# ----------------------------------------------------------------------------

def locate_in_csv(targets: List[Target], blocks, data_dir: str) -> None:
    """Fills target.edge_id / csv_label / n_candidates (or unmapped_reason) in place.

    format_kaggle_files.py wrote account_map / currency_map / payment_format_map / format_meta.json next to
    the csv; they are inverted here to encode the patterns-file row the way the csv does. The global edge id
    is the row position in the csv (what get_data / add_arange_ids / EdgeMetadataStore use)."""
    import numpy as np
    import pandas as pd

    inv = lambda name: {v: int(k) for k, v in load_json(os.path.join(data_dir, name)).items()}  # noqa: E731
    account_id, currency_id, format_id = inv("account_map.json"), inv("currency_map.json"), inv("payment_format_map.json")
    first_ts = load_json(os.path.join(data_dir, "format_meta.json"))["firstTs"]

    wanted = []
    for i, t in enumerate(targets):
        r = blocks[t.block_id]["rows"][t.target_row]
        ids = (account_id.get(r["src"]), account_id.get(r["dst"]),
               currency_id.get(r["receiving_currency"]), format_id.get(r["payment_format"]))
        if None in ids:
            t.unmapped_reason = "account / currency / payment format not in the encoding maps"
            continue
        epoch = calendar.timegm(r["timestamp"].timetuple())  # patterns-file timestamps are UTC, like the csv
        wanted.append({"i": i, "from_id": ids[0], "to_id": ids[1], "Received Currency": ids[2],
                       "Payment Format": ids[3], "Timestamp": int(round(epoch - first_ts)),
                       "amount": r["amount_received"]})
    if not wanted:
        return
    need = pd.DataFrame(wanted)

    csv_path = os.path.join(data_dir, "formatted_transactions.csv")
    logging.info(f"Locating {len(need)} pattern-file rows in {csv_path} ...")
    df = pd.read_csv(csv_path, usecols=["from_id", "to_id", "Timestamp", "Amount Received",
                                        "Received Currency", "Payment Format", "Is Laundering"])
    df["edge_id"] = np.arange(len(df))
    df = df[df["from_id"].isin(need["from_id"].unique())]  # cheap pre-filter, the csv has millions of rows
    keys = ["from_id", "to_id", "Timestamp", "Received Currency", "Payment Format"]
    m = need.merge(df, on=keys, how="left")
    m["diff"] = (m["amount"] - m["Amount Received"]).abs()
    m = m[m["diff"] <= AMOUNT_TOLERANCE]
    # prefer laundering rows, then the closest amount, then the smallest id (deterministic)
    m = m.sort_values(["i", "Is Laundering", "diff", "edge_id"], ascending=[True, False, True, True])

    used = set()
    for i, grp in m.groupby("i", sort=True):
        t = targets[int(i)]
        t.n_candidates = len(grp)
        free = grp[~grp["edge_id"].isin(used)]  # two identical rows must not collapse onto one csv row
        if free.empty:
            t.unmapped_reason = "all matching csv rows are already taken by another picked row"
            continue
        best = free.iloc[0]
        t.edge_id, t.csv_label = int(best["edge_id"]), int(best["Is Laundering"])
        used.add(t.edge_id)
    for t in targets:
        if t.edge_id is None and not t.unmapped_reason:
            t.unmapped_reason = "no matching row in formatted_transactions.csv"


def gnn_test_edge_ids(detect_flags: List[str]) -> set:
    """Global ids of the GNN's own (temporal) test split, computed exactly like detect.py / the predictor do."""
    sys.path.insert(0, str(REPO_ROOT))
    import detect  # noqa: E402  (light: torch is only imported inside detect.main)
    from prediction_model.data_loading import get_data  # heavy (torch_geometric)

    a = detect.build_parser(detect_flags).parse_args(detect_flags + ["--edge_ids", "0"])
    *_, te_inds = get_data(a, load_data_config(a.data_config))
    return set(te_inds.tolist())


# ----------------------------------------------------------------------------
# 3. Run detect.py
# ----------------------------------------------------------------------------

def run_detect(detect_script: str, detect_flags: List[str], edge_ids: List[int], out_path: str, log_path: str):
    cmd = [sys.executable, detect_script, *detect_flags, "--edge_ids", *map(str, edge_ids), "--out_path", out_path]
    shown = " ".join(shlex.quote(c) for c in cmd[:len(cmd) - len(edge_ids) - 3])
    logging.info(f"Running: {shown} --edge_ids <{len(edge_ids)} ids> --out_path {out_path}")
    if os.path.exists(out_path):
        os.remove(out_path)  # never score a stale file if detect.py dies early
    echo = True
    with open(log_path, "w") as log, subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace", bufsize=1) as proc:
        for line in proc.stdout:
            log.write(line)
            # detect.py ends by printing the whole result as JSON (a line "[" ... a line "]"); it is already
            # in --out_path, so do not flood the console with it (the full output stays in detect.log)
            if echo and line.rstrip() == "[":
                echo = False
            if echo:
                sys.stdout.write(line)
            elif line.rstrip() == "]":
                echo = True
        returncode = proc.wait()
    if returncode != 0 or not os.path.exists(out_path):
        raise RuntimeError(f"detect.py failed (exit code {returncode}); see {log_path}")


# ----------------------------------------------------------------------------
# 4. Scoring
# ----------------------------------------------------------------------------

_EDGE_RE = re.compile(r"amount:\s*([\d.]+)\s*\(currency:\s*(.+?)\)\s*\n\s*via:\s*(.+?)\s*\n\s*timestamp:\s*(\d{4}/\d\d/\d\d \d\d:\d\d)")


def subgraph_overlap(subgraph_text: str, block) -> Dict[str, float]:
    """How the subgraph the LLM was shown relates to the true block (edges matched on amount, currency,
    payment format and timestamp, which Subgraph.serialize() keeps)."""
    shown = Counter(_EDGE_RE.findall(subgraph_text))
    truth = Counter((f"{r['amount_received']:.2f}", r["receiving_currency"], r["payment_format"], r["timestamp_str"])
                    for r in block["rows"])
    in_block = sum((shown & truth).values())
    return {"block_edges_in_subgraph": in_block,
            "foreign_edges": sum(shown.values()) - in_block,   # unrelated transactions the GNN sampled
            "block_coverage": in_block / len(block["rows"])}


def score(targets: List[Target], detections: Dict[int, dict], blocks) -> "pd.DataFrame":  # noqa: F821
    import pandas as pd
    rows = []
    for t in targets:
        rec = detections.get(t.edge_id)
        row = {"block_id": t.block_id, "target_row": t.target_row, "edge_id": t.edge_id,
               "actual_pattern": t.pattern, "n_block_edges": t.n_block_edges, "csv_label": t.csv_label,
               "gnn_pred": None, "fraud_prob": None, "gnn_actual_label": None, "pred_pattern": NOT_FLAGGED,
               "llm_raw": None, "n_subgraph_edges": None, "block_edges_in_subgraph": None,
               "foreign_edges": None, "block_coverage": None, "error": None}
        if rec is None:
            row["status"], row["error"] = "no_result", "detect.py produced no record for this edge"
        else:
            row.update(gnn_pred=rec.get("gnn_pred"), fraud_prob=rec.get("fraud_prob"),
                       gnn_actual_label=rec.get("actual_label"), error=rec.get("error"))
            if rec.get("error"):
                row["status"] = "pipeline_error" if rec.get("gnn_pred") == 1 else "gnn_error"
            elif rec.get("gnn_pred") != 1:
                row["status"] = "gnn_miss"  # explanation models never run for a non-fraud prediction
            else:
                ex = rec["explanation"]
                row.update(status="explained", pred_pattern=ex["pattern"], llm_raw=ex.get("classifier_raw"),
                           n_subgraph_edges=ex.get("n_subgraph_edges"),
                           **subgraph_overlap(ex.get("subgraph", ""), blocks[t.block_id]))
        row["correct"] = row["pred_pattern"] == t.pattern
        rows.append(row)
    return pd.DataFrame(rows)


def report(df, args) -> dict:
    import pandas as pd
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

    n = len(df)
    n_flagged = int((df["gnn_pred"] == 1).sum())                # GNN said fraud (explained or pipeline error)
    exp = df[df["status"] == "explained"]
    n_unparsed = int((exp["pred_pattern"] == UNPARSED).sum())
    true_labels = [l for l in PATTERN_LABELS if l in set(df["actual_pattern"])]

    metrics = {
        "n_edges": n,
        "status_counts": df["status"].value_counts().to_dict(),
        "gnn_recall": n_flagged / n,                              # every picked edge is a laundering edge
        "n_explained": len(exp), "n_unparsed": n_unparsed,
        "pattern_accuracy_given_flagged": float(exp["correct"].mean()) if len(exp) else None,
        "e2e_accuracy": float(df["correct"].mean()),             # GNN misses / errors count as wrong
        "e2e_macro_f1": float(f1_score(df["actual_pattern"], df["pred_pattern"], labels=true_labels,
                                       average="macro", zero_division=0)),
    }
    if len(exp):
        metrics["pattern_macro_f1_given_flagged"] = float(f1_score(
            exp["actual_pattern"], exp["pred_pattern"], labels=true_labels, average="macro", zero_division=0))
        metrics["pattern_weighted_f1_given_flagged"] = float(f1_score(
            exp["actual_pattern"], exp["pred_pattern"], labels=true_labels, average="weighted", zero_division=0))
    bad_gt = int((df["gnn_actual_label"].notna() & (df["gnn_actual_label"] != 1)).sum())
    metrics["n_edges_whose_dataset_label_is_not_fraud"] = bad_gt

    log = logging.info
    log("\n" + "=" * 70 + "\nEND-TO-END RESULTS (detect.py: GNN -> sampled subgraph -> LLM)\n" + "=" * 70)
    log(f"Edges tested: {n} | status: {metrics['status_counts']}")
    if bad_gt:
        log(f"WARNING: for {bad_gt} edges detect.py reports a non-fraud dataset label -> the pattern-file row was "
            f"probably matched to the wrong csv row; check selected_edges.csv.")
    log(f"GNN recall on these (all fraud) edges      = {metrics['gnn_recall']:.4f}  ({n_flagged}/{n})")
    if len(exp):
        log(f"Pattern accuracy | flagged & explained      = {metrics['pattern_accuracy_given_flagged']:.4f}  "
            f"({int(exp['correct'].sum())}/{len(exp)}, unparseable LLM outputs: {n_unparsed})")
        log(f"Pattern macro-F1 | flagged & explained      = {metrics['pattern_macro_f1_given_flagged']:.4f}")
    log(f"End-to-end accuracy (misses count as wrong) = {metrics['e2e_accuracy']:.4f}")
    log(f"End-to-end macro-F1 (misses = '{NOT_FLAGGED}')  = {metrics['e2e_macro_f1']:.4f}")

    per = df.groupby("actual_pattern").agg(
        n=("edge_id", "size"), gnn_flagged=("gnn_pred", lambda s: int((s == 1).sum())),
        explained=("status", lambda s: int((s == "explained").sum())), correct=("correct", "sum"))
    per["gnn_recall"] = per["gnn_flagged"] / per["n"]
    per["acc_given_flagged"] = per["correct"] / per["explained"].where(per["explained"] > 0)
    per["e2e_acc"] = per["correct"] / per["n"]
    log("Per block pattern:\n" + per.reindex([l for l in PATTERN_LABELS if l in per.index]).round(3).to_string())
    metrics["per_pattern"] = per.round(4).to_dict(orient="index")

    if len(exp):
        y_true, y_pred = exp["actual_pattern"].tolist(), exp["pred_pattern"].tolist()
        tl = [l for l in PATTERN_LABELS if l in set(y_true)]
        cm_labels = tl + sorted({p for p in y_pred if p not in tl})
        log("Classification report (flagged & explained edges):\n"
            + classification_report(y_true, y_pred, labels=tl, zero_division=0, digits=4))
        cm = pd.DataFrame(confusion_matrix(y_true, y_pred, labels=cm_labels),
                          index=[f"true:{l}" for l in cm_labels], columns=[f"pred:{l}" for l in cm_labels])
        log("Confusion matrix (rows = block pattern, cols = predicted):\n" + cm.to_string())

        # does an incomplete / contaminated GNN neighbourhood explain the errors?
        ov = exp.assign(
            neighbourhood=lambda d: (d["foreign_edges"] > 0).map({True: "has foreign edges", False: "only block edges"}),
            complete=lambda d: (d["block_coverage"] >= 0.999).map({True: "whole block", False: "part of block"}))
        for col, title in (("neighbourhood", "unrelated transactions in the sampled subgraph"),
                           ("complete", "block edges present in the sampled subgraph")):
            g = ov.groupby(col)["correct"].agg(["size", "mean"]).rename(columns={"size": "n", "mean": "accuracy"})
            log(f"Accuracy by {title}:\n" + g.round(3).to_string())
        log(f"Mean block coverage of the sampled subgraph = {exp['block_coverage'].mean():.3f} | "
            f"mean #foreign edges = {exp['foreign_edges'].mean():.2f} | "
            f"mean #edges shown = {exp['n_subgraph_edges'].mean():.1f}")
        metrics["mean_block_coverage"] = float(exp["block_coverage"].mean())
        metrics["mean_foreign_edges"] = float(exp["foreign_edges"].mean())
    return metrics


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], allow_abbrev=False,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Every flag not listed here (--data, --model, --unique_name, --adapter_path, --load_in_4bit, "
               "--max_subgraph_edges, ...) is forwarded unchanged to detect.py.")
    p.add_argument("--patterns_file", required=True, help="Raw IBM AML *_Patterns.txt file")
    p.add_argument("--work_dir", default="./e2e_test", help="Where selected edges / detections / results are written")
    # how the test edges are picked: keep identical to the classifier's training run
    p.add_argument("--split_seed", type=int, default=1, help="Seed of the block split (= --seed of the classifier CLI)")
    p.add_argument("--n_samples", type=int, default=2000, help="Same --n_samples as the classifier CLI (-1 = all edges)")
    p.add_argument("--train_frac", type=float, default=0.8, help="Same --train_frac as the classifier CLI")
    p.add_argument("--test_jsonl", default=None,
                   help="Use exactly the edges of the classifier's test.jsonl instead of re-creating the split")
    p.add_argument("--n_test_edges", type=int, default=50,
                   help="How many test edges to run through detect.py (spread evenly over the patterns). -1 = all")
    p.add_argument("--only_gnn_test_edges", action="store_true",
                   help="Only use edges inside the GNN's own temporal test split (the GNN was trained on earlier "
                        "days, so its recall on those is otherwise optimistic). Loads the dataset (slow).")
    # stages
    p.add_argument("--detect_script", default=str(REPO_ROOT / "detect.py"))
    p.add_argument("--prepare_only", action="store_true", help="Only write selected_edges.csv and print the detect.py command")
    p.add_argument("--skip_detect", action="store_true",
                   help="Do not run detect.py: score the existing <work_dir>/selected_edges.csv + detections.json")
    return p


def forwarded_flags(argv_rest: List[str], parser) -> List[str]:
    bad = sorted({t.split("=")[0] for t in argv_rest} & RESERVED_DETECT_FLAGS)
    if bad:
        parser.error(f"{bad} are set by this script, do not pass them")
    return argv_rest


def write_selected(targets: List[Target], path: str):
    import pandas as pd
    pd.DataFrame([{"block_id": t.block_id, "target_row": t.target_row, "pattern": t.pattern,
                   "n_block_edges": t.n_block_edges, "edge_id": t.edge_id, "csv_label": t.csv_label,
                   "n_candidates": t.n_candidates} for t in targets]).to_csv(path, index=False)


def read_selected(path: str) -> List[Target]:
    import pandas as pd
    return [Target(int(r.block_id), int(r.target_row), r.pattern, int(r.n_block_edges), int(r.edge_id),
                   None if pd.isna(r.csv_label) else int(r.csv_label), int(r.n_candidates))
            for r in pd.read_csv(path).itertuples()]


def main():
    parser = build_parser()
    args, rest = parser.parse_known_args()
    detect_flags = forwarded_flags(rest, parser)
    mini = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    mini.add_argument("--data")
    mini.add_argument("--data_config", default=str(DEFAULT_DATA_CONFIG))
    known, _ = mini.parse_known_args(detect_flags)
    if not known.data and not args.skip_detect:
        parser.error("--data <dataset> is required (it is also forwarded to detect.py)")
    setup_logging()

    os.makedirs(args.work_dir, exist_ok=True)
    selected_path = os.path.join(args.work_dir, "selected_edges.csv")
    detections_path = os.path.join(args.work_dir, "detections.json")
    blocks = parse_patterns_file(args.patterns_file)

    # ---- 1-2. pick test edges and find their transaction ids -------------------------------------------------
    if args.skip_detect:
        targets = read_selected(selected_path)
        logging.info(f"--skip_detect: re-using {len(targets)} edges from {selected_path}")
    else:
        candidates = pick_test_targets(args, blocks)
        locate_in_csv(candidates, blocks, dataset_dir(load_data_config(known.data_config), known.data))
        mapped = [t for t in candidates if t.edge_id is not None]
        logging.info(f"Located {len(mapped)}/{len(candidates)} test edges in the transaction csv "
                     f"({sum(t.n_candidates > 1 for t in mapped)} had several identical csv rows)")
        for reason, c in Counter(t.unmapped_reason for t in candidates if t.edge_id is None).items():
            logging.warning(f"{c} test edges dropped: {reason}")
        not_fraud = sum(t.csv_label != 1 for t in mapped)
        if not_fraud:
            logging.warning(f"{not_fraud} located csv rows are labelled non-fraud (suspicious match)")
        if args.only_gnn_test_edges:
            te = gnn_test_edge_ids(detect_flags)
            before = len(mapped)
            mapped = [t for t in mapped if t.edge_id in te]
            logging.info(f"--only_gnn_test_edges: {len(mapped)}/{before} edges lie in the GNN test split")
        targets = stratified_cap(mapped, args.n_test_edges)
        if not targets:
            sys.exit("No test edge could be mapped to a transaction id; nothing to run.")
        write_selected(targets, selected_path)
        logging.info(f"Selected {len(targets)} edges (per pattern: {dict(Counter(t.pattern for t in targets))}) "
                     f"-> {selected_path}")
        if not args.only_gnn_test_edges:
            logging.warning("The GNN was trained on the first ~60% of the days; picked edges from that period were "
                            "seen by it, so GNN recall can be optimistic (see --only_gnn_test_edges). The pattern "
                            "accuracy of the LLM is not affected (its split is block-level).")

    # ---- 3. run the real pipeline -----------------------------------------------------------------------------
    edge_ids = list(dict.fromkeys(t.edge_id for t in targets))
    if args.prepare_only:
        print("python", args.detect_script, *map(shlex.quote, detect_flags), "--edge_ids", *edge_ids,
              "--out_path", detections_path)
        return
    if not args.skip_detect:
        run_detect(args.detect_script, detect_flags, edge_ids, detections_path,
                   os.path.join(args.work_dir, "detect.log"))

    # ---- 4. score ---------------------------------------------------------------------------------------------
    with open(detections_path) as f:
        detections = {r["edge_id"]: r for r in json.load(f)}
    df = score(targets, detections, blocks)
    df.to_csv(os.path.join(args.work_dir, "e2e_results.csv"), index=False)
    metrics = report(df, args)
    with open(os.path.join(args.work_dir, "e2e_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    logging.info(f"Per-edge results -> {os.path.join(args.work_dir, 'e2e_results.csv')}, "
                 f"metrics -> {os.path.join(args.work_dir, 'e2e_metrics.json')}")


if __name__ == "__main__":
    main()