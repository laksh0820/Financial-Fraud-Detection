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

    # Skip <think> generation entirely instead of stripping it afterward
    # (faster/cheaper, since no reasoning tokens get generated at all):
    python evaluate_llm_reasoning.py --data Small_HI --model gin --unique_name run1 \
        --n_samples 200 --stratify --llm_disable_thinking \
        --out_path llm_eval_results.csv

    # To (expensively) run over the *entire* test set instead of a sample:
    python evaluate_llm_reasoning.py --data Small_HI --model gin --unique_name run1 \
        --n_samples -1 --out_path llm_eval_full.csv

    # Just refresh the prompts cache (no LLM/GPU headroom needed for it):
    python evaluate_llm_reasoning.py --data Small_HI --model gin --unique_name run1 \
        --n_samples 200 --stratify --skip_llm_pass \
        --prompts_cache_path llm_prompts_cache.jsonl

    # Re-run (or try a different) LLM against an already-cached GNN pass,
    # without redoing the expensive sampling/explainer step:
    python evaluate_llm_reasoning.py --skip_gnn_pass \
        --prompts_cache_path llm_prompts_cache.jsonl \
        --llm_model Qwen/Qwen3-14B --out_path llm_eval_results.csv
"""

import gc
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
    build_edge_metadata_lookup, compute_subgraph_and_prompt, llm_infer_and_parse,
    LLMClient, _build_config,
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


def run_gnn_pass(eval_inds, te_data, model, device, args, transform,
                  edge_metadata_lookup, fewshot, cache_path):
    """Stage 1: GNN predict + explain + prompt-build for every eval edge."""
    n_ok, n_failed = 0, 0
    with open(cache_path, "w") as f:
        for i, edge_idx in enumerate(eval_inds):
            try:
                pre = compute_subgraph_and_prompt(
                    edge_idx, te_data, model, device, args, transform, args.reverse_mp,
                    edge_metadata_lookup, fewshot, max_edges=args.max_subgraph_edges,
                )
                f.write(json.dumps(pre) + "\n")
                f.flush()
                n_ok += 1
                logging.info(
                    f"[GNN pass {i + 1}/{len(eval_inds)}] edge={edge_idx} "
                    f"gnn_pred={pre['gnn_pred']} actual={pre['actual_label']}"
                )
            except RuntimeError as e:
                # e.g. sample_predict_explain couldn't locate exactly one seed edge
                logging.warning(f"Skipping edge {edge_idx} (sampling error): {e}")
                n_failed += 1
            except Exception as e:
                # covers explainer failures, serialization issues, etc.
                logging.warning(f"Skipping edge {edge_idx} (error): {e}")
                n_failed += 1

    logging.info(f"GNN pass done. {n_ok} cached to {cache_path}, {n_failed} failed/skipped.")
    return n_ok, n_failed


def run_llm_pass(cache_path, llm_client, out_path, checkpoint_every, sleep_between_calls):
    """Stage 2: reads the cached prompts from run_gnn_pass and runs only the
    LLM call + parsing. """
    with open(cache_path) as f:
        cached = [json.loads(line) for line in f if line.strip()]

    results = []
    n_failed = 0
    for i, pre in enumerate(cached):
        try:
            parsed = llm_infer_and_parse(llm_client, pre["prompt"])
            r = {
                'edge_id': pre['edge_id'],
                'actual_label': pre['actual_label'],
                'gnn_pred': pre['gnn_pred'],
                'llm_conclusion': parsed['conclusion'],
                'llm_pred': conclusion_to_label(parsed['conclusion']),
                'llm_pattern': parsed['observed_pattern'],
            }
            results.append(r)
            logging.info(
                f"[LLM pass {i + 1}/{len(cached)}] edge={pre['edge_id']} actual={r['actual_label']} "
                f"gnn_pred={r['gnn_pred']} llm_conclusion={r['llm_conclusion']!r} "
                f"llm_pred={r['llm_pred']}"
            )
        except Exception as e:
            # covers LLM API errors, malformed responses, etc.
            logging.warning(f"Skipping edge {pre['edge_id']} (LLM error): {e}")
            n_failed += 1

        if sleep_between_calls > 0:
            time.sleep(sleep_between_calls)

        if checkpoint_every and (i + 1) % checkpoint_every == 0:
            dump_results(results, out_path)
            logging.info(f"Checkpoint written to {out_path} ({len(results)} rows so far)")

    dump_results(results, out_path)
    logging.info(f"LLM pass done. {len(results)} scored, {n_failed} skipped/failed. Results -> {out_path}")
    return results


def main():
    parser = base_parser()
    parser.add_argument("--fewshot_path", default="fewshot_examples.json")
    parser.add_argument("--llm_provider", default="local", choices=["local"])
    parser.add_argument("--llm_model", default="Qwen/Qwen3-14B")
    parser.add_argument("--llm_max_new_tokens", type=int, default=512, help="Max tokens to generate")
    parser.add_argument("--llm_disable_thinking", action='store_true',
                         help="For hybrid thinking/non-thinking models (e.g. Qwen3): pass "
                              "enable_thinking=False so the model skips the <think>...</think> "
                              "block entirely (faster, fewer tokens). If not set, thinking stays "
                              "on by default and the <think> block is stripped from the output "
                              "after generation instead -- either way, llm_conclusion only ever "
                              "sees the final answer.")
    parser.add_argument("--max_subgraph_edges", type=int, default=10)
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
                         help="Write partial results to disk every N edges (LLM pass only).")
    parser.add_argument("--out_path", default="llm_eval_results.csv")
    parser.add_argument("--prompts_cache_path", default="llm_prompts_cache.jsonl",
                         help="Where stage 1 (GNN pass) writes its cached prompts/predictions, "
                              "and where stage 2 (LLM pass) reads them from.")
    parser.add_argument("--skip_gnn_pass", action='store_true',
                         help="Skip stage 1 and reuse an existing --prompts_cache_path "
                              "(e.g. to re-run just the LLM pass with a different model).")
    parser.add_argument("--skip_llm_pass", action='store_true',
                         help="Only run stage 1 (GNN pass) and exit -- useful for producing/"
                              "refreshing the prompts cache without needing GPU room for the LLM.")
    args = parser.parse_args()

    logger_setup()
    set_seed(args.seed)
    random.seed(args.seed)

    with open(args.fewshot_path) as f:
        fewshot = json.load(f)

    # ---------------------------------------------------------------
    # Stage 1: GNN predict + explain + prompt-build for every eval edge.
    # The LLM is NOT loaded during this stage, so the explainer gets the
    # whole GPU to itself.
    # ---------------------------------------------------------------
    if not args.skip_gnn_pass:
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

        edge_y = te_data['node', 'to', 'node'].y if args.reverse_mp else te_data.y
        eval_inds = select_eval_indices(te_inds, edge_y, args)

        n_fraud = sum(edge_y[i].item() for i in eval_inds)
        logging.info(f"Evaluating {len(eval_inds)} test edges "
                     f"({n_fraud} fraud / {len(eval_inds) - n_fraud} non-fraud).")

        run_gnn_pass(
            eval_inds, te_data, model, device, args, transform,
            edge_metadata_lookup, fewshot, args.prompts_cache_path,
        )

        # Explicitly drop everything GPU-resident from stage 1 before stage 2 loads the LLM
        del model, tr_loader, val_loader, te_loader, sample_batch, checkpoint
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logging.info("Freed GNN model/loaders from GPU memory before loading the LLM.")
    else:
        logging.info(f"--skip_gnn_pass set: reusing existing cache at {args.prompts_cache_path}")

    if args.skip_llm_pass:
        logging.info("--skip_llm_pass set: stopping after the GNN pass.")
        return

    # ---------------------------------------------------------------
    # Stage 2: LLM call + parsing over the cached prompts. The GNN model is
    # gone from the GPU by this point, so the LLM gets the full card.
    # ---------------------------------------------------------------
    llm_client = LLMClient(
        provider=args.llm_provider, model=args.llm_model,
        max_new_tokens=args.llm_max_new_tokens,
        enable_thinking=(False if args.llm_disable_thinking else None),
    )

    results = run_llm_pass(
        args.prompts_cache_path, llm_client, args.out_path,
        args.checkpoint_every, args.sleep_between_calls,
    )

    score_and_report(results)

if __name__ == "__main__":
    main()