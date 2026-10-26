"""Evaluation of the LLM pattern classifier on test.jsonl (accuracy, macro-F1, per-pattern P/R/F1, confusion matrix)."""
import json
import logging

from common.utils import read_jsonl
from explanation_model.classifier.llm.classifier import LLMPatternClassifier
from explanation_model.classifier.llm.prompts import parse_pattern
from explanation_model.patterns import PATTERN_LABELS, UNPARSED


def evaluate(args, paths):
    import pandas as pd
    from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

    if args.eval_zero_shot:
        logging.info("--eval_zero_shot: evaluating the base model WITHOUT the adapter.")
    classifier = LLMPatternClassifier(
        base_model=args.base_model,
        adapter_path=None if args.eval_zero_shot else paths["adapter"],
        load_in_4bit=args.load_in_4bit, max_new_tokens=args.max_new_tokens,
        batch_size=args.eval_batch_size,
    )

    test = read_jsonl(paths["test"])
    if args.max_eval_samples > 0:
        test = test[:args.max_eval_samples]
    logging.info(f"Evaluating on {len(test)} test samples")

    decoded = classifier.generate([ex["prompt"] for ex in test], log_progress=True)
    rows = []
    for ex, raw in zip(test, decoded):
        rows.append({
            "block_id": ex["block_id"], "target_row": ex["target_row"],
            "n_block_edges": ex["n_block_edges"],
            "actual_pattern": ex["pattern"], "llm_pattern": parse_pattern(raw), "llm_raw": raw.strip(),
        })

    df = pd.DataFrame(rows)
    df.to_csv(paths["results"], index=False)

    y_true, y_pred = df["actual_pattern"].tolist(), df["llm_pattern"].tolist()
    true_labels = [l for l in PATTERN_LABELS if l in set(y_true)]
    cm_labels = true_labels + sorted({p for p in y_pred if p not in true_labels})
    n_unparsed = sum(p == UNPARSED for p in y_pred)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=true_labels, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=true_labels, average="weighted", zero_division=0)

    logging.info("\n" + "=" * 60 + "\nRESULTS (LLM pattern classifier, "
                 f"{'zero-shot base model' if args.eval_zero_shot else 'fine-tuned'})\n" + "=" * 60)
    logging.info(f"Samples: {len(df)} | unparseable outputs (counted as wrong): {n_unparsed}")
    logging.info(f"Accuracy    = {acc:.4f}")
    logging.info(f"Macro F1    = {macro_f1:.4f}")
    logging.info(f"Weighted F1 = {weighted_f1:.4f}")
    logging.info("\n" + classification_report(y_true, y_pred, labels=true_labels, zero_division=0, digits=4))
    cm = pd.DataFrame(confusion_matrix(y_true, y_pred, labels=cm_labels),
                      index=[f"true:{l}" for l in cm_labels], columns=[f"pred:{l}" for l in cm_labels])
    logging.info("Confusion matrix (rows = actual, cols = predicted):\n" + cm.to_string())

    with open(paths["metrics"], "w") as f:
        json.dump({"n": len(df), "n_unparsed": n_unparsed, "accuracy": acc,
                   "macro_f1": macro_f1, "weighted_f1": weighted_f1,
                   "zero_shot": bool(args.eval_zero_shot)}, f, indent=2)
    logging.info(f"Per-sample results -> {paths['results']}, metrics -> {paths['metrics']}")
