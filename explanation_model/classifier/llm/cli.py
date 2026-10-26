"""CLI: build dataset -> LoRA fine-tune -> evaluate the LLM pattern classifier.

    python -m explanation_model.classifier.llm --patterns_file HI-Small_Patterns.txt \
        --work_dir ./pattern_clf --n_samples 2000 --base_model Qwen/Qwen3-14B --load_in_4bit

    # zero-shot baseline of the base model on the same test split (no adapter):
    python -m explanation_model.classifier.llm --patterns_file HI-Small_Patterns.txt --work_dir ./pattern_clf \
        --skip_build --skip_train --eval_zero_shot --load_in_4bit

    # re-run only the evaluation with the saved adapter:
    python -m explanation_model.classifier.llm --patterns_file HI-Small_Patterns.txt --work_dir ./pattern_clf \
        --skip_build --skip_train --load_in_4bit

Only the patterns file is needed (it already contains accounts, amounts, currencies, payment formats and
timestamps) -- no GNN, no formatted_transactions.csv.
"""
import argparse
import logging
import os
import random
from collections import Counter

from common.utils import set_seed, setup_logging, write_jsonl
from explanation_model.classifier.dataset import build_splits, parse_patterns_file
from explanation_model.classifier.llm.prompts import build_completion, build_prompt


def sample_to_record(s):
    """PatternSample -> one jsonl line of train.jsonl / test.jsonl."""
    return {
        "block_id": s.block_id,
        "target_row": s.target_row,
        "n_block_edges": s.n_block_edges,
        "pattern": s.pattern,
        "prompt": build_prompt(s.subgraph.serialize()),
        "completion": build_completion(s.pattern),
    }


def build_datasets(args, paths):
    blocks = parse_patterns_file(args.patterns_file)
    train_samples, test_samples = build_splits(blocks, args.n_samples, args.train_frac, args.max_block_edges)
    train_recs = [sample_to_record(s) for s in train_samples]
    test_recs = [sample_to_record(s) for s in test_samples]

    write_jsonl(train_recs, paths["train"])
    write_jsonl(test_recs, paths["test"])
    for name, recs in (("train", train_recs), ("test", test_recs)):
        logging.info(f"{name}: {len(recs)} samples from {len({r['block_id'] for r in recs})} blocks; "
                     f"per pattern: {dict(Counter(r['pattern'] for r in recs))}")
    logging.info(f"Wrote {paths['train']} and {paths['test']}")


def create_parser():
    p = argparse.ArgumentParser()
    # data
    p.add_argument("--patterns_file", required=True, help="Raw IBM AML *_Patterns.txt file")
    p.add_argument("--work_dir", default="./pattern_clf", help="Where datasets / adapter / results are written")
    p.add_argument("--n_samples", type=int, default=2000,
                   help="Total samples (train+test) to create. -1 = every edge of every block.")
    p.add_argument("--train_frac", type=float, default=0.8)
    p.add_argument("--max_block_edges", type=int, default=40,
                   help="Cap on edges serialized per block (window around the target edge).")
    # model / training
    p.add_argument("--base_model", default="Qwen/Qwen3-14B")
    p.add_argument("--load_in_4bit", action="store_true")
    p.add_argument("--max_seq_len", type=int, default=4096)
    p.add_argument("--n_epochs", type=int, default=3)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=16)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_target_modules", nargs="+",
                   default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    p.add_argument("--logging_steps", type=int, default=10)
    # eval
    p.add_argument("--eval_batch_size", type=int, default=4)
    p.add_argument("--max_new_tokens", type=int, default=24)
    p.add_argument("--max_eval_samples", type=int, default=-1)
    p.add_argument("--eval_zero_shot", action="store_true", help="Evaluate the base model without the adapter")
    # stages
    p.add_argument("--skip_build", action="store_true", help="Reuse existing train.jsonl / test.jsonl")
    p.add_argument("--skip_train", action="store_true", help="Reuse existing adapter")
    p.add_argument("--skip_eval", action="store_true")
    p.add_argument("--seed", type=int, default=1)
    return p




def main():
    args = create_parser().parse_args()
    setup_logging()
    set_seed(args.seed)
    random.seed(args.seed)

    os.makedirs(args.work_dir, exist_ok=True)
    paths = {
        "train": os.path.join(args.work_dir, "train.jsonl"),
        "test": os.path.join(args.work_dir, "test.jsonl"),
        "adapter": os.path.join(args.work_dir, "adapter"),
        "results": os.path.join(args.work_dir, "eval_results_zero_shot.csv" if args.eval_zero_shot
                                else "eval_results.csv"),
        "metrics": os.path.join(args.work_dir, "eval_metrics_zero_shot.json" if args.eval_zero_shot
                                else "eval_metrics.json"),
    }

    if not args.skip_build:
        build_datasets(args, paths)
    if not args.skip_train:
        from explanation_model.classifier.llm.finetune import train
        train(args, paths)
    if not args.skip_eval:
        from explanation_model.classifier.llm.evaluate import evaluate
        evaluate(args, paths)


if __name__ == "__main__":
    main()
