"""
Standalone LLM laundering-pattern classifier (no GNN involved).

Pipeline:
  1. BUILD : parse the IBM AML *_Patterns.txt file. Every "laundering attempt" block is
             one subgraph. Target edges are picked uniformly over patterns -> blocks
             -> edges (same round-robin idea as mine_finetune_data.pick_pattern_edges),
             but the *entire block* is serialized as the subgraph (no GNN sampling,
             no GNNExplainer, no importance scores). The label is the block's pattern.
             Blocks are split 80/20 into train/test BEFORE edges are picked, so no
             block appears in both splits.
  2. TRAIN : LoRA fine-tune of a causal LLM on train.jsonl (completion = pattern only).
  3. EVAL  : greedy generation on test.jsonl, parse the predicted pattern, report
             accuracy / macro-F1 / per-pattern P-R-F1 / confusion matrix.

Only needs the patterns file (every row already contains accounts, amount, currency,
payment format and timestamp), so formatted_transactions.csv is NOT required.

Dependencies:
    pip install transformers accelerate peft bitsandbytes scikit-learn pandas --break-system-packages

Usage:
    python llm_pattern_classifier.py --patterns_file HI-Small_Patterns.txt \
        --work_dir ./pattern_clf --n_samples 2000 --base_model Qwen/Qwen3-14B --load_in_4bit

    # zero-shot baseline of the base model on the same test split (no adapter):
    python llm_pattern_classifier.py --patterns_file HI-Small_Patterns.txt --work_dir ./pattern_clf \
        --skip_build --skip_train --eval_zero_shot --load_in_4bit

    # re-run only evaluation with the saved adapter:
    python llm_pattern_classifier.py --patterns_file HI-Small_Patterns.txt --work_dir ./pattern_clf \
        --skip_build --skip_train --load_in_4bit
"""

import argparse
import csv
import gc
import json
import logging
import os
import random
import re
from collections import Counter, defaultdict, deque
from datetime import datetime

from util import logger_setup, set_seed

TIMESTAMP_FORMAT = "%Y/%m/%d %H:%M"  # as in the IBM AML patterns file, e.g. 2022/09/01 00:20

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
PATTERN_LABELS = list(PATTERN_NAME_MAP.values())
UNPARSED = "<unparsed>"

SYSTEM_PROMPT = """You are an expert financial crime investigator. You are shown a subgraph of financial transactions that together form one money-laundering attempt. The data is represented as a graph, where:
- Nodes are accounts, anonymised as A1, A2, ... in order of first appearance. Each node lists its in-degree (number of distinct accounts that sent it money) and out-degree (number of distinct accounts it sent money to) within the subgraph.
- Edges are transfers (transfers_to), listed in chronological order, with amount, currency, payment method and timestamp. One transaction is marked [TARGET EDGE].

Your job is to identify which laundering typology the subgraph exhibits. The possible typologies are:
- fan-out: one account disperses funds to many distinct accounts.
- fan-in: many distinct accounts send funds to a single account.
- gather-scatter: funds converge on an intermediary from many sources, then are redistributed to many destinations.
- scatter-gather: funds fan out from one account to several intermediaries, then are reconsolidated into one account.
- simple cycle: funds travel along a chain of accounts and return to the originating account.
- random: irregular sequence of transfers that matches no clean canonical shape.
- bipartite: dense many-to-many transfers between two groups of accounts.
- stack: a linear chain of intermediary accounts (A to B to C to ...) with layering of hops."""

ANSWER_FORMAT = (
    "\nAnswer Format:\n"
    "- Observed Pattern: (one out of the following: {" + ", ".join(PATTERN_LABELS) + "})\n"
)


# ----------------------------------------------------------------------------
# 1. Patterns file parsing / serialization
# ----------------------------------------------------------------------------

def parse_patterns_file(path):
    """[{"pattern": "fan-out", "rows": [{timestamp, src, dst, amount_received,
    receiving_currency, payment_format, ...}, ...]}, ...]  -- one entry per
    BEGIN/END LAUNDERING ATTEMPT section."""
    fields = [
        "timestamp", "from_bank", "from_account", "to_bank", "to_account",
        "amount_received", "receiving_currency", "amount_paid",
        "payment_currency", "payment_format", "is_laundering",
    ]
    blocks, current = [], None
    with open(path, "r") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n").rstrip("\r")
            if not line:
                continue
            if line.startswith("BEGIN LAUNDERING ATTEMPT"):
                header = line[len("BEGIN LAUNDERING ATTEMPT - "):]
                pattern_raw = header.split(":", 1)[0].strip()
                pattern = PATTERN_NAME_MAP.get(pattern_raw, pattern_raw.lower())
                current = {"pattern": pattern, "rows": []}
            elif line.startswith("END LAUNDERING ATTEMPT"):
                if current is not None and current["rows"]:
                    blocks.append(current)
                current = None
            elif current is not None:
                parsed = next(csv.reader([line]))
                if len(parsed) != len(fields):
                    logging.warning(f"Skipping malformed pattern line: {line}")
                    continue
                row = dict(zip(fields, parsed))
                try:
                    row["amount_received"] = float(row["amount_received"])
                    row["amount_paid"] = float(row["amount_paid"])
                    row["is_laundering"] = int(row["is_laundering"])
                    # keep the original string for the prompt, a real datetime for ordering
                    row["timestamp_str"] = row["timestamp"]
                    row["timestamp"] = datetime.strptime(row["timestamp"], TIMESTAMP_FORMAT)
                except ValueError as e:
                    logging.warning(f"Skipping pattern line with unparseable value ({e}): {line}")
                    continue
                row["src"] = row["from_bank"] + row["from_account"]
                row["dst"] = row["to_bank"] + row["to_account"]
                current["rows"].append(row)

    counts = Counter(b["pattern"] for b in blocks)
    sizes = {p: (min(len(b["rows"]) for b in blocks if b["pattern"] == p),
                 max(len(b["rows"]) for b in blocks if b["pattern"] == p)) for p in counts}
    logging.info(f"Parsed {len(blocks)} blocks from {path}. Blocks per pattern: {dict(counts)}")
    logging.info(f"(min, max) edges per block by pattern: {sizes}")
    return blocks


def serialize_block(block, target_idx, max_edges):
    """Serializes the whole block (same text format as llm_reasoning.serialize_subgraph,
    minus importance scores). If the block has more than `max_edges` edges, keeps the
    `max_edges` chronologically contiguous edges centred on the target edge."""
    rows = block["rows"]
    order = sorted(range(len(rows)), key=lambda i: (rows[i]["timestamp"], i))  # chronological
    if len(order) > max_edges:
        pos = order.index(target_idx)
        start = max(0, min(pos - max_edges // 2, len(order) - max_edges))
        order = order[start:start + max_edges]

    alias, senders, receivers = {}, defaultdict(set), defaultdict(set)
    for i in order:
        r = rows[i]
        for acct in (r["src"], r["dst"]):
            alias.setdefault(acct, f"A{len(alias) + 1}")
        receivers[r["src"]].add(r["dst"])
        senders[r["dst"]].add(r["src"])

    edge_lines = []
    for i in order:
        r = rows[i]
        tag = " [TARGET EDGE]" if i == target_idx else ""
        edge_lines.append(
            f"- {alias[r['src']]} transfers_to {alias[r['dst']]}{tag}\n"
            f"  amount: {r['amount_received']:.2f} (currency: {r['receiving_currency']})\n"
            f"  via: {r['payment_format']}\n"
            f"  timestamp: {r['timestamp_str']}\n"
        )
    node_lines = [f"- {a} (type: Account, in-degree: {len(senders[acct])}, out-degree: {len(receivers[acct])})"
                  for acct, a in alias.items()]
    return "**Nodes:**\n" + "\n".join(node_lines) + "\n**Edges (chronological):**\n" + "\n".join(edge_lines)


def build_prompt(subgraph_text):
    task = (
        "Task: Given the subgraph below (the complete set of transactions of one laundering "
        "attempt, with one transaction marked [TARGET EDGE]), identify the laundering "
        "typology it exhibits. Base your answer on the structure of the subgraph (fan-out/"
        "fan-in, cycles, gather-scatter, stacked chains of intermediaries, ...) and the "
        "transaction values, currencies, payment formats and timing.\n"
    )
    return "\n".join([task, f"\nSubgraph:\n{subgraph_text}\n", ANSWER_FORMAT])


def build_completion(pattern):
    return f"- Observed Pattern: {pattern}\n"


# ----------------------------------------------------------------------------
# 2. Dataset creation (block-level split, uniform edge picking)
# ----------------------------------------------------------------------------

def split_blocks(blocks, train_frac):
    """Stratified (per pattern) split at the BLOCK level, so the same block never ends up
    in both train and test (several edges of one block would give near-identical prompts)."""
    by_pattern = defaultdict(list)
    for bid, b in enumerate(blocks):
        by_pattern[b["pattern"]].append(bid)
    train_ids, test_ids = [], []
    for pattern, ids in by_pattern.items():
        random.shuffle(ids)
        if len(ids) < 2:
            logging.warning(f"Pattern '{pattern}' has only {len(ids)} block -> train only, "
                            f"it will be missing from the test set.")
            train_ids += ids
            continue
        n_test = max(1, round(len(ids) * (1 - train_frac)))
        test_ids += ids[:n_test]
        train_ids += ids[n_test:]
    return train_ids, test_ids


def pick_uniform(blocks, block_ids, quota):
    """Round-robin over patterns -> over blocks within a pattern -> edges within a block
    (same idea as mine_finetune_data.pick_pattern_edges). A block only gets a second edge
    picked after every other block of its pattern has had one. Returns [(block_id, row_idx)]."""
    by_pattern = defaultdict(lambda: defaultdict(list))
    for bid in block_ids:
        b = blocks[bid]
        by_pattern[b["pattern"]][bid] = list(range(len(b["rows"])))

    queues = {}
    for pattern, bmap in by_pattern.items():
        items = [(bid, rows) for bid, rows in bmap.items()]
        for _, rows in items:
            random.shuffle(rows)
        random.shuffle(items)
        queues[pattern] = deque(items)
    active = list(queues)
    random.shuffle(active)

    picked = []
    while active and len(picked) < quota:
        for pattern in list(active):
            if len(picked) >= quota:
                break
            q = queues[pattern]
            bid, rows = q.popleft()
            picked.append((bid, rows.pop()))
            if rows:
                q.append((bid, rows))
            if not q:
                active.remove(pattern)
    return picked


def make_records(blocks, picked, max_edges):
    records = []
    for bid, row_idx in picked:
        b = blocks[bid]
        text = serialize_block(b, row_idx, max_edges)
        records.append({
            "block_id": bid,
            "target_row": row_idx,
            "n_block_edges": len(b["rows"]),
            "pattern": b["pattern"],
            "prompt": build_prompt(text),
            "completion": build_completion(b["pattern"]),
        })
    random.shuffle(records)
    return records


def write_jsonl(records, path):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def build_datasets(args, paths):
    blocks = parse_patterns_file(args.patterns_file)
    train_ids, test_ids = split_blocks(blocks, args.train_frac)
    logging.info(f"Block split: {len(train_ids)} train blocks / {len(test_ids)} test blocks")

    if args.n_samples == -1:
        n_train = n_test = 10 ** 9  # take every edge of every block
    else:
        n_train = int(args.n_samples * args.train_frac)
        n_test = args.n_samples - n_train

    train_picked = pick_uniform(blocks, train_ids, n_train)
    test_picked = pick_uniform(blocks, test_ids, n_test)
    train_recs = make_records(blocks, train_picked, args.max_block_edges)
    test_recs = make_records(blocks, test_picked, args.max_block_edges)

    write_jsonl(train_recs, paths["train"])
    write_jsonl(test_recs, paths["test"])
    for name, recs in (("train", train_recs), ("test", test_recs)):
        logging.info(f"{name}: {len(recs)} samples from {len({r['block_id'] for r in recs})} blocks; "
                     f"per pattern: {dict(Counter(r['pattern'] for r in recs))}")
    logging.info(f"Wrote {paths['train']} and {paths['test']}")


# ----------------------------------------------------------------------------
# 3. Model helpers
# ----------------------------------------------------------------------------

def pick_dtype():
    import torch
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    return (torch.bfloat16 if use_bf16 else torch.float16), use_bf16


def load_tokenizer(name):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_base_model(args, dtype):
    import torch
    from transformers import AutoModelForCausalLM
    kwargs = dict(device_map="auto", attn_implementation="sdpa")
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        )
    else:
        kwargs["torch_dtype"] = dtype
    logging.info(f"Loading base model {args.base_model} ...")
    return AutoModelForCausalLM.from_pretrained(args.base_model, **kwargs)


def render_prompt(tokenizer, user_prompt):
    """Thinking is always disabled (train and eval) so the template is identical in both."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}]
    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False, enable_thinking=False
    )


# ----------------------------------------------------------------------------
# 4. Fine-tuning
# ----------------------------------------------------------------------------

def tokenize_examples(examples, tokenizer, max_seq_len):
    out, n_dropped = [], 0
    for ex in examples:
        prompt_ids = tokenizer(render_prompt(tokenizer, ex["prompt"]), add_special_tokens=False)["input_ids"]
        completion_ids = tokenizer(ex["completion"] + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
        input_ids = prompt_ids + completion_ids
        if len(input_ids) > max_seq_len:
            n_dropped += 1
            continue
        out.append({
            "input_ids": input_ids,
            "labels": [-100] * len(prompt_ids) + completion_ids,  # loss on the answer only
            "attention_mask": [1] * len(input_ids),
        })
    if n_dropped:
        logging.warning(f"Dropped {n_dropped}/{len(examples)} training examples longer than "
                        f"{max_seq_len} tokens (lower --max_block_edges or raise --max_seq_len).")
    return out


def collate_fn(batch, pad_id):
    import torch
    max_len = max(len(b["input_ids"]) for b in batch)
    ids, labels, mask = [], [], []
    for b in batch:
        pad = max_len - len(b["input_ids"])
        ids.append([pad_id] * pad + b["input_ids"])
        labels.append([-100] * pad + b["labels"])
        mask.append([0] * pad + b["attention_mask"])
    return {"input_ids": torch.tensor(ids), "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(mask)}


def train(args, paths):
    import torch
    from torch.utils.data import Dataset
    from transformers import TrainingArguments, Trainer
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    dtype, use_bf16 = pick_dtype()
    tokenizer = load_tokenizer(args.base_model)

    examples = read_jsonl(paths["train"])
    data = tokenize_examples(examples, tokenizer, args.max_seq_len)
    logging.info(f"{len(data)} tokenized training examples "
                 f"(mean length {sum(len(d['input_ids']) for d in data) / max(1, len(data)):.0f} tokens)")

    class ListDataset(Dataset):
        def __len__(self): return len(data)
        def __getitem__(self, i): return data[i]

    model = load_base_model(args, dtype)
    model.config.use_cache = False
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.enable_input_require_grads()  # needed for gradient checkpointing + LoRA

    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules, bias="none", task_type="CAUSAL_LM",
    ))
    if not use_bf16:  # fp16 AMP needs fp32 trainable params
        for p in model.parameters():
            if p.requires_grad:
                p.data = p.data.float()
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=os.path.join(args.work_dir, "trainer_out"),
        num_train_epochs=args.n_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_strategy="no",           # only the final adapter is kept
        eval_strategy="no",           # test split is never touched during training
        bf16=use_bf16, fp16=not use_bf16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to=[],
        seed=args.seed,
    )
    trainer = Trainer(
        model=model, args=training_args, train_dataset=ListDataset(),
        data_collator=lambda b: collate_fn(b, tokenizer.pad_token_id),
    )
    logging.info("Starting fine-tuning ...")
    trainer.train()

    logging.info(f"Saving LoRA adapter to {paths['adapter']}")
    model.save_pretrained(paths["adapter"])
    tokenizer.save_pretrained(paths["adapter"])

    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()


# ----------------------------------------------------------------------------
# 5. Evaluation
# ----------------------------------------------------------------------------

def parse_pattern(text):
    """Maps the model's free text to one of PATTERN_LABELS (or UNPARSED)."""
    m = re.search(r"Observed Pattern\s*:\s*(.+)", text, re.IGNORECASE)
    cand = (m.group(1) if m else text).strip().lower()
    cand = re.sub(r"[^a-z\- ]", "", cand).strip()
    if cand in PATTERN_LABELS:
        return cand
    for lab in sorted(PATTERN_LABELS, key=len, reverse=True):  # tolerate e.g. "cycle", "fan out."
        if lab in cand or lab.replace("-", " ") in cand:
            return lab
    if "cycle" in cand:
        return "simple cycle"
    return UNPARSED


def evaluate(args, paths):
    import torch
    import pandas as pd
    from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

    dtype, _ = pick_dtype()
    tokenizer = load_tokenizer(args.base_model)
    tokenizer.padding_side = "left"

    model = load_base_model(args, dtype)
    if args.eval_zero_shot:
        logging.info("--eval_zero_shot: evaluating the base model WITHOUT the adapter.")
    else:
        from peft import PeftModel
        logging.info(f"Loading LoRA adapter from {paths['adapter']}")
        model = PeftModel.from_pretrained(model, paths["adapter"])
        if not args.load_in_4bit:
            model = model.merge_and_unload()
    model.eval()

    test = read_jsonl(paths["test"])
    if args.max_eval_samples > 0:
        test = test[:args.max_eval_samples]
    logging.info(f"Evaluating on {len(test)} test samples")

    rows = []
    for i in range(0, len(test), args.eval_batch_size):
        batch = test[i:i + args.eval_batch_size]
        texts = [render_prompt(tokenizer, ex["prompt"]) for ex in batch]
        enc = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                temperature=None, top_p=None, top_k=None, pad_token_id=tokenizer.pad_token_id,
            )
        decoded = tokenizer.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for ex, raw in zip(batch, decoded):
            rows.append({
                "block_id": ex["block_id"], "target_row": ex["target_row"],
                "n_block_edges": ex["n_block_edges"],
                "actual_pattern": ex["pattern"], "llm_pattern": parse_pattern(raw), "llm_raw": raw.strip(),
            })
        logging.info(f"[{min(i + args.eval_batch_size, len(test))}/{len(test)}] done")

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


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

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
    logger_setup()
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
        train(args, paths)
    if not args.skip_eval:
        evaluate(args, paths)


if __name__ == "__main__":
    main()