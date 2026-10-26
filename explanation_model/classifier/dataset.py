"""Training / test data for pattern classifiers, built from the IBM AML *_Patterns.txt file.

Classifier-agnostic: produces `PatternSample`s (a `Subgraph` + its pattern label). A concrete
classifier (e.g. llm/) turns them into whatever format it trains on.

  * every "laundering attempt" block of the patterns file is one subgraph (the entire block, no GNN
    sampling); its label is the block's pattern
  * blocks are split train/test per pattern BEFORE edges are picked, so no block is in both splits
  * target edges are picked uniformly over patterns -> blocks -> edges
"""
import csv
import logging
import random
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from datetime import datetime

from explanation_model.patterns import PATTERN_NAME_MAP
from explanation_model.subgraph import Subgraph

TIMESTAMP_FORMAT = "%Y/%m/%d %H:%M"  # as in the IBM AML patterns file, e.g. 2022/09/01 00:20


@dataclass
class PatternSample:
    block_id: int
    target_row: int        # row inside the block that is marked [TARGET EDGE]
    n_block_edges: int
    pattern: str
    subgraph: Subgraph


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
    """Round-robin over patterns -> over blocks within a pattern -> edges within a block.
    A block only gets a second edge picked after every other block of its pattern has had one.
    Returns [(block_id, row_idx)]."""
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


def build_samples(blocks, picked, max_edges):
    samples = []
    for bid, row_idx in picked:
        b = blocks[bid]
        samples.append(PatternSample(
            block_id=bid, target_row=row_idx, n_block_edges=len(b["rows"]), pattern=b["pattern"],
            subgraph=Subgraph.from_block_rows(b["rows"], row_idx, max_edges),
        ))
    random.shuffle(samples)
    return samples


def build_splits(blocks, n_samples, train_frac, max_edges):
    """-> (train_samples, test_samples). n_samples = total over both splits; -1 = every edge of
    every block. Uses the global `random` state, so seed it first for reproducible splits."""
    train_ids, test_ids = split_blocks(blocks, train_frac)
    logging.info(f"Block split: {len(train_ids)} train blocks / {len(test_ids)} test blocks")

    if n_samples == -1:
        n_train = n_test = 10 ** 9  # take every edge of every block
    else:
        n_train = int(n_samples * train_frac)
        n_test = n_samples - n_train

    train_picked = pick_uniform(blocks, train_ids, n_train)
    test_picked = pick_uniform(blocks, test_ids, n_test)
    return build_samples(blocks, train_picked, max_edges), build_samples(blocks, test_picked, max_edges)
