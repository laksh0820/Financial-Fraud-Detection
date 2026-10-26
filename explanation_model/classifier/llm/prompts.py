"""Prompt / answer format of the LLM pattern classifier.

NOTE: the fine-tuned adapter was trained on exactly these strings (system prompt, task text, answer
format). If you change anything here, rebuild the dataset and re-train.
"""
import re

from explanation_model.patterns import PATTERN_LABELS, UNPARSED

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



def render_prompt(tokenizer, user_prompt):
    """Thinking is always disabled (train and eval) so the template is identical in both."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}]
    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False, enable_thinking=False
    )
