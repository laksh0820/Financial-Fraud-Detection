"""Framework-neutral transaction subgraph + its text serialization.

A `Subgraph` is an ordered list of transfers with one of them marked as the *target* transaction.
It is what the classifier and the explainer consume, regardless of where it came from
(a laundering-attempt block of the patterns file for training, or the GNN-sampled neighborhood of a
transaction at detection time).
"""
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional


@dataclass
class Edge:
    src: str                       # sender account (bank + account id)
    dst: str                       # receiver account
    amount: float                  # amount received
    currency: str
    payment_format: str
    timestamp: datetime            # used for chronological ordering
    timestamp_str: str             # as printed in the prompt, e.g. "2022/09/01 00:20"
    edge_id: Optional[int] = None  # global transaction id (row in formatted_transactions.csv), if known
    importance: Optional[float] = None  # GNN-explainer importance, if computed (not serialized)


@dataclass
class Subgraph:
    edges: List[Edge]              # chronological
    target_idx: int                # index into `edges` of the transaction under review

    @property
    def target(self) -> Edge:
        return self.edges[self.target_idx]

    # ---- constructors -------------------------------------------------------------------------

    @classmethod
    def from_block_rows(cls, rows, target_idx, max_edges):
        """Builds a subgraph from one laundering-attempt block of the patterns file (list of parsed
        row dicts). If the block has more than `max_edges` edges, keeps the `max_edges`
        chronologically contiguous edges centred on the target row."""
        order = sorted(range(len(rows)), key=lambda i: (rows[i]["timestamp"], i))  # chronological
        if len(order) > max_edges:
            pos = order.index(target_idx)
            start = max(0, min(pos - max_edges // 2, len(order) - max_edges))
            order = order[start:start + max_edges]
        edges = [
            Edge(src=rows[i]["src"], dst=rows[i]["dst"], amount=rows[i]["amount_received"],
                 currency=rows[i]["receiving_currency"], payment_format=rows[i]["payment_format"],
                 timestamp=rows[i]["timestamp"], timestamp_str=rows[i]["timestamp_str"])
            for i in order
        ]
        return cls(edges, order.index(target_idx))

    @classmethod
    def from_edges(cls, edges: List[Edge], target_edge_id: int):
        """Builds a subgraph from edges that carry global `edge_id`s (e.g. a GNN-sampled neighborhood)."""
        ordered = sorted(edges, key=lambda e: (e.timestamp, -1 if e.edge_id is None else e.edge_id))
        for i, e in enumerate(ordered):
            if e.edge_id == target_edge_id:
                return cls(ordered, i)
        raise ValueError(f"target edge {target_edge_id} is not among the given edges")

    # ---- structure helpers (also handy for future subgraph-specific explainers) ----------------

    def node_table(self):
        """[(alias, account, in_degree, out_degree)] in order of first appearance. Aliases are
        A1, A2, ...; degrees count distinct counterparties within the subgraph."""
        alias, senders, receivers = {}, defaultdict(set), defaultdict(set)
        for e in self.edges:
            for acct in (e.src, e.dst):
                alias.setdefault(acct, f"A{len(alias) + 1}")
            receivers[e.src].add(e.dst)
            senders[e.dst].add(e.src)
        return [(a, acct, len(senders[acct]), len(receivers[acct])) for acct, a in alias.items()]

    # ---- serialization ------------------------------------------------------------------------

    def serialize(self) -> str:
        """Text used in the classifier prompt. NOTE: the LLM adapter was fine-tuned on exactly this
        format -- changing it requires re-building the dataset and re-training."""
        table = self.node_table()
        alias = {acct: a for a, acct, _, _ in table}
        edge_lines = []
        for i, e in enumerate(self.edges):
            tag = " [TARGET EDGE]" if i == self.target_idx else ""
            edge_lines.append(
                f"- {alias[e.src]} transfers_to {alias[e.dst]}{tag}\n"
                f"  amount: {e.amount:.2f} (currency: {e.currency})\n"
                f"  via: {e.payment_format}\n"
                f"  timestamp: {e.timestamp_str}\n"
            )
        node_lines = [f"- {a} (type: Account, in-degree: {i}, out-degree: {o})" for a, _, i, o in table]
        return "**Nodes:**\n" + "\n".join(node_lines) + "\n**Edges (chronological):**\n" + "\n".join(edge_lines)
