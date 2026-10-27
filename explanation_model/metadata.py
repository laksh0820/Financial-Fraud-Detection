"""Human-readable transaction metadata for the GNN's integer-encoded data.

The GNN trains on z-normalized features and integer-encoded accounts / currencies / payment formats.
To show a transaction to the classifier we need the original values back. `format_kaggle_files.py`
writes the encoding maps next to formatted_transactions.csv:

    currency_map.json, payment_format_map.json, account_map.json, format_meta.json (firstTs)

Global transaction ids are row positions in formatted_transactions.csv (the same ids that
`add_arange_ids` injects and `TransactionPredictor` returns in `sample['global_edge_ids']`).
"""
import json
import logging
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

from common.config import dataset_dir
from explanation_model.subgraph import Edge, Subgraph, select_edge_ids

TIMESTAMP_OUT_FORMAT = "%Y/%m/%d %H:%M"
_CSV_COLUMNS = ['from_id', 'to_id', 'Timestamp', 'Amount Received', 'Received Currency',
                'Payment Format', 'Is Laundering']


class EdgeMetadataStore:
    """Lazy lookup {global edge id -> Edge}. The CSV is only read the first time it is needed, so
    pipelines that never see a predicted-fraud transaction pay nothing."""

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self._cols = None

    @classmethod
    def from_data_config(cls, data_config, dataset):
        return cls(dataset_dir(data_config, dataset))

    # ---- loading ------------------------------------------------------------------------------

    def _load_map(self, filename):
        path = os.path.join(self.data_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found. Re-run prediction_model/preprocessing/format_kaggle_files.py "
                f"on the raw dataset to create the encoding maps.")
        with open(path) as f:
            return json.load(f)

    def _ensure_loaded(self):
        if self._cols is not None:
            return
        import pandas as pd
        logging.info(f"Loading transaction metadata from {self.data_dir} ...")
        self.currency_map = {int(k): v for k, v in self._load_map("currency_map.json").items()}
        self.payment_map = {int(k): v for k, v in self._load_map("payment_format_map.json").items()}
        self.account_map = {int(k): v for k, v in self._load_map("account_map.json").items()}
        self.first_ts = self._load_map("format_meta.json")["firstTs"]
        df = pd.read_csv(os.path.join(self.data_dir, "formatted_transactions.csv"), usecols=_CSV_COLUMNS)
        self._cols = {c: df[c].to_numpy() for c in _CSV_COLUMNS}
        self._n = len(df)

    def __len__(self):
        self._ensure_loaded()
        return self._n

    # ---- lookup -------------------------------------------------------------------------------

    def get_edge(self, edge_id: int, importance: Optional[float] = None) -> Edge:
        self._ensure_loaded()
        c = self._cols
        src_id, dst_id = int(c['from_id'][edge_id]), int(c['to_id'][edge_id])
        ts = datetime.fromtimestamp(self.first_ts + int(c['Timestamp'][edge_id]), tz=timezone.utc)
        cur_id, fmt_id = int(c['Received Currency'][edge_id]), int(c['Payment Format'][edge_id])
        return Edge(
            src=f"{self.account_map.get(src_id, src_id)}",
            dst=f"{self.account_map.get(dst_id, dst_id)}",
            amount=float(c['Amount Received'][edge_id]),
            currency=self.currency_map.get(cur_id, str(cur_id)),
            payment_format=self.payment_map.get(fmt_id, str(fmt_id)),
            timestamp=ts.replace(tzinfo=None),
            timestamp_str=ts.strftime(TIMESTAMP_OUT_FORMAT),
            edge_id=int(edge_id),
            importance=importance,
        )

    def build_subgraph(self, global_edge_ids: List[int], target_edge_id: int,
                       imp_by_id: Optional[Dict[int, float]] = None, max_edges: int = 40) -> Subgraph:
        """Subgraph of (at most `max_edges`) transactions around `target_edge_id`. When the sampled
        neighborhood is larger, the most important edges (per the GNN explainer) are kept, plus
        the target. Ordered chronologically."""
        self._ensure_loaded()
        valid = (e for e in global_edge_ids if 0 <= int(e) < self._n)
        ids = select_edge_ids(valid, target_edge_id, imp_by_id, max_edges)  # shared with the plot
        imp_by_id = imp_by_id or {}
        edges = [self.get_edge(e, imp_by_id.get(e)) for e in ids]
        return Subgraph.from_edges(edges, int(target_edge_id))
