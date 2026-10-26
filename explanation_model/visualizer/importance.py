"""Edge importance of a sampled transaction subgraph.

GNNExplainer for homogeneous models, CaptumExplainer (IntegratedGradients) for heterogeneous /
reverse-message-passing models (PyG's GNNExplainer doesn't reliably support heterogeneous link-level
models, see https://github.com/pyg-team/pytorch_geometric/discussions/7963; needs `pip install captum`).
"""
import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch_geometric.explain import CaptumExplainer, Explainer, GNNExplainer

from prediction_model.predictor import REV_REL, TO_REL

_MODEL_CONFIG = dict(mode='multiclass_classification', task_level='graph', return_type='raw')


@dataclass
class EdgeImportance:
    imp_lookup: Dict[Tuple[int, int], float]  # (src, dst) in batch-local node ids, original transaction direction -> importance
    imp_by_id: Dict[int, float]               # global transaction id -> importance ('to' relation)


class _SeedLogits(torch.nn.Module):
    """Returns only the seed edge's logits, so the explainer can use task_level='graph'."""

    def __init__(self, base_model, seed_pos, relation=None):
        super().__init__()
        self.base_model = base_model
        self.seed_pos = seed_pos
        self.relation = relation

    def forward(self, x, edge_index, edge_attr, **kwargs):
        out = self.base_model(x, edge_index, edge_attr)
        if self.relation is not None:
            out = out[self.relation]
        return out[self.seed_pos:self.seed_pos + 1]


def _max_merge(lookup, key, value):
    lookup[key] = max(lookup.get(key, 0.0), float(value))


def compute_edge_importance(model, sample, is_hetero) -> EdgeImportance:
    """Runs the explainer on a sample returned by prediction_model.predictor.sample_and_predict."""
    batch = sample['batch']
    seed_pos = sample['seed_pos']
    imp_lookup, imp_by_id = {}, {}

    if not is_hetero:
        device = batch.x.device
        wrapper = _SeedLogits(model, seed_pos).to(device).eval()
        explainer = Explainer(
            model=wrapper, algorithm=GNNExplainer(epochs=100), explanation_type='model',
            node_mask_type='attributes', edge_mask_type='object', model_config=_MODEL_CONFIG,
        )
        try:
            explanation = explainer(x=batch.x, edge_index=batch.edge_index, edge_attr=batch.edge_attr)
        finally:
            model.eval()

        logging.info("Edge mask:\n%s", explanation.edge_mask.detach().cpu().numpy())
        logging.info("Node feature mask:\n%s", explanation.node_mask.detach().cpu().numpy())

        edge_index_np = batch.edge_index.cpu().numpy()
        edge_imp = explanation.edge_mask.cpu().detach().numpy()
        for j in range(edge_index_np.shape[1]):
            _max_merge(imp_lookup, (int(edge_index_np[0, j]), int(edge_index_np[1, j])), edge_imp[j])
        for gid, imp in zip(sample['global_edge_ids'], edge_imp.tolist()):
            _max_merge(imp_by_id, int(gid), imp)

    else:
        device = batch['node'].x.device
        wrapper = _SeedLogits(model, seed_pos, relation=TO_REL).to(device).eval()
        explainer = Explainer(
            model=wrapper, algorithm=CaptumExplainer('IntegratedGradients', internal_batch_size=1),
            explanation_type='model', node_mask_type='attributes', edge_mask_type='object',
            model_config=_MODEL_CONFIG,
        )
        try:
            explanation = explainer(x=batch.x_dict, edge_index=batch.edge_index_dict, edge_attr=batch.edge_attr_dict)
        finally:
            model.eval()

        logging.info("Node feature mask (per node type):")
        for ntype, node_mask in explanation.node_mask_dict.items():
            logging.info(f"  {ntype}: {node_mask.detach().cpu().numpy()}")
        logging.info("Edge mask (per canonical edge type):")
        for etype, edge_mask in explanation.edge_mask_dict.items():
            logging.info(f"  {etype}: {edge_mask.detach().cpu().numpy()}")

        to_edge_index_np = batch[TO_REL].edge_index.cpu().numpy()
        rev_edge_index_np = batch[REV_REL].edge_index.cpu().numpy()
        to_imp = explanation.edge_mask_dict[TO_REL].cpu().detach().numpy()
        rev_imp = explanation.edge_mask_dict[REV_REL].cpu().detach().numpy()

        # merge both relations into one edge set keyed by the original transaction direction:
        # rev_to edge (a, b) <=> to edge (b, a); keep the larger importance if an edge is in both.
        for j in range(to_edge_index_np.shape[1]):
            _max_merge(imp_lookup, (int(to_edge_index_np[0, j]), int(to_edge_index_np[1, j])), to_imp[j])
        for j in range(rev_edge_index_np.shape[1]):
            _max_merge(imp_lookup, (int(rev_edge_index_np[1, j]), int(rev_edge_index_np[0, j])), rev_imp[j])
        for gid, imp in zip(sample['global_edge_ids'], to_imp.tolist()):
            _max_merge(imp_by_id, int(gid), imp)

    return EdgeImportance(imp_lookup=imp_lookup, imp_by_id=imp_by_id)
