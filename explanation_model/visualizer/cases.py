"""Finding one example of each confusion-matrix case (TP / FP / FN / TN) on the test set."""
import logging

import torch
from torch_geometric.loader import LinkNeighborLoader


# (predicted_class, actual_class) for each of the 4 confusion-matrix cases
CASE_DEFINITIONS = {
    'TP': (1, 1),
    'FP': (1, 0),
    'FN': (0, 1),
    'TN': (0, 0),
}

CASE_LABELS = {
    'TP': 'Predicted Fraud & Actually Fraud',
    'FP': 'Predicted Fraud & Actually Non-Fraud',
    'FN': 'Predicted Non-Fraud & Actually Fraud',
    'TN': 'Predicted Non-Fraud & Actually Non-Fraud',
}


@torch.no_grad()
def find_case_examples(inds, wanted_case_keys, te_data, model, device, args, transform, is_hetero):
    """Batched scan (same masking trick as evaluate_homo/evaluate_hetero in
    train_util.py) that finds one seed-edge ID per case in
    `wanted_case_keys`. Much faster than predicting edge-by-edge since each
    batch is a single GPU forward pass. Returns {case_key: edge_idx}.
    """
    if len(inds) == 0:
        return {}

    to_rel = ('node', 'to', 'node')
    loader_kwargs = dict(num_neighbors=args.num_neighs, batch_size=args.batch_size,
                          shuffle=False, transform=transform)
    if is_hetero:
        loader = LinkNeighborLoader(te_data, edge_label_index=(to_rel, te_data[to_rel].edge_index[:, inds]),
                                     edge_label=te_data[to_rel].y[inds], **loader_kwargs)
    else:
        loader = LinkNeighborLoader(te_data, edge_label_index=te_data.edge_index[:, inds],
                                     edge_label=te_data.y[inds], **loader_kwargs)

    inds_cpu = inds.detach().cpu()
    found = {}
    for batch in loader:
        if is_hetero:
            batch_edge_inds = inds_cpu[batch[to_rel].input_id.detach().cpu()]
            batch_edge_ids = loader.data[to_rel].edge_attr.detach().cpu()[batch_edge_inds, 0]
            mask = torch.isin(batch[to_rel].edge_attr[:, 0].detach().cpu(), batch_edge_ids)
            ids_masked = batch[to_rel].edge_attr[:, 0][mask]
            actual_masked = batch[to_rel].y[mask]
            batch[to_rel].edge_attr = batch[to_rel].edge_attr[:, 1:]
            batch['node', 'rev_to', 'node'].edge_attr = batch['node', 'rev_to', 'node'].edge_attr[:, 1:]
            batch = batch.to(device)
            out = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)[to_rel]
        else:
            batch_edge_inds = inds_cpu[batch.input_id.detach().cpu()]
            batch_edge_ids = loader.data.edge_attr.detach().cpu()[batch_edge_inds, 0]
            mask = torch.isin(batch.edge_attr[:, 0].detach().cpu(), batch_edge_ids)
            ids_masked = batch.edge_attr[:, 0][mask]
            actual_masked = batch.y[mask]
            batch.edge_attr = batch.edge_attr[:, 1:]
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr)

        preds_masked = out[mask.to(device)].argmax(dim=-1).cpu()

        for eid, pred, actual in zip(ids_masked.int().tolist(), preds_masked.tolist(), actual_masked.int().tolist()):
            for case_key in wanted_case_keys:
                if case_key in found:
                    continue
                if (pred, actual) == CASE_DEFINITIONS[case_key]:
                    found[case_key] = eid
                    logging.info(f"Found example for case '{CASE_LABELS[case_key]}': edge {eid}")

        if len(found) == len(wanted_case_keys):
            break

    return found
