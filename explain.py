import torch
import logging
import os
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import to_hetero
from torch_geometric.explain import Explainer, GNNExplainer, CaptumExplainer
from train_util import extract_param, AddEgoIds, get_loaders, add_arange_ids
from training import get_model


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


def _sample_and_predict(edge_idx, te_data, model, device, args, transform=None):
    """Sample the k-hop subgraph around `edge_idx` and return the model's
    prediction for it, along with everything needed to run GNNExplainer on
    it later without resampling.

    `transform` must match the one used to build the loaders that sized the
    model (e.g. AddEgoIds() if args.ego), otherwise batch.x's feature dim
    won't match what node_emb expects.
    """
    loader = LinkNeighborLoader(
        te_data,
        num_neighbors=args.num_neighs,
        edge_label_index=te_data.edge_index[:, edge_idx:edge_idx + 1],
        edge_label=te_data.y[edge_idx:edge_idx + 1],
        batch_size=1,
        shuffle=False,
        transform=transform,
    )
    batch = next(iter(loader))
    batch = batch.to(device)

    # locate the seed edge via its global ID (first column of edge_attr)
    mask = (batch.edge_attr[:, 0] == edge_idx)
    seed_positions = mask.nonzero(as_tuple=False)
    if seed_positions.numel() != 1:
        raise RuntimeError(f"Expected exactly one seed edge, found {seed_positions.numel()}")
    seed_pos = seed_positions.item()

    # drop the ID column now that the seed edge has been located
    batch.edge_attr = batch.edge_attr[:, 1:]

    seed_u, seed_v = batch.edge_index[0, seed_pos].item(), batch.edge_index[1, seed_pos].item()

    with torch.no_grad():
        out = model(batch.x, batch.edge_index, batch.edge_attr)
        pred = out[seed_pos:seed_pos + 1].argmax(dim=-1).item()

    actual = int(te_data.y[edge_idx].item())

    return {
        'batch': batch,
        'seed_pos': seed_pos,
        'seed_u': seed_u,
        'seed_v': seed_v,
        'pred': pred,
        'actual': actual,
    }


def _sample_and_predict_hetero(edge_idx, te_data, model, device, args, transform=None):
    """Heterogeneous version of `_sample_and_predict`. Our HeteroData has one
    node type ('node') and two edge types: ('node','to','node') for the
    original transaction direction and ('node','rev_to','node') for its
    reverse. `edge_idx` indexes the ('node','to','node') relation.
    """
    to_rel = ('node', 'to', 'node')

    loader = LinkNeighborLoader(
        te_data,
        num_neighbors=args.num_neighs,
        edge_label_index=(to_rel, te_data[to_rel].edge_index[:, edge_idx:edge_idx + 1]),
        edge_label=te_data[to_rel].y[edge_idx:edge_idx + 1],
        batch_size=1,
        shuffle=False,
        transform=transform,
    )
    batch = next(iter(loader))
    batch = batch.to(device)

    # locate the seed edge via its global ID (first column of edge_attr)
    mask = (batch[to_rel].edge_attr[:, 0] == edge_idx)
    seed_positions = mask.nonzero(as_tuple=False)
    if seed_positions.numel() != 1:
        raise RuntimeError(f"Expected exactly one seed edge, found {seed_positions.numel()}")
    seed_pos = seed_positions.item()

    # drop the ID column from both relations now that the seed edge is located
    batch[to_rel].edge_attr = batch[to_rel].edge_attr[:, 1:]
    batch['node', 'rev_to', 'node'].edge_attr = batch['node', 'rev_to', 'node'].edge_attr[:, 1:]

    seed_u = batch[to_rel].edge_index[0, seed_pos].item()
    seed_v = batch[to_rel].edge_index[1, seed_pos].item()

    with torch.no_grad():
        out = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)
        out_to = out[to_rel]
        pred = out_to[seed_pos:seed_pos + 1].argmax(dim=-1).item()

    actual = int(te_data[to_rel].y[edge_idx].item())

    return {
        'batch': batch,
        'seed_pos': seed_pos,
        'seed_u': seed_u,
        'seed_v': seed_v,
        'pred': pred,
        'actual': actual,
    }


@torch.no_grad()
def _find_case_examples(inds, wanted_case_keys, te_data, model, device, args, transform, is_hetero):
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


def _focus_subgraph_and_layout(imp_lookup, seed_u, seed_v, args):
    """Trim a full {(u, v): importance} edge dict down to the neighborhood
    around the seed edge, so the seed edge stays visually central instead of
    buried in the full sampled subgraph (which can have 100+ nodes).

    `args.explain_focus_hops` (default 2) controls neighborhood size;
    `args.explain_max_nodes` (default 40) caps it further by keeping the
    seed edge plus the highest-importance incident edges. Returns
    (focused DiGraph, layout positions).
    """
    hops = getattr(args, 'explain_focus_hops', 2)
    max_nodes = getattr(args, 'explain_max_nodes', 40)

    g_full = nx.DiGraph()
    for (u, v), imp in imp_lookup.items():
        g_full.add_edge(u, v, importance=imp)

    ug = g_full.to_undirected()
    close_nodes = {seed_u, seed_v}
    for seed in (seed_u, seed_v):
        if seed in ug:
            close_nodes.update(nx.single_source_shortest_path_length(ug, seed, cutoff=hops))

    g = g_full.subgraph(close_nodes).copy()

    if g.number_of_nodes() > max_nodes:
        ranked = sorted(g.edges(data='importance'), key=lambda e: e[2] or 0.0, reverse=True)
        keep_nodes = {seed_u, seed_v}
        for u, v, _ in ranked:
            if len(keep_nodes) >= max_nodes:
                break
            keep_nodes.update([u, v])
        g = g.subgraph(keep_nodes).copy()

    k = 1.5 / (len(g) ** 0.5) if len(g) > 1 else 0.5
    pos = nx.spring_layout(g, seed=42, k=k, iterations=200)
    return g, pos


def _explain_and_plot(model, sample, edge_idx, args, case_key, case_label):
    """Run GNNExplainer on an already-sampled subgraph and save the plot."""
    device = sample['batch'].x.device
    batch = sample['batch']
    seed_pos = sample['seed_pos']
    seed_u, seed_v = sample['seed_u'], sample['seed_v']
    target = sample['pred']

    # returns only the seed edge's logits, so we can use task_level='graph'
    class ExplainWrapper(torch.nn.Module):
        def __init__(self, base_model, seed_pos):
            super().__init__()
            self.base_model = base_model
            self.seed_pos = seed_pos

        def forward(self, x, edge_index, edge_attr, **kwargs):
            out = self.base_model(x, edge_index, edge_attr)
            return out[self.seed_pos:self.seed_pos + 1]

    wrapper = ExplainWrapper(model, seed_pos).to(device)

    explainer = Explainer(
        model=wrapper,
        algorithm=GNNExplainer(epochs=100),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(
            mode='multiclass_classification',
            task_level='graph',
            return_type='raw',
        ),
    )

    explanation = explainer(
        x=batch.x,
        edge_index=batch.edge_index,
        edge_attr=batch.edge_attr,
    )

    logging.info(f"[{case_key}] Edge mask:")
    logging.info(explanation.edge_mask.detach().cpu().numpy())
    logging.info(f"[{case_key}] Node feature mask:")
    logging.info(explanation.node_mask.detach().cpu().numpy())

    # ---------- visualization ----------
    edge_index_np = batch.edge_index.cpu().numpy()
    edge_imp = explanation.edge_mask.cpu().detach().numpy()
    imp_lookup = {}
    for j in range(edge_index_np.shape[1]):
        key = (int(edge_index_np[0, j]), int(edge_index_np[1, j]))
        imp_lookup[key] = max(imp_lookup.get(key, 0.0), float(edge_imp[j]))

    g, pos = _focus_subgraph_and_layout(imp_lookup, seed_u, seed_v, args)
    plt.figure(figsize=(12, 10))
    nx.draw_networkx_nodes(g, pos, node_color='lightblue', node_size=600)

    edges = list(g.edges())
    edge_colors = [g[u][v]['importance'] for u, v in edges]

    nx.draw_networkx_edges(
        g, pos, edgelist=edges, edge_color=edge_colors,
        edge_cmap=plt.cm.Reds, edge_vmin=0, edge_vmax=1, width=2,
        arrows=True, arrowstyle='->', arrowsize=15
    )

    # highlight the seed edge
    if (seed_u, seed_v) in g.edges():
        nx.draw_networkx_edges(
            g, pos, edgelist=[(seed_u, seed_v)],
            edge_color='red', width=5, arrows=True, arrowstyle='->', arrowsize=20
        )

    nx.draw_networkx_labels(g, pos, font_size=8)
    plt.title(
        f"Explanation for edge {edge_idx} (seed in red)\n"
        f"{g.number_of_nodes()}-node neighborhood (~{getattr(args, 'explain_focus_hops', 2)}-hop)\n"
        f"Case: {case_label}\n"
        f"Predicted class: {'Fraud' if target == 1 else 'Legitimate'}"
    )

    sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    plt.colorbar(sm, ax=plt.gca(), label='Edge Importance')
    plt.tight_layout()

    out_dir = args.explain_plot_dir
    os.makedirs(out_dir, exist_ok=True)
    plot_path = f"{out_dir}/explanation_{case_key}_{edge_idx}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"[{case_key}] Explanation plot saved to {plot_path}")


def _explain_and_plot_hetero(model, sample, edge_idx, args, case_key, case_label):
    """Heterogeneous version of `_explain_and_plot`.

    Uses CaptumExplainer instead of GNNExplainer: PyG's GNNExplainer doesn't
    reliably support heterogeneous link-level models (see
    https://github.com/pyg-team/pytorch_geometric/discussions/7963).
    Requires `captum` (`pip install captum`).

    Since each transaction exists as both a ('node','to','node') edge and a
    mirrored ('node','rev_to','node') edge, the sampled subgraph naturally
    includes a seed node's outgoing edges as well as incoming ones. We merge
    both relations back into one directed picture for the plot.
    """
    to_rel = ('node', 'to', 'node')
    rev_rel = ('node', 'rev_to', 'node')

    device = sample['batch']['node'].x.device
    batch = sample['batch']
    seed_pos = sample['seed_pos']
    seed_u, seed_v = sample['seed_u'], sample['seed_v']
    target = sample['pred']

    class ExplainWrapperHetero(torch.nn.Module):
        def __init__(self, base_model, seed_pos):
            super().__init__()
            self.base_model = base_model
            self.seed_pos = seed_pos

        def forward(self, x, edge_index, edge_attr, **kwargs):
            out = self.base_model(x, edge_index, edge_attr)
            out_to = out[to_rel]
            return out_to[self.seed_pos:self.seed_pos + 1]

    wrapper = ExplainWrapperHetero(model, seed_pos).to(device)

    explainer = Explainer(
        model=wrapper,
        algorithm=CaptumExplainer('IntegratedGradients'),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(
            mode='multiclass_classification',
            task_level='graph',
            return_type='raw',
        ),
    )

    explanation = explainer(
        x=batch.x_dict,
        edge_index=batch.edge_index_dict,
        edge_attr=batch.edge_attr_dict,
    )

    logging.info(f"[{case_key}] Node feature mask (per node type):")
    for ntype, node_mask in explanation.node_mask_dict.items():
        logging.info(f"  {ntype}: {node_mask.detach().cpu().numpy()}")
    logging.info(f"[{case_key}] Edge mask (per canonical edge type):")
    for etype, edge_mask in explanation.edge_mask_dict.items():
        logging.info(f"  {etype}: {edge_mask.detach().cpu().numpy()}")

    # ---------- visualization ----------
    to_edge_index_np = batch[to_rel].edge_index.cpu().numpy()
    rev_edge_index_np = batch[rev_rel].edge_index.cpu().numpy()
    to_imp = explanation.edge_mask_dict[to_rel].cpu().detach().numpy()
    rev_imp = explanation.edge_mask_dict[rev_rel].cpu().detach().numpy()

    # merge both relations into one edge set, keyed by the original
    # transaction direction: rev_to edge (a,b) <=> to edge (b,a).
    # keep the larger importance if an edge appears in both.
    imp_lookup = {}
    for j in range(to_edge_index_np.shape[1]):
        key = (int(to_edge_index_np[0, j]), int(to_edge_index_np[1, j]))
        imp_lookup[key] = max(imp_lookup.get(key, 0.0), float(to_imp[j]))
    for j in range(rev_edge_index_np.shape[1]):
        a, b = int(rev_edge_index_np[0, j]), int(rev_edge_index_np[1, j])
        key = (b, a)
        imp_lookup[key] = max(imp_lookup.get(key, 0.0), float(rev_imp[j]))

    g, pos = _focus_subgraph_and_layout(imp_lookup, seed_u, seed_v, args)
    plt.figure(figsize=(12, 10))
    nx.draw_networkx_nodes(g, pos, node_color='lightblue', node_size=600)

    edges = list(g.edges())
    edge_colors = [g[u][v]['importance'] for u, v in edges]

    nx.draw_networkx_edges(
        g, pos, edgelist=edges, edge_color=edge_colors,
        edge_cmap=plt.cm.Reds, edge_vmin=0, edge_vmax=1, width=2,
        arrows=True, arrowstyle='->', arrowsize=15
    )

    if (seed_u, seed_v) in g.edges():
        nx.draw_networkx_edges(
            g, pos, edgelist=[(seed_u, seed_v)],
            edge_color='red', width=5, arrows=True, arrowstyle='->', arrowsize=20
        )

    nx.draw_networkx_labels(g, pos, font_size=8)
    plt.title(
        f"Hetero explanation for edge {edge_idx} (seed in red)\n"
        f"{g.number_of_nodes()}-node neighborhood (~{getattr(args, 'explain_focus_hops', 2)}-hop)\n"
        f"Case: {case_label}\n"
        f"Predicted class: {'Fraud' if target == 1 else 'Legitimate'}"
    )

    sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    plt.colorbar(sm, ax=plt.gca(), label='Edge Importance (max of to/rev_to relations)')
    plt.tight_layout()

    out_dir = args.explain_plot_dir
    os.makedirs(out_dir, exist_ok=True)
    plot_path = f"{out_dir}/explanation_hetero_{case_key}_{edge_idx}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"[{case_key}] Explanation plot saved to {plot_path}")


def run_explain(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, args, data_config):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    class Config:
        pass
    config = Config()
    config.epochs = args.n_epochs
    config.batch_size = args.batch_size
    config.model = args.model
    config.data = args.data
    config.num_neighbors = args.num_neighs
    config.lr = extract_param("lr", args)
    config.n_hidden = extract_param("n_hidden", args)
    config.n_gnn_layers = extract_param("n_gnn_layers", args)
    config.w_ce1 = extract_param("w_ce1", args)
    config.w_ce2 = extract_param("w_ce2", args)
    config.dropout = extract_param("dropout", args)
    config.final_dropout = extract_param("final_dropout", args)
    if args.model == 'gat':
        config.n_heads = extract_param("n_heads", args)

    transform = AddEgoIds() if args.ego else None

    add_arange_ids([tr_data, val_data, te_data])

    tr_loader, val_loader, te_loader = get_loaders(
        tr_data, val_data, te_data, tr_inds, val_inds, te_inds, transform, args
    )
    sample_batch = next(iter(te_loader))

    model = get_model(sample_batch, config, args)
    if args.reverse_mp:
        model = to_hetero(model, te_data.metadata(), aggr='mean')

    is_hetero = args.reverse_mp
    predict_fn = _sample_and_predict_hetero if is_hetero else _sample_and_predict
    explain_fn = _explain_and_plot_hetero if is_hetero else _explain_and_plot
    edge_y = te_data['node', 'to', 'node'].y if is_hetero else te_data.y

    checkpoint_path = f"{data_config['paths']['model_to_load']}/checkpoint_{args.unique_name}.tar"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # explicit edge requested: explain just that one, skip the 4-case search
    if args.explain_edge_idx != -1:
        edge_idx = args.explain_edge_idx
        logging.info(f"Explaining user-specified edge {edge_idx}")
        try:
            sample = predict_fn(edge_idx, te_data, model, device, args, transform=transform)
        except RuntimeError as e:
            logging.error(f"Could not sample/predict edge {edge_idx}: {e}")
            return
        explain_fn(model, sample, edge_idx, args, 'USER', 'User-specified edge')
        return

    # find one example each of TP / FP / FN / TN. Fraud is rare, so we split
    # te_inds by actual label up front and scan each group only for the
    # cases it can produce, instead of one flat scan over all of te_inds.
    found = {key: None for key in CASE_DEFINITIONS}

    labels = edge_y[te_inds]
    fraud_inds = te_inds[labels == 1]
    non_fraud_inds = te_inds[labels == 0]

    logging.info(
        f"Test set: {len(fraud_inds)} fraud edges, {len(non_fraud_inds)} non-fraud edges "
        f"(of {len(te_inds)} total)."
    )

    seed_gen = torch.Generator().manual_seed(getattr(args, 'seed', 42))
    fraud_inds = fraud_inds[torch.randperm(len(fraud_inds), generator=seed_gen)]
    non_fraud_inds = non_fraud_inds[torch.randperm(len(non_fraud_inds), generator=seed_gen)]

    found_ids = {}
    found_ids.update(_find_case_examples(fraud_inds, ['TP', 'FN'], te_data, model, device, args, transform, is_hetero))
    found_ids.update(_find_case_examples(non_fraud_inds, ['FP', 'TN'], te_data, model, device, args, transform, is_hetero))

    # only resample individually for the (up to 4) edges we're actually going to explain
    for case_key, edge_idx in found_ids.items():
        try:
            found[case_key] = (edge_idx, predict_fn(edge_idx, te_data, model, device, args, transform=transform))
        except RuntimeError as e:
            logging.warning(f"Could not resample edge {edge_idx} for explanation: {e}")

    for case_key, label in CASE_LABELS.items():
        entry = found[case_key]
        if entry is None:
            logging.info(f"No example found for case: '{label}'. Skipping.")
            continue
        edge_idx, sample = entry
        logging.info(f"Explaining case '{label}' using edge {edge_idx}")
        explain_fn(model, sample, edge_idx, args, case_key, label)