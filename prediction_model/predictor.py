"""Single-transaction inference on top of a trained GNN checkpoint.

This is the entry point the explanation framework uses: give it a transaction (edge) id, get back
the GNN's fraud / non-fraud prediction together with the sampled computation subgraph, which the
explanation_model (visualizer / classifier / explainer) can then work on.

The CLI flags describing the architecture (--model, --emlps, --reverse_mp, --ports, --tds, --ego,
--num_neighs, ...) must match the ones used when the checkpoint was trained.
"""
import logging

import torch
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import to_hetero

from prediction_model.data_loading import get_data
from prediction_model.train_util import AddEgoIds, extract_param, add_arange_ids, get_loaders
from prediction_model.training import get_model

TO_REL = ('node', 'to', 'node')
REV_REL = ('node', 'rev_to', 'node')


def build_config(args):
    """Model hyper-parameters from model_settings.json, in the attribute style get_model() expects."""
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
    return config


def _locate_seed_edge(edge_attr, edge_idx):
    seed_positions = (edge_attr[:, 0] == edge_idx).nonzero(as_tuple=False)
    if seed_positions.numel() != 1:
        raise RuntimeError(f"Expected exactly one seed edge, found {seed_positions.numel()}")
    return seed_positions.item()


def _sample_and_predict_homo(edge_idx, te_data, model, device, args, transform=None):
    loader = LinkNeighborLoader(
        te_data,
        num_neighbors=args.num_neighs,
        edge_label_index=te_data.edge_index[:, edge_idx:edge_idx + 1],
        edge_label=te_data.y[edge_idx:edge_idx + 1],
        batch_size=1,
        shuffle=False,
        transform=transform,
    )
    batch = next(iter(loader)).to(device)

    # locate the seed edge via its global ID (first column of edge_attr)
    seed_pos = _locate_seed_edge(batch.edge_attr, edge_idx)
    global_edge_ids = batch.edge_attr[:, 0].detach().cpu().int().tolist()

    # drop the ID column now that the seed edge has been located
    batch.edge_attr = batch.edge_attr[:, 1:]

    seed_u, seed_v = batch.edge_index[0, seed_pos].item(), batch.edge_index[1, seed_pos].item()

    with torch.no_grad():
        out = model(batch.x, batch.edge_index, batch.edge_attr)
        logits = out[seed_pos]
        pred = logits.argmax(dim=-1).item()
        fraud_prob = torch.softmax(logits, dim=-1)[1].item()

    return {
        'batch': batch, 'seed_pos': seed_pos, 'seed_u': seed_u, 'seed_v': seed_v,
        'pred': pred, 'fraud_prob': fraud_prob, 'actual': int(te_data.y[edge_idx].item()),
        'global_edge_ids': global_edge_ids,
    }


def _sample_and_predict_hetero(edge_idx, te_data, model, device, args, transform=None):
    """Heterogeneous (reverse message passing) version. One node type ('node'), two edge types:
    ('node','to','node') = original direction, ('node','rev_to','node') = reversed. `edge_idx`
    indexes the 'to' relation."""
    loader = LinkNeighborLoader(
        te_data,
        num_neighbors=args.num_neighs,
        edge_label_index=(TO_REL, te_data[TO_REL].edge_index[:, edge_idx:edge_idx + 1]),
        edge_label=te_data[TO_REL].y[edge_idx:edge_idx + 1],
        batch_size=1,
        shuffle=False,
        transform=transform,
    )
    batch = next(iter(loader)).to(device)

    seed_pos = _locate_seed_edge(batch[TO_REL].edge_attr, edge_idx)
    global_edge_ids = batch[TO_REL].edge_attr[:, 0].detach().cpu().int().tolist()

    # drop the ID column from both relations now that the seed edge is located
    batch[TO_REL].edge_attr = batch[TO_REL].edge_attr[:, 1:]
    batch[REV_REL].edge_attr = batch[REV_REL].edge_attr[:, 1:]

    seed_u = batch[TO_REL].edge_index[0, seed_pos].item()
    seed_v = batch[TO_REL].edge_index[1, seed_pos].item()

    with torch.no_grad():
        out = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)[TO_REL]
        logits = out[seed_pos]
        pred = logits.argmax(dim=-1).item()
        fraud_prob = torch.softmax(logits, dim=-1)[1].item()

    return {
        'batch': batch, 'seed_pos': seed_pos, 'seed_u': seed_u, 'seed_v': seed_v,
        'pred': pred, 'fraud_prob': fraud_prob, 'actual': int(te_data[TO_REL].y[edge_idx].item()),
        'global_edge_ids': global_edge_ids,
    }


def sample_and_predict(edge_idx, te_data, model, device, args, transform=None, is_hetero=None):
    """Samples the k-hop subgraph around `edge_idx` (same sampler as in training) and predicts it.

    Returns a dict with: batch (PyG batch, edge-id column already removed), seed_pos / seed_u / seed_v
    (location of the transaction inside the batch), pred (0/1), fraud_prob, actual (ground truth),
    global_edge_ids (global transaction id of every edge of the 'to' relation of the batch, same order).
    `transform` must match the one used while training (AddEgoIds() if args.ego).
    """
    is_hetero = args.reverse_mp if is_hetero is None else is_hetero
    fn = _sample_and_predict_hetero if is_hetero else _sample_and_predict_homo
    return fn(edge_idx, te_data, model, device, args, transform)


class TransactionPredictor:
    """Loads the dataset + a trained checkpoint and predicts individual transactions.

        predictor = TransactionPredictor(args, data_config)
        sample = predictor.predict(edge_idx)      # sample['pred'] == 1  -> predicted fraud
    """

    def __init__(self, args, data_config, device=None):
        self.args = args
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.is_hetero = bool(args.reverse_mp)

        logging.info("Loading dataset ...")
        (self.tr_data, self.val_data, self.te_data,
         self.tr_inds, self.val_inds, self.te_inds) = get_data(args, data_config)

        self.transform = AddEgoIds() if args.ego else None
        add_arange_ids([self.tr_data, self.val_data, self.te_data])
        tr_loader, _, _ = get_loaders(self.tr_data, self.val_data, self.te_data,
                                      self.tr_inds, self.val_inds, self.te_inds, self.transform, args)

        sample_batch = next(iter(tr_loader))
        model = get_model(sample_batch, build_config(args), args)
        if self.is_hetero:
            model = to_hetero(model, self.te_data.metadata(), aggr='mean')

        logging.info("Loading model checkpoint ...")
        ckpt_path = f"{data_config['paths']['model_to_load']}/checkpoint_{args.unique_name}.tar"
        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        self.model = model.to(self.device).eval()

    @property
    def edge_labels(self):
        return self.te_data[TO_REL].y if self.is_hetero else self.te_data.y

    def predict(self, edge_idx):
        return sample_and_predict(edge_idx, self.te_data, self.model, self.device,
                                  self.args, self.transform, self.is_hetero)
