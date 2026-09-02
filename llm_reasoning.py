"""
Usage:
    pip install transformers accelerate torch --break-system-packages
    python llm_reasoning.py --data Small_HI --model gin --unique_name run1 \
        --explain_edge_idx 12345 --llm_provider local --llm_model Qwen/Qwen3-14B
"""

import json
import logging
import os
import re
import sys

import pandas as pd
import torch
from torch_geometric.explain import Explainer, GNNExplainer, CaptumExplainer
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import to_hetero

from train_util import AddEgoIds, extract_param, add_arange_ids, get_loaders
from training import get_model


# 1. Sampling + prediction + GNNExplainer, keeping global edge ids alive

def compute_edge_importance(model, batch, seed_pos, is_hetero):
    """
    Runs GNNExplainer (homogeneous) or CaptumExplainer (heterogeneous) on an
    already-sampled subgraph and returns (edge_index_np, edge_importance_np),
    in the exact row order of `batch.edge_index` / `batch[to_rel].edge_index`
    """
    if not is_hetero:
        device = batch.x.device

        class Wrap(torch.nn.Module):
            def __init__(self, m, pos):
                super().__init__()
                self.m, self.pos = m, pos

            def forward(self, x, edge_index, edge_attr, **kw):
                return self.m(x, edge_index, edge_attr)[self.pos:self.pos + 1]

        wrapper = Wrap(model, seed_pos).to(device)
        explainer = Explainer(
            model=wrapper,
            algorithm=GNNExplainer(epochs=100),
            explanation_type='model',
            node_mask_type='attributes',
            edge_mask_type='object',
            model_config=dict(mode='multiclass_classification', task_level='graph', return_type='raw'),
        )
        explanation = explainer(x=batch.x, edge_index=batch.edge_index, edge_attr=batch.edge_attr)
        edge_index_np = batch.edge_index.detach().cpu().numpy()
        edge_imp_np = explanation.edge_mask.detach().cpu().numpy()

    else:
        to_rel = ('node', 'to', 'node')
        device = batch['node'].x.device

        class WrapH(torch.nn.Module):
            def __init__(self, m, pos):
                super().__init__()
                self.m, self.pos = m, pos

            def forward(self, x, edge_index, edge_attr, **kw):
                out = self.m(x, edge_index, edge_attr)[to_rel]
                return out[self.pos:self.pos + 1]

        wrapper = WrapH(model, seed_pos).to(device)
        explainer = Explainer(
            model=wrapper,
            algorithm=CaptumExplainer('IntegratedGradients'),
            explanation_type='model',
            node_mask_type='attributes',
            edge_mask_type='object',
            model_config=dict(mode='multiclass_classification', task_level='graph', return_type='raw'),
        )
        explanation = explainer(x=batch.x_dict, edge_index=batch.edge_index_dict, edge_attr=batch.edge_attr_dict)
        edge_index_np = batch[to_rel].edge_index.detach().cpu().numpy()
        edge_imp_np = explanation.edge_mask_dict[to_rel].detach().cpu().numpy()

    return edge_index_np, edge_imp_np


def sample_predict_explain(edge_idx, te_data, model, device, args, transform, is_hetero):
    """
    Samples the k-hop subgraph around a single test edge, runs the GNN
    forward pass, and runs the explainer on it
    """
    if not is_hetero:
        loader = LinkNeighborLoader(
            te_data, num_neighbors=args.num_neighs,
            edge_label_index=te_data.edge_index[:, edge_idx:edge_idx + 1],
            edge_label=te_data.y[edge_idx:edge_idx + 1],
            batch_size=1, shuffle=False, transform=transform,
        )
        batch = next(iter(loader)).to(device)

        mask = (batch.edge_attr[:, 0] == edge_idx)
        seed_positions = mask.nonzero(as_tuple=False)
        if seed_positions.numel() != 1:
            raise RuntimeError(f"Expected exactly one seed edge, found {seed_positions.numel()}")
        seed_pos = seed_positions.item()

        global_edge_ids = batch.edge_attr[:, 0].detach().cpu().int().tolist()

        # drop the id column
        batch.edge_attr = batch.edge_attr[:, 1:]

        with torch.no_grad():
            out = model(batch.x, batch.edge_index, batch.edge_attr)
            pred = out[seed_pos:seed_pos + 1].argmax(dim=-1).item()
        actual = int(te_data.y[edge_idx].item())

        edge_index_np, edge_imp_np = compute_edge_importance(model, batch, seed_pos, is_hetero=False)

    else:
        to_rel = ('node', 'to', 'node')
        loader = LinkNeighborLoader(
            te_data, num_neighbors=args.num_neighs,
            edge_label_index=(to_rel, te_data[to_rel].edge_index[:, edge_idx:edge_idx + 1]),
            edge_label=te_data[to_rel].y[edge_idx:edge_idx + 1],
            batch_size=1, shuffle=False, transform=transform,
        )
        batch = next(iter(loader)).to(device)

        mask = (batch[to_rel].edge_attr[:, 0] == edge_idx)
        seed_positions = mask.nonzero(as_tuple=False)
        if seed_positions.numel() != 1:
            raise RuntimeError(f"Expected exactly one seed edge, found {seed_positions.numel()}")
        seed_pos = seed_positions.item()

        global_edge_ids = batch[to_rel].edge_attr[:, 0].detach().cpu().int().tolist()

        batch[to_rel].edge_attr = batch[to_rel].edge_attr[:, 1:]
        batch['node', 'rev_to', 'node'].edge_attr = batch['node', 'rev_to', 'node'].edge_attr[:, 1:]

        with torch.no_grad():
            out = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)[to_rel]
            pred = out[seed_pos:seed_pos + 1].argmax(dim=-1).item()
        actual = int(te_data[to_rel].y[edge_idx].item())

        edge_index_np, edge_imp_np = compute_edge_importance(model, batch, seed_pos, is_hetero=True)

    # `edge_index_np` / `edge_imp_np` are in the same row order as
    # `batch.edge_attr` was at the point global_edge_ids was captured
    # (nothing reorders rows between capture and the explainer call), so we
    # can zip them directly
    imp_by_id = {}
    for gid, imp in zip(global_edge_ids, edge_imp_np.tolist()):
        imp_by_id[int(gid)] = max(imp_by_id.get(int(gid), 0.0), float(imp))

    return {
        'pred': pred,
        'actual': actual,
        'global_edge_ids': [int(g) for g in global_edge_ids],
        'imp_by_id': imp_by_id,
        'target_edge_id': int(edge_idx),
    }


# 2. Raw-CSV metadata lookup (human-readable amount/currency/payment/time)

def load_serialization_maps(args, data_config):
    data_dir = f"{data_config['paths']['aml_data']}/{args.data}"
 
    def _load_int_keyed(filename):
        path = os.path.join(data_dir, filename)
        if not os.path.exists(path):
            logging.error(f"{filename} not found in {data_dir}")
            sys.exit(1)
        with open(path) as f:
            raw = json.load(f)
        return {int(k): v for k, v in raw.items()}
 
    currency_map = _load_int_keyed("currency_map.json")
    payment_map = _load_int_keyed("payment_format_map.json")
    account_map = _load_int_keyed("account_map.json")
 
    meta_path = os.path.join(data_dir, "format_meta.json")
    first_ts = None
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            first_ts = json.load(f).get("firstTs")
    else:
        logging.error(f"format_meta.json not found in {data_dir}")
        sys.exit(1)
 
    return currency_map, payment_map, account_map, first_ts


def build_edge_metadata_lookup(args, data_config):
    """
    Reads the raw `formatted_transactions.csv` to recover human-readable amount / currency / 
    payment / timestamp / account fields for serialization, since the tensors used for
    GNN training are z-normalized and account/currency/payment ids are
    integer-encoded by `format_kaggle_files.py`.
 
    Returns {edge_id (row position, matches the ids `add_arange_ids`
    injects): {src, dst, amount, currency, payment_format, timestamp,
    timestamp_str, label}}.
    """
    currency_map, payment_map, account_map, first_ts = load_serialization_maps(args, data_config)

    path = f"{data_config['paths']['aml_data']}/{args.data}/formatted_transactions.csv"
    df = pd.read_csv(path)
 
    lookup = {}
    for i, row in df.iterrows():
        cur_id = int(row['Received Currency'])
        fmt_id = int(row['Payment Format'])
        src_id = int(row['from_id'])
        dst_id = int(row['to_id'])
 
        lookup[i] = {
            "src": f"{account_map.get(src_id, src_id)}",
            "dst": f"{account_map.get(dst_id, dst_id)}",
            "amount": float(row['Amount Received']),
            "currency": currency_map.get(cur_id, str(cur_id)),
            "payment_format": payment_map.get(fmt_id, str(fmt_id)),
            "timestamp": _fmt_timestamp(int(row['Timestamp']), first_ts),
            "label": int(row['Is Laundering']),
        }
    return lookup
 
 
def _fmt_timestamp(rel_seconds, first_ts=None):
    from datetime import datetime, timezone
    return datetime.fromtimestamp(first_ts + rel_seconds, tz=timezone.utc).strftime("%Y/%m/%d %H:%M")

# 3. Serialization

def serialize_subgraph(edge_metadata_lookup, edge_ids, target_edge_id, imp_lookup_by_id=None, max_edges=40):
    edge_ids = list(dict.fromkeys(edge_ids))  # de-dup, preserve order
    if target_edge_id not in edge_ids:
        edge_ids.append(target_edge_id)

    if len(edge_ids) > max_edges:
        if imp_lookup_by_id:
            ranked = sorted(edge_ids, key=lambda e: imp_lookup_by_id.get(e, 0.0), reverse=True)
        else:
            ranked = edge_ids
        keep = set(ranked[:max_edges])
        keep.add(target_edge_id)
        edge_ids = [e for e in edge_ids if e in keep]

    nodes = {}
    edge_lines = []
    for eid in edge_ids:
        meta = edge_metadata_lookup.get(eid)
        if meta is None:
            continue
        nodes[meta["src"]] = "Account"
        nodes[meta["dst"]] = "Account"

        tag = " [TARGET EDGE]" if eid == target_edge_id else ""
        line = f"- {meta['src']} transfers_to {meta['dst']}{tag}\n"
        line += f"  amount: {meta['amount']:.2f} (currency: {meta['currency']})\n"
        line += f"  via: {meta['payment_format']}\n"
        line += f"  timestamp: {meta['timestamp']}\n"
        if imp_lookup_by_id is not None:
            imp = imp_lookup_by_id.get(eid)
            if imp is not None:
                line += (
                    f"  importance: {imp:.3f}   "
                    f"# GNNExplainer edge importance score; higher = more "
                    f"influential to the GNN's own prediction for the target edge\n"
                )
        edge_lines.append(line)

    node_lines = [f"- {name} (type: {ntype})" for name, ntype in nodes.items()]
    return "**Nodes:**\n" + "\n".join(node_lines) + "\n**Edges:**\n" + "\n".join(edge_lines)


# 4. Few-shot prompt

SYSTEM_PROMPT = """You are an expert financial crime investigator reviewing patterns of financial activities and behaviors of involved accounts to identify potential cases of money laundering. The data is represented as a graph, where:
- Nodes are of type Account or Bank.
- Edges represent relationships of type transfers_to or belongs_to, and include metadata such as amount, currency, payment method, and timestamp.
- Some edges also include an `importance` score produced by a graph neural network's explainability module (GNNExplainer), indicating how influential that edge was to the GNN's own prediction for the target transaction. Higher importance means the GNN relied on that edge more heavily when making its prediction.

For training purposes, you will be shown examples of subgraph typologies that are known to be either suspicious (indicative of laundering tactics) or non-suspicious (routine financial activity). These typologies illustrate common structural patterns in financial networks."""

def build_prompt(fewshot, test_subgraph_text, target_edge_id, gnn_pred=None):
    parts = ["Few-shot Examples:\n"]
    for ex in fewshot.get("suspicious", []):
        parts.append(ex["subgraph_text"])
        parts.append(f"Explanation: {ex['explanation']}\n")

    parts.append("\nnon-suspicious Examples:\n")
    for ex in fewshot.get("non_suspicious", []):
        parts.append(ex["subgraph_text"])
        parts.append(f"Explanation: {ex['explanation']}\n")

    task = (
        f"\nTask: Given a transaction (edge) with Transaction ID {target_edge_id}, "
        "along with its surrounding subgraph, determine whether the transaction is "
        "suspicious, or not suspicious. Your reasoning should be based on whether the "
        "surrounding subgraph resembles any of the suspicious typologies provided in "
        "the examples. Where edges include an `importance` score, weigh higher-"
        "importance edges more heavily in your reasoning, but treat that score as one "
        "signal among the structural and value-based patterns you observe -- not as "
        "ground truth.\n"
    )
    if gnn_pred is not None:
        task += (
            "\n(For reference only, not to be treated as ground truth: the upstream "
            f"GNN classifier predicted this transaction as "
            f"{'Suspicious' if gnn_pred == 1 else 'Not Suspicious'}.)\n"
        )
    parts.append(task)
    parts.append(f"\nTest Example:\n{test_subgraph_text}\n")
    parts.append(
        "\nAnswer Format:\n"
        "- Conclusion: Suspicious or Not Suspicious\n"
        "- Explanation: (2-3 sentences reasoning)\n"
        "- Observed Pattern: (e.g., gather-scatter)\n"
    )
    return "\n".join(parts)


def parse_llm_response(text):
    """Extracts Conclusion / Explanation / Observed Pattern from the LLM's
    reply. Falls back to None for any field it can't find, so callers can
    detect and log malformed responses instead of silently mis-parsing."""
    def grab(field):
        m = re.search(rf"{field}\s*:\s*(.+)", text, re.IGNORECASE)
        return m.group(1).strip() if m else None

    return {
        "conclusion": grab("Conclusion"),
        "explanation": grab("Explanation"),
        "observed_pattern": grab("Observed Pattern"),
        "raw": text,
    }


# 5. LLM client

class LLMClient:
    def __init__(self, provider="local", model="Qwen/Qwen3-14B", temperature=0.0,
                 max_new_tokens=1024, enable_thinking=None):
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.enable_thinking = enable_thinking
 
        if provider == "local":
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
 
            logging.info(f"Loading tokenizer for {model} ...")
            self.tokenizer = AutoTokenizer.from_pretrained(model)
 
            load_kwargs = dict(torch_dtype=torch.float16, device_map="auto")
            logging.info(f"Loading model weights for {model} "
                         f"(first run downloads from the Hub, this can take a while) ...")
            self.hf_model = AutoModelForCausalLM.from_pretrained(model, **load_kwargs)
            self.hf_model.eval()
 
        else:
            raise ValueError(f"Unknown provider: {provider!r}")
 
    def complete(self, system_prompt, user_prompt):
        if self.provider == "local":
            import torch
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            template_kwargs = dict(add_generation_prompt=True, return_tensors="pt")
            if self.enable_thinking is not None:
                template_kwargs["enable_thinking"] = self.enable_thinking
            input_ids = self.tokenizer.apply_chat_template(messages, **template_kwargs).to(self.hf_model.device)
 
            with torch.no_grad():
                output_ids = self.hf_model.generate(
                    input_ids,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=self.temperature > 0,
                    temperature=max(self.temperature, 1e-5),
                    pad_token_id=self.tokenizer.eos_token_id,
                )
 
            # strip the input prompt tokens off, keep only the new tokens
            new_tokens = output_ids[0][input_ids.shape[-1]:].tolist()
            return self._strip_thinking(new_tokens)

    def _strip_thinking(self, new_tokens):
        """Splits off a Qwen3-style <think>...</think> reasoning block and
        returns only the final answer text.
        """
        THINK_END_TOKEN_ID = 151668  # </think>, per the Qwen3 model card
        try:
            # rindex: last occurrence, in case the model repeats the tag
            split_at = len(new_tokens) - new_tokens[::-1].index(THINK_END_TOKEN_ID)
        except ValueError:
            split_at = 0  # no </think> found -- e.g. enable_thinking=False

        thinking = self.tokenizer.decode(new_tokens[:split_at], skip_special_tokens=True).strip("\n")
        content = self.tokenizer.decode(new_tokens[split_at:], skip_special_tokens=True).strip("\n")

        if thinking:
            logging.debug(f"[local LLM thinking content, discarded]\n{thinking}")

        return content
 

# 6. Orchestration

def explain_with_llm(edge_idx, te_data, model, device, args, transform, is_hetero,
                      edge_metadata_lookup, fewshot, llm_client, max_edges=20):
    """Runs the full pipeline for one transaction: GNN predict + explain ->
    serialize -> few-shot prompt -> LLM call -> parse. Returns a dict
    suitable for logging/audit or appending to a results table."""
    sampled = sample_predict_explain(edge_idx, te_data, model, device, args, transform, is_hetero)

    subgraph_text = serialize_subgraph(
        edge_metadata_lookup,
        sampled['global_edge_ids'],
        sampled['target_edge_id'],
        imp_lookup_by_id=sampled['imp_by_id'],
        max_edges=max_edges,
    )

    prompt = build_prompt(fewshot, subgraph_text, edge_idx, gnn_pred=sampled['pred'])
    response_text = llm_client.complete(SYSTEM_PROMPT, prompt)
    parsed = parse_llm_response(response_text)

    return {
        "edge_id": edge_idx,
        "gnn_pred": sampled['pred'],
        "actual_label": sampled['actual'],
        "subgraph_text": subgraph_text,
        "llm_conclusion": parsed["conclusion"],
        "llm_explanation": parsed["explanation"],
        "llm_pattern": parsed["observed_pattern"],
        "llm_raw": parsed["raw"],
    }

# 7. Standalone CLI entry point

def _build_config(args):
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


def main():
    from util import create_parser as base_parser, set_seed, logger_setup
    from data_loading import get_data

    parser = base_parser()
    parser.add_argument("--explain_edge_idx", type=int, required=True,
                        help="Global test-set edge id (row in formatted_transactions.csv) to review")
    parser.add_argument("--fewshot_path", default="fewshot_examples.json")
    parser.add_argument("--llm_provider", default="local", choices=["local"])
    parser.add_argument("--llm_model", default="Qwen/Qwen3-14B")
    parser.add_argument("--llm_max_new_tokens", type=int, default=1024, help="Max tokens to generate")
    parser.add_argument("--llm_disable_thinking", action='store_true',
                        help="For hybrid thinking/non-thinking models (e.g. Qwen3): pass "
                              "enable_thinking=False so the model skips the <think>...</think> "
                              "block entirely (faster, fewer tokens). If not set, thinking stays "
                              "on by default and the <think> block is stripped from the output "
                              "after generation instead -- either way, llm_conclusion only ever "
                              "sees the final answer.")
    parser.add_argument("--max_subgraph_edges", type=int, default=20)
    parser.add_argument("--out_path", default=None, help="Optional path to dump the result JSON")
    args = parser.parse_args()

    logger_setup()
    set_seed(args.seed)

    with open("data_config.json") as f:
        data_config = json.load(f)

    logging.info("Loading dataset ...")
    tr_data, val_data, te_data, tr_inds, val_inds, te_inds = get_data(args, data_config)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    transform = AddEgoIds() if args.ego else None
    add_arange_ids([tr_data, val_data, te_data])
    tr_loader, val_loader, te_loader = get_loaders(
        tr_data, val_data, te_data, tr_inds, val_inds, te_inds, transform, args
    )

    config = _build_config(args)
    sample_batch = next(iter(tr_loader))
    model = get_model(sample_batch, config, args)
    if args.reverse_mp:
        model = to_hetero(model, te_data.metadata(), aggr='mean')

    logging.info("Loading model checkpoint ...")
    ckpt_path = f'{data_config["paths"]["model_to_load"]}/checkpoint_{args.unique_name}.tar'
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    logging.info("Building raw-CSV metadata lookup ...")
    edge_metadata_lookup = build_edge_metadata_lookup(args, data_config)

    with open(args.fewshot_path) as f:
        fewshot = json.load(f)

    llm_client = LLMClient(
        provider=args.llm_provider, model=args.llm_model,
        max_new_tokens=args.llm_max_new_tokens,
        enable_thinking=(False if args.llm_disable_thinking else None),
    )

    result = explain_with_llm(
        args.explain_edge_idx, te_data, model, device, args, transform, args.reverse_mp,
        edge_metadata_lookup, fewshot, llm_client, max_edges=args.max_subgraph_edges,
    )

    print(json.dumps(result, indent=2))
    if args.out_path:
        with open(args.out_path, "w") as f:
            json.dump(result, f, indent=2)
        logging.info(f"Result written to {args.out_path}")


if __name__ == "__main__":
    main()