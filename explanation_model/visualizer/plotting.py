"""Drawing the explained neighborhood of a transaction (matplotlib / networkx)."""
import logging
import os

import matplotlib.pyplot as plt
import networkx as nx


def add_visualizer_args(parser):
    parser.add_argument("--explain_plot_dir", default="explanations", help="Directory to save explanation plots (PNG files).")
    parser.add_argument("--explain_focus_hops", type=int, default=2, help="Neighborhood radius (hops) drawn around the transaction.")
    parser.add_argument("--explain_max_nodes", type=int, default=40, help="Max nodes drawn (keeps the most important edges).")


def focus_subgraph_and_layout(imp_lookup, seed_u, seed_v, args):
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


def plot_explanation(importance, sample, edge_idx, args, case_key, case_label, is_hetero):
    """Saves a PNG of the focused neighborhood (edge colour = importance, seed transaction in red)
    and returns its path. `importance` is an EdgeImportance from compute_edge_importance."""
    seed_u, seed_v = sample['seed_u'], sample['seed_v']
    target = sample['pred']
    hops = getattr(args, 'explain_focus_hops', 2)

    g, pos = focus_subgraph_and_layout(importance.imp_lookup, seed_u, seed_v, args)
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
        f"{'Hetero explanation' if is_hetero else 'Explanation'} for edge {edge_idx} (seed in red)\n"
        f"{g.number_of_nodes()}-node neighborhood (~{hops}-hop)\n"
        f"Case: {case_label}\n"
        f"Predicted class: {'Fraud' if target == 1 else 'Legitimate'}"
    )

    sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    plt.colorbar(sm, ax=plt.gca(),
                 label='Edge Importance (max of to/rev_to relations)' if is_hetero else 'Edge Importance')
    plt.tight_layout()

    out_dir = getattr(args, 'explain_plot_dir', 'explanations')
    os.makedirs(out_dir, exist_ok=True)
    plot_path = f"{out_dir}/explanation_{'hetero_' if is_hetero else ''}{case_key}_{edge_idx}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"[{case_key}] Explanation plot saved to {plot_path}")
    return plot_path
