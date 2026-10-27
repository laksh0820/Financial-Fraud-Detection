"""Drawing the explained neighborhood of a transaction (matplotlib / networkx).

The plot shows exactly the transactions the pattern classifier sees: the seed transaction plus the
`max_edges - 1` most important other transactions of the GNN-sampled neighborhood (the rule lives in
`explanation_model.subgraph.select_edge_ids`, shared with `EdgeMetadataStore.build_subgraph`).

This module must stay free of torch imports: detect.py imports `add_visualizer_args` before the
(heavy) prediction model is loaded.
"""
import logging
import math
import os
from collections import defaultdict, deque

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.lines import Line2D

from explanation_model.subgraph import select_edge_ids

_TO_REL = ('node', 'to', 'node')  # same as prediction_model.predictor.TO_REL (not imported: torch)

SEED_NODE_COLOR = '#ffd54f'
SEED_EDGE_COLOR = '#111111'
NODE_COLOR = '#a9d6e5'
DETACHED_NODE_COLOR = '#d9d9d9'
# Reds without its near-white end, so low-importance edges stay visible on a white background
IMPORTANCE_CMAP = mcolors.LinearSegmentedColormap.from_list('reds_visible', plt.cm.Reds(np.linspace(0.25, 1.0, 256)))


def add_visualizer_args(parser):
    parser.add_argument("--explain_plot_dir", default="explanations", help="Directory to save explanation plots (PNG files).")
    parser.add_argument("--explain_max_edges", type=int, default=40,
                        help="Max transactions drawn: the seed plus the most important other ones, selected exactly as "
                             "for the classifier's subgraph. (detect.py always uses --max_subgraph_edges instead, "
                             "so the plot matches what the classifier saw.)")


# ---- which edges are drawn ----------------------------------------------------------------------

def _to_numpy(t):
    return t.detach().cpu().numpy() if hasattr(t, 'detach') else np.asarray(t)


def _sampled_edge_locations(sample, is_hetero):
    """{global transaction id: (src, dst)} in batch-local node ids, for every edge of the sampled
    neighborhood. `sample['global_edge_ids']` is aligned with the columns of the 'to' relation."""
    batch = sample['batch']
    edge_index = _to_numpy(batch[_TO_REL].edge_index if is_hetero else batch.edge_index)
    gids = sample['global_edge_ids']
    if edge_index.shape[1] != len(gids):
        raise ValueError(f"edge_index has {edge_index.shape[1]} edges but there are {len(gids)} global edge ids")
    return {int(g): (int(edge_index[0, j]), int(edge_index[1, j])) for j, g in enumerate(gids)}


def build_plot_graph(importance, sample, edge_idx, is_hetero, max_edges):
    """The DiGraph to draw: one arrow per (src, dst) pair among the selected transactions.

    Node attrs: none. Edge attrs: `importance` (max over to/rev_to relations, what the colour shows)
    and `count` (number of selected transactions on that pair). Returns (graph, n_sampled), where
    n_sampled is the size of the neighborhood before selection."""
    locations = _sampled_edge_locations(sample, is_hetero)
    locations[int(edge_idx)] = (sample['seed_u'], sample['seed_v'])  # the seed is always present

    selected = select_edge_ids(locations.keys(), edge_idx, importance.imp_by_id, max_edges)

    g = nx.DiGraph()
    for gid in selected:
        u, v = locations[gid]
        imp = importance.imp_lookup.get((u, v), importance.imp_by_id.get(gid, 0.0))
        if g.has_edge(u, v):
            g[u][v]['count'] += 1
            g[u][v]['importance'] = max(g[u][v]['importance'], imp)
        else:
            g.add_edge(u, v, importance=imp, count=1)
    return g, len(locations)


# ---- layout -------------------------------------------------------------------------------------

def radial_layout(g, seed_u, seed_v, seed_gap=3.2, ring_gap=1.7, node_gap=1.3):
    """Deterministic radial layout, in units where a node is ~1 wide.

    The seed transaction sits in the middle (seed_u left, seed_v right of the origin). Every other node
    is placed on a ring according to its hop distance from the seed, inside an angular wedge that is
    proportional to the size of the BFS subtree it belongs to, so neighbors of the same account stay
    together and edges stay short. Ring radii grow just enough to keep nodes `node_gap` apart (a
    force-directed layout instead collapses a hub-and-spoke neighborhood into one blob).
    Nodes not connected to the seed through the drawn edges form their own wedges on the first ring.
    Returns (pos, detached_nodes)."""
    ug = g.to_undirected()
    roots = list(dict.fromkeys([seed_u, seed_v]))
    depth = {r: 0 for r in roots}
    children = defaultdict(list)

    def grow(sources):
        queue = deque(sources)
        while queue:
            n = queue.popleft()
            for m in sorted((m for m in ug.neighbors(n) if m != n), key=lambda m: (-ug.degree(m), m)):
                if m not in depth:
                    depth[m] = depth[n] + 1
                    children[n].append(m)
                    queue.append(m)

    grow(roots)
    detached_roots = []
    while len(depth) < len(ug):
        left = [n for n in ug if n not in depth]
        root = max(left, key=lambda n: (ug.degree(n), -n))  # most connected account of the remaining ones
        depth[root] = 1
        detached_roots.append(root)
        grow([root])
    detached = {n for r in detached_roots for n in _subtree(children, r)}

    leaves = {}

    def count_leaves(n):
        leaves[n] = sum(count_leaves(c) for c in children[n]) if children[n] else 1
        return leaves[n]

    top = roots + detached_roots
    for r in top:
        count_leaves(r)
    total = sum(leaves[r] for r in top)

    angle = {}

    def place_children(n, a0, a1):
        cur = a0
        for c in children[n]:
            width = (a1 - a0) * leaves[c] / leaves[n]
            angle[c] = cur + width / 2
            place_children(c, cur, cur + width)
            cur += width

    # u's wedge is centred on the left, so the seed edge reads left -> right through the middle
    wedge = {r: 2 * math.pi * leaves[r] / total for r in top}
    start = math.pi - wedge[roots[0]] / 2
    for r in top:
        a0, a1 = start, start + wedge[r]
        if r not in roots:
            angle[r] = (a0 + a1) / 2
        place_children(r, a0, a1)
        start = a1

    pos = {}
    if len(roots) == 2:
        pos[seed_u], pos[seed_v] = (-seed_gap / 2, 0.0), (seed_gap / 2, 0.0)
    else:
        pos[seed_u] = (0.0, 0.0)

    # ring radii: grow until the closest pair of nodes on a ring is `node_gap` apart
    by_depth = defaultdict(list)
    for n, d in depth.items():
        if d > 0:
            by_depth[d].append(n)
    radius = {0: seed_gap / 2 if len(roots) == 2 else 0.0}
    for d in sorted(by_depth):
        angs = sorted(angle[n] % (2 * math.pi) for n in by_depth[d])
        gaps = [b - a for a, b in zip(angs, angs[1:])] + [angs[0] + 2 * math.pi - angs[-1]]
        smallest = min(gaps) if len(angs) > 1 else math.pi
        needed = node_gap / (2 * math.sin(min(smallest, math.pi) / 2)) if len(angs) > 1 else 0.0
        radius[d] = max(radius[d - 1] + ring_gap, needed)
        for n in by_depth[d]:
            pos[n] = (radius[d] * math.cos(angle[n]), radius[d] * math.sin(angle[n]))
    return pos, detached


def _subtree(children, root):
    out, stack = [], [root]
    while stack:
        n = stack.pop()
        out.append(n)
        stack.extend(children[n])
    return out


# ---- drawing ------------------------------------------------------------------------------------

def _curve_offset(p1, p2, rad):
    """Midpoint of matplotlib's arc3 curve minus the chord midpoint (for placing a label on it)."""
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    return 0.5 * rad * dy, -0.5 * rad * dx


def plot_explanation(importance, sample, edge_idx, args, case_key, case_label, is_hetero, max_edges=None):
    """Saves a PNG of the transactions the classifier sees (edge colour = importance, seed transaction
    highlighted) and returns its path. `importance` is an EdgeImportance from compute_edge_importance.
    `max_edges` overrides args.explain_max_edges (the pipeline passes its --max_subgraph_edges)."""
    seed_u, seed_v = sample['seed_u'], sample['seed_v']
    target = sample['pred']
    if max_edges is None:
        max_edges = getattr(args, 'explain_max_edges', 40)

    g, n_sampled = build_plot_graph(importance, sample, edge_idx, is_hetero, max_edges)
    assert g.has_edge(seed_u, seed_v), "the seed edge must always be drawn"
    n_tx = sum(d['count'] for _, _, d in g.edges(data=True))
    pos, detached = radial_layout(g, seed_u, seed_v)

    # arrows between the same two accounts in both directions are drawn as two arcs, the rest straight
    curved = {(u, v) for u, v in g.edges() if u != v and g.has_edge(v, u)}
    rad = 0.22

    fig, ax = plt.subplots(figsize=(13, 10))
    xs, ys = [p[0] for p in pos.values()], [p[1] for p in pos.values()]
    pad = 1.6
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_aspect('equal')
    ax.set_axis_off()
    fig.canvas.draw()
    bbox = ax.get_window_extent()
    pts_per_unit = min(bbox.width / (ax.get_xlim()[1] - ax.get_xlim()[0]),
                       bbox.height / (ax.get_ylim()[1] - ax.get_ylim()[0])) * 72.0 / fig.dpi
    diameter = float(np.clip(0.85 * pts_per_unit, 16, 46))  # node diameter in points
    node_size = diameter ** 2
    font_size = float(np.clip(diameter * 0.38, 6, 11))

    max_imp = max((d['importance'] for _, _, d in g.edges(data=True)), default=0.0)
    norm = mcolors.Normalize(vmin=0.0, vmax=max_imp if max_imp > 0 else 1.0)

    def draw_arrows(edges, **style):
        if edges:
            nx.draw_networkx_edges(g, pos, edgelist=edges, ax=ax, node_size=node_size, arrows=True,
                                   arrowstyle='-|>', arrowsize=max(10, diameter * 0.5), **style)

    # least important first, so the important ones end up on top
    others = sorted((e for e in g.edges() if e != (seed_u, seed_v)), key=lambda e: g.edges[e]['importance'])
    for group, style in (([e for e in others if e not in curved], dict(connectionstyle='arc3,rad=0')),
                         ([e for e in others if e in curved], dict(connectionstyle=f'arc3,rad={rad}'))):
        draw_arrows(group, edge_color=[g.edges[e]['importance'] for e in group], edge_cmap=IMPORTANCE_CMAP,
                    edge_vmin=norm.vmin, edge_vmax=norm.vmax,
                    width=[1.6 + 2.4 * norm(g.edges[e]['importance']) for e in group], **style)

    seed_style = f'arc3,rad={rad}' if (seed_u, seed_v) in curved else 'arc3,rad=0'
    draw_arrows([(seed_u, seed_v)], edge_color=SEED_EDGE_COLOR, width=4.5, connectionstyle=seed_style)

    # "x2" badge where several selected transactions share the same (src, dst) pair
    for u, v, d in g.edges(data=True):
        if d['count'] > 1 and u != v:
            off = _curve_offset(pos[u], pos[v], rad) if (u, v) in curved else (0.0, 0.0)
            ax.text((pos[u][0] + pos[v][0]) / 2 + off[0], (pos[u][1] + pos[v][1]) / 2 + off[1], f"x{d['count']}",
                    fontsize=font_size - 1, ha='center', va='center', zorder=5,
                    bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='0.6', lw=0.5))

    seeds = {seed_u, seed_v}
    regular = [n for n in g.nodes() if n not in seeds]
    nx.draw_networkx_nodes(g, pos, nodelist=regular, ax=ax, node_size=node_size, edgecolors='0.35', linewidths=0.8,
                           node_color=[DETACHED_NODE_COLOR if n in detached else NODE_COLOR for n in regular])
    nx.draw_networkx_nodes(g, pos, nodelist=sorted(seeds), ax=ax, node_size=node_size * 1.15,
                           node_color=SEED_NODE_COLOR, edgecolors=SEED_EDGE_COLOR, linewidths=2.5)
    nx.draw_networkx_labels(g, pos, ax=ax, font_size=font_size)

    handles = [Line2D([], [], color=SEED_EDGE_COLOR, lw=4, label='seed transaction'),
               Line2D([], [], marker='o', ls='', mfc=SEED_NODE_COLOR, mec=SEED_EDGE_COLOR, mew=2, ms=10, label='seed accounts'),
               Line2D([], [], marker='o', ls='', mfc=NODE_COLOR, mec='0.35', ms=10, label='other accounts')]
    if detached:
        handles.append(Line2D([], [], marker='o', ls='', mfc=DETACHED_NODE_COLOR, mec='0.35', ms=10,
                              label='not linked to the seed by the drawn edges'))
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.01), ncol=len(handles),
              fontsize=9, frameon=False)

    ax.set_title(
        f"{'Hetero explanation' if is_hetero else 'Explanation'} for edge {edge_idx} (seed = gold accounts, black arrow)\n"
        f"{n_tx} of {n_sampled} sampled transactions (seed + most important, max {max_edges}) "
        f"between {g.number_of_nodes()} accounts - the classifier's subgraph\n"
        f"Case: {case_label}\n"
        f"Predicted class: {'Fraud' if target == 1 else 'Legitimate'}",
        fontsize=12)

    sm = plt.cm.ScalarMappable(cmap=IMPORTANCE_CMAP, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.8,
                 label='Edge Importance (max of to/rev_to relations)' if is_hetero else 'Edge Importance')

    out_dir = getattr(args, 'explain_plot_dir', 'explanations')
    os.makedirs(out_dir, exist_ok=True)
    plot_path = f"{out_dir}/explanation_{'hetero_' if is_hetero else ''}{case_key}_{edge_idx}.png"
    fig.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    logging.info(f"[{case_key}] Explanation plot saved to {plot_path}")
    return plot_path
