from typing import Dict, Generator, List, Any, Optional
import argparse, os
from collections import defaultdict, Counter
import pathlib as pl
import itertools as it
import networkx as nx
import igraph as ig
from igraph.drawing.colors import Palette
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from renard.pipeline.core import PipelineState
from renard.graph_utils import cumulative_graph
from comsplot import igraph_plot, zp_scatter, CategoricalPalette
from extract import extract_from_tei, group_minor_characters_


# HACK: community colors are inconsistent across igraph versions, so
# we force them with hints.
_5CLIQUE_COLOR_HINTS = [
    ("Agnolo", (0.0, 1.0, 1.0, 1.0)),
    ("Tebaldeo", (0.0, 1.0, 0.0, 1.0)),
    ("Catherine", (1.0, 0.0, 1.0, 1.0)),
    ("Philippe Strozzi", (0.0, 0.0, 1.0, 1.0)),
    ("Les Soldats", (1.0, 0.0, 0.0, 1.0)),
    ("Le Petit Salviati", (1.0, 1.0, 0.0, 1.0)),
]
_KCLIQUE_COLOR_HINTS = {5: _5CLIQUE_COLOR_HINTS}


def groupby_act_(out: PipelineState):
    graphs = defaultdict(list)
    for G in out.character_network:
        graphs[G.graph["act"]].append(G)
    out.character_network = [cumulative_graph(graphs[k])[-1] for k in sorted(graphs)]


def annotate_kclique_coms_(G: nx.Graph, k: int) -> List[set]:
    partitions = list(nx.community.k_clique_communities(G, k))

    def get_coms(node) -> List[int]:
        coms = []
        for i, p in enumerate(partitions):
            if node in p:
                coms.append(i)
        return coms

    for node in G.nodes:
        G.nodes[node]["dcoms"] = get_coms(node)

    return partitions


def dynplot_kclique_coms(
    graphs: List[nx.Graph], coms: Dict[Any, List[int]], **kwargs
) -> Generator:
    for G in graphs:
        for node, dcoms in coms.items():
            try:
                G.nodes[node]["dcoms"] = dcoms
            except KeyError:
                continue

    for G in graphs:
        _, ax = plt.subplots(figsize=(15, 12))
        igraph_plot(G, ax=ax, **kwargs)
        yield ax
        plt.clf()


def plot_coms_in_time(
    graphs: List[nx.Graph],
    coms: Dict[Any, List[int]],
    com_palette: Optional[Palette] = None,
):
    all_coms = set()
    for _, ccoms in coms.items():
        for com in ccoms:
            all_coms.add(com)

    # { com => [act_1_presence, ..., act_n_presence] }
    coms_presence = {k: [] for k in all_coms}

    for graph in graphs:
        d = {k: 0 for k in all_coms}
        for node in graph.nodes:
            for com in coms[node]:
                d[com] += 1
        total_presence = sum(d.values())
        for k, c in d.items():
            coms_presence[k].append(c / total_presence)

    if com_palette is None:
        com_palette = ig.RainbowPalette(n=len(all_coms))
    _, ax = plt.subplots(figsize=(15, 8))
    ax.stackplot(
        [1 + i for i in range(len(graphs))],
        coms_presence.values(),
        colors=[com_palette.get(comid) for comid in coms_presence.keys()],  # type: ignore
    )
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Acte")
    ax.set_ylabel("Influence Normalisée")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output-dir", type=pl.Path)
    parser.add_argument(
        "-g",
        "--graph-type",
        type=str,
        default="co-occurrence",
        help="One of 'co-occurrence', 'mention'",
    )
    parser.add_argument("-r", "--group", action="store_true")
    parser.add_argument("-k", type=int)
    args = parser.parse_args()

    matplotlib.rc("font", size=14)

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    # extract graphs
    # --------------
    if args.graph_type == "co-occurrence":
        out = extract_from_tei(
            "./lorenzaccio.tei.xml",
            "co-occurrence",
            dynamic=True,
            dynamic_window=1,
            dynamic_overlap=0,
        )
        groupby_act_(out)
    else:
        out = extract_from_tei(
            "./lorenzaccio.tei.xml",
            "mention",
            dynamic=True,
            dynamic_window=1,
            dynamic_overlap=0,
        )
        groupby_act_(out)

    if args.group:
        group_minor_characters_(out, args.graph_type)

    if args.graph_type == "mention":
        # not supported by networkx kclique community detection
        out.character_network = [G.to_undirected() for G in out.character_network]

    G = cumulative_graph(out.character_network)[-1]
    partitions = annotate_kclique_coms_(G, args.k)
    g_c = ig.Graph.from_networkx(G)
    layout = {
        char: coords
        for char, coords in zip(g_c.vs["_nx_name"], g_c.layout_kamada_kawai())
    }
    coms = {node: G.nodes[node]["dcoms"] for node in G.nodes}
    com_palette = CategoricalPalette.with_hints(G, _KCLIQUE_COLOR_HINTS.get(args.k, []))

    # plot static graph communities
    # -----------------------------
    fig, ax = plt.subplots(figsize=(15, 12))
    igraph_plot(G, ax=ax, layout_dict=layout, com_palette=com_palette)
    plt.tight_layout()
    if args.output_dir:
        plt.savefig(args.output_dir / "coms.pdf")
    else:
        plt.show()

    # zp plot
    # -------
    zp_scatter(G, partitions)
    if args.output_dir:
        plt.savefig(args.output_dir / "zp.pdf")
    else:
        plt.show()

    # plot each act graph
    # -------------------
    for i, _ in enumerate(
        dynplot_kclique_coms(
            out.character_network, coms, layout_dict=layout, com_palette=com_palette
        )
    ):
        if args.output_dir:
            plt.savefig(args.output_dir / f"coms_act{i+1}.pdf")
        else:
            plt.show()

    # plot communities in time
    # ------------------------
    plot_coms_in_time(out.character_network, coms, com_palette=com_palette)
    if args.output_dir:
        plt.savefig(args.output_dir / "coms_in_time.pdf")
    else:
        plt.show()
