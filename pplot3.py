from typing import Dict, Tuple, Optional, Callable
import argparse, math, os
import pathlib as pl
import igraph as ig
import networkx as nx
import matplotlib.pyplot as plt
from eng import eng_name_style
from extract import extract_from_tei, group_minor_characters_
from renard.graph_utils import cumulative_graph, graph_with_names
from renard.pipeline.character_unification import Character

LANG_TO_NAMESTYLE = {"fra": "most_frequent", "eng": eng_name_style}


def pplot_graph(
    G: nx.Graph,
    layout: Dict[Character, Tuple[float, float]],
    ax=None,
):
    nx.draw_networkx_nodes(
        G,
        layout,
        ax=ax,
        node_color=[degree for _, degree in G.degree],
        cmap="YlOrRd",
        node_size=[1 + degree * 10 for _, degree in G.degree],
        edgecolors="black",
        linewidths=0.5,
    )

    nx.draw_networkx_edges(
        G,
        layout,
        ax=ax,
        edge_color=["royalblue" if e[0] != e[1] else "red" for e in G.edges],
        connectionstyle="arc3,rad=0.2" if isinstance(G, nx.DiGraph) else "arc3",
        width=[1 + math.log(d["weight"]) for _, _, d in G.edges.data()],
        alpha=0.5,
    )

    nx.draw_networkx_labels(
        G,
        pos=layout,
        ax=ax,
        verticalalignment="center",
        font_size=10,
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output-dir", type=pl.Path)
    parser.add_argument("-r", "--group", action="store_true")
    parser.add_argument(
        "-l", "--lang", type=str, help="one of 'fra', 'eng'", default="fra"
    )
    args = parser.parse_args()

    # get the layout from the co-occurrence network
    co_occurrence_out = extract_from_tei(
        "./lorenzaccio.tei.xml",
        "co-occurrence",
        dynamic=True,
        dynamic_window=1,
        dynamic_overlap=0,
    )
    if args.group:
        group_minor_characters_(co_occurrence_out, "co-occurrence")
    G_co_occurrence = cumulative_graph(co_occurrence_out.character_network)[-1]
    G_co_occurrence = graph_with_names(
        G_co_occurrence, name_style=LANG_TO_NAMESTYLE[args.lang]
    )
    g_c = ig.Graph.from_networkx(graph_with_names(G_co_occurrence))
    layout = {
        char: coords
        for char, coords in zip(g_c.vs["_nx_name"], g_c.layout_kamada_kawai())
    }

    mention_out = extract_from_tei(
        "./lorenzaccio.tei.xml",
        "mention",
        dynamic=True,
        dynamic_window=1,
        dynamic_overlap=0,
    )
    if args.group:
        group_minor_characters_(mention_out, "mention")
    G_mention = cumulative_graph(mention_out.character_network)[-1]
    G_mention = graph_with_names(G_mention, name_style=LANG_TO_NAMESTYLE[args.lang])

    conversation_out = extract_from_tei(
        "./lorenzaccio.tei.xml",
        "conversation",
        dynamic=True,
        dynamic_window=1,
        dynamic_overlap=0,
    )
    if args.group:
        group_minor_characters_(conversation_out, "conversation")
    G_conversation = cumulative_graph(conversation_out.character_network)[-1]
    G_conversation = graph_with_names(
        G_conversation, name_style=LANG_TO_NAMESTYLE[args.lang]
    )

    if args.output_dir:
        x_values, y_values = zip(*layout.values())
        x_max = max(x_values)
        x_min = min(x_values)
        margin = (x_max - x_min) * 0.1
        os.makedirs(args.output_dir, exist_ok=True)
        plt.figure(figsize=(6, 5))
        pplot_graph(G_co_occurrence, layout)
        plt.tight_layout()
        plt.xlim(x_min - margin, x_max + margin)
        plt.savefig(f"{args.output_dir}/co_occurrence.pdf")
        plt.clf()
        pplot_graph(G_mention, layout)
        plt.tight_layout()
        plt.xlim(x_min - margin, x_max + margin)
        plt.savefig(f"{args.output_dir}/mention.pdf")
        plt.clf()
        pplot_graph(G_conversation, layout)
        plt.tight_layout()
        plt.xlim(x_min - margin, x_max + margin)
        plt.savefig(f"{args.output_dir}/conversation.pdf")
    else:
        fig = plt.figure(layout="constrained", figsize=(14, 12))
        gs = fig.add_gridspec(2, 4)
        ax_co_occurrence = fig.add_subplot(gs[0, 1:3])
        if args.lang == "eng":
            ax_co_occurrence.set_title("Co-occurrence network (a)")
        else:
            ax_co_occurrence.set_title("Graphe de co-occurrence (a)")
        ax_mention = fig.add_subplot(gs[1, 0:2])
        if args.lang == "eng":
            ax_mention.set_title("Mention network (b)")
        else:
            ax_mention.set_title("Graphe de mention (b)")

        ax_conversation = fig.add_subplot(gs[1, 2:4])
        if args.lang == "eng":
            ax_conversation.set_title("Conversational network (c)")
        else:
            ax_conversation.set_title("Graphe conversationnel (c)")
        pplot_graph(G_co_occurrence, layout, ax=ax_co_occurrence)
        pplot_graph(G_mention, layout, ax=ax_mention)
        pplot_graph(G_conversation, layout, ax=ax_conversation)
        plt.show()
