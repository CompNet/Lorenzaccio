import random
import itertools as it
from typing import Optional, Dict, List, TypeVar, Tuple, Callable
from statistics import mean, stdev
import igraph as ig
from igraph.drawing.colors import Palette
import networkx as nx
import matplotlib.pyplot as plt
from adjustText import adjust_text
from renard.pipeline.character_unification import Character


def get_all_coms(graphs: List[nx.Graph]) -> List[int]:
    all_coms = set()
    for G in graphs:
        for node in G.nodes:
            for com in G.nodes[node].get("dcoms", []):
                all_coms.add(com)
    return list(sorted(all_coms))


T = TypeVar("T")
K = TypeVar("K")


def alist_get(alist: List[Tuple[K, T]], key: K) -> T:
    for akey, avalue in alist:
        if akey == key:
            return avalue
    raise IndexError


class CategoricalPalette(Palette):
    def __init__(self, coms: List[int]) -> None:
        super().__init__(len(coms))
        if len(coms) > 0:
            rp = ig.RainbowPalette(n=len(coms))
            self._colors = [(com, rp._get(i)) for i, com in enumerate(coms)]
        else:
            self._colors = []

    def _get(self, v: int) -> tuple:
        return self._colors[v][1]

    def subpalette(self, coms: List[int]) -> "CategoricalPalette":
        # some useless computations but that's fine
        cp = CategoricalPalette(coms)
        cp._colors = [(com, alist_get(self._colors, com)) for com in coms]
        return cp

    @staticmethod
    def with_hints(
        G: nx.Graph,
        hints: List[Tuple[str, Tuple[float, float, float, float]]],
    ) -> "CategoricalPalette":
        """
        :param G: a graph, annotated with :func:`annotate_kclique_coms_`
        :param hints: a list of tuple of the form (NAME COLOR)
        :return: a palette that hopefully respects ``hints``
        """
        coms = get_all_coms([G])
        palette = CategoricalPalette(coms)
        for name, color in hints:
            for node in G.nodes:
                if name in node.names:
                    char_coms = G.nodes[node]["dcoms"]
                    palette._colors[char_coms[0]] = (char_coms[0], color)
                    break
        return palette


def igraph_plot(
    G: nx.Graph,
    ax=None,
    com_palette: Optional[CategoricalPalette] = None,
    layout_dict: Optional[Dict[Character, List[int]]] = None,
    name_style: Optional[Callable[[Character], str]] = None,
):
    if ax is None:
        _, ax = plt.subplots()

    all_coms = get_all_coms([G])

    if com_palette is None:
        com_palette = ig.RainbowPalette(n=len(all_coms))

    # when converting from nx to igraph, node types are changed. In
    # igraph, it seems nodes can only be int. The nx node itself is
    # converted to the node attribute "_nx_name".
    g = ig.Graph.from_networkx(G)

    if name_style is None:
        name_style = lambda char: char.most_frequent_name()
    g.vs["label"] = [name_style(char) for char in g.vs["_nx_name"]]

    if layout_dict:
        layout = ig.layout.Layout(coords=[layout_dict[v["_nx_name"]] for v in g.vs])
    else:
        layout = g.layout_kamada_kawai()

    if len(all_coms) == 0:
        ig.plot(g, layout=layout, target=ax)
        return

    clusters = [[] for _ in all_coms]
    for v_i, v in enumerate(g.vs):
        character = v["_nx_name"]
        coms = G.nodes[character].get("dcoms", [])
        for com in coms:
            com_i = all_coms.index(com)
            clusters[com_i].append(v_i)
    cover = ig.VertexCover(g, clusters)

    # see
    # https://python.igraph.org/en/0.11.6/api/igraph.Graph.html#__plot__
    # for all options
    ig.plot(
        cover,
        layout=layout,
        target=ax,
        vertex_size=15,
        # COMPLETE HACK! ig.plot does not take 'palette' into account
        # in that case (why?). So we retorts to
        # 'mark_groups'. However, 'mark_groups' should take a dict but
        # call iteritems() on it (not good in python 3), so using
        # 'items()' change the handling by
        # 'igraph.clustering._handle_mark_groups_arg_for_clustering'
        mark_groups={i: com_palette._get(i) for i, _ in enumerate(all_coms)}.items(),  # type: ignore
    )


def zp_scatter(
    G: nx.Graph,
    coms: List[set],
    name_style: Optional[Callable[[Character], str]] = None,
):
    if name_style is None:
        name_style = lambda c: c.most_frequent_name()

    degrees = {n: d for n, d in G.degree}

    labels = []
    comids = []
    pscores = []
    zscores = []

    for com_i, com in enumerate(coms):

        for node in com:
            labels.append(name_style(node))
            comids.append(com_i)

        # pscores
        for node in com:
            s = 0
            for ocom in coms:
                s += (
                    len([n for n in G.neighbors(node) if n in ocom]) / degrees[node]
                ) ** 2
            pscores.append(1 - s)

        # zscore
        kscores = []
        for node in com:
            kscores.append(len([n for n in G.neighbors(node) if n in com]))
        mean_k = mean(kscores)
        stdev_k = stdev(kscores)
        zscores += [(k - mean_k) / stdev_k if stdev_k > 0 else 0 for k in kscores]

    com_palette = ig.RainbowPalette(n=len(coms))
    fig, ax = plt.subplots(figsize=(15, 12))
    ax.scatter(pscores, zscores, c=[com_palette.get(comid) for comid in comids])
    texts = []
    for label, x, y in zip(labels, pscores, zscores):
        texts.append(ax.text(x, y, label))
    adjust_text(texts, arrowprops={"arrowstyle": "->"}, force_text=(0.01, 0.01))

    ax.set_xlabel("Participation coefficient P")
    ax.set_ylabel("Within-module degree z")
    plt.legend()


def zp_scatter_star(
    G: nx.Graph,
    coms: List[set],
    name_style: Optional[Callable[[Character], str]] = None,
    lang: str = "fra",
):
    if name_style is None:
        name_style = lambda c: c.most_frequent_name()

    labels = []
    pscores = []
    zscores = []

    def com_degree(node, com: set) -> int:
        return len([n for n in G.neighbors(node) if n in com])

    for node in G.nodes:

        # find the node community, which is the union of all the
        # community he belongs to
        node_com = set()
        for com in coms:
            if node in com:
                node_com = node_com.union(com)

        if len(node_com) == 0:
            zscores.append(0)
        else:
            k = len([n for n in G.neighbors(node) if n in node_com])
            # compute number of links for nodes in node_com
            kscores = []
            for onode in node_com:
                kscores.append(len([n for n in G.neighbors(onode) if n in node_com]))
            # compute zscore
            mean_k = mean(kscores)
            stdev_k = stdev(kscores)
            zscores.append((k - mean_k) / stdev_k if stdev_k > 0 else 0)

        s = 0
        for com in coms:
            s += (
                com_degree(node, com) / sum(com_degree(node, ocom) for ocom in coms)
            ) ** 2
        pscores.append(1 - s)

        labels.append(name_style(node))

    fig, ax = plt.subplots(figsize=(15, 12))
    ax.scatter(pscores, zscores)

    tested_pos = set()

    def find_textpos(
        pos: Tuple[float, float], xoff: float, yoff: float
    ) -> Tuple[float, float]:

        if not pos in tested_pos:
            tested_pos.add(pos)
            return pos

        return find_textpos(
            random.choice(
                list(
                    it.product(
                        [pos[0], pos[0] + xoff, pos[0] - xoff],
                        [pos[1], pos[1] + yoff, pos[1] - yoff],
                    )
                )
            ),
            xoff,
            yoff,
        )

    for label, x, y in zip(labels, pscores, zscores):
        textpos = find_textpos((x, y + 0.05), 0.0, 0.1)
        ax.annotate(
            label,
            (x, y),
            textpos,
            ha="center",
            # arrowprops={"arrowstyle": "-"} if textpos != (x, y + 0.1) else None,
        )

    if lang == "fra":
        ax.set_xlabel("Coefficient de participation $P^*$")
        ax.set_ylabel("Degré intra-module $z^*$")
    elif lang == "eng":
        ax.set_xlabel("Participation coefficient $P^*$")
        ax.set_ylabel("Intra-module degree $z^*$")
    plt.legend()
