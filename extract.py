from __future__ import annotations
import argparse
from typing import Dict, List, Literal, Optional, Set, Tuple
from collections import Counter

import itertools as it
import xml.etree.ElementTree as ET
from dataclasses import dataclass
import networkx as nx

from more_itertools import windowed
from renard.pipeline.core import Pipeline, PipelineState, PipelineStep
from renard.pipeline.ner import NEREntity, BertNamedEntityRecognizer
from renard.pipeline.character_unification import Character
from renard.pipeline.tokenization import NLTKTokenizer
from renard.gender import Gender

GraphType = Literal["co-occurrence", "mention", "conversation"]


def case_variations(aliases: set[str]) -> set[str]:
    return aliases.union({alias.upper() for alias in aliases})


LORENZACCIO_CHAR_ALIASES = {
    "le-duc": {"Le Duc", "Alexandre", "Alexandre de Médicis"},
    "lorenzo": {"Lorenzo", "Lorenzaccio", "Lorenzo de Médicis", "Renzo"},
    "giomo": {"Giomo"},
    "maffio": {"Maffio"},
    "l-orfevre": {"L'Orfevre", "Mondella"},
    "le-marchand-de-soieries": {"Le Marchand de Soieries"},
    "le-marchand": {"Le Marchand"},
    "le-provediteur": {
        "Le Provediteur",
        "Roberto Corsini",
        "Corsini",
        "seigneur Corsini",
    },
    "salviati": {"Salviati", "Julien Salviati"},
    "julien": {"Julien"},
    "louise": {"Louise", "Louise Strozzi"},
    "le-marquis": {"Le Marquis"},
    "la-marquise": {"La Marquise", "Ricciarda", "Ricciarda Cibo"},
    "le-cardinal-cibo": {"Malaspina", "Cardinal Cibo"},
    "ascanio": {"Ascanio"},
    "agnolo": {"Agnolo"},
    "valori": {"Valori", "Baccio"},
    "sire-maurice": {"Sire Maurice", "Maurice"},
    "le-prieur": {"Le Prieur", "Léon"},
    "catherine": {"Catherine", "Cattina"},
    "marie": {"Marie"},
    "philippe": {"Philippe", "Philippe Strozzi"},
    "pierre": {"Pierre", "Pierre Strozzi"},
    "tebaldeo": {"Tebaldeo"},
    "bindo": {"Bindo"},
    "venturi": {"Venturi"},
    "scoronconcolo": {"Scoronconcolo"},
    "thomas": {"Thomas", "Thomas Strozzi", "Masaccio"},
    "guicciardini": {"Guicciardini"},
    "vettori": {"Vettori"},
    "capponi": {"Capponi"},
    "acciaiuoli": {"Acciaiuoli"},
    "ruccellai": {"Ruccellai", "Palla Ruccellai"},
    "canigiani": {"Canigiani"},
    "corsi": {"Corsi"},
    "come": {"Côme", "Côme de Médicis"},
}


@dataclass
class Scene:
    #: each tuple is of the form (speaker_id, content)
    utterances: List[Tuple[str, str]]


@dataclass
class Act:
    scenes: List[Scene]


@dataclass
class Play:
    acts: List[Act]
    # { character_id => character }
    characters: Dict[str, Character]

    def scenes(self) -> List[Scene]:
        return [scene for act in self.acts for scene in act.scenes]


class PlayTEIParser(PipelineStep):
    def __init__(self) -> None:
        super().__init__()

    def __call__(
        self,
        text: str,
        character_aliases: Optional[Dict[str, Set[str]]] = None,
        merge_characters: Optional[List[List[str]]] = None,
        **kwargs,
    ) -> dict:
        root = ET.fromstring(text)

        body = root.find(".//body")
        assert not body is None

        act_nodes = body.findall("./div[@type='act']") + body.findall(
            "./div[@type='acte']"
        )
        acts = [self._act_from_node(node) for node in act_nodes]

        list_person_node = root.find(".//listPerson")
        assert not list_person_node is None
        characters = self._characters_from_node(list_person_node, character_aliases)

        return {"play": Play(acts, characters)}

    def _characters_from_node(
        self,
        list_person_node: ET.Element,
        character_aliases: Optional[Dict[str, Set[str]]],
    ) -> Dict[str, Character]:
        char_nodes = list_person_node.findall("./person")
        characters = {}
        for char_node in char_nodes:
            char_id = char_node.attrib["{http://www.w3.org/XML/1998/namespace}id"]
            char_sex = char_node.attrib["sex"]
            char_gender = self._gender_from_tei_sex(char_sex)
            char_name_nodes = char_node.findall("./persName")
            names = frozenset([node.text for node in char_name_nodes])
            if not character_aliases is None:
                names = names.union(frozenset(character_aliases.get(char_id, set())))
            characters[char_id] = Character(frozenset(names), [], char_gender)
        return characters

    def _gender_from_tei_sex(self, sex: str) -> Gender:
        if sex == "MALE":
            return Gender.MALE
        elif sex == "FEMALE":
            return Gender.FEMALE
        else:
            return Gender.UNKNOWN

    def _act_from_node(self, act_node: ET.Element) -> Act:
        scene_nodes = act_node.findall("./div[@type='scene']")
        scenes = [self._scene_from_node(node) for node in scene_nodes]
        return Act(scenes)

    def _scene_from_node(self, scene_node: ET.Element) -> Scene:
        utterances = []
        for ut_node in scene_node.findall("./sp"):
            speaker_id = ut_node.attrib["who"][1:]
            content = []
            for s_node in ut_node.findall(".//s"):
                content.append(s_node.text)
            content = "".join(content)
            utterances.append((speaker_id, content))
        return Scene(utterances)

    def needs(self) -> Set[str]:
        return {"text"}

    def optional_needs(self) -> Set[str]:
        return {"character_aliases"}

    def production(self) -> Set[str]:
        return {"play"}

    def supported_langs(self) -> str:
        return "any"


class PlayNamedEntityRecognizer(PipelineStep):
    def __init__(self, tokenization_step: PipelineStep, ner_step: PipelineStep) -> None:
        self.tokenization_step = tokenization_step
        self.ner_step = ner_step
        self._pipeline = Pipeline(
            [tokenization_step, ner_step], progress_report=None, warn=False
        )

    def _pipeline_init_(self, lang: str, **kwargs):
        super()._pipeline_init_(lang, **kwargs)

    def __call__(self, play: Play, **kwargs) -> dict:
        scenes_entities = []
        for scene in self._progress_(play.scenes()):
            scenes_entities.append([])
            for speaker, speech in scene.utterances:
                out = self._pipeline(speech)
                scenes_entities[-1].append(out.entities)
        return {"scenes_entities": scenes_entities}

    def needs(self) -> Set[str]:
        return {"play"}

    def production(self) -> Set[str]:
        return {"scenes_entities"}

    def supported_langs(self) -> Set[str]:
        tokenization_langs = self.tokenization_step.supported_langs()
        ner_langs = self.ner_step.supported_langs()
        if tokenization_langs == "any":
            return ner_langs
        if ner_langs == "any":
            return tokenization_langs
        return tokenization_langs.intersection(ner_langs)


class PlayGraphExtractor(PipelineStep):
    def __init__(
        self,
        graph_type: GraphType,
        dynamic: bool = False,
        dynamic_window: Optional[int] = None,
        dynamic_overlap: int = 0,
    ) -> None:
        """
        :param graph_type: the type of the graph to extract.  With
            ``"co-occurrence"``, extract a graph where characters interact
            if they talk in the same scene.  With ``"mention"``,
            extract a graph two characters interact if one talk about
            the other.  With ``"conversation"``, extract a
            conversational network.

        :param dynamic: if ``False``, extract a single static
            :class:`nx.Graph`.  If ``True``, several :class:`nx.Graph`
            are extracted, and ``dynamic_window`` and
            ``dynamic_overlap`` can be specified.

        :param dynamic_window: dynamic window, in number of scenes.  A
            dynamic window of `n` means that each returned graph will
            be formed by `n` scenes.

        :param dynamic_overlap: overlap, in number of scenes.
        """
        self.graph_type = graph_type

        if dynamic:
            assert not dynamic_window is None
            assert dynamic_window > dynamic_overlap
            assert dynamic_overlap >= 0
        self.dynamic = dynamic
        self.dynamic_window = dynamic_window
        self.dynamic_overlap = dynamic_overlap

    def __call__(
        self,
        play: Play,
        scenes_entities: Optional[List[List[List[NEREntity]]]] = None,
        **kwargs,
    ) -> dict:
        graphs = []

        if self.graph_type == "mention":
            assert not scenes_entities is None
            assert len(scenes_entities) == len(play.scenes())

        if self.dynamic:
            assert not self.dynamic_window is None
            graphs = []

            for act_i, act in enumerate(play.acts):
                act_start = sum(len(prev_act.scenes) for prev_act in play.acts[:act_i])
                act_end = act_start + len(act.scenes)

                for scene_indices in windowed(
                    range(len(act.scenes)),
                    self.dynamic_window,
                    step=self.dynamic_window - self.dynamic_overlap,
                ):
                    scene_indices = [i for i in scene_indices if not i is None]
                    scenes = [act.scenes[i] for i in scene_indices]

                    if self.graph_type == "mention":
                        assert not scenes_entities is None
                        act_scenes_entities = scenes_entities[act_start:act_end]
                        scene_entities = [act_scenes_entities[i] for i in scene_indices]
                    else:
                        scene_entities = None

                    G = self._extract_graph(play, scenes, scene_entities)
                    G.graph["act"] = act_i
                    graphs.append(G)

            characters = set()
            for G in graphs:
                for n in G.nodes:
                    characters.add(n)

            return {"character_network": graphs, "characters": list(characters)}

        else:
            G = self._extract_graph(play, play.scenes(), scenes_entities)
            return {"character_network": G, "characters": list(G.nodes)}

    def _extract_graph(
        self,
        play: Play,
        scenes: List[Scene],
        scenes_entities: Optional[List[List[List[NEREntity]]]],
    ) -> nx.Graph:
        if self.graph_type == "co-occurrence":
            G = nx.Graph()
        elif self.graph_type in ("mention", "conversation"):
            G = nx.DiGraph()
        else:
            raise ValueError(self.graph_type)

        for scene in scenes:
            if self.graph_type == "co-occurrence":
                speakers = set([play.characters[ut[0]] for ut in scene.utterances])
                for char in speakers:
                    G.add_node(char)
                for c1, c2 in it.combinations(speakers, 2):
                    if (c1, c2) in G.edges:
                        G.edges[(c1, c2)]["weight"] += 1
                    else:
                        G.add_edge(c1, c2, weight=1)

            elif self.graph_type == "conversation":
                # we suppose each speaker is talking to all the other
                # characters present in the scene
                speakers = set([play.characters[ut[0]] for ut in scene.utterances])
                for char in speakers:
                    G.add_node(char)
                for speaker_id, _ in scene.utterances:
                    speaker = play.characters[speaker_id]
                    for listener in speakers - {speaker}:
                        if (speaker, listener) in G.edges:
                            G.edges[(speaker, listener)]["weight"] += 1
                        else:
                            G.add_edge(speaker, listener, weight=1)

            elif self.graph_type == "mention":

                assert not scenes_entities is None
                assert len(scenes) == len(scenes_entities)

                for scene, scene_entities in zip(scenes, scenes_entities):

                    for ut, ut_entities in zip(scene.utterances, scene_entities):

                        speaker, _ = ut
                        speaker = play.characters[speaker]
                        # this would add speaker mentionning no-one
                        # G.add_node(speaker)

                        char_entities = [e for e in ut_entities if e.tag == "PER"]
                        mentions = [" ".join(ent.tokens) for ent in char_entities]
                        mentioned_characters = set()
                        for mention in mentions:
                            char_id = self._match_mention_to_character(
                                mention, play.characters
                            )
                            if char_id is None:
                                continue
                            character = play.characters[char_id]
                            mentioned_characters.add(character)

                        for mc in mentioned_characters:
                            if (speaker, mc) in G.edges:
                                G.edges[(speaker, mc)]["weight"] += 1
                            else:
                                G.add_edge(speaker, mc, weight=1)

        return G

    def _match_mention_to_character(
        self, mention: str, characters: Dict[str, Character]
    ) -> Optional[str]:
        # TODO: naive
        for character_id, character in characters.items():
            if mention in character.names:
                return character_id
        return None

    def needs(self) -> Set[str]:
        return {"play"}

    def production(self) -> Set[str]:
        return {"character_network", "characters"}

    def supported_langs(self) -> str:
        return "any"


def extract_from_tei(
    tei_path: str,
    graph_type: GraphType,
    dynamic: bool,
    dynamic_window: Optional[int] = None,
    dynamic_overlap: int = 0,
) -> PipelineState:
    if graph_type in ("co-occurrence", "conversation"):
        pipeline = Pipeline(
            [
                PlayTEIParser(),
                PlayGraphExtractor(
                    graph_type=graph_type,  # type: ignore
                    dynamic=dynamic,
                    dynamic_window=dynamic_window,
                    dynamic_overlap=dynamic_overlap,
                ),
            ],
            lang="fra",
        )
    else:
        pipeline = Pipeline(
            [
                PlayTEIParser(),
                PlayNamedEntityRecognizer(
                    tokenization_step=NLTKTokenizer(),
                    ner_step=BertNamedEntityRecognizer(),
                ),
                PlayGraphExtractor(
                    graph_type=graph_type,  # type: ignore
                    dynamic=dynamic,
                    dynamic_window=dynamic_window,
                    dynamic_overlap=dynamic_overlap,
                ),
            ],
            lang="fra",
        )

    with open(tei_path) as f:
        xml = f.read()
    out = pipeline(text=xml, character_aliases=LORENZACCIO_CHAR_ALIASES)
    # fix some issues in the TEI file
    merge_characters_(out, "Julien Salviati", ["Julien", "Julien Salviati"], graph_type)
    merge_characters_(
        out,
        "Le Marchand de Soieries",
        ["Le Marchand", "Le Marchand de Soieries"],
        graph_type,
    )
    try:
        delete_character_(out, "Les Huit")
    except ValueError:
        pass

    return out


def delete_character_(out: PipelineState, name: str):
    character = out.get_character(name)
    if character is None:
        raise ValueError(f"unkown character for name: {character}")
    out.characters.remove(character)
    for G in out.character_network:
        try:
            G.remove_node(character)
        except nx.exception.NetworkXError:
            continue


def merge_characters_(
    out: PipelineState,
    new_name: str,
    names: List[str],
    graph_type: GraphType,
):
    assert len(names) > 0
    assert graph_type in ("mention", "co-occurrence", "conversation")

    characters = [
        c for c in {out.get_character(name) for name in names} if not c is None
    ]
    if all(not c in out.characters for c in characters):
        print(f"merge_characters_: nothing to do for {new_name}.")
        return

    newc = Character(
        [new_name],
        list(it.accumulate([c.mentions for c in characters]))[-1],
        (
            characters[0].gender
            if len(set(c.gender for c in characters)) == 1
            else Gender.UNKNOWN
        ),
    )

    for G in out.character_network:

        # NOTE: in_edge used by default for graph_type == "co-occurrence"
        in_edges = {}  # { neighbor => weight }
        out_edges = {}  # { neighbor => weight }

        for c in characters:
            if not c in G.nodes:
                continue
            if graph_type == "co-occurrence":
                for neighbor in G.neighbors(c):
                    if neighbor in characters:
                        continue
                    # NOTE: this is an underestimated approximation of
                    # interactions with merged characters. Hopefully it is
                    # pretty close to reality, as merged characters tend to
                    # appear together.
                    in_edges[neighbor] = max(
                        in_edges.get(neighbor, 0), G.edges[c, neighbor]["weight"]
                    )
            elif graph_type in ("conversation", "mention"):
                for successor in G.successors(c):
                    if successor in characters:
                        continue
                    out_edges[successor] = (
                        out_edges.get(successor, 0) + G.edges[c, successor]["weight"]
                    )
                for predecessor in G.predecessors(c):
                    if predecessor in characters:
                        continue
                    in_edges[predecessor] = (
                        in_edges.get(predecessor, 0) + G.edges[predecessor, c]["weight"]
                    )

        for c in characters:
            if c in G.nodes:
                G.remove_node(c)

        if graph_type == "co-occurrence":
            for neighbor, weight in in_edges.items():
                G.add_edge(newc, neighbor, weight=weight)
        elif graph_type in ("mention", "conversation"):
            for neighbor, weight in in_edges.items():
                G.add_edge(neighbor, newc, weight=weight)
            for neighbor, weight in out_edges.items():
                G.add_edge(newc, neighbor, weight=weight)

    for c in characters:
        out.characters.remove(c)
    out.characters.append(newc)


def group_minor_characters_(out: PipelineState, graph_type: GraphType):

    merge_characters_(
        out, "Les Étudiants", ["Un Autre Etudiant", "L Etudiant"], graph_type
    )
    merge_characters_(
        out,
        "Les Bannis",
        [
            "Une Voix",
            "Premier Banni",
            "Deuxieme Banni",
            "Second Banni",
            "Le Deuxieme Banni",
            "Troiseme Banni",
            "Troisieme Banni",
            "Quatrieme Banni",
            "Une Autre Banni",
            "Tous les Bannis",
        ],
        graph_type,
    )
    merge_characters_(
        out, "Les Convives", ["Convive", "Les Convives", "Un Convive"], graph_type
    )
    merge_characters_(
        out,
        "Deux Gentilhommes",
        ["Premier Gentilhomme", "Deuxieme Gentilhomme"],
        graph_type,
    )
    merge_characters_(
        out, "Deux Écoliers", ["Premier Ecolier", "Second Ecolier"], graph_type
    )
    merge_characters_(
        out,
        "Les Bourgeois",
        [
            "Le Bourgeois",
            "Premier Bourgeois",
            "Un des Bourgeois",
            "Second Bourgeois",
            "Deuxieme Bourgeois",
            "Le Deuxieme Bourgeois",
        ],
        graph_type,
    )
    merge_characters_(
        out, "Deux Cavaliers", ["Le Premier Cavalier", "Un Autre Cavalier"], graph_type
    )
    merge_characters_(out, "Deux Dames", ["Premiere Dame", "Deuxieme Dame"], graph_type)
    merge_characters_(out, "Les Soldats", ["Un Soldat", "Les Soldats"], graph_type)
    merge_characters_(
        out,
        "Deux Précepteurs",
        ["Premier Precepteur", "Deuxieme Precepteur"],
        graph_type,
    )
    merge_characters_(out, "Le Peuple", ["Un Homme du Peuple", "Le Peuple"], graph_type)
    merge_characters_(
        out,
        "Les Courtisans",
        ["Le Seigneur", "Plusieurs Seigneurs", "Les Courtisans"],
        graph_type,
    )


def char_name_style(char: Character) -> str:
    c = Counter([" ".join(mention.tokens) for mention in char.mentions])
    c = {c: count for c, count in c.items() if c in char.names and not c.isupper()}
    if len(c) == 0:
        name = char.longest_name()
    else:
        name = max(c, key=c.get)  # type: ignore
    return " ".join([elt.capitalize() for elt in name.split(" ")])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-i", "--input-file", type=str, default="./lorenzaccio.tei.xml")
    parser.add_argument("-o", "--output-file", type=str, default="./lorenzaccio.gexf")
    parser.add_argument(
        "-g",
        "--graph-type",
        type=str,
        default="co-occurrence",
        help="one of: 'co-occurrence', 'mention', 'conversation'",
    )
    parser.add_argument("-d", "--dynamic", action="store_true")
    parser.add_argument("-w", "--dynamic-window", type=int, default=None)
    parser.add_argument("-v", "--dynamic-overlap", type=int, default=0)
    parser.add_argument("-r", "--group-minor-characters", action="store_true")
    args = parser.parse_args()

    out = extract_from_tei(
        args.input_file,
        args.graph_type,
        args.dynamic,
        args.dynamic_window,
        args.dynamic_overlap,
    )
    if args.group_minor_characters:
        group_minor_characters_(out, args.graph_type)
    out.export_graph_to_gexf(args.output_file, name_style=char_name_style)
