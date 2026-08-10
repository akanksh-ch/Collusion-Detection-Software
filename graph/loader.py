"""
This file takes in multiple submission root directories and spits out submission paths. Files or Directories.
"""

import glob
import os
import re
import networkx as nx

# Joern's export.dot files only ever contain two statement shapes: a quoted node-id
# with a bracketed attr list, or a quoted edge with a bracketed attr list. That
# regularity is what lets a small parser stand in for pydot/pyparsing below.
_NODE_RE = re.compile(r'^"(?P<id>[^"]+)"\s*\[(?P<attrs>.*)\]$', re.DOTALL)
_EDGE_RE = re.compile(r'^"(?P<src>[^"]+)"\s*->\s*"(?P<dst>[^"]+)"\s*\[(?P<attrs>.*)\]$', re.DOTALL)
_ATTR_RE = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"', re.DOTALL)  # handles \" and \\ escapes inside values


def _split_statements(text: str):
    # scanning: walk the text once, tracking whether we're inside a "..." string (honoring \" escapes),
    # so a CODE attribute's embedded newlines/brackets/semicolons never get mistaken for real syntax
    start = 0
    in_string = False
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if in_string:
            if c == "\\":
                i += 2  # skip escaped char, it can't end the string
                continue
            if c == '"':
                in_string = False
        else:
            if c == '"':
                in_string = True
            elif c == ";":
                stmt = text[start:i].strip()
                if stmt:
                    yield stmt
                start = i + 1
        i += 1

    tail = text[start:].strip()
    if tail:
        yield tail


def _parse_attrs(attr_str: str) -> dict:
    # unescaping: DOT escapes literal quotes/backslashes inside quoted values as \" and \\
    return {
        k: v.replace('\\"', '"').replace('\\\\', '\\')
        for k, v in _ATTR_RE.findall(attr_str)
    }


def _fast_read_dot(dot_file: str) -> nx.MultiDiGraph:
    # reading: pydot's grammar-rebuild cost dominates on Joern's per-method dot files (tiny files, huge
    # call count), and its parser isn't thread-safe. This has no shared state, so no lock is needed
    with open(dot_file, "r", encoding="utf-8") as f:
        text = f.read().strip()

    # unwrapping: strip the outer "digraph { ... }" — it isn't ';'-terminated, so it has to come off before statement splitting
    if text.startswith("digraph"):
        text = text[text.index("{") + 1:]
    if text.endswith("}"):
        text = text[:-1]

    # building: match each statement as either an edge or a node, and fail loudly on anything else instead of silently dropping data
    g = nx.MultiDiGraph()
    for stmt in _split_statements(text):
        m = _EDGE_RE.match(stmt)
        if m:
            g.add_edge(m["src"], m["dst"], **_parse_attrs(m["attrs"]))
            continue
        m = _NODE_RE.match(stmt)
        if m:
            g.add_node(m["id"], **_parse_attrs(m["attrs"]))
            continue
        raise ValueError(f"Unrecognized dot statement in {dot_file}: {stmt!r}")

    return g


def load_graph(path: str) -> nx.MultiGraph:

    dot_files = sorted(glob.glob(os.path.join(path, "**", "export.dot"), recursive=True))
    if not dot_files:
        raise FileNotFoundError(f"No export.dot files found under {path}")

    combined = nx.MultiGraph()
    node_offset = 0

    for dot_file in dot_files:
        g = _fast_read_dot(dot_file)

        if g.number_of_nodes() == 0:
            # empty method graph (e.g. an implicit _init_ with no body) —
            # legitimately contributes nothing, not a parse failure
            continue

        sorted_nodes = sorted(g.nodes())
        mapping = {old: node_offset + i for i, old in enumerate(sorted_nodes)}
        g = nx.relabel_nodes(g, mapping, copy=True)

        # force to MultiGraph explicitly — to_undirected() on a MultiDiGraph
        # stays a MultiGraph too, but being explicit avoids relying on that
        ug = nx.MultiGraph(g)

        combined.add_nodes_from(sorted(ug.nodes(data=True)))
        combined.add_edges_from(ug.edges(data=True))

        node_offset += g.number_of_nodes()

    return combined
