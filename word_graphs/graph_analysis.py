"""
Functions for analyzing word graphs and locating subgraphs 
using NetworkX.

Functions:

    calculate_greatest_unconnected_distance, get_minimal_external_paths,
    find_minimal_external_paths
"""

import networkx as nx

from .io import retrieve_word_graph, store_external_paths


def calculate_greatest_unconnected_distance(weak_graph, strong_graph):
    if type(weak_graph) == nx.DiGraph:
        weak_graph = nx.Graph(weak_graph)
    if type(strong_graph) == nx.DiGraph:
        strong_graph = nx.Graph(strong_graph)

    max_distance = 0
    for vertex1 in strong_graph:
        for vertex2 in strong_graph:
            if not strong_graph.has_path(vertex1, vertex2):
                max_distance = max(max_distance, 
                    nx.shortest_path_length(weak_graph, vertex1, vertex2))

    return max_distance


def get_minimal_external_paths(weak_graph, strong_graph):
    if type(weak_graph) == nx.DiGraph:
        weak_graph = nx.Graph(weak_graph)
    if type(strong_graph) == nx.DiGraph:
        strong_graph = nx.Graph(strong_graph)

    minimal_external_paths = {}
    for vertex1 in strong_graph:
        for vertex2 in strong_graph:
            if (vertex1 in minimal_external_paths 
                    and vertex2 in minimal_external_paths[vertex1]):
                continue
            if not nx.has_path(strong_graph, vertex1, vertex2):
                path_dictionary = minimal_external_paths.setdefault(vertex1, {})
                path_dictionary[vertex2] = nx.all_shortest_paths(
                    weak_graph, vertex1, vertex2)
                minimal_external_paths[vertex1] = path_dictionary

    return minimal_external_paths


def find_minimal_external_paths(ascending_order, sizes, name_base):
    external_paths_container = {}
    for size in sizes:
        strong_graph = retrieve_word_graph(
            ascending_order, size, name_base, "strong")
        weak_graph = retrieve_word_graph(
            ascending_order, size, name_base, "weak")
        try:
            minimal_external_paths = get_minimal_external_paths(
                weak_graph, strong_graph)
        except nx.exception.NetworkXNoPath:
            minimal_external_paths = {}
        external_paths_container[size] = minimal_external_paths

    store_external_paths(external_paths_container, ascending_order, name_base)
