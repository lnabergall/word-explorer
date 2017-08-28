"""
Functions for analyzing word graphs and locating subgraphs 
using NetworkX.
"""

import re
from time import  time

import networkx as nx

from word_explorer.objects import Word
from subgraph_finder import extract_word_graph


def get_word_graph_filename(ascending_order, size, subgraph=None):
    file_name = "word_graph_size"
    if ascending_order:
        file_name = "ao" + file_name
    if size:
        file_name = file_name + str(size)
    if subgraph:
        file_name += "_" + subgraph

    return file_name + ".txt"


def load_word_graph(graph_file_name, word_class=Word):
    with open(graph_file_name, "r") as graph_file:
        word_graph_dict = extract_word_graph(graph_file)
    edges = []
    for word, neighbors in word_graph_dict.items():
        for neighbor in list(neighbors):
            edges.append((word, neighbor))

    word_graph = nx.DiGraph()
    word_graph.add_edges_from(edges)

    return word_graph


def load_word_graphs(graph_file_names, word_class=Word):
    word_graphs = {}
    for file_name in graph_file_names:
        word_graphs[file_name] = load_word_graph(
            file_name, word_class=word_class)

    return word_graphs


def find_greatest_unconnected_distance(weak_graph, strong_graph):
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


if __name__ == '__main__':
    start_time =  time()
    data_weak_file_names = []
    data_strong_file_names = []
    for size in range(3, 6):
        dataD_file_name_weak = (
            "loops_removed_dataD_rearrangements_subgraph_size" 
            + str(size) + "_weak.txt")
        random_file_name_weak = (
            "loops_removed_dataD_rearrangements_random1_subgraph_size"
            + str(size) + "_weak.txt")
        dataD_file_name_strong = (
            "loops_removed_dataD_rearrangements_subgraph_size" 
            + str(size) + "_strong.txt")
        random_file_name_strong = (
            "loops_removed_dataD_rearrangements_random1_subgraph_size"
            + str(size) + "_strong.txt")
        data_weak_file_names.extend([dataD_file_name_weak, random_file_name_weak])
        data_strong_file_names.extend([dataD_file_name_strong, random_file_name_strong])

    word_graphs = load_word_graphs(data_weak_file_names + data_strong_file_names)

    external_paths_dict = {}
    for graph_file_name, word_graph in word_graphs.items():
        size = int(re.search(r"size(\d)", graph_file_name).group(1))
        print("\nComputing the minimal external paths of")
        print(graph_file_name)
        if "strong" in graph_file_name:
            strong_graph = word_graph
            weak_graph = word_graphs[graph_file_name.replace("weak", "strong")]
        elif "weak" in graph_file_name:
            continue
        try:
            minimal_external_paths = get_minimal_external_paths(
                weak_graph, strong_graph)
        except nx.exception.NetworkXNoPath:
            pass
        external_paths_dict[graph_file_name] = minimal_external_paths

    external_paths_file_name = ("loops_removed_dataD_rearrangements_subgraph"
                                "_minimal_external_paths.txt")
    with open(external_paths_file_name, "w") as paths_file:
        for graph_file_name in external_paths_dict:
            print("\n\n" + graph_file_name + "\n", file=paths_file)
            minimal_external_paths = external_paths_dict[graph_file_name]
            for vertex1, paths_by_vertex in minimal_external_paths.items():
                for vertex2, minimal_external_paths in paths_by_vertex.items():
                    print(file=paths_file)
                    print(vertex1, "-->", vertex2, ":", file=paths_file)
                    for path in minimal_external_paths:
                        print(path, file=paths_file)

    print("duration:", time() - start_time)