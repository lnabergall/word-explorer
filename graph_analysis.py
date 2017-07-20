"""
Functions for analyzing word graphs and locating subgraphs 
using NetworkX.
"""

import networkx as nx

from objects import Word
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


if __name__ == '__main__':
    data_graph_file_names = []
    for size in range(3, 8):
        dataD_file_name = ("loops_removed_dataD_rearrangements_subgraph_size" 
                           + str(size) + "_weak.txt")
        random_file_name = ("loops_removed_dataD_rearrangements_random1_subgraph_size"
                           + str(size) + "_weak.txt")
        data_graph_file_names.extend([dataD_file_name, random_file_name])

    full_graph_file_names = []
    for size in range(2, 6):
        full_graph_file_names.append(get_word_graph_filename(True, size))

    word_graphs = load_word_graphs(data_graph_file_names + full_graph_file_names)

    properties = {}
    for graph_file_name, word_graph in word_graphs.items():
        properties[graph_file_name] = {}
        print("\nComputing the connectivity and diameter of")
        print(graph_file_name)
        properties[graph_file_name]["weakly_connected"] = (
            nx.is_weakly_connected(word_graph))
        properties[graph_file_name]["strongly_connected"] = (
            nx.is_strongly_connected(word_graph))
        try:
            properties[graph_file_name]["diameter"] = (
                nx.diameter(nx.Graph(word_graph)))
        except nx.exception.NetworkXError:
            properties[graph_file_name]["diameter"] = "infinity"

    for graph_file_name, properties in properties.items():
        print("\n" + graph_file_name)
        print("Weakly connected:", properties["weakly_connected"])
        print("Strongly connected:", properties["strongly_connected"])
        print("Diameter:", properties["diameter"])