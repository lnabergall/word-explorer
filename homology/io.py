"""
Input/output functions for the homology API---in particular, for loading
maximal simplices and storing homology data.

Functions:

    get_word_labels, retrieve_maximal_simplices, store_homology_data
"""

from word_explorer.io import retrieve_data, store_data
from word_explorer.word_graphs.word_graphs import expand_word_graph
from word_explorer.word_graphs.io import (retrieve_word_graph,
                                          retrieve_word_subgraphs,
                                          get_word_graph_filename)


FACE_SQUARE_STRUCTURES = ("((0, 1), (1, 2), (3, 2), (0, 3))",
                          "((0, 1), (1, 2), (2, 3), (0, 3))", 
                          "((0, 1), (2, 1), (3, 2), (0, 3))")


def get_word_labels(expanded_word_graph):
    words = sorted(list(expanded_word_graph))
    return {word: i for i, word in enumerate(words)}


def retrieve_maximal_simplices(ascending_order, size, 
                               name_base="word_graph_size", name_suffix="",
                               max_simplex_types=["triangle"]):
    maximal_simplices = []
    for subgraph_type in max_simplex_types:
        # Add maximal k-simplices, k >= 2
        if subgraph_type != "square":
            maximal_simplices.extend(retrieve_word_subgraphs(
                ascending_order, size, subgraph_type, 
                name_base, name_suffix))
        else:
            simplices = retrieve_word_subgraphs(
                ascending_order, size, subgraph_type, 
                name_base, name_suffix, sorted_=True)
            for subgraph_class, simplex_list in simplices.items():
                if subgraph_class in FACE_SQUARE_STRUCTURES:
                    maximal_simplices.extend(simplex_list)
        word_graph = retrieve_word_graph(
            ascending_order, size, name_base, name_suffix)
        words = sorted(list(expand_word_graph(word_graph)))
        word_labels = {word: i for i, word in enumerate(words)}

        # Add maximal 1-simplices (edges)
        for vertex, neighbors in word_graph.items():
            for neighbor in neighbors:
                for subgraph in maximal_simplices:
                    if vertex in subgraph and neighbor in subgraph:
                        break
                else:
                    maximal_simplices.append([vertex, neighbor])

    # Label simplices
    labeled_maximal_simplices = []
    for subgraph in maximal_simplices:
        simplex = [word_labels[word] for word in subgraph]
        labeled_maximal_simplices.append(simplex)

    return labeled_maximal_simplices


def store_homology_data(ascending_order, size, homology_data, 
                        max_simplex_types, name_base="word_graph_size",
                        name_suffix=""):
    file_name = get_word_graph_filename(ascending_order, size, 
                                        name_base, name_suffix)[:-4]
    file_name += "_homology_" + "_".join(max_simplex_types) + ".txt"
    print_data = []
    for i, homology_group in enumerate(homology_data):
        print_data.append("\nBetti number " + i + " : " + homology_group[0])
        if len(homology_group) > 1:
            print_data.append("\nBoundary matrix of dimension " + i)
            print_data.append(homology_group[1])
            print_data.append("\nReduced boundary matrix of dimension " + i)
            print_data.append(homology_group[2])
        print_data.append("\n--------------------------\n")

    store_data(print_data, file_name)
