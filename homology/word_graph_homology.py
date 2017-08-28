"""
Computing the homology of simplicial complexes 
corresponding to word graphs.
"""

import os
import sys
import threading

import numpy as np
from simplex import SimplicialComplex

from word_explorer.objects import Word
from word_explorer.word_graphs.subgraph_finder import (
    extract_word_graph, expand_word_graph)

 
FACE_SQUARE_STRUCTURES = ("((0, 1), (1, 2), (3, 2), (0, 3))",
                          "((0, 1), (1, 2), (2, 3), (0, 3))", 
                          "((0, 1), (2, 1), (3, 2), (0, 3))")


def get_word_labels(file_path):
    with open(file_path + ".txt", "r") as graph_file:
        word_graph = expand_word_graph(extract_word_graph(graph_file, Word))
    words = sorted(list(word_graph))
    return {word: i for i, word in enumerate(words)}


def get_maximal_simplices(file_path, word_labels, max_simplex_types=["triangle"]):
    maximal_simplices = []
    if "triangle" in max_simplex_types:
        triangles = []
        with open(file_path + "_triangles.txt", "r") as triangle_file:
            for line in triangle_file:
                if line.startswith("(") or line.startswith("["):
                    triangle_string = line.strip()[2:-2].split("', '")
                    triangle = [Word(word) for word in triangle_string]
                    triangles.append(triangle)
        maximal_simplices.extend(triangles)

    if "square" in max_simplex_types:
        # Collect parallel + (3,1) squares
        directed_structure = None
        squares = []
        with open(file_path + "_squares_sorted.txt", "r") as sorted_squares_file:
            for line in sorted_squares_file:
                if line.strip().endswith("):"):
                    directed_structure = line.strip()[:-1]
                if (directed_structure in FACE_SQUARE_STRUCTURES 
                        and line.startswith("('")):
                    square_string = line.strip()[2:-2].split("', '")
                    square = [Word(word) for word in square_string]
                    squares.append(square)
        maximal_simplices.extend(squares)

    # Get maximal 1-simplices (edges)
    with open(file_path + ".txt", "r") as graph_file:
        word_graph = expand_word_graph(extract_word_graph(graph_file, Word))
    for vertex, neighbors in word_graph.items():
        for neighbor in neighbors:
            maximal = True
            for subgraph in maximal_simplices:
                if vertex in subgraph and neighbor in subgraph:
                    maximal = False
                    break
            if maximal:
                maximal_simplices.append([vertex, neighbor])

    relabeled_maximal_simplices = []
    for subgraph in maximal_simplices:
        simplex = [word_labels[word] for word in subgraph]
        relabeled_maximal_simplices.append(simplex)

    return relabeled_maximal_simplices


def main():
    print("This script calculates the homology of a simplicial complex"
          + " corresponding to a given ascending order word graph or subgraph.\n")
    size = int(input("Word graph size? ").strip())
    graph_type = input("Word graph or subgraph? ")
    if graph_type.strip().lower().startswith("s"):
        word_graph_file_name = input("Enter name of file containing a word subgraph" 
                                     + "(excluding extension): ").strip()
        file_path = os.path.join(os.pardir, word_graph_file_name)
    else:
        file_prefix = "aoword_graph_size"
        file_path = os.path.join(os.pardir, file_prefix + str(size))
    word_labels = get_word_labels(file_path)
    maximal_simplices = get_maximal_simplices(
        file_path, word_labels, max_simplex_types=["triangle"])
    print("Maximal 1-simplices:", len([simplex for simplex in maximal_simplices 
                                       if len(simplex) == 2]))
    simplicial_complex = SimplicialComplex(maximal_simplices)
    with open(file_path + "_homology_triangle.txt", "w") as homology_file:
        np.set_printoptions(threshold=100000000000)
        for i in range(4):
            # boundary_matrix = simplicial_complex.get_boundary_matrix(i, 2)
            # print("\nBoundary matrix of dimension", i, file=homology_file)
            # print(boundary_matrix, file=homology_file)
            # print("\nReduced boundary matrix of dimension", i, file=homology_file)
            # print(simplicial_complex.reduce_matrix(boundary_matrix)[0], 
            #       file=homology_file)
            print("\nBetti number", i, "=", simplicial_complex.betti_number(i, 2), 
                  file=homology_file)
            print("\n--------------------------\n", file=homology_file)


if __name__ == '__main__':
    threading.stack_size(67108864) # 64MB stack
    sys.setrecursionlimit(2**22) # something real big, 64MB limit hit first

    # only new threads get the redefined stack size
    thread = threading.Thread(target=main)
    thread.start()