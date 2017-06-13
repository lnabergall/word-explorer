"""
Computing the homology of simplicial complexes 
corresponding to word graphs.
"""

import os
import sys
import threading

import numpy as np
from simplex import SimplicialComplex

from objects import Word
from subgraph_finder import extract_word_graph, expand_word_graph


def get_word_labels(file_prefix, size):
    with open(os.path.join(os.pardir, 
            file_prefix + str(size) + ".txt"), "r") as graph_file:
        word_graph = expand_word_graph(extract_word_graph(graph_file, Word))
    words = sorted(list(word_graph))
    return {word: i for i, word in enumerate(words)}


def get_maximal_simplices(file_prefix, size, word_labels):
    """Assumes triangles are the maximal simplices."""
    triangles = []
    with open(os.path.join(os.pardir, 
            file_prefix + str(size) + "_triangles.txt"), "r") as triangle_file:
        for line in triangle_file:
            if line.startswith("(") or line.startswith("["):
                triangle_string = line.strip()[2:-2].split("', '")
                triangle = [Word(word) for word in triangle_string]
                triangles.append(triangle)

    maximal_simplices = []
    for triangle in triangles:
        simplex = [word_labels[word] for word in triangle]
        maximal_simplices.append(simplex)

    return maximal_simplices


def main():
    print("This script calculates the homology of a simplicial complex"
          + " corresponding to a given ascending order word graph.\n")
    size = int(input("Word graph size? ").strip())
    file_prefix = "aoword_graph_size"
    word_labels = get_word_labels(file_prefix, size)
    maximal_simplices = get_maximal_simplices(file_prefix, size, word_labels)
    simplicial_complex = SimplicialComplex(maximal_simplices)
    with open(file_prefix + str(size) + "_homology.txt", "w") as homology_file:
        np.set_printoptions(threshold=100000000000)
        for i in range(4):
            boundary_matrix = simplicial_complex.get_boundary_matrix(i, 2)
            print("\nBoundary matrix of dimension", i, file=homology_file)
            print(boundary_matrix, file=homology_file)
            print("\nReduced boundary matrix of dimension", i, file=homology_file)
            print(simplicial_complex.reduce_matrix(boundary_matrix)[0], 
                  file=homology_file)
            print("\nBetti number", i, "=", simplicial_complex.betti_number(i, 2), 
                  file=homology_file)
            print("\n--------------------------\n", file=homology_file)


if __name__ == '__main__':
    threading.stack_size(67108864) # 64MB stack
    sys.setrecursionlimit(2**22) # something real big, 64MB limit hit first

    # only new threads get the redefined stack size
    thread = threading.Thread(target=main)
    thread.start()