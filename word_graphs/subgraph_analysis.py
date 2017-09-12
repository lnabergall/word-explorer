"""
Functions for analyzing subgraphs and properties of their edges;
currently only supports square subgraphs. Use 'analyze_square_subgraphs'
for an easy interface.

Functions:
    
    extract_directed_edges, classify_square, classify_squares, 
    get_all_classes, analyze_square_subgraphs
"""

from itertools import product

from word_explorer.objects import Word, GeneralizedPattern
from word_explorer.operations.reduction_flipping import get_deletions
from .io import retrieve_word_subgraphs


def extract_directed_edges(square, directed_structure):
    edges = []
    for direction in directed_structure:
        vertex1 = square[direction[0]]
        vertex2 = square[direction[1]]
        if len(vertex1) < len(vertex2):
            edges.append([vertex1, vertex2])
        else:
            edges.append([vertex2, vertex1])

    return edges


def classify_square(square, directed_structure):
    edges = extract_directed_edges(square, directed_structure)
    repeat_pattern = GeneralizedPattern(
        (("a", ""), ("a", "")), name="repeat_pattern")
    return_pattern = GeneralizedPattern(
        (("a", ""), ("a", "R")), name="repeat_pattern")
    classification = []
    for edge in edges:
        edge_classification = [0, False, False]
        insertion_length = len(edge[1]) - len(edge[0])
        edge_classification[0] = insertion_length
        repeat_deletions = get_deletions(edge[1], repeat_pattern)
        return_deletions = get_deletions(edge[1], return_pattern)
        if edge[0] in repeat_deletions:
            edge_classification[1] = True
        if edge[0] in return_deletions:
            edge_classification[2] = True
        classification.append(tuple(edge_classification))

    return tuple(classification)


def classify_squares(sorted_squares):
    classified_squares = {}
    for directed_structure in sorted_squares:
        classifications = set()
        for square in sorted_squares[directed_structure]:
            classification = classify_square(square, directed_structure)
            classifications.add(classification)
        classified_squares[directed_structure] = classifications

    return classified_squares


def get_all_classes(length):
    classes = set()    
    for start_length in range(2, length-2, 2):
        for i in range(2, length, 2):
            for j in range(2, length, 2):
                if start_length + i <= length - 2 and start_length + j <= length - 2:
                    for pattern_class in product([True, False], repeat=8):
                        classification = (
                            (i, pattern_class[0], pattern_class[1]), 
                            (length - (start_length + i), 
                             pattern_class[2], pattern_class[3]), 
                            (j, pattern_class[4], pattern_class[5]), 
                            (length - (start_length + j), 
                             pattern_class[6], pattern_class[7]),
                        )
                        classes.add(classification)

    return classes


def analyze_square_subgraphs(ascending_order, size, name_base):
    sorted_squares = retrieve_word_subgraphs(
        ascending_order, size, "square", name_base, sorted_=True)
    classified_squares = classify_squares(sorted_squares)
    class_count = sum(len(classified_squares[directed_structure]) 
                      for directed_structure in classified_squares)
    possible_class_count = len(get_all_classes(size*2))

    return classified_squares, class_count, possible_class_count
