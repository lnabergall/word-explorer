"""
Functions for finding and storing certain subgraphs of word graphs, 
including 3-paths, 4-paths, triangles, squares, and cubes. 
Use 'find_subgraph' for an easy interface.

Functions:

    find_3paths, find_4paths, find_triangles, filter_subgraphs, 
    find_squares, extract_directed_structure, sort_directed_squares, 
    extract_squares, find_cubes, filter_cubes, find_subgraphs
"""

from itertools import combinations, permutations

from word_explorer.objects import Word
from .io import (retrieve_word_graph, store_word_subgraphs, 
                 retrieve_word_subgraphs)
from .word_graphs import expand_word_graph
from .subgraphs_gpu import find_subgraphs as find_subgraphs_gpu


# Currently supported subgraphs
SUBGRAPH_TYPES = ["3-path", "4-path", "triangle", "square", "cube"]


def find_3paths(word_graph, directed=True):
    paths = set()
    for word in word_graph:
        for neighbor1 in list(word_graph[word]):
            for neighbor2 in list(word_graph[neighbor1]):
                if (neighbor2 not in word_graph.get(word, []) 
                        and word not in word_graph.get(neighbor2, [])
                        and ((not directed and word != neighbor2) 
                            or (len(word) <= len(neighbor1) <= len(neighbor2)))):
                    paths.add((word, neighbor1, neighbor2))

    return list(paths)


def find_4paths(word_graph, length3_paths):
    paths = set()
    for path in length3_paths:
        for word in word_graph[path[2]]:
            if (len(word) >= len(path[2]) 
                    and word not in word_graph.get(path[0], [])
                    and path[0] not in word_graph.get(word, [])
                    and word not in word_graph.get(path[1], [])
                    and path[1] not in word_graph.get(word, [])):
                paths.add(path + (word, ))

    return list(paths)


def find_triangles(word_graph):
    triangles = set()
    for word in word_graph:
        for neighbor1, neighbor2 in combinations(list(word_graph[word]), 2):
            if neighbor2 in word_graph.get(neighbor1, []):
                triangles.add((word, neighbor1, neighbor2))
            elif neighbor1 in word_graph.get(neighbor2, []):
                triangles.add((word, neighbor2, neighbor1))

    return triangles


def filter_subgraphs(subgraphs):
    subgraphs = list(subgraphs)
    subgraph_sets = set()
    filtered_subgraphs = []
    for subgraph in subgraphs:
        subgraph_set = frozenset(subgraph)
        if subgraph_set not in subgraph_sets:
            filtered_subgraphs.append(subgraph)
            subgraph_sets.add(subgraph_set)

    return filtered_subgraphs


def find_squares(word_graph):
    paths = {}
    paths_reversed = {}
    for word in word_graph:
        for neighbor1, neighbor2 in combinations(list(word_graph[word]), 2):
            if (neighbor2 not in word_graph.get(neighbor1, []) and 
                    neighbor1 not in word_graph.get(neighbor2, [])):
                if (not paths_reversed.get((neighbor1, neighbor2)) and 
                        not paths_reversed.get((neighbor2, neighbor1))):
                    paths_reversed[(neighbor1, neighbor2)] = set([word])
                    neighbors = paths.get(word, set())
                    neighbors.add((neighbor1, neighbor2))
                    paths[word] = neighbors
                elif paths_reversed.get((neighbor1, neighbor2)):
                    paths_reversed[(neighbor1, neighbor2)].add(word)
                    neighbors = paths.get(word, set())
                    neighbors.add((neighbor1, neighbor2))
                    paths[word] = neighbors
                else:
                    paths_reversed[(neighbor2, neighbor1)].add(word)
                    neighbors = paths.get(word, set())
                    neighbors.add((neighbor2, neighbor1))
                    paths[word] = neighbors
    squares = set()
    for word1 in paths:
        for pair in paths[word1]:
            for word2 in paths_reversed[pair]:
                if (word1 != word2 and word2 not in word_graph.get(word1, []) and
                        word1 not in word_graph.get(word2, [])):
                    squares.add((word1, pair[0], word2, pair[1]))

    return list(squares)


def extract_directed_structure(square):
    lengths = [len(word) for word in square]
    directed_edges = (
        (0, 1) if lengths[0] < lengths[1] else (1, 0),
        (1, 2) if lengths[1] < lengths[2] else (2, 1),
        (2, 3) if lengths[2] < lengths[3] else (3, 2),
        (0, 3) if lengths[0] < lengths[3] else (3, 0)
    )
    return directed_edges


def sort_directed_squares(squares):
    sorted_squares = {}
    for square in squares:
        directed_edges = extract_directed_structure(square)
        for permutation in permutations(directed_edges):
            if permutation in sorted_squares:
                sorted_squares[permutation].append(square)
                break
        else:
            sorted_squares[directed_edges] = [square]

    sorted_squares_final = {}
    for directed_structure1 in sorted_squares:
        match = False
        for directed_structure2 in sorted_squares_final:
            for i in range(4):
                directed_structure2_relabeled = (((edge[0]+i) % 4, (edge[1]+i) % 4) 
                                                 for edge in directed_structure2)
                if set(directed_structure1) == set(directed_structure2_relabeled):
                    match = True
                    break
            if match:
                sorted_squares_final[directed_structure2].extend(
                    sorted_squares[directed_structure1])
                break
        if not match:
            sorted_squares_final[directed_structure1] = sorted_squares[
                directed_structure1]

    return sorted_squares_final


SQUARE_STRUCTURES = [
    set(((1, 0), (2, 1), (2, 3), (3, 0))), 
    set(((1, 0), (1, 2), (2, 3), (0, 3))),
    set(((0, 1), (1, 2), (3, 2), (0, 3))),
    set(((0, 1), (2, 1), (3, 2), (3, 0))),
]


def extract_squares(sorted_squares):
    squares = []
    for directed_structure in sorted_squares:
        squares.extend(sorted_squares[directed_structure])
    return squares


def find_cubes(word_graph, squares):
    cubes = set()
    for square1 in squares:
        for square2 in squares:
            square1_words = set(square1)
            square2_words = set(square2)
            cube_edges = []
            unassigned = set()
            invalid = False
            for word in square1:
                neighbors = word_graph[word] & square2_words
                if len(neighbors) > 1:
                    invalid = True
                    break
                elif len(neighbors) == 0:
                    unassigned.add(word)
                else:
                    cube_edges.append((word, neighbors.pop()))
            if not invalid and len(cube_edges) != 4:
                for word in square2:
                    neighbors = word_graph[word] & square1_words
                    if len(neighbors) > 1:
                        invalid = True
                        break
                    elif len(neighbors) == len(unassigned & neighbors) == 1:
                        cube_edges.append((word, neighbors.pop()))
            if not invalid and len(cube_edges) == 4:
                cube = square1 + tuple(edge[1] for edge in cube_edges)
                if len(set(cube)) == 8:
                    cubes.add(cube)

    return list(cubes)


def filter_cubes(cubes):
    cubes_list = filter_subgraphs(cubes)
    sorted_cubes = {"Linearly Ordered": [], "Not Linearly Ordered": []}
    for cube in cubes_list:
        directed_structure1 = extract_directed_structure(
            (cube[0], cube[1], cube[2], cube[3]))
        directed_structure2 = extract_directed_structure(
            (cube[4], cube[5], cube[6], cube[7]))
        linearly_ordered = (set(directed_structure1) in SQUARE_STRUCTURES 
            and set(directed_structure2) in SQUARE_STRUCTURES)
        vertical_structure1 = list(filter(
            lambda i: len(cube[i]) >= len(cube[i+4]), range(4)))
        vertical_structure2 = list(filter(
            lambda i: len(cube[i]) <= len(cube[i+4]), range(4)))
        linearly_ordered = linearly_ordered and (not vertical_structure1 
            or not vertical_structure2)
        if linearly_ordered:
            sorted_cubes["Linearly Ordered"].append(cube)
        else:
            sorted_cubes["Not Linearly Ordered"].append(cube)

    return sorted_cubes


SUBGRAPH_FINDERS = {
    "triangle": find_triangles,
    "3-path": find_3paths,
    "4-path": find_4paths,
    "square": find_squares,
    "cube": find_cubes,
}


def find_subgraphs(ascending_order, sizes, subgraph_types, 
                   name_base="word_graph_size", 
                   include_sorting=False, use_gpu=True):
    for size in sizes:
        word_graph = expand_word_graph(
            retrieve_word_graph(ascending_order, size, name_base))
        for subgraph_type in subgraph_types:
            finder = SUBGRAPH_FINDERS[subgraph_type]
            if subgraph_type == "4-path":
                length_3paths = retrieve_word_subgraphs(
                    ascending_order, size, "3-path", name_base)
                if use_gpu:
                    subgraphs = find_subgraphs_gpu(
                        subgraph_type, word_graph, ascending_order, 
                        Word, data=length_3paths)
                else:
                    subgraphs = finder(word_graph, length_3paths)
            elif subgraph_type == "cube":
                squares = retrieve_word_subgraphs(
                    ascending_order, size, "square", name_base)
                if use_gpu:
                    subgraphs = filter_cubes(find_subgraphs_gpu(
                        subgraph_type, word_graph, ascending_order, 
                        Word, data=squares))
                else:
                    subgraphs = filter_cubes(finder(word_graph, squares))
            else:
                if use_gpu:
                    subgraphs = find_subgraphs_gpu(
                        subgraph_type, word_graph, ascending_order, Word)
                else:
                    subgraphs = finder(word_graph)
            if include_sorting and subgraph_type == "square":
                squares = retrieve_word_subgraphs(
                    ascending_order, size, "square", name_base)
                subgraphs = sort_directed_squares(squares)
            store_word_subgraphs(subgraphs, subgraph_type, ascending_order, 
                                 size, name_base)