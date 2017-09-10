"""
"""

from itertools import combinations, permutations

from word_explorer.objects import Word
from .subgraph_finder_gpu import find_subgraphs as find_subgraphs_gpu


def expand_word_graph(word_graph):
    """Expands word graph dictionary to include all words as keys."""
    expanded_word_graph = word_graph.copy()
    for word1 in word_graph:
        for word2 in list(word_graph[word1]):
             expanded_word_graph[word2] = (expanded_word_graph.get(word2, set())
                | set([word1]))

    return expanded_word_graph


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


SPECIAL_STRUCTURES = [
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
        linearly_ordered = (set(directed_structure1) in SPECIAL_STRUCTURES 
            and set(directed_structure2) in SPECIAL_STRUCTURES)
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


def extract_word_graph(graph_file, word_class=Word):
    word_graph = {}
    for i, line in enumerate(graph_file):
        if i >= 4:
            colon_index = line.find(":")
            word_graph[word_class(line[:colon_index])] = {word_class(word) for word in 
                line.strip()[colon_index+3:len(line)-2].split(", ")}
    return word_graph


def get_3paths(word_graph_file_name, directed=True):
    if directed:
        file_name_end = "_3paths.txt"
    else:
        file_name_end = "_3paths_undirected.txt"
    with open(word_graph_file_name + file_name_end, "r") as paths_file:
        paths = []
        for line in paths_file:
            if line.startswith("("):
                path = tuple(word_class(word) 
                    for word in line[2:len(line)-3].split("', '"))
                paths.append(path)

    return paths


def get_squares(file_name):
    with open(file_name, "r") as squares_file:
            squares = []
            for line in squares_file:
                if line.startswith("('"):
                    square = tuple(word_class(word) 
                        for word in line[2:len(line)-3].split("', '"))
                    squares.append(square)
    return squares


if __name__ == "__main__":
    ascending_order = input("Ascending order words? ('Y' or 'N') ")
    request = input("\nFind squares in a graph, sort directed squares, " 
                    "find cubes from directed squares, find " 
                    "triangles, find 3-paths, or find 4-paths? " 
                    "('1', '2', '3', '4', '5', or '6'): ")
    if request == "5":
        directed = input("\nDirected or undirected 3-paths? ")
        directed = False if directed.strip().lower().startswith("u") else True
    graph_type = input("Word graph or subgraph? ")
    if graph_type.strip().lower().startswith("s"):
        graph_type = "word_subgraph"
        word_graph_file_name = input("Enter name of file containing a word subgraph" 
                                     + "(excluding extension): ").strip()
        gpu = False
    else:
        graph_type = "word_graph"
        graph_size = input("\nWord graph size? ")
        gpu = input("\nUse GPU? ")
        gpu = False if gpu.strip().lower().startswith("n") else True
    if ascending_order.strip().lower().startswith("n"):
        ascending_order = False
        if graph_type == "word_graph":
            file_prefix = "word_graph_size"
        word_class = Word
    else:
        ascending_order = True
        if graph_type == "word_graph":
            file_prefix = "aoword_graph_size"
        word_class = Word_eq

    if graph_type == "word_graph":
        word_graph_file_name = file_prefix + graph_size

    if request in ["1", "3", "4", "5", "6"]:
        with open(word_graph_file_name + ".txt", "r") as graph_file:
            word_graph = extract_word_graph(graph_file, word_class)

    if request == "1":
        word_graph = expand_word_graph(word_graph)
        if gpu:
            paths = get_3paths(word_graph_file_name, directed=False)
            squares = filter_subgraphs(find_subgraphs_gpu(
                "square", word_graph, ascending_order, word_class, data=paths))
        else:
            squares = filter_subgraphs(find_squares(word_graph))
        squares.sort(key=lambda square: len(square[0]))
        with open(word_graph_file_name + "_squares.txt", "w") as squares_file:
            print("Square subgraph count: " + str(len(squares)) + "\n\n", 
                  file=squares_file)
            for square in squares:
                print(square, file=squares_file)

    if request == "2":
        file_name = input("\nEnter name of file containing directed "  
                          "squares (including extension): ")
        squares = get_squares(file_name)
        sorted_squares = sort_directed_squares(squares)
        with open(word_graph_file_name + "_squares_sorted.txt", "w") \
                as sorted_squares_file:
            square_count = 0
            for directed_labeling in sorted_squares:
                square_count += len(sorted_squares[directed_labeling])
            print("Square subgraph count: " + str(square_count) + "\n", 
                  file=sorted_squares_file)
            for directed_labeling in sorted_squares:
                print(str(directed_labeling) + " square subgraph count: " + 
                      str(len(sorted_squares[directed_labeling])), 
                      file=sorted_squares_file)
            print("\n\n", file=sorted_squares_file)
            for directed_labeling in sorted_squares:
                print(str(directed_labeling) + ":\n", file=sorted_squares_file)
                for square in sorted_squares[directed_labeling]:
                    print(square, file=sorted_squares_file)
                print("\n\n", file=sorted_squares_file)

    if request == "3":
        if not gpu or int(graph_size) < 5:
            file_name = input("\nEnter name of file containing sorted directed "  
                              "squares (including extension): ")
            with open(file_name, "r") as sorted_squares_file:
                sorted_squares = {}
                for line in sorted_squares_file:
                    line = line.strip()
                    if line.endswith(":"):
                        directed_structure = tuple((int(edge[0]), int(edge[3])) 
                            for edge in line[2:len(line)-3].split("), ("))
                        sorted_squares[directed_structure] = []
                    if line.startswith("('"):
                        square = tuple(word_class(word) 
                            for word in line[2:len(line)-2].split("', '"))
                        sorted_squares[directed_structure].append(square)
            squares = extract_squares(sorted_squares)
            
        if gpu:
            if int(graph_size) >= 5:
                file_name = input("\nEnter name of file containing directed "  
                                  "squares (including extension): ")
                squares = get_squares(file_name)
            sorted_cubes = filter_cubes(find_subgraphs_gpu(
                "cube", expand_word_graph(word_graph), 
                ascending_order, word_class, data=squares))
        else:
            sorted_cubes = filter_cubes(find_cubes(
                expand_word_graph(word_graph), squares))
        with open(word_graph_file_name + "_cubes.txt", "w") as cubes_file:
            cube_count = 0
            for directed_labeling in sorted_cubes:
                cube_count += len(sorted_cubes[directed_labeling])
            print("Cube subgraph count: " + str(cube_count) + "\n", 
                  file=cubes_file)
            for directed_labeling in sorted_cubes:
                print(str(directed_labeling) + " cube subgraph count: " + 
                      str(len(sorted_cubes[directed_labeling])), file=cubes_file)
            print("\n\n", file=cubes_file)
            for directed_labeling in sorted_cubes:
                print(str(directed_labeling) + ":\n", file=cubes_file)
                for cube in sorted_cubes[directed_labeling]:
                    print(cube, file=cubes_file)
                print("\n\n", file=cubes_file)

    if request == "4":
        if gpu:
            triangles = filter_subgraphs(find_subgraphs_gpu(
                "triangle", expand_word_graph(word_graph), 
                ascending_order, word_class))
        else:
            triangles = filter_subgraphs(
                find_triangles(expand_word_graph(word_graph)))
        with open(word_graph_file_name + "_triangles.txt", "w") \
                as triangles_file:
            print("Triangle subgraph count: " + str(len(triangles)) + "\n\n",
                  file=triangles_file)
            for triangle in triangles:
                print(triangle, file=triangles_file)

    if request == "5":
        if gpu:
            paths = find_subgraphs_gpu(
                "3-path", expand_word_graph(word_graph), 
                ascending_order, word_class, data=directed)
        else:
            paths = find_3paths(expand_word_graph(word_graph), directed)
        if not directed:
            paths = filter_subgraphs(paths)
        paths.sort(key=lambda path: len(path[0]))
        if directed:
            file_name_end = "_3paths.txt"
        else:
            file_name_end = "_3paths_undirected.txt"
        with open(word_graph_file_name + file_name_end, "w") as paths_file:
            print("3-path subgraph count: " + str(len(paths)) + "\n\n",
                  file=paths_file)
            for path in paths:
                print(path, file=paths_file)

    if request == "6":
        paths = get_3paths(word_graph_file_name)
        if gpu:
            paths = find_subgraphs_gpu("4-path", expand_word_graph(word_graph),
                                       ascending_order, word_class, data=paths)
        else:
            paths = find_4paths(expand_word_graph(word_graph), paths)
        paths.sort(key=lambda path: len(path[0]))
        with open(word_graph_file_name + "_4paths.txt", "w") as paths_file:
            print("4-path subgraph count: " + str(len(paths)) + "\n\n",
                  file=paths_file)
            for path in paths:
                print(path, file=paths_file)
