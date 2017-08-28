"""Functions for analyzing subgraphs and properties of their edges."""

from itertools import product

from word_explorer.objects import Word, GeneralizedPattern
from word_explorer.operations.reduction_flipping import get_deletions


BASE_PREFIX = "word_graph_size"


def make_subgraph_filename(subgraph_type, size, ascending_order):
    file_name = BASE_PREFIX + str(size) + "_" + subgraph_type + "s_sorted"
    if ascending_order:
        file_name = "ao" + file_name

    return file_name + ".txt"


def load_subgraphs(subgraph_type, size, ascending_order=True):
    if subgraph_type == "square":
        file_name = make_subgraph_filename(subgraph_type, size, ascending_order)
        with open(file_name, "r") as sorted_squares_file:
            sorted_squares = {}
            for line in sorted_squares_file:
                line = line.strip()
                if line.endswith(":"):
                    directed_structure = tuple((int(edge[0]), int(edge[3])) 
                        for edge in line[2:len(line)-3].split("), ("))
                    sorted_squares[directed_structure] = []
                if line.startswith("('"):
                    square = tuple(Word(word) 
                        for word in line[2:len(line)-2].split("', '"))
                    sorted_squares[directed_structure].append(square)
        subgraphs = sorted_squares

    return subgraphs


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
                            (length - (start_length + i), pattern_class[2], pattern_class[3]), 
                            (j, pattern_class[4], pattern_class[5]), 
                            (length - (start_length + j), pattern_class[6], pattern_class[7]),
                        )
                        classes.add(classification)

    return classes


def find_missing_classes(classified_squares):
    missing_classes = {}
    for directed_structure in classified_squares:
        pass


if __name__ == '__main__':
    squares = load_subgraphs("square", 5)
    classified_squares = classify_squares(squares)
    for directed_structure in classified_squares:
        print("Square classes for directed structure:", directed_structure)
        print(classified_squares[directed_structure])
    print("Number of classes:", sum(len(classified_squares[directed_structure]) 
                                    for directed_structure in classified_squares))
    print("Total possible classes:", len(get_all_classes(10)))

