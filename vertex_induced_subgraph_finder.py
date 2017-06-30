"""
Tools for extracting induced subgraphs from a graph 
given a set of vertices.
"""

import re

from subgraph_finder import (Word, Word_eq, extract_word_graph, 
                             expand_word_graph)


def get_vertices(vertex_file, word_class):
    vertices = []
    for line in vertex_file:
        if line:
            if "," in line:
                letters = re.findall(r"\d+", line)
                digit_strings = [str(i) for i in range(1, 10)]
                for i, letter in enumerate(letters):
                    if letter not in digit_strings:
                        letters[i] = chr(int(letter) + 87)
                word = word_class("".join(letters))
            else:
                word = word_class(line.strip())
            if word is None:
                print("Encountered a non-DOW!")
                print(letters)
            else:
                vertices.append(word)

    return vertices


def extract_vertex_induced_subgraphs(word_graph, vertices):
    vertex_induced_subgraph = {}
    weakly_vertex_induced_subgraph = {}
    for vertex in word_graph:
        if vertex in vertices:
            weakly_vertex_induced_subgraph[vertex] = word_graph[vertex]
            for neighbor in list(word_graph[vertex]):
                if neighbor in vertices:
                    prev_neighborhood = vertex_induced_subgraph.get(vertex, set())
                    prev_neighborhood.add(neighbor)
                    vertex_induced_subgraph[vertex] = prev_neighborhood

    return vertex_induced_subgraph, weakly_vertex_induced_subgraph


def store_subgraph(subgraph, file_name, graph_size, unique_name):
    subgraph_file_name = (file_name[:-4] + "_subgraph_size" 
                          + graph_size + "_" + unique_name + ".txt")
    with open(subgraph_file_name, "w") as output_file:
        print("Vertex count: " + str(len(subgraph)), file=output_file)
        edge_count = sum(len(subgraph[vertex]) for vertex in subgraph)
        print("Edge count: " + str(edge_count) + "\n\n", file=output_file)
        words = list(subgraph.keys())
        words.sort()
        for word in words:
            if subgraph[word]:
                neighborhood = word + ": " + str(subgraph[word])
                print(neighborhood, file=output_file)
    with open(subgraph_file_name, "r") as output_file:
        text = output_file.read()
        text = text.replace("\'", "")
    with open(subgraph_file_name, "w") as output_file:
        output_file.write(text)


if __name__ == "__main__":
    ascending_order = input("Ascending order words? ('Y' or 'N') ")
    graph_size = input("\nWord graph size? ")
    if ascending_order.strip().lower().startswith("n"):
        ascending_order = False
        file_prefix = "word_graph_size"
        word_class = Word
    else:
        ascending_order = True
        file_prefix = "aoword_graph_size"
        word_class = Word_eq

    vertex_file_name = input("Enter name of file containing" 
                             " vertices (including extension): ")
    with open(vertex_file_name, "r") as vertex_file:
        vertices = get_vertices(vertex_file, word_class)
        vertices = [word for word in vertices if word.size <= int(graph_size)]

    with open(file_prefix + graph_size + ".txt", "r") as graph_file:
        word_graph = extract_word_graph(graph_file, word_class)

    induced_subgraph, weakly_induced_subgraph = extract_vertex_induced_subgraphs(
        word_graph, vertices)

    store_subgraph(induced_subgraph, vertex_file_name, graph_size, "strong")
    store_subgraph(weakly_induced_subgraph, vertex_file_name, graph_size, "weak")
