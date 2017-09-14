"""
Tools for extracting subgraphs induced by a given set of vertices from 
a word graph. There are two types: weakly and strongly induced subgraphs.
The latter is the usual notion of induced subgraph. The former is 
the subgraph constructed by taking the union of the neighborhood of 
each vertex in the given set of vertices and including in 
the subgraph any edge with both its endpoints in this union. 

Use 'find_vertex_induced_subgraphs' as an interface.

Functions:
    
    extract_vertex_induced_subgraphs, find_vertex_induced_subgraphs
"""

from word_explorer.objects import Word
from word_explorer.objects.io import retrieve_words
from .word_graphs import expand_word_graph
from .io import store_word_graph, get_word_subgraph_filename, retrieve_word_graph


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


def find_vertex_induced_subgraphs(ascending_order, sizes, name_base):
    for size in sizes:
        # Find the subgraphs
        vertices = retrieve_words(vertex_file_name, include_empty_word=False, 
                                  ascending_order=ascending_order)
        vertices = [word for word in vertices if word.size <= size]
        word_graph = retrieve_word_graph(ascending_order, size)
        induced_subgraph, weakly_induced_subgraph = extract_vertex_induced_subgraphs(
            word_graph, vertices)

        # Store them
        strong_file_name = get_word_subgraph_filename(
            ascending_order, graph_size, vertex_file_name, "strong")
        weak_file_name = get_word_subgraph_filename(
            ascending_order, graph_size, vertex_file_name, "weak")
        store_word_graph(induced_subgraph, strong_file_name)
        store_word_graph(weakly_induced_subgraph, weak_file_name)