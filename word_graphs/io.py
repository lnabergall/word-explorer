"""
Input/output functions for the word_graphs API, including 
functions for storing and retrieving word graphs and subgraphs.

Functions:

    get_word_graph_filename, get_word_subgraph_filename, store_word_graph, 
    retrieve_word_graph, store_word_subgraphs, retrieve_word_subgraphs,
    store_external_paths
"""

from itertools import chain

from word_explorer.objects import Word
from word_explorer.io import store_data, retrieve_data


def get_word_graph_filename(ascending_order, size, name_base, name_suffix=""):
    file_name = name_base + str(size) + name_suffix + ".txt"
    if ascending_order:
        file_name = "ao" + file_name
    return file_name


def get_word_subgraph_filename(ascending_order, size, name_base, 
                               subgraph_type, sorted_=False):
    file_name = get_word_graph_filename(ascending_order, size, name_base)[:-4]
    file_name += "_" + subgraph_type 
    file_name = file_name + "_sorted" if sorted_ else file_name
    return file_name + ".txt"


def store_word_graph(word_graph, file_name=None):
    from .word_graphs import WordGraph, expand_word_graph  # For circular import
    if file_name is None:
        file_name = os.path.join("word_graphs", word_graph.file_name)
    if isinstance(word_graph, WordGraph):
        vertex_count = word_graph.vertex_count
        edge_count = word_graph.edge_count
        words = list(word_graph.directed_neighborhoods.keys())
    else:
        vertex_count = len(expand_word_graph(word_graph))
        edge_count = sum(len(neighors) for neighors in word_graph.values())
        words = list(word_graph)
    word_graph_data = ["Vertex count: " + str(vertex_count), 
                       "Edge count: " + str(edge_count) + "\n\n"]
    words.sort()
    for word in words:
        if (isinstance(word_graph, WordGraph) 
                and word_graph.directed_neighborhoods[word]):
            neighborhood = word + ": " + str(
                word_graph.directed_neighborhoods[word]).replace("\'", "")
        elif type(word_graph) == dict and word_graph[word]:
            neighborhood = word + ": " + str(word_graph[word]).replace("\'", "")
        word_graph_data.append(neighborhood)
    store_data(word_graph_data, file_name)


def retrieve_word_graph(ascending_order, size, 
                        name_base="word_graph_size", name_suffix=""):
    file_name = get_word_graph_filename(
        ascending_order, size, name_base, name_suffix)
    file_name = os.path.join("word_graphs", file_name)
    word_graph = {}
    for i, line in enumerate(retrieve_data(file_name)):
        if i >= 4:
            colon_index = line.find(":")
            word_graph[word_class(line[:colon_index])] = {
                word_class(word) for word 
                in line.strip()[colon_index+3:-2].split(", ")}

    return word_graph


def store_word_subgraphs(subgraphs, subgraph_type, ascending_order, 
                         size, name_base="word_graph_size"):
    sorted_ = True if type(subgraphs) == dict else False
    subgraph_count = (len(subgraphs) if type(subgraphs) == list 
                      else sum(len(subgraph_list) for subgraph_list 
                               in subgraphs.values()))
    subgraph_data = [subgraph_type.title() + " subgraph count: " 
                     + str(subgraph_count) + "\n"]
    if (subgraph_type in ["triangle", "3-path", "4-path", "square"] and not sorted_):
        subgraph_data += subgraphs
    elif sorted_:
        for subgraph_class, subgraph_list in subgraphs.items():
            subgraph_data.append(str(subgraph_class) + " " + subgraph_type 
                                 + " subgraph count: " + str(len(subgraph_list)))
        subgraph_data.append("\n\n")
        for subgraph_class, subgraph_list in subgraphs.items():
            subgraph_data.append(str(subgraph_class) + ":\n")
            subgraph_data += subgraph_list
            subgraph_data.append("\n\n")

    subgraph_file_name = get_word_subgraph_filename(
        ascending_order, size, name_base, subgraph_type, sorted_)
    subgraph_file_name = os.path.join("word_graphs", subgraph_file_name)
    store_data(subgraph_data, subgraph_file_name)


def retrieve_word_subgraphs(ascending_order, size, subgraph_type, 
                            name_base="word_graph_size", sorted_=False):
    subgraph_file_name = get_word_subgraph_filename(
        ascending_order, size, name_base, subgraph_type, sorted_)
    subgraph_file_name = os.path.join("word_graphs", subgraph_file_name)
    if sorted_:
        subgraphs = {}
    else:
        subgraphs = []
    for line in retrieve_data(subgraph_file_name):
        line = line.strip()
        if sorted_ and line.endswith(":"):
            if line.startswith("((") or line.startswith("[("):
                # Assumes it's a 'directed structure'
                subgraph_class = tuple((int(edge[0]), int(edge[3])) 
                    for edge in line[2:-3].split("), ("))
                subgraphs[subgraph_class] = []
            else:
                subgraph_class = line[:-1]
        if (line.startswith("(") or line.startswith("[")):
            if sorted_:
                square = tuple(Word(word, ascending_order=ascending_order, 
                                    optimize=ascending_order) 
                               for word in line[2:-2].split("', '"))
                subgraphs[subgraph_class].append(square)
            else:
                subgraph = tuple(Word(word, ascending_order=ascending_order, 
                                      optimize=ascending_order)
                                 for word in line[2:-2].split("', '"))
                subgraphs.append(subgraph)

    return subgraphs
    

def store_external_paths(external_paths_container, ascending_order, name_base):
    file_name = name_base + "_minimal_external_paths.txt"
    if ascending_order:
        file_name = "ao" + file_name
    file_name = os.path.join("word_graphs", file_name)

    external_path_data = []
    for size, minimal_external_paths in external_paths_container.items():
        identifier = get_word_graph_filename(ascending_order, size, name_base)
        external_path_data.append("\n\n" + identifier + "\n")
        for vertex1 in paths_by_vertex in minimal_external_paths.items():
            for vertex2, paths in paths_by_vertex.items():
                external_path_data.append("\n" + vertex1 + " --> " + vertex2 + ":")
                for path in paths:
                    external_path_data.append(path)

    store_data(external_path_data, file_name)
