"""
Input/output functions for the word_graphs API, including 
functions for storing and retrieving word graphs and subgraphs.

Functions:

    get_word_graph_filename, get_word_subgraph_filename, store_word_graph, 
    retrieve_word_graph, store_word_subgraphs, retrieve_word_subgraphs
"""

from itertools import chain

from word_explorer.objects import Word
from word_explorer.io import store_data, retrieve_data


def get_word_graph_filename(ascending_order, size, name_base):
    file_name = name_base + str(size) + ".txt"
    if ascending_order:
        file_name = "ao" + file_name
    return file_name


def get_word_subgraph_filename(ascending_order, size, name_base, 
                               subgraph_type, sorted_=False):
    file_name = get_word_graph_filename(ascending_order, size, name_base)[:-4]
    file_name += "_" + subgraph_type 
    file_name = file_name + "_sorted" if sorted_ else file_name
    return file_name + ".txt"


def store_word_graph(word_graph):
    file_name = os.path.join("word_graphs", word_graph.file_name)
    word_graph_data = ["Vertex count: " + str(word_graph.vertex_count), 
                       "Edge count: " + str(word_graph.edge_count) + "\n\n"]
    words = list(word_graph.directed_neighborhoods.keys())
    words.sort()
    for word in words:
        if word_graph.directed_neighborhoods[word]:
            neighborhood = word + ": " + str(
                word_graph.directed_neighborhoods[word]).replace("\'", "")
            word_graph_data.append(neighborhood)
    store_data(word_graph_data, file_name)


def retrieve_word_graph(ascending_order, size, name_base="word_graph_size"):
    file_name = get_word_graph_filename(ascending_order, size, name_base)
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
    