"""
Input/output functions for the word_graphs API, including 
functions for storing and retrieving word graphs and subgraphs.

Functions:

    get_integer, get_word_graph_filename, get_word_subgraph_filename, 
    store_word_graph, retrieve_word_graph, retrieve_word_graph_statistics,
    store_word_subgraphs, retrieve_word_subgraphs, 
    retrieve_subgraph_statistics, store_subgraph_statistics,
    store_external_paths
"""

from itertools import chain

from word_explorer.objects import Word
from word_explorer.io import store_data, retrieve_data


def get_integer(string):
    integers = re.findall(r"\d+", string)
    if not integers:
        raise NotImplementedError
    else:
        return int(integers[-1])


def get_word_graph_filename(ascending_order, size, name_base, name_suffix=""):
    file_name = name_base + str(size) + "_" + name_suffix + ".txt"
    if ascending_order:
        file_name = "ao" + file_name
    return file_name


def get_word_subgraph_filename(ascending_order, size, subgraph_type, 
                               name_base="word_graph_size", name_suffix="", 
                               sorted_=False):
    file_name = get_word_graph_filename(
        ascending_order, size, name_base, name_suffix)[:-4]
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


def retrieve_word_graph_statistics(ascending_order, size, 
                                   name_base="word_graph_size", name_suffix=""):
    file_name = get_word_graph_filename(
        ascending_order, size, name_base, name_suffix)
    file_name = os.path.join("word_graphs", file_name)
    for line in retrieve_data(file_name):
        if line.startswith("Vertex count"):
            vertex_count = get_integer(line)
        elif line.startswith("Edge count"):
            edge_count = get_integer(line)

    return {"vertices": vertex_count, "edges": edge_count}


def store_word_subgraphs(subgraphs, subgraph_type, ascending_order, 
                         size, name_base="word_graph_size", name_suffix=""):
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
        ascending_order, size, subgraph_type, 
        name_base, name_suffix, sorted_)
    subgraph_file_name = os.path.join("word_graphs", subgraph_file_name)
    store_data(subgraph_data, subgraph_file_name)


def retrieve_word_subgraphs(ascending_order, size, subgraph_type, 
                            name_base="word_graph_size", name_suffix="", 
                            sorted_=False):
    subgraph_file_name = get_word_subgraph_filename(
        ascending_order, size, subgraph_type, 
        name_base, name_suffix, sorted_)
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


def retrieve_subgraph_statistics(ascending_order, sizes, 
                                 name_base="word_graph_size", name_suffix=""):
    from .subgraphs import SUBGRAPH_TYPES  # For circular import
    statistics = {}
    for size in sizes:
        statistics[size] = retrieve_word_graph_statistics(
            ascending_order, size, name_base, name_suffix)
        for subgraph_type in SUBGRAPH_TYPES:
            sorted_ = True if subgraph_type == "square" else False
            file_name = get_word_subgraph_filename(
                ascending_order, size, subgraph_type, 
                name_base, name_suffix, sorted_)
            statistics[size][subgraph_type] = {}
            try:
                for line in retrieve_data(file_name):
                    if line.contains("count"):
                        if line.startswith("((") or line.startswith("[("):
                            subgraph_class = line[:line.find("))")+2]
                            statistics[size][subgraph_type][subgraph_class] = (
                                get_integer(line))
                        else:
                            statistics[size][subgraph_type]["total"] = (
                                get_integer(line))
            except FileNotFoundError:
                pass

    return statistics


def store_subgraph_statistics(subgraph_statistics, ascending_order, 
                              name_base="word_graph", name_suffix=""):
    file_name = name_base + "_" + name_suffix + "_subgraph_statistics.txt"
    file_name = os.path.join("word_graphs", file_name)
    title = (name_base + name_suffix).replace("_", " ").title()
    if ascending_order:
        "Ascending Order " + title
    subgraph_data = ["\n" + title]
    for size in subgraph_statistics:
        subgraph_data.append("\n\n---------------- Words of size <= " 
                             + size + "----------------")
        subgraph_data.append("\nVertices: " + subgraph_statistics[size]["vertices"])
        subgraph_data.append("Edges: " + subgraph_statistics[size]["edges"])
        for subgraph_type, statistics_dict in subgraph_statistics[size].items():
            for subgraph_class, value in statistics_dict.items():
                if subgraph_class == "total":
                    subgraph_data.append("\n" + subgraph_type.title() 
                                         + "s: " + str(value))
                else:
                    subgraph_data.append(subgraph_class.title() + " " 
                                         + subgraph_type + "s: " + str(value))

    store_data(subgraph_data, file_name)
    

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
