"""
"""

from word_explorer.io import store_data, retrieve_data


def get_word_graph_filename(ascending_order, size, name_base="word_graph_size"):
    file_name = name_base + str(size) + ".txt"
    if ascending_order:
        file_name = "ao" + file_name
    return file_name


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
                in line.strip()[colon_index+3:len(line)-2].split(", ")}

    return word_graph


def store_word_subgraphs():
    pass