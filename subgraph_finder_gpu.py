"""
CUDA Python version of subgraph_finder.py to support parallel computations via GPU.
"""

from accelerate.cuda import cuda
from numba import vectorize
from numpy import array, zeros, int32

from word_graph_gpu import *


@cuda.jit("int32[:](int32[:,:], int32, int32)", device=True)
def get_row(array2D, index, offset):
    num_columns = array2D.shape(1)
    row = zeros(num_columns, dtype=int32)
    for i in range(num_columns):
        if i >= offset:
            row[i-offset] = array2D[index, i]
    return row


@cuda.jit("int32(int32[:,:], int32)", device=True)
def index_from_dict(dict_array, element):
    for i in range(dict_array.shape(0)):
        if dict_array[i, 0] == element:
            index = i
            break
    return index


@cuda.jit("bool(int32[:], int32)", device=True)
def contains(flat_array, element):
    for i in range(flat_array.size):
        if flat_array[i] == element:
            return True
    return False


@cuda.jit("int32(int32[:,:])", device=True)
def maximum_key_size(dict_array):
    max_size = 0
    for i in range(dict_array.shape(0)):
        size = size(dict_array[i, 0])
        if size >= max_size:
            max_size = size
    return max_size


@cuda.jit("int32(int32[:,:])", device=True)
def max_object_count(word_graph):
    max_word_size = maximum_key_size(word_graph)    # Assumes fully expanded word graph
    return 100 * pow(10, max_word_size)


@cuda.jit("int32[:](int32[:,:], int32)", device=True)
def get_value(dict_array, key):
    index = index_from_dict(dict_array, key)
    return remove_zeros(get_row(dict_array, index, 1))


@cuda.jit("void(int32[:,:])")
def find_3paths(word_graph):
    paths = zeros((max_object_count(word_graph), 3), dtype=int32)
    paths_index = 0
    for i in range(word_graph.shape(0)):
        word = word_graph[i, 0]
        neighbors1 = remove_zeros(get_row(word_graph, i, 1))
        for j in range(neighbors1.size):
            neighbor1 = neighbors1[j]
            neighbors2 = get_value(word_graph, neighbor1)
            for k in range(neighbors2.size):
                neighbor2 = neighbors2[k]
                neighbors3 = get_value(word_graph, neighbor2)
                if (not contains(neighbors1, neighbor2) 
                        and not contains(neighbors3, word)
                        and length(word) <= length(neighbor1) 
                        and length(neighbor1) <= length(neighbor2)):
                    paths[paths_index, 0] = word
                    paths[paths_index, 1] = neighbor1
                    paths[paths_index, 2] = neighbor2
                    paths_index += 1
    return paths    # CAN'T RETURN VALUE, NEED TO SEND TO GPU FROM EXTERNAL CODE


@cuda.jit("void(int32[:,:], int32[:])")
def find_4paths(word_graph, length3_paths):
    num_3paths = length3_paths.shape(0)
    paths = zeros((num_3paths, 4), dtype=int32)
    paths_index = 0
    for i in range(num_3paths):
        neighbors2 = get_value(word_graph, num_3paths[i, 2])
        for j in range(neighbors2.size):
            word = neighbors2[j]
            neighbors = get_value(word_graph, word)
            neighbors0 = get_value(word_graph, num_3paths[i, 0])
            neighbors1 = get_value(word_graph, num_3paths[i, 1])
            if (length(word) >= length(num_3paths[i, 2])
                    and word not in neighbors0
                    and num_3paths[i, 0] not in neighbors
                    and word not in neighbors1
                    and num_3paths[i, 1] not in neighbors):
                for k in range(3):
                    paths[paths_index, k] = num_3paths[i, k]
                paths[paths_index, 3] = word
                paths_index += 1
    return paths    # CAN'T RETURN VALUE, NEED TO SEND TO GPU FROM EXTERNAL CODE 


@cuda.jit("void(int32[:,:])")
def find_triangles(word_graph):
    triangles = zeros((max_object_count(word_graph), 3), dtype=int32)
    index = 0
    for i in range(word_graph.shape(0)):
        word = word_graph[i, 0]
        neighbors = remove_zeros(get_row(word_graph, i, 1))
        for j in range(neighbors.size):
            neighbor1 = neighbors[j]
            neighbors1 = get_value(word_graph, neighbor1)
            for k in range(neighbors.size):
                neighbor2 = neighbors[k]
                neighbors2 = get_value(word_graph, neighbor2)
                if neighbor2 in neighbors1:
                    triangles[index, 0] = word
                    triangles[index, 1] = neighbor1
                    triangles[index, 2] = neighbor2
                    index += 1
                elif neighbor1 in neighbors2:
                    triangles[index, 0] = word
                    triangles[index, 1] = neighbor2
                    triangles[index, 2] = neighbor1
                    index += 1
    return triangles    # CAN'T RETURN VALUE..., and need to remove duplicates


@cuda.jit("void(int32[:,:])")
def find_squares(word_graph):
    pass