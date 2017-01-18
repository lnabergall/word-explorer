"""
CUDA Python version of subgraph_finder.py to support parallel computations via GPU.
"""

from accelerate.cuda import cuda
from numba import vectorize
from numpy import array, zeros, int32

from word_graph_gpu import *

def get_row(array2D, index, offset):
    num_columns = array2D.shape(1)
    row = zeros(num_columns, dtype=int32)
        for i in range(num_columns):
            if i >= offset:
                row[i-offset] = array2D[index, i]
    return row

def index_from_dict(dict_array, element):
    for i in range(word_graph.shape(0)):
        if word_graph[i, 0] == element:
            index = i
            break
    return index

def contains(flat_array, element):
    for i in range(flat_array.size):
        if flat_array[i] == element:
            return True
    return False

@cuda.jit("void(int32[:,:])")
def find_3paths(word_graph):
    num_vertices = word_graph.shape(0)
    num_columns = word_graph.shape(1)
    for i in range(num_vertices):
        neighbors1 = remove_zeros(get_row(word_graph, i, 1))
        for j in range(neighbors1.size):
            index1 = index_from_dict(word_graph, neighbors1[j])
            neighbors2 = remove_zeros(get_row(word_graph, index1, 1))
            for k in range(neighbors2.size):
                if (not contains(neighbors1, neighbors2[k]) 
                        not )

