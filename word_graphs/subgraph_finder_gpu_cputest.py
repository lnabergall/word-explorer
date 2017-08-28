"""
CUDA Python version of subgraph_finder.py to support parallel computations via GPU.
"""

from time import time
from math import factorial, floor, log10
from collections import Counter

from numpy import array, zeros, int64, float64
int64_py = int64
zeros_py = zeros
# from numba import cuda
# from numba.types import int64

#from word_graph_gpu2 import *
from word_graph_gpu_cputest import *


NEIGHBOR_MAX = 150
PATH3_MAX = 5500


def find_subgraphs(subgraph_type, word_graph, 
                   ascending_order, word_class, data=None):
    Word = word_class
    start_time = time()
    max_nbhd_size = max(len(word_graph[word]) for word in word_graph)
    print(max_nbhd_size)
    max_word_size = max(len(word)//2 for word in word_graph)
    word_graph_array = zeros_py((len(word_graph), max_nbhd_size+1), dtype=int64_py)
    word_list = list(word_graph.keys())
    word_list.sort(key=lambda word: len(word))
    for i, word in enumerate(word_list):
        word_integer = word_from_letters_list(list(map(int, list(word))))
        neighborhood = [word_from_letters_list(list(map(int, list(neighbor))))
                        for neighbor in list(word_graph[word])]
        for j in range(word_graph_array.shape[1]):
            if j == 0:
                if word_integer == 0:
                    word_integer = -1
                word_graph_array[i, j] = word_integer
            else:
                try:
                    neighbor = neighborhood[j-1]
                    if neighbor == 0:
                        neighbor = -1
                    word_graph_array[i, j] = neighbor
                except IndexError:
                    pass
    threads_perblock = 32
    blocks_perdim = ((word_graph_array.shape[0] 
                      + (threads_perblock - 1)) // threads_perblock)
    # device_word_graph_array = cuda.to_device(word_graph_array)
    if subgraph_type == "3-path":
        max_root_size = max_word_size - 2
        path_root_count = 1
        for i in range(max_root_size):
            root_size = i+1
            if ascending_order:
                path_root_count += (factorial(2*root_size) 
                                    // ((2**root_size) * factorial(root_size)))
            else:
                raise NotImplementedError
        paths = zeros_py((path_root_count, PATH3_MAX, 3), dtype=int64_py)
        # device_paths = cuda.to_device(paths)
        # blocks_perdim = ((path_root_count
        #                   + (threads_perblock - 1)) // threads_perblock)
        path_num_array = zeros_py(path_root_count, dtype=int64_py)
        # device_path_num_array = cuda.to_device(path_num_array)
        # find_3paths[blocks_perdim, threads_perblock](
        #     device_word_graph_array, device_paths, device_path_num_array)
        # paths_found = device_paths.copy_to_host()
        # path_num_array = device_path_num_array.copy_to_host()
        paths_found, path_num_array = find_3paths(
            word_graph_array, paths, path_num_array)
        print(path_num_array)
        paths_list = paths_found.tolist()
        end_time = time()
        paths = []
        for i in range(paths_found.shape[0]):
            for j in range(paths_found.shape[1]):
                if paths_found[i, j, 0] != 0:
                    word1 = (Word(str(paths_found[i, j, 0])) 
                        if paths_found[i, j, 0] != -1 else Word(""))
                    word2 = (Word(str(paths_found[i, j, 1])) 
                        if paths_found[i, j, 1] != -1 else Word(""))
                    word3 = (Word(str(paths_found[i, j, 2])) 
                        if paths_found[i, j, 2] != -1 else Word(""))
                    if word3 is None and i < 1:
                        print(word3, i, j)
                    paths.append((word1, word2, word3))
        path_counts = Counter(paths)
        duplicated_paths = list(set([path for path in paths if path_counts[path] > 1]))
        duplicated_paths.sort()
        count_repetitions = [0, 0]
        for path in duplicated_paths:
            if str(path[0]) == "122133" and str(path[1]) == "12231434":
                count_repetitions[1] += 1
            elif str(path[0]) == "1212" and str(path[1]) == "12312434":
                count_repetitions[0] += 1
            else:
                raise ValueError("Unexpected path type!")
            print(path, path_counts[path])
        print(count_repetitions)
        print("\n", end_time - start_time, "\n")
        return paths
    elif subgraph_type == "4-path":
        length3_paths = data
        paths = zeros_py((len(length3_paths)*15, 4), dtype=int64_py)
        device_paths = cuda.to_device(paths)
        length3_paths_array = zeros_py((len(length3_paths), 3), dtype=int64_py)
        for i, path in enumerate(length3_paths):
            for j, word in enumerate(path):
                word_integer = word_from_letters_list(list(map(int, list(word))))
                if word_integer == 0:
                    word_integer = -1
                length3_paths_array[i, j] = word_integer
        device_3paths = cuda.to_device(length3_paths_array)
        blocks_perdim = ((len(length3_paths) 
                         + (threads_perblock - 1)) // threads_perblock)
        find_4paths[blocks_perdim, threads_perblock](
            device_word_graph_array, device_3paths, device_paths)
        paths_found = device_paths.copy_to_host()
        paths_list = paths_found.tolist()
        end_time = time()
        paths = []
        for i in range(paths_found.shape[0]):
            if paths_found[i, 0] != 0 and paths_found[i, 1] != 0:
                word1 = Word("".join(list(map(
                    str, letters_from_int(paths_found[i, 0])))))
                word2 = Word("".join(list(map(
                    str, letters_from_int(paths_found[i, 1])))))
                word3 = Word("".join(list(map(
                    str, letters_from_int(paths_found[i, 2])))))
                word4 = Word("".join(list(map(
                    str, letters_from_int(paths_found[i, 3])))))
                paths.append((word1, word2, word3, word4))
        print(end_time - start_time)
        return paths
    elif subgraph_type == "triangle":
        pass
    elif subgraph_type == "square":
        pass
    elif subgraph_type == "cube":
        pass


# @cuda.jit("int64[:](int64[:,:], int64, int64, int64[:])", device=True)
def get_row(array2D, index, offset, flat_array):
    for i in range(array2D.shape[1]):
        if i >= offset:
            flat_array[i-offset] = array2D[index, i]
    return flat_array


# @cuda.jit("int64(int64[:,:], int64)", device=True)
def index_from_dict(dict_array, element):
    for i in range(dict_array.shape[0]):
        if dict_array[i, 0] == element:
            index = i
            break
    return index


# @cuda.jit("boolean(int64[:], int64)", device=True)
def contains(flat_array, element):
    for i in range(flat_array.size):
        if flat_array[i] == element:
            return True
    return False


# @cuda.jit("int64(int64[:,:])", device=True)
# def maximum_key_size(dict_array):
#     max_size = 0
#     for i in range(dict_array.shape[0]):
#         size = size(dict_array[i, 0])
#         if size >= max_size:
#             max_size = size
#     return max_size


# @cuda.jit("int64(int64[:,:])", device=True)
# def max_object_count(word_graph):
#     max_word_size = maximum_key_size(word_graph)    # Assumes fully expanded word graph
#     return 100 * pow(10, max_word_size)


# @cuda.jit("int64[:](int64[:,:], int64, int64[:])", device=True)
def get_value(dict_array, key, flat_array):
    index = index_from_dict(dict_array, key)
    for i in range(dict_array.shape[1]):
        if i >= 1:
            flat_array[i-1] = dict_array[index, i]
    return flat_array


# @cuda.jit("int64(int64)", device=True)
def digit_count(integer):
    if integer <= 0:
        return 0
    else:
        return int64(floor(log10(float64(integer))) + 1)


# @cuda.jit("int64(int64)", device=True)
def length(word):
    return digit_count(word)


# @cuda.jit("void(int64[:,:], int64[:,:,:], int64[:])")
def find_3paths(word_graph, paths, path_num_array):
    # thread_num = cuda.grid(1)
    for i in range(paths.shape[0]):
        # if i == thread_num:
        paths_index = 0
        word = word_graph[i, 0]
        # neighbors1_array = cuda.local.array(NEIGHBOR_MAX, int64)
        neighbors1_array = zeros_py(NEIGHBOR_MAX, int64_py)
        neighbors1 = get_row(word_graph, i, 1, neighbors1_array)
        for j in range(neighbors1.size):
            neighbor1 = neighbors1[j]
            if neighbor1 != 0:
                # neighbors2_array = cuda.local.array(NEIGHBOR_MAX, int64)
                neighbors2_array = zeros_py(NEIGHBOR_MAX, int64_py)
                neighbors2 = get_value(word_graph, neighbor1, neighbors2_array)
                for k in range(neighbors2.size):
                    neighbor2 = neighbors2[k]
                    if neighbor2 != 0:
                        # neighbors3_array = cuda.local.array(NEIGHBOR_MAX, int64)
                        neighbors3_array = zeros_py(NEIGHBOR_MAX, int64_py)
                        neighbors3 = get_value(word_graph, neighbor2, neighbors3_array)
                        if (not contains(neighbors1, neighbor2) 
                                and not contains(neighbors3, word)
                                and length(word) < length(neighbor1) 
                                and length(neighbor1) < length(neighbor2)):
                            paths[i, paths_index, 0] = word
                            paths[i, paths_index, 1] = neighbor1
                            paths[i, paths_index, 2] = neighbor2
                            paths_index += 1
        path_num_array[i] = paths_index

    return paths, path_num_array

"""
@cuda.jit("void(int64[:,:], int64[:,:], int64[:,:])")
def find_4paths(word_graph, length3_paths, paths):
    thread_num = cuda.grid(1)
    num_3paths = length3_paths.shape[0]
    for i in range(num_3paths):
        if i == thread_num:
            neighbors2_array = cuda.local.array(NEIGHBOR_MAX, int64)
            neighbors2 = get_value(word_graph, length3_paths[i, 2], neighbors2_array)
            for j in range(neighbors2.size):
                word = neighbors2[j]
                if word != 0:
                    neighbors_array = cuda.local.array(NEIGHBOR_MAX, int64)
                    neighbors = get_value(word_graph, word, neighbors_array)
                    neighbors0_array = cuda.local.array(NEIGHBOR_MAX, int64)
                    neighbors0 = get_value(word_graph, length3_paths[i, 0], neighbors0_array)
                    neighbors1_array = cuda.local.array(NEIGHBOR_MAX, int64)
                    neighbors1 = get_value(word_graph, length3_paths[i, 1], neighbors1_array)
                    if (length(word) >= length(length3_paths[i, 2])
                            and not contains(neighbors0, word)
                            and not contains(neighbors, length3_paths[i, 0])
                            and not contains(neighbors1, word)
                            and not contains(neighbors, length3_paths[i, 1])):
                        for k in range(paths.shape[0]):
                            if paths[k, 0] != 0 and paths[k, 1] != 0:
                                index = k
                                break
                        for k in range(3):
                            paths[index, k] = length3_paths[i, k]
                        paths[index, 3] = word


@cuda.jit("void(int64[:,:], int64[:,:])")
def find_triangles(word_graph, triangles):
    # triangles = zeros_py((max_object_count(word_graph), 3), dtype=int64)
    thread_num = cuda.grid(1)
    index = 0
    for i in range(word_graph.shape[0]):
        if i == thread_num:
            word = word_graph[i, 0]
            neighbors_array = cuda.local.array(NEIGHBOR_MAX, int64)
            neighbors = get_row(word_graph, i, 1, neighbors_array)
            for j in range(neighbors.size):
                neighbor1 = neighbors[j]
                if neighbor1 != 0:
                    neighbors1_array = cuda.local.array(NEIGHBOR_MAX, int64)
                    neighbors1 = get_value(word_graph, neighbor1, neighbors1_array)
                    for k in range(neighbors.size):
                        neighbor2 = neighbors[k]
                        if neighbor2 != 0:
                            neighbors2_array = cuda.local.array(NEIGHBOR_MAX, int64)
                            neighbors2 = get_value(word_graph, neighbor2, neighbors2_array)
                            if contains(neighbors1, neighbor2):
                                triangles[index, 0] = word
                                triangles[index, 1] = neighbor1
                                triangles[index, 2] = neighbor2
                                index += 1
                            elif contains(neighbors2, neighbor1):
                                triangles[index, 0] = word
                                triangles[index, 1] = neighbor2
                                triangles[index, 2] = neighbor1
                                index += 1


@cuda.jit("void(int64[:,:], int64[:,:], int64[:,:])")
def find_squares(word_graph, length3_paths, squares):
    #squares = zeros_py(max_object_count(word_graph), 4), dtype=int64)
    thread_num = cuda.grid(1)
    index = 0
    for i in range(length3_paths.shape[0]):
        for j in range(length3_paths.shape[0]):
            if i*length3_paths.shape[0] + j == thread_num:
                path1_array = cuda.local.array(3, int64)
                path1 = get_row(length3_paths, i, 0, path1_array)
                path2_array = cuda.local.array(3, int64)
                path2 = get_row(length3_paths, j, 0, path2_array)
                if ((path1[0] == path2[0] and path1[2] == path2[2])
                        or (path1[0] == path2[2] and path1[2] == path2[0])):
                    neighbors1_array = cuda.local.array(NEIGHBOR_MAX, int64)
                    neighbors1 = get_value(word_graph, path1[1], neighbors1_array)
                    neighbors2_array = cuda.local.array(NEIGHBOR_MAX, int64)
                    neighbors2 = get_value(word_graph, path2[1], neighbors2_array)
                    if (not contains(neighbors1, path2[1]) 
                            and not contains(neighbors2, path1[1])):
                        squares[index, 0] = path1[0]
                        squares[index, 1] = path1[1]
                        squares[index, 2] = path1[2]
                        squares[index, 3] = path2[1]
                        index += 1


@cuda.jit("void(int64[:,:], int64[:,:], int64[:,:])")
def find_cubes(word_graph, sorted_squares, cubes):
    pass
"""