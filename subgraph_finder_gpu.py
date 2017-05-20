"""
CUDA Python version of subgraph_finder.py to support parallel computations via GPU.
"""

from time import time
from math import factorial
from collections import Counter

from numpy import array, zeros, int64
int64_py = int64
zeros_py = zeros
from numba import cuda
from numba.types import int64

from word_graph_gpu2 import *


NEIGHBOR_MAX = 250
PATH3_MAX = 20000


def create_paths_array(paths, length):
    paths_array = zeros_py((len(paths), length), dtype=int64_py)
    for i, path in enumerate(paths):
        for j, word in enumerate(path):
            word_integer = word_from_letters_list(list(map(int, list(word))))
            if word_integer == 0:
                word_integer = -1
            paths_array[i, j] = word_integer
    
    return paths_array


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
    device_word_graph_array = cuda.to_device(word_graph_array)
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
        device_paths = cuda.to_device(paths)
        blocks_perdim = ((path_root_count
                          + (threads_perblock - 1)) // threads_perblock)
        path_num_array = zeros_py(path_root_count, dtype=int64_py)
        device_path_num_array = cuda.to_device(path_num_array)
        print("Starting GPU computations...")
        find_3paths[blocks_perdim, threads_perblock](
            device_word_graph_array, device_paths, device_path_num_array)
        print("Finished GPU computations.")
        paths_found = device_paths.copy_to_host()
        path_num_array = device_path_num_array.copy_to_host()
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
        all_paths = []
        batches = 175
        for k in range(batches):
            print("Batch", k)
            length3_paths_batch = length3_paths[k*len(length3_paths) // batches: 
                                                (k+1)*len(length3_paths) // batches]
            paths = zeros_py((len(length3_paths_batch), 275, 4), dtype=int64_py)
            device_paths = cuda.to_device(paths)
            length3_paths_array = create_paths_array(length3_paths_batch, 3)
            device_3paths = cuda.to_device(length3_paths_array)
            blocks_perdim = ((len(length3_paths_batch) 
                             + (threads_perblock - 1)) // threads_perblock)
            paths_per = zeros_py(len(length3_paths_batch), dtype=int64_py)
            device_paths_per = cuda.to_device(paths_per)
            print("Starting GPU computations...")
            find_4paths[blocks_perdim, threads_perblock](
                device_word_graph_array, device_3paths, 
                device_paths, device_paths_per)
            print("Finished GPU computations.")
            paths_found = device_paths.copy_to_host()
            paths_per = device_paths_per.copy_to_host().tolist()
            print(max(paths_per))
            end_time = time()
            paths = []
            for i in range(paths_found.shape[0]):
                for j in range(paths_found.shape[1]):
                    if paths_found[i, j, 0] != 0 and paths_found[i, j, 1] != 0:
                        word1 = (Word(str(paths_found[i, j, 0])) 
                                 if paths_found[i, j, 0] != -1 else Word(""))
                        word2 = (Word(str(paths_found[i, j, 1])) 
                                 if paths_found[i, j, 1] != -1 else Word(""))
                        word3 = (Word(str(paths_found[i, j, 2])) 
                                 if paths_found[i, j, 2] != -1 else Word(""))
                        word4 = (Word(str(paths_found[i, j, 3])) 
                                 if paths_found[i, j, 3] != -1 else Word(""))
                        paths.append((word1, word2, word3, word4))
            all_paths.extend(paths)
        print(end_time - start_time)
        return all_paths
    elif subgraph_type == "triangle":
        all_triangles = []
        batches = 40
        for k in range(batches):
            print("Batch", k)
            word_batch_indices = list(range(k*word_graph_array.shape[0] // batches, 
                (k+1)*word_graph_array.shape[0] // batches))
            batch_indices_array = zeros_py(len(word_batch_indices), dtype=int64_py)
            for i, index in enumerate(word_batch_indices):
                batch_indices_array[i] = index
            device_batch_indices = cuda.to_device(batch_indices_array)
            triangles = zeros_py((len(word_batch_indices), 2000, 3), dtype=int64_py)
            device_triangles = cuda.to_device(triangles)
            triangles_per = zeros_py(word_graph_array.shape[0], dtype=int64_py)
            device_triangles_per = cuda.to_device(triangles_per)
            blocks_perdim = ((len(word_batch_indices)
                + (threads_perblock - 1)) // threads_perblock)
            print("Starting GPU computations...")
            find_triangles[blocks_perdim, threads_perblock](
                device_word_graph_array, batch_indices_array, 
                device_triangles, device_triangles_per)
            print("Finished GPU computations.")
            triangles_found = device_triangles.copy_to_host()
            triangles_per = device_triangles_per.copy_to_host().tolist()
            print(max(triangles_per))
            end_time = time()
            triangles = []
            for i in range(triangles_found.shape[0]):
                for j in range(triangles_found.shape[1]):
                    if triangles_found[i, j, 0] != 0 and triangles_found[i, j, 1] != 0:
                        word1 = (Word(str(triangles_found[i, j, 0])) 
                                 if triangles_found[i, j, 0] != -1 else Word(""))
                        word2 = (Word(str(triangles_found[i, j, 1])) 
                                 if triangles_found[i, j, 1] != -1 else Word(""))
                        word3 = (Word(str(triangles_found[i, j, 2])) 
                                 if triangles_found[i, j, 2] != -1 else Word(""))
                        triangles.append((word1, word2, word3))
            all_triangles.extend(triangles)
        print(end_time - start_time)
        return all_triangles
    elif subgraph_type == "square":
        length3_paths = data
        length3_paths_array = create_paths_array(length3_paths, 3)
        device_all_3paths = cuda.to_device(length3_paths_array)
        all_squares = []
        batches = 1
        for k in range(batches):
            print("Batch", k)
            length3_paths_batch = length3_paths[k*len(length3_paths) // batches: 
                                                (k+1)*len(length3_paths) // batches]
            squares = zeros_py(
                (len(length3_paths)*len(length3_paths_batch), 4), dtype=int64_py)
            device_squares = cuda.to_device(squares)
            batch_array = create_paths_array(length3_paths_batch, 3)
            device_3paths_batch = cuda.to_device(batch_array)
            blocks_perdim = ((len(length3_paths)*len(length3_paths_batch)
                             + (threads_perblock - 1)) // threads_perblock)
            print("Starting GPU computations...")
            find_squares[blocks_perdim, threads_perblock](
                device_word_graph_array, device_all_3paths, 
                device_3paths_batch, device_squares)
            print("Finished GPU computations.")
            squares_found = device_squares.copy_to_host()
            end_time = time()
            squares = []
            for i in range(squares_found.shape[0]):
                if squares_found[i, 0] != 0 and squares_found[i, 1] != 0:
                    word1 = (Word(str(squares_found[i, 0])) 
                             if squares_found[i, 0] != -1 else Word(""))
                    word2 = (Word(str(squares_found[i, 1])) 
                             if squares_found[i, 1] != -1 else Word(""))
                    word3 = (Word(str(squares_found[i, 2])) 
                             if squares_found[i, 2] != -1 else Word(""))
                    word4 = (Word(str(squares_found[i, 3])) 
                             if squares_found[i, 3] != -1 else Word(""))
                    squares.append((word1, word2, word3, word4))
            all_squares.extend(squares)
        print(end_time - start_time)
        return all_squares
    elif subgraph_type == "cube":
        pass


@cuda.jit("int64[:](int64[:,:], int64, int64, int64[:])", device=True)
def get_row(array2D, index, offset, flat_array):
    for i in range(array2D.shape[1]):
        if i >= offset:
            flat_array[i-offset] = array2D[index, i]
    return flat_array


@cuda.jit("int64(int64[:,:], int64)", device=True)
def index_from_dict(dict_array, element):
    for i in range(dict_array.shape[0]):
        if dict_array[i, 0] == element:
            index = i
            break
    return index


@cuda.jit("boolean(int64[:], int64)", device=True)
def contains(flat_array, element):
    for i in range(flat_array.size):
        if flat_array[i] == element:
            return True
    return False


@cuda.jit("int64[:](int64[:,:], int64, int64[:])", device=True)
def get_value(dict_array, key, flat_array):
    index = index_from_dict(dict_array, key)
    for i in range(dict_array.shape[1]):
        if i >= 1:
            flat_array[i-1] = dict_array[index, i]
    return flat_array


@cuda.jit("void(int64[:,:], int64[:,:,:], int64[:])")
def find_3paths(word_graph, paths, path_num_array):
    thread_num = cuda.grid(1)
    for i in range(paths.shape[0]):
        if i == thread_num:
            paths_index = 0
            word = word_graph[i, 0]
            neighbors1_array = zeros1D(cuda.local.array(NEIGHBOR_MAX, int64))
            neighbors1 = get_row(word_graph, i, 1, neighbors1_array)
            for j in range(neighbors1.size):
                neighbor1 = neighbors1[j]
                if neighbor1 != 0:
                    neighbors2_array = zeros1D(cuda.local.array(NEIGHBOR_MAX, int64))
                    neighbors2 = get_value(word_graph, neighbor1, neighbors2_array)
                    for k in range(neighbors2.size):
                        neighbor2 = neighbors2[k]
                        if neighbor2 != 0:
                            neighbors3_array = zeros1D(
                                cuda.local.array(NEIGHBOR_MAX, int64))
                            neighbors3 = get_value(
                                word_graph, neighbor2, neighbors3_array)
                            if (not contains(neighbors1, neighbor2) 
                                    and not contains(neighbors3, word)
                                    and length(word) < length(neighbor1) 
                                    and length(neighbor1) < length(neighbor2)):
                                paths[i, paths_index, 0] = word
                                paths[i, paths_index, 1] = neighbor1
                                paths[i, paths_index, 2] = neighbor2
                                paths_index += 1
            path_num_array[i] = paths_index


@cuda.jit("void(int64[:,:], int64[:,:], int64[:,:,:], int64[:])")
def find_4paths(word_graph, length3_paths, paths, paths_per):
    thread_num = cuda.grid(1)
    for i in range(length3_paths.shape[0]):
        if i == thread_num:
            path_count = 0
            neighbors2_array = zeros1D(cuda.local.array(NEIGHBOR_MAX, int64))
            neighbors2 = get_value(word_graph, length3_paths[i, 2], neighbors2_array)
            for j in range(neighbors2.size):
                word = neighbors2[j]
                if word != 0:
                    neighbors_array = zeros1D(cuda.local.array(NEIGHBOR_MAX, int64))
                    neighbors = get_value(word_graph, word, neighbors_array)
                    neighbors0_array = zeros1D(cuda.local.array(NEIGHBOR_MAX, int64))
                    neighbors0 = get_value(
                        word_graph, length3_paths[i, 0], neighbors0_array)
                    neighbors1_array = zeros1D(cuda.local.array(NEIGHBOR_MAX, int64))
                    neighbors1 = get_value(
                        word_graph, length3_paths[i, 1], neighbors1_array)
                    if (length(word) >= length(length3_paths[i, 2])
                            and not contains(neighbors0, word)
                            and not contains(neighbors, length3_paths[i, 0])
                            and not contains(neighbors1, word)
                            and not contains(neighbors, length3_paths[i, 1])):
                        for k in range(3):
                            paths[i, path_count, k] = length3_paths[i, k]
                        paths[i, path_count, 3] = word
                        path_count += 1
            paths_per[i] = path_count


@cuda.jit("void(int64[:,:], int64[:], int64[:,:,:], int64[:])")
def find_triangles(word_graph, word_indices, triangles, triangles_per):
    thread_num = cuda.grid(1)
    for i in range(word_indices.size):
        if i == thread_num:
            triangle_count = 0
            i = word_indices[i]
            word = word_graph[i, 0]
            neighbors_array = zeros1D(cuda.local.array(NEIGHBOR_MAX, int64))
            neighbors = get_row(word_graph, i, 1, neighbors_array)
            for j in range(neighbors.size):
                neighbor1 = neighbors[j]
                if neighbor1 != 0:
                    neighbors1_array = zeros1D(cuda.local.array(NEIGHBOR_MAX, int64))
                    neighbors1 = get_value(word_graph, neighbor1, neighbors1_array)
                    for k in range(neighbors.size):
                        neighbor2 = neighbors[k]
                        if neighbor2 != 0:
                            neighbors2_array = zeros1D(
                                cuda.local.array(NEIGHBOR_MAX, int64))
                            neighbors2 = get_value(
                                word_graph, neighbor2, neighbors2_array)
                            if contains(neighbors1, neighbor2):
                                triangles[i, triangle_count, 0] = word
                                triangles[i, triangle_count, 1] = neighbor1
                                triangles[i, triangle_count, 2] = neighbor2
                                triangle_count += 1
                            elif contains(neighbors2, neighbor1):
                                triangles[i, triangle_count, 0] = word
                                triangles[i, triangle_count, 1] = neighbor2
                                triangles[i, triangle_count, 2] = neighbor1
                                triangle_count += 1
            triangles_per[i] = triangle_count


@cuda.jit("void(int64[:,:], int64[:,:], int64[:,:], int64[:,:])")
def find_squares(word_graph, length3_paths, length3_path_batch, squares):
    thread_num = cuda.grid(1)
    for i in range(length3_paths.shape[0]):
        for j in range(length3_path_batch.shape[0]):
            if i*length3_paths.shape[0] + j == thread_num:
                path1_array = zeros1D(cuda.local.array(3, int64))
                path1 = get_row(length3_paths, i, 0, path1_array)
                path2_array = zeros1D(cuda.local.array(3, int64))
                path2 = get_row(length3_path_batch, j, 0, path2_array)
                if path1[1] != path2[1] and ((path1[0] == path2[0] 
                        and path1[2] == path2[2]) or (path1[0] == path2[2] 
                        and path1[2] == path2[0])):
                    neighbors1_array = zeros1D(cuda.local.array(NEIGHBOR_MAX, int64))
                    neighbors1 = get_value(word_graph, path1[1], neighbors1_array)
                    neighbors2_array = zeros1D(cuda.local.array(NEIGHBOR_MAX, int64))
                    neighbors2 = get_value(word_graph, path2[1], neighbors2_array)
                    if (not contains(neighbors1, path2[1]) 
                            and not contains(neighbors2, path1[1])):
                        squares[thread_num, 0] = path1[0]
                        squares[thread_num, 1] = path1[1]
                        squares[thread_num, 2] = path1[2]
                        squares[thread_num, 3] = path2[1]


@cuda.jit("void(int64[:,:], int64[:,:], int64[:,:])")
def find_cubes(word_graph, sorted_squares, cubes):
    pass