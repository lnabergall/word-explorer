"""
Defines a CUDA Python version of WordGraph.find_adjacent_vertices
to support parallel computations via an Nvidia GPU.
"""

from time import time
from math import log10, floor, fmod, ceil

from numba import cuda
from numpy import array, zeros, int64, int8
int64_py = int64
int8_py = int8
from numba.types import int8, int64, float64
zeros_py = zeros


REPEAT_WORD_AO = array((
    [1, 12, 123, 1234, 12345, 123456, 1234567, 12345678],
    [1, 12, 123, 1234, 12345, 123456, 1234567, 12345678],
), dtype=int64_py)
RETURN_WORD_AO = array((
    [1, 12, 123, 1234, 12345, 123456, 1234567, 12345678],
    [1, 21, 321, 4321, 54321, 654321, 7654321, 87654321],
), dtype=int64_py)
NUM_NEIGHBOR_LIMIT = 15000
SMALL_ARRAY_LENGTH = 16
LARGE_ARRAY_LENGTH = 5500
MAX_ARRAY_LENGTH = 5500


def find_adjacent_vertices(word_list, size_limit, ascending_order=False):
    if ascending_order:
        from word_graph import Word_eq as Word
    else:
        from word_graph import Word     # Avoid circular dependency
    words = word_list.copy()
    for i, word in enumerate(words):
        word_integer = word_from_letters_list(list(map(int, list(word))))
        words[i] = int64(word_integer)
    start_time = time()
    neighborhoods_all = []
    for i in range(200):
        if i == 199:
            word_batch = words[i*len(words) // 200 : ]
        else:
            word_batch = words[i*len(words) // 200 : (i+1)*len(words) // 200]
        word_array = array(word_batch, dtype=int64_py)
        pattern_instance_count = (REPEAT_WORD_AO.size // REPEAT_WORD_AO.ndim 
                                  + RETURN_WORD_AO.size // RETURN_WORD_AO.ndim)
        threads_perblock = 32
        blocks_perdim = (len(word_batch) + (threads_perblock - 1)) // threads_perblock
        device_word_array = cuda.to_device(word_array)
        device_repeats = cuda.to_device(REPEAT_WORD_AO)
        device_returns = cuda.to_device(RETURN_WORD_AO)
        neighbors_array = zeros_py((word_array.size, NUM_NEIGHBOR_LIMIT), int64_py)
        insertions = zeros_py((
            word_array.size, pattern_instance_count, NUM_NEIGHBOR_LIMIT), int64_py)
        instances = zeros_py((
            word_array.size, pattern_instance_count, LARGE_ARRAY_LENGTH, 2), int64_py)
        tuple_array = zeros_py((word_array.size, pattern_instance_count, 
                                MAX_ARRAY_LENGTH, SMALL_ARRAY_LENGTH), int8_py)
        permutation_array = zeros_py((word_array.size, pattern_instance_count, 
                                      LARGE_ARRAY_LENGTH, SMALL_ARRAY_LENGTH), int8_py)
        device_neighors_array = cuda.to_device(neighbors_array)
        device_insertions = cuda.to_device(insertions)
        device_instances = cuda.to_device(instances)
        device_tuple_array = cuda.to_device(tuple_array)
        device_permutation_array = cuda.to_device(permutation_array)
        test_emptyword = zeros_py(6, int64_py)
        device_test_emptyword = cuda.to_device(test_emptyword)
        compute_neighbors[blocks_perdim, threads_perblock](
            device_word_array, device_neighors_array,
            size_limit, device_repeats, device_returns, device_insertions,
            device_instances, device_tuple_array, device_permutation_array,
            device_test_emptyword)
        test_emptyword = device_test_emptyword.copy_to_host()
        print("Max pattern instance size considered:", test_emptyword.tolist())
        neighborhoods_found = device_neighors_array.copy_to_host()
        end_time = time()
        neighborhoods = neighborhoods_found.tolist()
        neighborhoods = [set([Word("".join(list(map(str, letters_from_int(neighbor))))) 
                             for neighbor in neighbors if neighbor != 0]) 
                         for neighbors in neighborhoods]
        neighborhoods_all.extend(neighborhoods)
    print("GPU time:", end_time - start_time)
    return neighborhoods_all


def word_from_letters_list(letters):
    word = 0
    for i in range(len(letters)):
        word += letters[i] * int(pow(10, len(letters)-i-1))
    return word


def digit_count_int(integer):
    if integer <= 0:
        return 0
    else:
        return int(floor(log10(float(integer)) + 1))


def ith_digit(integer, i):
    return int(fmod(integer, pow(10,i+1)) - fmod(integer, pow(10,i))) // pow(10,i)


def letters_from_int(integer):
    word_length = digit_count_int(integer)
    letters = list(range(word_length))
    for i in range(word_length):
        letters[word_length-i-1] = ith_digit(integer, i)
    return letters


@cuda.jit("int64[:](int64[:])", device=True)
def zeros1D(zeros_array):
    for i in range(zeros_array.size):
        zeros_array[i] = 0
    return zeros_array


@cuda.jit("int8[:](int8[:])", device=True)
def zeros1D8(zeros_array):
    for i in range(zeros_array.size):
        zeros_array[i] = 0
    return zeros_array


@cuda.jit("int64[:,:](int64[:,:])", device=True)
def zeros2D(zeros_array):
    for i in range(zeros_array.shape[0]):
        for j in range(zeros_array.shape[1]):
            zeros_array[i, j] = 0
    return zeros_array


@cuda.jit("int8[:,:](int8[:,:])", device=True)
def zeros2D8(zeros_array):
    for i in range(zeros_array.shape[0]):
        for j in range(zeros_array.shape[1]):
            zeros_array[i, j] = 0
    return zeros_array


@cuda.jit("boolean(int64[:,:])", device=True)
def nonzero2D(array):
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if array[i, j] != 0:
                return True
    return False


@cuda.jit("boolean(int8[:,:,:,:], int64, int64)", device=True)
def nonzero4D_2D(array, index1, index2):
    for i in range(array.shape[2]):
        for j in range(array.shape[3]):
            if array[index1, index2, i, j] != 0:
                return True
    return False


@cuda.jit("int64(int64[:])", device=True)
def nonzeros_count(flat_array):
    nonzero_elements = 0
    for i in range(flat_array.size):
        if flat_array[i] != 0:
            nonzero_elements += 1
    return nonzero_elements


@cuda.jit("int64(int64[:,:], int64)", device=True)
def nonzeros_count2D_1D(array, index):
    nonzero_elements = 0
    for i in range(array.shape[1]):
        if array[index, i] != 0:
            nonzero_elements += 1
    return nonzero_elements


@cuda.jit("int64(int64[:,:,:], int64, int64)", device=True)
def nonzeros_count3D_1D(array, index1, index2):
    nonzero_elements = 0
    for i in range(array.shape[2]):
        if array[index1, index2, i] != 0:
            nonzero_elements += 1
    return nonzero_elements


@cuda.jit("int64(int8[:,:,:,:], int64, int64, int64)", device=True)
def nonzeros_count4D_1D(array, index1, index2, index3):
    nonzero_elements = 0
    for i in range(array.shape[3]):
        if array[index1, index2, index3, i] != 0:
            nonzero_elements += 1
    return nonzero_elements


@cuda.jit("int64(int8[:])", device=True)
def nonzeros_count8(flat_array):
    nonzero_elements = 0
    for i in range(flat_array.size):
        if flat_array[i] != 0:
            nonzero_elements += 1
    return nonzero_elements


@cuda.jit("int64[:](int64[:], int64[:])", device=True)
def remove_zeros(flat_array, filtered_array):
    for i in range(flat_array.size):
        if flat_array[i] != 0:
            for j in range(filtered_array.size):
                if filtered_array[j] == 0:
                    filtered_array[j] = flat_array[i]
                    break
    return filtered_array


@cuda.jit("int64(int64)", device=True)
def digit_count(integer):
    if integer <= 0:
        return 0
    else:
        return int64(floor(log10(float64(integer))) + 1)


@cuda.jit("int64[:](int64, int64[:])", device=True)
def digits(integer, digits_array):
    order = digits_array.size
    for i in range(order):
        digits_array[order-i-1] = int64(
            int64(fmod(integer, pow(10,i+1)) - fmod(integer, pow(10,i))) // pow(10,i))
    return digits_array


@cuda.jit("int64(int64)", device=True)
def length(word):
    return digit_count(word)


@cuda.jit("int64(int64, int64)", device=True)
def ith_letter(word, i):
    return int64(
        int64(fmod(word, pow(10,i+1)) - fmod(word, pow(10,i))) // pow(10,i))


@cuda.jit("int64(int64, int64)", device=True)
def letter(word, index):
    return ith_letter(word, index)


@cuda.jit("int64(int64[:])", device=True)
def length_word_array(letters_array):
    return nonzeros_count(letters_array)


@cuda.jit("int64(int8[:])", device=True)
def length_word_array8(letters_array):
    return nonzeros_count8(letters_array)


@cuda.jit("int64[:](int64, int64[:])", device=True)
def letters(word, letters_array):
    word_length = length(word)
    for i in range(word_length):
        letters_array[word_length-i-1] = ith_letter(word, i)
    return letters_array


@cuda.jit("int64(int64)", device=True)
def size(word):
    return length(word) // 2


@cuda.jit("int64(int64, int64)", device=True)
def get_letter(integer, reverse_index):
    return integer * int64(pow(10, reverse_index))


@cuda.jit("int64[:](int64[:], int64[:], int64, int64)", device=True)
def array_slice(flat_array, slice_array, start, end):
    step_size = 1   # assumed
    if flat_array.size == 0:
        return flat_array
    for i in range(slice_array.size):
        if start + i*step_size >= end:
            break
        else:
            slice_array[i] = flat_array[start + i*step_size]
    return slice_array


@cuda.jit("int64(int64[:])", device=True)
def word_from_letters(letters):
    word = 0
    size = length_word_array(letters)
    for i in range(letters.size):
        if letters[i] != 0:
            word += get_letter(letters[i], size-i-1)
    return word


@cuda.jit("int64(int8[:])", device=True)
def word_from_letters8(letters):
    word = 0
    size = length_word_array8(letters)
    for i in range(letters.size):
        if letters[i] != 0:
            word += get_letter(letters[i], size-i-1)
    return word


@cuda.jit("int64(int64[:], int64[:], int64, int64)", device=True)
def word_slice(word_letters, slice_array, start, end):
    return word_from_letters(array_slice(
        word_letters, slice_array, start, end))


@cuda.jit("int64(int8[:])", device=True)
def reverse_word(word_letters):
    word_reversed = 0
    for i in range(word_letters.size):
        if word_letters[i] != 0:
            word_reversed += word_letters[i]*int64(pow(10, i))
    return word_reversed


@cuda.jit("int64(int64, int64)", device=True)
def concatenate(word1, word2):
    if word1 == 0:
        return word2
    elif word2 == 0:
        return word1
    else:
        return word1*int64(pow(10, length(word2))) + word2


@cuda.jit("int64(int64, int64)", device=True)
def permutation_count(n, k):
    if n < k or k < 0:
        return 0
    else:
        permutation_count = 1
        for i in range(k):
            permutation_count *= (n - i)
        return permutation_count


@cuda.jit("int64(int64, int64)", device=True)
def tuple_count(n, k):
    if n < k or k < 0:
        return 0
    else:
        return int64(pow(n, k))


@cuda.jit("int8[:,:,:,:](int64[:], int64, int8[:,:,:,:], "
          + "int8[:,:,:,:], int64, int64, int64[:])", device=True)
def permutations(flat_array, size, tuple_array, 
                 permutation_array, index1, index2, test_emptyword):
    for i in range(tuple_array.shape[2]):
        for j in range(tuple_array.shape[3]):
            tuple_array[index1, index2, i, j] = 0
    for h in range(size):
        size_step = h+1
        num_generated = 0
        if size_step == 1:
            for i in range(nonzeros_count(flat_array)):
                tuple_array[index1, index2, i, 0] = flat_array[i]
        else:
            for i in range(tuple_array.shape[2]):
                all_zeros = True
                for j in range(size_step-1):
                    if tuple_array[index1, index2, i, j] != 0:
                        all_zeros = False
                if not all_zeros:
                    num_generated += 1
            offset = 0
            for i in range(nonzeros_count(flat_array)):
                for j in range(num_generated):
                    invalid = False
                    for k in range(size_step):
                        if tuple_array[index1, index2, j, k] == flat_array[i]:
                            invalid = True
                    if invalid:
                        offset += 1
                        continue
                    for k in range(size_step):
                        if k != size_step-1:
                            tuple_array[
                                index1, index2, i*num_generated + j - offset, k] = (
                                    tuple_array[index1, index2, j, k])
                        else:
                            tuple_array[
                                index1, index2, i*num_generated + j - offset, k] = flat_array[i]
    # Remove 'permutations' with repetition
    offset = 0
    for i in range(tuple_array.shape[2]):
        invalid = False
        for j in range(tuple_array.shape[3]):
            for k in range(tuple_array.shape[3]):
                if (tuple_array[index1, index2, i, j] == 
                        tuple_array[index1, index2, i, k] != 0 
                        and j != k):
                    invalid = True
                    offset += 1
                    break
            if invalid:
                break
        if not invalid:
            for j in range(tuple_array.shape[3]):
                permutation_array[index1, index2, i-offset, j] = (
                    tuple_array[index1, index2, i, j])
    return permutation_array


@cuda.jit("void(int64[:], int64[:,:], int64, int64[:,:], int64[:,:], " + 
          "int64[:,:,:], int64[:,:,:,:], int8[:,:,:,:], int8[:,:,:,:], int64[:])")
def compute_neighbors(word_array, neighbors_array, size_limit, 
                      repeat_words, return_words, insertions, instances, 
                      tuple_array, permutation_array, test_emptyword):
    """
    Args:
        word_array: A 1D numpy.ndarray.
        size_limit: Integer.
    Usage: Input numpy.ndarray of integers, returns numpy.ndarray 
           of numpy.ndarrays with integer elements.
    """
    thread_num = cuda.grid(1)
    for i in range(word_array.size):
        if i == thread_num:
            word = word_array[i]
            pattern_instance_count = (repeat_words.size // repeat_words.ndim 
                                      + return_words.size // return_words.ndim)
            word_length = length(word)
            for j in range(pattern_instance_count):
                pattern_instance = zeros1D(cuda.local.array(2, int64))
                if j <= pattern_instance_count//2 - 1:
                    pattern_instance[0] = repeat_words[0, j]
                    pattern_instance[1] = repeat_words[1, j]
                else:
                    j = j - pattern_instance_count//2
                    pattern_instance[0] = return_words[0, j]
                    pattern_instance[1] = return_words[1, j]
                instance_length = (length(pattern_instance[0]) 
                                   + length(pattern_instance[1]))
                if word_length//2 + instance_length//2 <= size_limit:
                    word_letters = letters(word, 
                        zeros1D(cuda.local.array(SMALL_ARRAY_LENGTH, int64)))
                    new_letters = zeros1D(
                        cuda.local.array(SMALL_ARRAY_LENGTH, int64))
                    offset = 0
                    for k in range(size_limit):
                        letter_taken = False
                        for l in range(word_letters.size):
                            if word_letters[l] == k+1:
                                letter_taken = True
                                offset += 1
                                break
                        if not letter_taken:
                            new_letters[k-offset] = k+1

                    instance1_array = zeros1D(
                        cuda.local.array(SMALL_ARRAY_LENGTH, int64))
                    instance_letters = letters(pattern_instance[0], instance1_array)
                    return_word = False
                    instance_size = length(pattern_instance[0])
                    for k in range(instance_size):
                        if (letter(pattern_instance[0], k) 
                                != letter(pattern_instance[1], instance_size-k-1)):
                            return_word = True

                    permutation_num = permutation_count(
                        nonzeros_count(new_letters), nonzeros_count(instance_letters))
                    tuple_num = tuple_count(nonzeros_count(new_letters), 
                        nonzeros_count(instance_letters))
                    permutations_array = permutations(
                        new_letters, nonzeros_count(instance_letters), 
                        tuple_array, permutation_array, i, j, test_emptyword)

                    # Generate all pattern instance labelings
                    last_seen = 0
                    for k in range(permutation_num):
                        indices = zeros1D8(
                            cuda.local.array(SMALL_ARRAY_LENGTH, int8))
                        if permutations_array[i, j, k, 0] == 0:
                            continue
                        for l in range(permutations_array.shape[3]):
                            indices[l] = permutations_array[i, j, k, l]
                        # if i == 0 and instance_length == 10 and k == 50:
                        #     for l in range(indices.size):
                        #         if indices[l] != 0:
                        #             test_emptyword[l] = indices[l]
                        for l in range(pattern_instance.size):
                            relabeled_part = word_from_letters8(indices)
                            if return_word and l == 1:
                                relabeled_part = reverse_word(indices)
                            pattern_instance[l] = relabeled_part
                        instances[i,j,k,0] = pattern_instance[0]
                        instances[i,j,k,1] = pattern_instance[1]
                        if i == 0 and j == 4:
                            test_emptyword[0] = pattern_instance[0]
                            test_emptyword[1] = pattern_instance[1]

                    # Generate all possible insertions
                    word_length = length(word)
                    for k in range(instances.shape[2]):
                        if instances[i,j,k,0] == 0:
                            continue
                        for l in range(word_length+1):
                            for m in range(word_length+1):
                                if l < m:
                                    slice1_array = zeros1D(
                                        cuda.local.array(SMALL_ARRAY_LENGTH, int64))
                                    slice2_array = zeros1D(
                                        cuda.local.array(SMALL_ARRAY_LENGTH, int64))
                                    slice3_array = zeros1D(
                                        cuda.local.array(SMALL_ARRAY_LENGTH, int64))
                                    new_word = concatenate(
                                        concatenate(
                                            concatenate(
                                                concatenate(
                                                    word_slice(word_letters, slice1_array, 0, l), 
                                                    instances[i,j,k,0]),
                                                word_slice(word_letters, slice2_array, l, m)),
                                            instances[i,j,k,1]), 
                                        word_slice(word_letters, slice3_array, m, word_length)
                                    )
                                else:
                                    slice1_array = zeros1D(
                                        cuda.local.array(SMALL_ARRAY_LENGTH, int64))
                                    slice2_array = zeros1D(
                                        cuda.local.array(SMALL_ARRAY_LENGTH, int64))
                                    slice3_array = zeros1D(
                                        cuda.local.array(SMALL_ARRAY_LENGTH, int64))
                                    new_word = concatenate(
                                        concatenate(
                                            concatenate(
                                                concatenate(
                                                    word_slice(word_letters, slice1_array, 0, m), 
                                                    instances[i,j,k,1]),
                                                word_slice(word_letters, slice2_array, m, l)),
                                            instances[i,j,k,0]), 
                                        word_slice(word_letters, slice3_array, l, word_length)
                                    )
                                insertions[i, j, (word_length+1)*(word_length+1)*k 
                                           + (word_length+1)*l + m] = new_word
                    some_neighbors = insertions
                    ###
                    neighbor_count = nonzeros_count2D_1D(neighbors_array, i)
                    new_neighbor_count = nonzeros_count3D_1D(some_neighbors, i, j)
                    for k in range(new_neighbor_count):
                        neighbors_array[i, neighbor_count+k] = some_neighbors[i, j, k]
