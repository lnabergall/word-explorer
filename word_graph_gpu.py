"""
CUDA Python version of word_graph.py to support parallel computations via GPU.
"""

from math import log10, floor, fmod, ceil
from numba import cuda
from numpy import array, zeros, int32
int32_py = int32
from numba.types import int32, float64
zeros_py = zeros


REPEAT_WORD_AO = array((
    [1, 12, 123, 1234, 12345, 123456, 1234567, 12345678, 12345678, 123456789],
    [1, 12, 123, 1234, 12345, 123456, 1234567, 12345678, 12345678, 123456789],
), dtype=int32_py)
RETURN_WORD_AO = array((
    [1, 12, 123, 1234, 12345, 123456, 1234567, 12345678, 12345678, 123456789],
    [1, 21, 321, 4321, 54321, 654321, 7654321, 87654321, 87654321, 987654321],
), dtype=int32_py)
NUM_NEIGHBOR_LIMIT = 1000
SMALL_ARRAY_LENGTH = 30
MAX_ARRAY_LENGTH = 1000


def find_adjacent_vertices(word_list, size_limit):
    from word_graph import Word     # Avoid circular dependency
    words = word_list.copy()
    for i, pair in enumerate(words):
        word_integer = word_from_letters_list(list(map(int, list(pair))))
        words[i] = int32(word_integer)
    word_array = array(words, dtype=int32_py)
    # Likely <10000 neighbors
    neighborhoods = zeros_py((len(words), NUM_NEIGHBOR_LIMIT), dtype=int32_py)   
    threads_perblock = 32
    blocks_perdim = (len(words) + (threads_perblock - 1)) // threads_perblock
    device_word_array = cuda.to_device(word_array)
    device_neighborhoods = cuda.to_device(neighborhoods)
    device_repeats = cuda.to_device(REPEAT_WORD_AO)
    device_returns = cuda.to_device(RETURN_WORD_AO)
    compute_neighbors[blocks_perdim, threads_perblock](
        device_word_array, device_neighborhoods, 
        size_limit, device_repeats, device_returns)
    neighborhoods_found = device_neighborhoods.copy_to_host()
    neighborhoods = neighborhoods_found.tolist()
    # zero_count = 0
    # nonzero_count = 0
    # for i, neighborhood in enumerate(neighborhoods):
    #     print(word_list[i])
    #     nonzero_neighbors = [neighbor for neighbor in neighborhood if neighbor != 0]
    #     print(nonzero_neighbors, "\n")
    #     nonzero_count += len(nonzero_neighbors)
    #     zero_count += len(neighborhood) - len(nonzero_neighbors)
    # print(nonzero_count, zero_count)
    neighborhoods = [set([Word("".join(list(map(str, letters_from_int(neighbor))))) 
                         for neighbor in neighbors if neighbor != 0]) 
                     for neighbors in neighborhoods]
    return neighborhoods


def word_from_letters_list(letters):
    word = 0
    for i in range(len(letters)):
        word += letters[i] * int(pow(10, len(letters)-i-1))
    return word


def digit_count_int(integer):
    if integer == 0:
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


@cuda.jit("int32[:](int32[:])", device=True)
def zeros1D(zeros_array):
    for i in range(zeros_array.size):
        zeros_array[i] = 0
    return zeros_array


@cuda.jit("int32[:,:](int32[:,:])", device=True)
def zeros2D(zeros_array):
    for i in range(zeros_array.shape[0]):
        for j in range(zeros_array.shape[1]):
            zeros_array[i, j] = 0
    return zeros_array


@cuda.jit("boolean(int32[:,:])", device=True)
def nonzero2D(array):
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if array[i, j] != 0:
                return True
    return False


@cuda.jit("int32(int32[:])", device=True)
def nonzeros_count(flat_array):
    nonzero_elements = 0
    for i in range(flat_array.size):
        if flat_array[i] != 0:
            nonzero_elements += 1
    return nonzero_elements


@cuda.jit("int32[:](int32[:], int32[:])", device=True)
def remove_zeros(flat_array, filtered_array):
    for i in range(flat_array.size):
        if flat_array[i] != 0:
            for j in range(filtered_array.size):
                if filtered_array[j] == 0:
                    filtered_array[j] = flat_array[i]
                    break
    return filtered_array


@cuda.jit("int32(int32)", device=True)
def digit_count(integer):
    if integer == 0:
        return 0
    else:
        return int32(floor(log10(float64(integer))) + 1)


@cuda.jit("int32[:](int32, int32[:])", device=True)
def digits(integer, digits_array):
    order = digits_array.size
    for i in range(order):
        digits_array[order-i-1] = int32(
            int32(fmod(integer, pow(10,i+1)) - fmod(integer, pow(10,i))) // pow(10,i))
    return digits_array


@cuda.jit("int32(int32)", device=True)
def length(word):
    return digit_count(word)


@cuda.jit("int32(int32, int32)", device=True)
def ith_letter(word, i):
    return int32(
        int32(fmod(word, pow(10,i+1)) - fmod(word, pow(10,i))) // pow(10,i))


@cuda.jit("int32(int32, int32)", device=True)
def letter(word, index):
    return ith_letter(word, index)


@cuda.jit("int32(int32[:])", device=True)
def length_word_array(letters_array):
    return nonzeros_count(letters_array)


@cuda.jit("int32[:](int32, int32[:])", device=True)
def letters(word, letters_array):
    word_length = length(word)
    for i in range(word_length):
        letters_array[word_length-i-1] = ith_letter(word, i)
    return letters_array


@cuda.jit("int32(int32)", device=True)
def size(word):
    return length(word) // 2


@cuda.jit("int32(int32, int32)", device=True)
def get_letter(integer, reverse_index):
    return integer * int32(pow(10, reverse_index))


@cuda.jit("int32[:](int32[:], int32[:], int32, int32)", device=True)
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


@cuda.jit("int32(int32[:])", device=True)
def word_from_letters(letters):
    word = 0
    size = length_word_array(letters)
    for i in range(letters.size):
        if letters[i] != 0:
            word += get_letter(letters[i], size-i-1)
    return word


@cuda.jit("int32(int32[:], int32[:], int32, int32)", device=True)
def word_slice(word_letters, slice_array, start, end):
    return word_from_letters(array_slice(
        word_letters, slice_array, start, end))


@cuda.jit("int32(int32[:])", device=True)
def reverse_word(word_letters):
    word_reversed = 0
    for i in range(word_letters.size):
        if word_letters[i] != 0:
            word_reversed += word_letters[i]*int32(pow(10, i))
    return word_reversed


@cuda.jit("int32(int32, int32)", device=True)
def concatenate(word1, word2):
    if word1 == 0:
        return word2
    elif word2 == 0:
        return word1
    else:
        return word1*int32(pow(10, length(word2))) + word2


@cuda.jit("int32(int32, int32)", device=True)
def permutation_count(n, k):
    if n < k or k < 0:
        return 0
    else:
        permutation_count = 1
        for i in range(k):
            permutation_count *= (n - i)
        return permutation_count


@cuda.jit("int32(int32, int32)", device=True)
def tuple_count(n, k):
    if n < k or k < 0:
        return 0
    else:
        return int32(pow(n, k))


@cuda.jit("int32[:,:](int32[:], int32, int32[:,:], int32[:,:])", device=True)
def permutations(flat_array, size, tuple_array, permutation_array):
    for h in range(size):
        size_step = h+1
        num_generated = 0
        if nonzero2D(tuple_array):
            for i in range(tuple_array.shape[0]):
                all_zeros = True
                for j in range(size_step-1):
                    if tuple_array[i, j] != 0:
                        all_zeros = False
                if not all_zeros:
                    num_generated += 1
        if not nonzero2D(tuple_array):
            for i in range(nonzeros_count(flat_array)):
                tuple_array[i,0] = flat_array[i]
        else:
            for i in range(nonzeros_count(flat_array)):
                for j in range(num_generated):
                    for k in range(size_step):
                        if k != size_step-1:
                            tuple_array[
                                i*num_generated + j, k] = tuple_array[j,k]
                        else:
                            tuple_array[
                                i*num_generated + j, k] = flat_array[i]
    # Remove 'permutations' with repetition
    offset = 0
    for i in range(tuple_array.shape[0]):
        invalid = False
        for j in range(tuple_array.shape[1]):
            for k in range(tuple_array.shape[1]):
                if (tuple_array[i, j] == tuple_array[i, k] != 0 
                        and j != k):
                    invalid = True
                    offset += 1
                    break
            if invalid:
                break
        if not invalid:
            for j in range(tuple_array.shape[1]):
                permutation_array[i-offset, j] = tuple_array[i, j]
    return permutation_array


@cuda.jit("void(int32[:], int32[:,:], int32, int32[:,:], int32[:,:])")
def compute_neighbors(word_array, neighborhoods, size_limit, 
                      repeat_words, return_words):
    """s
    Args:
        word_array: A 1D numpy.ndarray.
        neighborhoods: A 2D numpy.ndarray with the same number of rows 
            as word_array.
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
            neighbors_array = zeros1D(cuda.local.array(NUM_NEIGHBOR_LIMIT, int32))
            for j in range(pattern_instance_count):
                pattern_instance = zeros1D(cuda.local.array(2, int32))
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
                        zeros1D(cuda.local.array(SMALL_ARRAY_LENGTH, int32)))
                    new_letters = zeros1D(
                        cuda.local.array(SMALL_ARRAY_LENGTH, int32))
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
                        cuda.local.array(SMALL_ARRAY_LENGTH, int32))
                    instance_letters = letters(pattern_instance[0], instance1_array)
                    return_word = True
                    instance_size = length(pattern_instance[0])
                    for k in range(instance_size):
                        if (letter(pattern_instance[0], k) 
                                != letter(pattern_instance[1], instance_size-k-1)):
                            return_word = False
                    insertions = zeros1D(cuda.local.array(NUM_NEIGHBOR_LIMIT, int32))
                    instances = zeros2D(cuda.local.array((MAX_ARRAY_LENGTH, 2), int32))

                    permutation_num = permutation_count(
                        nonzeros_count(new_letters), nonzeros_count(instance_letters))
                    tuple_num = tuple_count(nonzeros_count(new_letters), 
                        nonzeros_count(instance_letters))
                    tuple_array = zeros2D(cuda.local.array(
                        (MAX_ARRAY_LENGTH, SMALL_ARRAY_LENGTH), int32))
                    permutation_array = zeros2D(cuda.local.array(
                        (MAX_ARRAY_LENGTH, SMALL_ARRAY_LENGTH), int32))
                    permutations_array = permutations(
                        new_letters, nonzeros_count(instance_letters), 
                        tuple_array, permutation_array)

                    # Generate all pattern instance labelings
                    for k in range(permutation_num):
                        indices = zeros1D(
                            cuda.local.array(SMALL_ARRAY_LENGTH, int32))
                        for l in range(permutations_array.shape[1]):
                            indices[l] = permutations_array[k, l]
                        for l in range(pattern_instance.size):
                            relabeled_part = word_from_letters(indices)
                            if return_word and l == 1:
                                relabeled_part = reverse_word(indices)
                            pattern_instance[l] = relabeled_part
                        instances[k,0] = pattern_instance[0]
                        instances[k,1] = pattern_instance[1]

                    # Generate all possible insertions
                    word_length = length(word)
                    for k in range(instances.shape[0]):
                        if instances[k,0] == 0:
                            continue
                        for l in range(word_length+1):
                            for m in range(word_length+1):
                                if l < m:
                                    slice1_array = zeros1D(
                                        cuda.local.array(SMALL_ARRAY_LENGTH, int32))
                                    slice2_array = zeros1D(
                                        cuda.local.array(SMALL_ARRAY_LENGTH, int32))
                                    slice3_array = zeros1D(
                                        cuda.local.array(SMALL_ARRAY_LENGTH, int32))
                                    new_word = concatenate(
                                        concatenate(
                                            concatenate(
                                                concatenate(
                                                    word_slice(word_letters, slice1_array, 0, l), 
                                                    instances[k,0]),
                                                word_slice(word_letters, slice2_array, l, m)),
                                            instances[k,1]), 
                                        word_slice(word_letters, slice3_array, m, word_length)
                                    )
                                else:
                                    slice1_array = zeros1D(
                                        cuda.local.array(SMALL_ARRAY_LENGTH, int32))
                                    slice2_array = zeros1D(
                                        cuda.local.array(SMALL_ARRAY_LENGTH, int32))
                                    slice3_array = zeros1D(
                                        cuda.local.array(SMALL_ARRAY_LENGTH, int32))
                                    new_word = concatenate(
                                        concatenate(
                                            concatenate(
                                                concatenate(
                                                    word_slice(word_letters, slice1_array, 0, m), 
                                                    instances[k,1]),
                                                word_slice(word_letters, slice2_array, m, l)),
                                            instances[k,0]), 
                                        word_slice(word_letters, slice3_array, l, word_length)
                                    )
                                insertions[(word_length+1)*(word_length+1)*k 
                                           + (word_length+1)*l + m] = new_word
                    some_neighbors = insertions
                    ###
                    neighbor_count = nonzeros_count(neighbors_array)
                    new_neighbor_count = nonzeros_count(some_neighbors)
                    for k in range(new_neighbor_count):
                        neighbors_array[neighbor_count+k] = some_neighbors[k]
            for j in range(neighbors_array.size):
                neighborhoods[i, j] = neighbors_array[j]
