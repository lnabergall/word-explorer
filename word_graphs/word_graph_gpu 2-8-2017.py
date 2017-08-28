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
NUM_NEIGHBOR_LIMIT = 10000


def find_adjacent_vertices(word_list, size_limit):
    from word_graph import Word     # Avoid circular dependency
    for i, pair in enumerate(word_list):
        word_integer = word_from_letters_list(list(map(int, list(pair[0]))))
        word_list[i] = word_integer
    word_array = array(word_list, dtype=int32)
    # Likely <10000 neighbors
    neighborhoods = zeros_py((len(word_list), NUM_NEIGHBOR_LIMIT), dtype=int32_py)   
    threads_perblock = 32
    blocks_perdim = (len(word_list) + (threads_perblock - 1)) // threads_perblock
    compute_neighbors[blocks_perdim, threads_perblock](
        word_array, neighborhoods, size_limit)
    neighborhoods = neighborhoods.tolist()
    neighborhoods = [set([Word("".join(list(map(str, letters(neighbor))))) 
                         for neighbor in neighbors]) for neighbors in neighborhoods]
    return neighborhoods


def word_from_letters_list(letters):
    word = 0
    for i in range(len(letters)):
        word += get_letter(letters[i], len(letters)-i-1)
    return word


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


@cuda.jit("bool(int32[:,:])", device=True)
def nonzero2D(array):
    for i in range(array.shape(0)):
        for j in range(array.shape(1)):
            if array[i, j] != 0:
                return True
    return False


@cuda.jit("int32(int32[:])", device=True)
def nonzero_element_count(flat_array):
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
    for i in range(order):
        digits_array[order-i-1] = int32(
            int32(fmod(integer, pow(10,i+1)) - fmod(integer, pow(10,i))) // pow(10,i))
    return digits_array


@cuda.jit("int32(int32)", device=True)
def length(word):
    return digit_count(word)


@cuda.jit("int32[:](int32, int32[:])", device=True)
def letters(word, letters_array):
    return digits(word, letters_array)


@cuda.jit("int32(int32)", device=True)
def size(word):
    return length(word) // 2


@cuda.jit("int32[:](int32, int32, int32[:])", device=True)
def letter(word, index, letters_array):
    # letters_array = zeros1D(cuda.shared.array(length(word), int32))
    word_letters = letters(word, letters_array)
    return word_letters(word)[index]


@cuda.jit("int32(int32, int32)", device=True)
def get_letter(integer, reverse_index):
    return integer * int32(pow(10, reverse_index))


@cuda.jit("int32(int32[:], int32[:], int32, int32, int32)", device=True)
def array_slice(flat_array, slice_array, start, end, step_size=1):
    if flat_array.size == 0:
        return flat_array
    for i in range(slice_array.size):
        slice_array[i] = flat_array[start + i*step_size]
    return slice_array


@cuda.jit("int32(int32[:], int32[:], int32, int32, int32)", device=True)
def slice(word_letters, slice_array, start, end, step_size=1):
    # slice_array = zeros1D(cuda.shared.array(ceil((end-start)/step_size), int32))
    return word_from_letters(array_slice(
        word_letters, slice_array, start, end, step_size))


@cuda.jit("int32(int32[:])", device=True)
def word_from_letters(letters):
    word = 0
    for i in range(letters.size):
        word += get_letter(letters[i], letters.size-i-1)
    return word


@cuda.jit("int32(int32[:])", device=True)
def reverse_word(word_letters):
    for i in range(word_letters.size):
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
    if not nonzero2D(tuple_array):
        size_step = 1
    else:
        size_step = tuple_array.shape[1] + 1
    num_generated = 0
    if nonzero2D(tuple_array):
        for i in range(tuple_array.shape(0)):
            all_zeros = True
            for j in range(size_step-1):
                if tuple_array[i, j] != 0:
                    all_zeros = False
            if not all_zeros:
                num_generated += 1

    for i in range(flat_array.size):
        if not nonzero2D(tuple_array):
            tuple_array[i,0] = flat_array[i]
        else:
            for j in range(num_generated):
                for k in range(size_step):
                    if k != size_step-1:
                        tuple_array[
                            i*num_generated + j, k] = tuple_array[j,k]
                    else:
                        tuple_array[
                            i*num_generated + j, k] = flat_array[i]

    if size_step == size:
        # Remove 'permutations' with repetition
        offset = 0
        for i in range(tuple_array.shape[0]):
            invalid = False
            for j in range(tuple_array.shape[1]):
                for k in range(tuple_array.shape[1]):
                    if (tuple_array[i, j] == tuple_array[i, k] 
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
    else:
        return permutations(flat_array, size, tuple_array, permutation_array)


@cuda.jit("int32[:](int32, int32[:], int32)", device=True)
def generate_insertions(word, pattern_instance, size_limit):
    # Generate new letter choices
    word_letters = letters(word)
    new_letters = zeros1D(cuda.shared.array(size_limit - size(word), int32))
    offset = 0
    for i in range(size_limit):
        letter_taken = False
        for j in range(word_letters.size):
            if word_letters[j] == i+1:
                letter_taken = True
                offset += 1
                break
        if not letter_taken:
            new_letters[i-offset] = i+1

    instance_letters = letters(pattern_instance[0])
    return_word = True
    instance_size = length(pattern_instance[0])
    for i in range(instance_size):
        if (letter(pattern_instance[0], i) 
                != letter(pattern_instance[1], instance_size-i-1)):
            return_word = False
    insertions = zeros1D(cuda.shared.array(NUM_NEIGHBOR_LIMIT, int32))
    instances = zeros2D(cuda.shared.array((NUM_NEIGHBOR_LIMIT, 2), int32))

    permutation_num = permutation_count(new_letters.size, instance_letters.size)
    tuple_num = tuple_count(new_letters.size, instance_letters.size)
    tuple_array = zeros2D(
        cuda.shared.array((tuple_num, instance_letters.size), int32))
    permutation_array = zeros2D(
        cuda.shared.array((permutation_num, instance_letters.size), int32))
    permutations_array = permutations(
        new_letters, instance_letters.size, tuple_array, permutation_array)

    # Generate all pattern instance labelings
    for i in range(permutation_num):
        indices = zeros1D(cuda.shared.array(permutations_array.shape[1], int32))
        for j in range(permutations_array.shape[1]):
            indices[j] = permutations_array[i, j]
        for j in range(pattern_instance.size):
            relabeled_part = word_from_letters(indices)
            if return_word and j == 1:
                relabeled_part = reverse_word(indices)
            pattern_instance[j] = relabeled_part
        instances[i,0] = pattern_instance[0]
        instances[i,1] = pattern_instance[1]

    # Generate all possible insertions
    word_length = length(word)
    for i in range(instances.shape[0]):
        if instances[i,0] == 0:
            continue
        for j in range(word_length+1):
            for k in range(word_length+1):
                if j < k:
                    new_word = concatenate(
                        concatenate(
                            concatenate(
                                concatenate(slice(word, 0, j), instances[i,0]),
                                slice(word, j, k)),
                            instances[i,1]), 
                        slice(word, k, word_length)
                    )
                else:
                    new_word = concatenate(
                        concatenate(
                            concatenate(
                                concatenate(slice(word, 0, k), instances[i,1]),
                                slice(word, k, j)),
                            instances[i,0]), 
                        slice(word, j, word_length)
                    )
                insertions[(word_length+1)*(word_length+1)*i 
                           + (word_length+1)*j + k] = new_word
    return insertions


@cuda.jit("void(int32[:], int32[:], int32)")
def compute_neighbors(word_array, neighborhoods, size_limit):
    """
    Args:
        word_array: A 1D numpy.ndarray.
        neighborhoods: A 2D numpy.ndarray with the same number of rows 
            as word_array.
    Usage: Input numpy.ndarray of integers, returns numpy.ndarray 
           of numpy.ndarrays with integer elements.
    """
    thread_num = cuda.grid(1)
    for i in range(word_array.shape[0]):
        if i == thread_num:
            word = word_array[i]
            pattern_instance_count = (REPEAT_WORD_AO.size // REPEAT_WORD_AO.ndim 
                                      + RETURN_WORD_AO.size // RETURN_WORD_AO.ndim)
            word_length = length(word)
            neighbors = zeros_py(neighborhoods.shape[1], dtype=int32) 
            for j in range(pattern_instance_count):
                if j <= pattern_instance_count//2 - 1:
                    pattern_instance = array(
                        (REPEAT_WORD_AO[0,j], REPEAT_WORD_AO[1,j]), dtype=int32)
                else:
                    j = j - pattern_instance_count//2
                    pattern_instance = array(
                        (RETURN_WORD_AO[0,j], RETURN_WORD_AO[1,j]), dtype=int32)
                instance_length = (length(pattern_instance[0]) 
                                   + length(pattern_instance[1]))
                if word_length//2 + instance_length//2 <= size_limit:
                    some_neighbors = generate_insertions(
                        word, pattern_instance, size_limit)
                    neighbor_count = nonzero_element_count(neighbors)
                    new_neighbor_count = nonzero_element_count(some_neighbors)
                    for k in range(new_neighbor_count):
                        neighbors[neighbor_count+k] = some_neighbors[k]
            nonzero_count = nonzero_element_count(neighbors)
            neighbors_nozeros = zeros1D(cuda.shared.array(nonzero_count, int32))
            neighbors = remove_zeros(neighbors, neighbors_nozeros)
            for j in range(size(neighbors)):
                neighborhoods[i, j] = neighbors[j]
