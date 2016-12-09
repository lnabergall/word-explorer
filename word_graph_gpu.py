"""
CUDA Python version of word_graph.py to support parallel computations via GPU.
"""

from math import log10, floor, fmod, ceil
from accelerate.cuda import cuda
from numba import vectorize
from numpy import array, zeros, int32, asscalar


REPEAT_WORD_AO = array((
    [1, 12, 123, 1234, 12345, 123456, 1234567, 12345678, 12345678, 123456789],
    [1, 12, 123, 1234, 12345, 123456, 1234567, 12345678, 12345678, 123456789],
), dtype=int32)
RETURN_WORD_AO = array((
    [1, 12, 123, 1234, 12345, 123456, 1234567, 12345678, 12345678, 123456789],
    [1, 21, 321, 4321, 54321, 654321, 7654321, 87654321, 87654321, 987654321],
), dtype=int32)
NUM_NEIGHBOR_LIMIT = 10000


def find_adjacent_vertices(word, size_limit):
    from word_graph import Word     # Avoid circular dependency
    word_integer = word_from_letters_list(list(map(int, list(word))))
    word_array = array([word_integer, size_limit], dtype=int32)
    neighbors = compute_neighbors(word_array).tolist()
    neighbors = set([Word("".join(list(map(str, letters(neighbor))))) 
                     for neighbor in neighbors])
    return neighbors


#@vectorize(["int32[:](int32[:])"], target="gpu")
def compute_neighbors(word_array):
    """
    Args:
        word_array: An numpy.ndarray of the form [word, size_limit], 
            with size_limit < 10.
    Usage: Input numpy.ndarray of integers, returns numpy.ndarray 
           of numpy.ndarrays with integer elements.
    """
    neighbors = zeros(NUM_NEIGHBOR_LIMIT, dtype=int32)   # Likely <10000 neighbors
    word = word_array[0]
    size_limit = word_array[1]
    pattern_instance_count = (REPEAT_WORD_AO.size // REPEAT_WORD_AO.ndim 
                              + RETURN_WORD_AO.size // RETURN_WORD_AO.ndim)
    word_length = length(word) 
    for i in range(pattern_instance_count):
        if i <= pattern_instance_count//2 - 1:
            pattern_instance = array(
                (REPEAT_WORD_AO[0,i], REPEAT_WORD_AO[1,i]), dtype=int32)
        else:
            j = i - pattern_instance_count//2
            pattern_instance = array(
                (RETURN_WORD_AO[0,j], RETURN_WORD_AO[1,j]), dtype=int32)
        instance_length = (length(pattern_instance[0]) 
                           + length(pattern_instance[1]))
        if word_length//2 + instance_length//2 <= size_limit:
            some_neighbors = generate_insertions(word, pattern_instance)
            neighbor_count = nonzero_element_count(neighbors)
            new_neighbor_count = nonzero_element_count(some_neighbors)
            for i in range(new_neighbor_count):
                neighbors[neighbor_count+i] = some_neighbors[i]

    neighbors = remove_zeros(neighbors)
    return neighbors


def nonzero_element_count(flat_array):
    nonzero_elements = 0
    for i in range(flat_array.size):
        if flat_array[i] != 0:
            nonzero_elements += 1
    return nonzero_elements


def remove_zeros(flat_array):
    nonzero_count = nonzero_element_count(flat_array)
    filtered_array = zeros(nonzero_count, dtype=int32)
    for i in range(flat_array.size):
        if flat_array[i] != 0:
            for j in range(filtered_array.size):
                if filtered_array[j] == 0:
                    filtered_array[j] = flat_array[i]
    return filtered_array


def digit_count(integer):
    if type(integer) != int:
        integer = asscalar(integer)
    if integer == 0:
        return 0
    else:
        return floor(log10(integer)) + 1 


def digits(integer):
    if type(integer) != int:
        integer = asscalar(integer)
    order = digit_count(integer)
    digits = zeros(order, dtype=int32)
    for i in range(order):
        digits[order-i-1] = (
            int(fmod(integer, pow(10,i+1)) - fmod(integer, pow(10,i))) // pow(10,i))
    return digits


def length(word):
    return digit_count(word)


def letters(word):
    return digits(word)


def size(word):
    return length(word) // 2


def letter(word, index):
    return letters(word)[index]


def get_letter(integer, reverse_index):
    return integer * pow(10, reverse_index)


def array_slice(flat_array, start, end, step_size=1):
    if start >= end or flat_array.size == 0:
        return zeros(0, dtype=int32)
    slice_array = zeros(ceil((end-start)/step_size), dtype=int32)
    for i in range(slice_array.size):
        slice_array[i] = flat_array[start + i*step_size]
    return slice_array


def slice(word, start, end, step_size=1):
    return word_from_letters(array_slice(letters(word), start, end, step_size))


def word_from_letters(letters):
    word = 0
    for i in range(letters.size):
        word += get_letter(letters[i], letters.size-i-1)
    return word


def word_from_letters_list(letters):
    word = 0
    for i in range(len(letters)):
        word += get_letter(letters[i], len(letters)-i-1)
    return word


def reverse_word(word):
    word_reversed = 0
    word_letters = letters(word)
    for i in range(word_letters.size):
        word_reversed += word_letters[i]*pow(10, i)
    return word_reversed


def concatenate(word1, word2):
    if word1 == 0:
        return word2
    if word2 == 0:
        return word1
    letters1 = letters(word1)
    letters2 = letters(word2)
    all_letters = zeros(letters1.size + letters2.size, dtype=int32)
    for i in range(letters1.size + letters2.size):
        if i < letters1.size:
            all_letters[i] = letters1[i]
        else:
            all_letters[i] = letters2[i - letters1.size]

    return word_from_letters(all_letters)


def permutation_count(n, k):
    if n < k or k < 0:
        return 0
    else:
        permutation_count = 1
        for i in range(k):
            permutation_count *= (n - i)
        return permutation_count


def tuple_count(n, k):
    if n < k or k < 0:
        return 0
    else:
        return int(pow(n, k))


def permutations(flat_array, size, permutation_array):
    tuple_num = tuple_count(flat_array.size, size)
    permutation_num = permutation_count(flat_array.size, size)
    if permutation_array.size == 0:
        size_step = 1
    else:
        size_step = permutation_array.shape[1] + 1
    new_permutation_array = zeros(
        (tuple_num, size_step), dtype=int32)
    num_generated = 0
    if permutation_array.size != 0:
        for i in range(tuple_num):
            all_zeros = True
            for j in range(size_step-1):
                if permutation_array[i, j] != 0:
                    all_zeros = False
            if not all_zeros:
                num_generated += 1

    for i in range(flat_array.size):
        #print(i, size, size_step, flat_array, permutation_array, new_permutation_array)
        if permutation_array.size == 0:
            new_permutation_array[i,0] = flat_array[i]
        else:
            for j in range(num_generated):
                for k in range(size_step):
                    if k != size_step-1:
                        new_permutation_array[
                            i*num_generated + j, k] = permutation_array[j,k]
                    else:
                        new_permutation_array[
                            i*num_generated + j, k] = flat_array[i]

    if size_step == size:
        # Remove 'permutations' with repetition
        offset = 0
        permutation_array = zeros((permutation_num, size), dtype=int32)
        for i in range(new_permutation_array.shape[0]):
            invalid = False
            for j in range(new_permutation_array.shape[1]):
                for k in range(new_permutation_array.shape[1]):
                    if (new_permutation_array[i, j] == new_permutation_array[i, k] 
                            and j != k):
                        invalid = True
                        offset += 1
                        break
                if invalid:
                    break
            if not invalid:
                for j in range(new_permutation_array.shape[1]):
                    permutation_array[i-offset, j] = new_permutation_array[i, j]
        print(permutation_array)
        return permutation_array
    else:
        return permutations(flat_array, size, new_permutation_array)


def generate_insertions(word, pattern_instance):
    # Generate new letter choices
    word_letters = letters(word)
    new_letter_count = size(word) + length(pattern_instance[0])
    new_letters = zeros(length(pattern_instance[0]), dtype=int32)
    offset = 0
    for i in range(new_letter_count):
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
    insertions = zeros(NUM_NEIGHBOR_LIMIT, dtype=int32)
    instances = zeros((NUM_NEIGHBOR_LIMIT, 2), dtype=int32)

    permutation_num = permutation_count(new_letters.size, instance_letters.size)
    permutations_array = permutations(new_letters, instance_letters.size, 
                                      zeros(0, dtype=int32))

    # Generate all pattern instance labelings
    for i in range(permutation_num):
        indices = zeros(permutations_array.shape[1], dtype=int32)
        for j in range(permutations_array.shape[1]):
            indices[j] = permutations_array[i, j]
        for j in range(pattern_instance.size):
            relabeled_part = word_from_letters(indices)
            if return_word and i == 1:
                relabeled_part = reverse_word(relabeled_part)
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
