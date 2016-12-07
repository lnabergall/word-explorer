"""
CUDA Python version of word_graph.py to support parallel computations via GPU.
"""

from math import log10, floor, fmod, ceil
from accelerate.cuda import cuda
from accelerate.numba import vectorize
from numpy import array, zeros, int32


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
    word_integer = word_from_letters(list(map(int, list(word))))
    word_array = array([word_integer, size_limit], dtype=int32)
    neighbors = compute_neighbors(word_array).tolist()
    neighbors = [Word("".join(list(map(str, letters(neighbor))))) 
                 for neighbor in neighbors]
    return neighbors


@vectorize(["int32[:](int32[:])"], target="gpu")     
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
                (REPEAT_WORD_AO[1,i], REPEAT_WORD_AO[2,i]), dtype=int32)
        else:
            j = i - pattern_instance_count//2
            pattern_instance = array(
                (RETURN_WORD_AO[1,j], RETURN_WORD_AO[2,j]), dtype=int32)
        instance_length = (length(pattern_instance[0]) 
                           + length(pattern_instance[1]))
        if word_length//2 + instance_length//2 <= size_limit:
            some_neighbors = generate_insertions(
                word, pattern_instance, size_limit)
            for i in range(NUM_NEIGHBOR_LIMIT):
                if neighbors[i] == 0:
                    first_index = i
                    break
            for i in range(some_neighbors.size):
                neighbors[first_index+i] = some_neighbors[i]

    return neighbors


def digit_count(integer):
    return floor(log10(integer)) + 1 


def digits(integer):
    digit_count = digit_count(integer)
    digits = zeros(digit_count, dtype=int32)
    for i in range(digit_count):
        digits[digit_count-i-1] = int(fmod(integer, pow(10,i)))
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
    if start >= end:
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


def reverse_word(word):
    word_reversed = 0
    letters = letters(word)
    for i in range(letters.size):
        word_reversed += letters[i]*pow(10, i)
    return word_reversed


def concatenate(word1, word2):
    letters1 = letters(word1)
    letters2 = letters(word2)
    all_letters = zeros(letters1.size + letters2.size, dtype=int32)
    for i in range(letters1.size + letters2.size):
        if i < letters1.size:
            all_letters[i] = letters1[i]
        else:
            all_letters[letters2.size + i] = letters1[i]

    return word_from_letters(all_letters)


def permutation_count(n, k):
    permutation_count = 1
    for i in range(k):
        permutation_count *= (n - i)
    return permutation_count


def permutations(flat_array, size, permutation_array):
    permutation_count = permutation_count(flat_array.size, size)
    new_permutation_array = zeros(permutation_count, dtype=int32)
    for i in range(flat_array.size):
        if permutation_array.size == 0:
            new_permutation_array[i] = array((flat_array[i]), dtype=int32)
        for j in range(permutation_array.size):
            permutation = zeros(permutation_array[j].size + 1, dtype=int32)
            for k in range(permutation.size):
                if k != permutation.size - 1:
                    permutation[k] = permutation_array[j][k]
                else:
                    permutation[k] = flat_array[i]
            new_permutation_array[i*permutation_array.size + j] = permutation
    if new_permutation_array[0].size == size:
        return new_permutation_array
    else:
        return permutations(flat_array, size, new_permutation_array)


def product(flat_array1, flat_array2):
    product = zeros(flat_array1.size*flat_array2.size, dtype=int32)
    for i in range(flat_array1.size):
        for j in range(flat_array2.size):
            product[flat_array2.size*i + j] = array(
                (flat_array1[i], flat_array2[j]), dtype=int32)
    return product


def generate_insertions(word, pattern_instance, size_limit):
    # Generate new letter choices
    word_letters = letters(word)
    new_letters = zeros(size_limit, dtype=int32)
    for i in range(size_limit):
        new_letters[i] = i+1
        for j in range(word_letters.size):
            if word_letters[j] == i+1:
                new_letters[i] = 0

    instance_letters = letters(pattern_instance[0])
    return_word = True
    instance_size = length(pattern_instance[0])
    for i in range(instance_size):
        if pattern_instance[0][i] != pattern_instance[1][instance_size-i-1]:
            return_word = False
    insertions = zeros(NUM_NEIGHBOR_LIMIT, dtype=int32)
    instances = zeros(NUM_NEIGHBOR_LIMIT, dtype=int32)

    new_letter_count = 0
    for i in range(new_letters.size):
        if new_letters[i] != 0:
            new_letter_count += 1

    permutation_count = permutation_count(new_letter_count, 
                                          instance_letters.size)
    permutations = permutations(new_letters, instance_letters.size, 
                                zeros(0, dtype=int32))

    # Generate all pattern instance labelings
    for i in range(permutation_count):
        indices = permutations[i]
        for j in range(pattern_instance.size):
            relabeled_part = word_from_letters(indices)
            if return_word and i == 1:
                relabeled_part = reverse_word(relabeled_part)
            pattern_instance[j] = relabeled_part
        instances[i] = pattern_instance.copy()

    # Generate all possible insertions
    for i in range(instances.size):
        for j in range(word.size+1):
            for k in range(word.size+1):
                if j < k:
                    new_word = concatenate(
                        concatenate(
                            concatenate(
                                concatenate(slice(word, 0, j), instances[i][0]), 
                                slice(word, j, k)),
                            instances[i][1]), 
                        slice(word, k, length(word))
                    )
                else:
                    new_word = concatenate(
                        concatenate(
                            concatenate(
                                concatenate(slice(word, 0, k), instances[i][1]), 
                                slice(word, k, j)),
                            instances[i][0]), 
                        slice(word, j, length(word))
                    )
                insertions[(word.size+1)*(word.size+1)*i + (word.size+1)*j + k] = new_word

    return new_word

