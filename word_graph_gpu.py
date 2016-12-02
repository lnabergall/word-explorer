"""
CUDA Python version of word_graph.py to support parallel computations via GPU.
"""

from math import log10, floor, fmod
from accelerate.cuda import cuda
from accelerate.numba import vectorize, guvectorize
from numpy import array, zeros, int32


REPEAT_WORD_AO = array([
    [1, 12, 123, 1234, 12345, 123456, 1234567, 12345678, 12345678, 123456789],
    [1, 12, 123, 1234, 12345, 123456, 1234567, 12345678, 12345678, 123456789],
])
RETURN_WORD_AO = array([
    [1, 12, 123, 1234, 12345, 123456, 1234567, 12345678, 12345678, 123456789],
    [1, 21, 321, 4321, 54321, 654321, 7654321, 87654321, 87654321, 987654321],
])
NUM_NEIGHBOR_LIMIT = 10000


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
    word_length = digit_count(word) 
    for i in range(pattern_instance_count):
        if i <= pattern_instance_count//2 - 1:
            pattern_instance = array([REPEAT_WORD_AO[1,i], REPEAT_WORD_AO[2,i]])
        else:
            j = i - pattern_instance_count//2
            pattern_instance = array([RETURN_WORD_AO[1,j], RETURN_WORD_AO[2,j]])
        instance_length = (digit_count(pattern_instance[0]) 
                           + digit_count(pattern_instance[1]))
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
        digits[digit_count-i-1] = int(fmod(integer, pow(10,i+1)))
    return digits


def length(word):
    return digit_count(word)


def letters(word):
    return digits(integer)


def size(word):
    return length(word) // 2


def letter(word, index):
    return letters(word)[index]


def word_from_letters(letters):
    word = 0
    for i in range(letters.size):
        word += letters[i] * pow(10, letters.size-i-1)
    return word


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
            new_permutation_array[i] = array([flat_array[i]])
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


def generate_insertions(word, pattern_instance, size_limit):
    # Generate new letter choices
    word_letters = digits(word)
    new_letters = zeros(size_limit, dtype=int32)
    for i in range(size_limit):
        new_letters[i] = i+1
        for j in range(word_letters.size):
            if word_letters[j] == i+1:
                new_letters[i] = 0

    instance_letters = digits(pattern_instance[0])
    return_word = True
    instance_size = digit_count(pattern_instance[0])
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
        indices = permutation_count[i]
        for j in range(pattern_instance.size):
            relabeled_part = 0
            for k in range(instance_letters.size):
                relabeled_part += indices[k]*pow(10, k)
            if return_word and i == 1:
                relabeled_part = 0
                for i in range(digits(relabeled_part).size):
                    relabeled_part += relabeled_part[i]*pow(10, k)
            pattern_instance[j] = relabeled_part
        instances[i] = pattern_instance.copy()

    # Generate all possible insertions
    for i in range(instances.size):
        pass


# Write series of functions that replicate string (word) methods, 
# e.g. length, size, concatenation, indexing 
# (something like 'integer_word = 1234554321; letter(integer_word, 6) = 5'), etc.
