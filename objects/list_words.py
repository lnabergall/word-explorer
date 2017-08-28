"""
Functions for generating lists of words with various properties.

Functions:

    get_all_words, get_all_dows, get_all_dows_noeq, get_random_sample
"""

from random import shuffle
from itertools import permutations, combinations, product

from .objects import Word
from .ascending_order import convert_to_ascending_order


def get_all_words(max_length, alphabet_size):
    """
    Args:
        max_length: Integer.
        alphabet_size: Integer.
    Returns:
        A sorted list containing all words of length at most max_length
        from the alphabet {1, 2, ..., alphabet_size}. 
    """
    if alphabet_size**max_length > 30000000:
        raise ValueError(
            "Too many words to compute without likely exceeding memory limits!")
    elif alphabet_size > 9:
        raise NotImplementedError("Need to expand allowed alphabets!")

    word_list = []
    for length in range(1, max_length+1):
        letter_lists = product(range(1,alphabet_size+1), repeat=length)
        words = [Word("".join(str(i) for i in letters), double_occurrence=False) 
                 for letters in letter_lists]
        word_list.extend(words)

    word_list.sort()
    word_list.sort(key=lambda word: len(word))
    return word_list


def get_all_dows(max_size):
    """
    Args:
        max_size: Integer.
    Returns:
        A sorted list containing all double occurrence words 
        without bijective equivalance, where each word of size n 
        is constructed from the alphabet {1, 2, ..., n}.
    """
    word_list = set()
    for size in range(1, max_size+1):
        print("generating words of size " + str(size) + "...")
        letters = [str(i) for i in range(1, min(size,9)+1)]
        if size > 9:
            letters.extend(chr(i+96) for i in range(10, size+1))
        words = set(Word("".join(tupl), double_occurrence=False) 
                    for tupl in permutations(letters*2))
        word_list |= words
        
    word_list = list(word_list)
    word_list.sort()
    word_list.sort(key=lambda word: len(word))
    return word_list


def get_all_dows_noeq(max_size):
    """
    Args:
        max_size: Integer.
    Returns:
        A sorted list containing all double occurrence words 
        without bijective equivalence. 
    """
    word_list = set()
    letter_list = [str(i) for i in range(1, min(max_size,9)+1)]
    if max_size > 9:
        letter_list.extend(chr(i+96) for i in range(10, max_size+1))
    for size in range(1, max_size+1):
        print("generating words of size " + str(size) + "...")
        for letters in combinations(letter_list, size):
            words = set(Word("".join(tupl), double_occurrence=False) 
                        for tupl in permutations(letters*2))
            word_list |= words

    word_list = list(word_list)
    word_list.sort()
    word_list.sort(key=lambda word: len(word))
    return word_list


def get_random_sample(word_list):
    """
    Args:
        word_list: List, contains instances of Word.
    Returns:
        A list of words---if word_list contains m words of size n, 
        the returned list contains m uniformly sampled words of size n.
    """
    random_sample = []
    for word in word_list:
        letters = [letter for letter in word]
        shuffle(letters)
        random_word = Word(convert_to_ascending_order("".join(letters)))
        random_sample.append(random_word)

    return random_sample