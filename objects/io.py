"""
Input/output for objects API---includes functions for storing and retrieving
lists of words.

Functions:

    get_word_filename, store_words, retrieve_words
"""

from .objects import max_word_size, max_word_length
from word_explorer.io import store_data, retrieve_data


def get_word_filename(list_type, size=None, length=None):
    if length is not None:
        file_name = "words_up_to_length" + str(length)
    else:
        file_name = "words_up_to_size" + str(size)
    return file_name + "_" + list_type + ".txt"


def store_words(word_list, list_type="dow"):
    if list_type = "all":
        length = max_word_length(word_list)
        size = None
    else:
        size = max_word_size(word_list)
        length = None
    file_name = get_word_filename(list_type, size, length)
    store_data(word_list, file_name)


def retrieve_words(list_type, size=None, length=None):
    file_name = get_word_filename(list_type, size, length)
    word_list = []
    for line in retrieve_data(file_name):
        double_occurrence = False if list_type == "all" else True
        word = Word(line, double_occurrence=double_occurrence)
        word_list.append(word)

    return word_list