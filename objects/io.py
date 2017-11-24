"""
Input/output for objects API---includes functions for storing and retrieving
lists of words.

Functions:

    get_word_filename, store_words, retrieve_words
"""

from .objects import max_word_size, max_word_length
from word_explorer.io import store_data, retrieve_data


def get_word_filename(list_type, size=None, length=None, extra_suffix=""):
    if length is not None:
        file_name = "words_up_to_length" + str(length)
    else:
        file_name = "words_up_to_size" + str(size)
    return file_name + "_" + list_type + "_" + extra_suffix + ".txt"


def store_words(word_list, list_type="dow", extra_suffix=""):
    if list_type == "all":
        length = max_word_length(word_list)
        size = None
    else:
        size = max_word_size(word_list)
        length = None
    file_name = get_word_filename(
        list_type, size, length, extra_suffix=extra_suffix)
    store_data(word_list, file_name)


def retrieve_words(list_type, file_name=None, size=None, length=None, 
                   optimize=False, include_empty_word=True, ascending_order=None):
    if file_name is None:
        file_name = get_word_filename(list_type, size, length)
    double_occurrence = False if list_type == "all" else True
    if ascending_order is None:
        ascending_order = True if list_type == "ao" else False
    word_list = []
    for line in retrieve_data(file_name, add_output_dir=False):
        if "," in line:
            letters = re.findall(r"\d+", line)
            digit_strings = [str(i) for i in range(1, 10)]
            for i, letter in enumerate(letters):
                if letter not in digit_strings:
                    letters[i] = chr(int(letter) + 87)
            word = Word("".join(letters), ascending_order=ascending_order, 
                        optimize=optimize)
        else:
            word = Word(line.strip(), double_occurrence=double_occurrence, 
                        ascending_order=ascending_order, optimize=optimize)
        word_list.append(word)
    if "" not in word_list and include_empty_word:
        empty_word = Word("", double_occurrence=double_occurrence, 
                          ascending_order=ascending_order, optimize=optimize)
        word_list = [empty_word] + word_list

    return word_list