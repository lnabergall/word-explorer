"""Utility functions for handling words in ascending order."""

from objects import Word as Word_eq


def convert_to_ascending_order(word_collection):
    collection_type = type(word_collection)
    if collection_type == dict:
        converted_collection = {}
        for key in word_collection:
            if type(key) == Word_eq:
                key = convert_to_ascending_order(key)
            converted_value = convert_to_ascending_order(word_collection[key])
            converted_collection[key] = converted_value
        return converted_collection
    elif collection_type == list:
        for i, element in enumerate(word_collection):
            word_collection[i] = convert_to_ascending_order(element)
        return word_collection
    elif collection_type == tuple:
        return tuple(convert_to_ascending_order(list(word_collection)))
    elif collection_type == set:
        return set(convert_to_ascending_order(list(word_collection)))
    elif collection_type == Word_eq or collection_type == str:
        translation = {}
        new_word = ""
        for char in word_collection:
            try:
                new_letter = translation[char]
            except KeyError:
                if len(translation) < 9:
                    new_letter = str(len(translation)+1)
                else:
                    offset = len(translation) - 9
                    new_letter = chr(97 + offset)
                translation[char] = new_letter
            finally:
                new_word += new_letter
        if collection_type == Word_eq:
            return Word_eq(new_word)
        elif collection_type == str:
            return new_word
    else:
        raise TypeError("Invalid word collection type!")
