"""
Checks whether one word could be generated from another 
with the insertion of a repeat word or return word.
"""

from itertools import product, permutations

from objects import Word


def generate_insertions(word, pattern_instance):
    pattern_parts = pattern_instance.split("...")
    insertions = set()
    insertion_indices = {}
    for i, j in product(range(len(word)+1), range(len(word)+1)):
        if i < j:
            new_word = Word(word[:i] + pattern_parts[0] + word[i:j] 
                            + pattern_parts[1] + word[j:])
            insertion_indices[new_word] = (i, j)
        else:
            new_word = Word(word[:j] + pattern_parts[1] + word[j:i]
                            + pattern_parts[0] + word[i:])
            insertion_indices[new_word] = (j, i)
        insertions.add(new_word)

    return insertions, insertion_indices


def find_adjacent_vertices(word, repeat_word, return_word):
    repeat_neighbors, repeat_insertion_indices = generate_insertions(
        word, repeat_word)
    return_neighbors, return_insertion_indices = generate_insertions(
        word, return_word)

    return (repeat_neighbors, repeat_insertion_indices, 
            return_neighbors, return_insertion_indices)


def is_insertion(word1, word2):
    """
    Args:
        word1: String, a double occurrence word.
        word2: String, a double occurrence word.
    """
    word1 = Word(word1)
    word2 = Word(word2)
    if word1 is None or word2 is None:
        raise ValueError("Only accepts double occurrence words!")
    if len(word2) < len(word1):
        return False
    if len(word1) == len(word2):
        return word1 == word2
    else:
        insertion_letters = list(set(word2) - set(word1))
        insertion_letters.sort()
        pattern_part = "".join(insertion_letters)
        repeat_word = pattern_part + "..." + pattern_part
        return_word = pattern_part + "..." + pattern_part[::-1]
        (repeat_neighbors, repeat_insertion_indices, 
         return_neighbors, return_insertion_indices) = find_adjacent_vertices(
            word1, repeat_word, return_word)
        if word2 in repeat_neighbors:
            return True, repeat_insertion_indices[word2]
        elif word2 in return_neighbors:
            return True, return_insertion_indices[word2]
        else:
            return False

if __name__ == '__main__':
    word1 = input("Double occurrence word: ")
    word2 = input("Another double occurrence word: ")
    print(is_insertion(word1, word2))