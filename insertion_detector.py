"""
Checks whether one word could be generated from another 
with the insertion of a repeat word or return word.
"""

from itertools import product, permutations

from objects import Word


def generate_insertions(word, pattern_instance, size_limit):
    new_letters = list(range(1, size_limit + 1))
    new_letters = [str(letter) for letter in new_letters 
                   if str(letter) not in word]
    instance_letters = list(set(pattern_instance.replace("...", "")))
    instance_letters.sort()
    pattern_parts = pattern_instance.split("...")
    return_word = pattern_parts[0] == pattern_parts[1][::-1]
    insertions = set()
    insertion_indices = {}
    instances = []

    for indices in permutations(new_letters, len(instance_letters)):
        for i in range(len(pattern_parts)):
            relabeled_part = ""
            for j in range(len(instance_letters)):
                relabeled_part += indices[j]
            if return_word and i == 1:
                relabeled_part = relabeled_part[::-1]   # Reverses relabeled_part
            pattern_parts[i] = relabeled_part
        instances.append(pattern_parts[:])

    for instance in instances:
        for i, j in product(range(len(word)+1), range(len(word)+1)):
            if i < j:
                new_word = Word(word[:i] + instance[0] + word[i:j] 
                                + instance[1] + word[j:])
                insertion_indices[new_word] = (i, j)
            else:
                new_word = Word(word[:j] + instance[1] + word[j:i]
                                + instance[0] + word[i:])
                insertion_indices[new_word] = (j, i)
            insertions.add(new_word)

    return insertions, insertion_indices


def find_adjacent_vertices(word, repeat_word, return_word, size_limit):
    repeat_neighbors, repeat_insertion_indices = generate_insertions(
        word, repeat_word, size_limit)
    return_neighbors, return_insertion_indices = generate_insertions(
        word, return_word, size_limit)

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
        size_limit = word2.size
        insertion_size = word2.size - word1.size
        if insertion_size < 10:
            pattern_part = "".join([str(i) for i in range(insertion_size)])
        else:
            pattern_part = "".join([str(i) for i in range(10)])
            for i in range(insertion_size-9):
                pattern_part += chr(i+97)

        repeat_word = pattern_part + "..." + pattern_part
        return_word = pattern_part + "..." + pattern_part[::-1]
        (repeat_neighbors, repeat_insertion_indices, 
         return_neighbors, return_insertion_indices) = find_adjacent_vertices(
            word1, repeat_word, return_word, size_limit)
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