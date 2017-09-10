"""
Tools for generating and manipulating insertions.

Functions:

    generate_insertions, find_adjacent_vertices, insertion
"""

from itertools import product, permutations

from word_explorer.objects import Word


def generate_insertions(word, pattern_instance, alphabet_size, 
                        double_occurrence=True, ascending_order=False):
    """
    Args:
        word: Instance of Word.
        pattern_instance: Instance of Pattern with two factors. 
        alphabet_size: Integer or None. 
        double_occurrence: Bool, defaults to True.
        ascending_order: Bool, defaults to False.
    Returns:
        A list containing all words constructed by inserting pattern_instance
        into word and a dictionary containing a pair of insertion indices 
        for each word. If double_occurrence = False, then the labels of word 
        and pattern_instance are preserved. If ascending_order = True, 
        we assume that word is in ascending order and relabel pattern_instance 
        into ascending order---this speeds up the algorithm.
    """
    if not double_occurrence:
        instances = [pattern_instance]
    else:
        instance_letters = list(set(pattern_instance.replace("...", "")))
        instance_letters.sort()
        if alphabet_size is None:
            alphabet_size = word.size + len(instance_letters)
        pattern_parts = pattern_instance.split("...")
        return_word = pattern_parts[0] == pattern_parts[1][::-1]
        new_letters = list(range(1, alphabet_size + 1))
        new_letters = [str(letter) for letter in new_letters 
                       if str(letter) not in word]
        instances = []

        for i, indices in enumerate(
                permutations(new_letters, len(instance_letters))):
            if ascending_order and i != 0:
                break
            for j in range(len(pattern_parts)):
                relabeled_part = ""
                for k in range(len(instance_letters)):
                    relabeled_part += indices[k]
                if return_word and j == 1:
                    relabeled_part = relabeled_part[::-1]   # Reverses relabeled_part
                pattern_parts[j] = relabeled_part
            instances.append(pattern_parts[:])

    insertion_indices = {}
    insertions = set()
    for instance in instances:
        for i, j in product(range(len(word)+1), range(len(word)+1)):
            if i < j:
                new_word = Word(word[:i] + instance[0] + word[i:j] 
                                + instance[1] + word[j:], 
                                double_occurrence=double_occurrence)
                insertion_indices[new_word] = (i, j)
            else:
                new_word = Word(word[:j] + instance[1] + word[j:i]
                                + instance[0] + word[i:], 
                                double_occurrence=double_occurrence)
                insertion_indices[new_word] = (j, i)
            insertions.add(new_word)

    return insertions, insertion_indices


def find_adjacent_vertices(word, repeat_word, return_word):
    repeat_neighbors, repeat_insertion_indices = generate_insertions(
        word, repeat_word, None, ascending_order=True)
    return_neighbors, return_insertion_indices = generate_insertions(
        word, return_word, None, ascending_order=True)

    return (repeat_neighbors, repeat_insertion_indices, 
            return_neighbors, return_insertion_indices)


def insertion(word1, word2):
    """
    Args:
        word1: String, a double occurrence word.
        word2: String, a double occurrence word.
    Returns:
        A boolean indicating whether word2 can be constructed by 
        inserting a repeat word or return word into word1. If it can, 
        then the two indices one of possible insertion are also returned. 
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
    print("Insertion?", is_insertion(word1, word2))