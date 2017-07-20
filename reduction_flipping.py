"""
Functions for constructing reductions and reverse reductions 
on arbitrary words, primarily for testing the conditions 
under which paths can be 'flipped'.
"""

from itertools import product

from objects import GeneralizedPattern, Word
from ascending_order import convert_to_ascending_order
from insertions import generate_insertions
from list_words import get_all_words


def generate_pattern_instances(pattern, alphabet_size, max_length):
    if (len(pattern.variables)*(alphabet_size**max_length)
            // pattern.max_variable_repetitions > 50000000):
        raise ValueError("Too many pattern instances to compute " 
                         "without likely exceeding memory limits!")

    possible_images = get_all_words(
        max_length // pattern.max_variable_repetitions, alphabet_size)
    variables = list(pattern.variables)
    morphisms = product(possible_images, repeat=len(variables))
    pattern_instances = []
    for morphism in morphisms:
        morphism_dict = {variables[i]: morphism[i] 
                         for i in range(len(variables))}
        pattern_instance = pattern.instance_from_morphism(morphism_dict)
        if len("".join(pattern_instance)) <= max_length:
            pattern_instances.append(pattern_instance)

    return pattern_instances


def get_deletions(word, pattern):
    instance_indices = word.find_instances(pattern)
    return [word.perform_reduction(instance) for instance in instance_indices]


def get_insertions(word, pattern, alphabet_size, max_word_length):
    insertions = set()
    if type(pattern) == GeneralizedPattern:
        pattern_instances = generate_pattern_instances(
            pattern, alphabet_size, max_word_length-len(word))
    else:
        raise NotImplementedError(
            "Pattern should be of type 'GeneralizedPattern'!")
    for pattern_instance in pattern_instances:
        instance_insertions = generate_insertions(
            word, pattern_instance, alphabet_size, 
            Word, double_occurrence=False)
        insertions |= instance_insertions

    return list(insertions)


def generate_paths(start_word, pattern, alphabet_size, 
                   max_path_length, max_word_length):
    paths = [[start_word],]
    for i in range(max_path_length):
        extended_paths = []
        for j, path in enumerate(paths):
            last_word = path[-1]
            deletions = get_deletions(last_word, pattern)
            insertions = get_insertions(last_word, pattern, 
                                        alphabet_size, max_word_length)
            extended_paths.extend(
                [path + [word] for word in deletions + insertions])
        paths.extend(extended_paths)

    return paths


def filter_paths(paths):
    paths_without_repetitions = []
    for path in paths:
        if len(set(path)) == len(path):
            paths_without_repetitions.append(path)

    return paths_without_repetitions


def test_flipping(start_word, pattern, alphabet_size, max_path_length, 
                  max_word_length, ascending_order=False):
    paths = generate_paths(start_word, pattern, alphabet_size, 
                           max_path_length, max_word_length)
    if ascending_order:
        paths = filter_paths(convert_to_ascending_order(paths))

    path_types = {}
    for i, path in enumerate(paths):
        for j in range(len(path)):
            if (j >= 2 and len(path[j-2]) < len(path[j-1]) 
                    and len(path[j-1]) > len(path[j])):
                types_seen = path_types.get(path[-1], set())
                types_seen.add(("peak", i))
                path_types[path[-1]] = types_seen
                break
        else:
            types_seen = path_types.get(path[-1], set())
            types_seen.add(("no_peak", i))
            path_types[path[-1]] = types_seen

    counterexamples = 0
    for end_word in path_types:
        if (all(type_tuple[0] == "peak" for type_tuple in path_types[end_word]) 
                or min([len(paths[type_tuple[1]]) for type_tuple in path_types[end_word]
                        if type_tuple[0] == "peak"], default=100000)
                < min([len(paths[type_tuple[1]]) for type_tuple in path_types[end_word]
                       if type_tuple[0] == "no_peak"], default=100000)):
            print("Counterexample found!")
            print(end_word)
            if end_word == "123342":
                for type_tuple in path_types[end_word]:
                    print(paths[type_tuple[1]])
            if end_word == "1234555":
                for type_tuple in path_types[end_word]:
                    print(paths[type_tuple[1]])
            counterexamples += 1

    print("\nStart word:", "'" + start_word + "'")
    print("Paths:", len(paths))
    print("End words:", len(path_types))
    print("Counterexamples:", counterexamples)


if __name__ == '__main__':
    word = Word("1212", double_occurrence=False)
    pattern = GeneralizedPattern((("a", ""), ("a", "")))
    test_flipping(word, pattern, 6, 3, 9, ascending_order=True)
