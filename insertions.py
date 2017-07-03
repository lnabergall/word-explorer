"""Functions for generating insertions."""

from itertools import product, permutations


def generate_insertions(word, pattern_instance, alphabet_size,
                        word_class, double_occurrence=True, 
                        ascending_order=False):
        if not double_occurrence:
            instances = [pattern_instance]
        else:
            new_letters = list(range(1, alphabet_size + 1))
            new_letters = [str(letter) for letter in new_letters 
                           if str(letter) not in word]
            instance_letters = list(set(pattern_instance.replace("...", "")))
            instance_letters.sort()
            pattern_parts = pattern_instance.split("...")
            return_word = pattern_parts[0] == pattern_parts[1][::-1]
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

        insertions = set()
        for instance in instances:
            for i, j in product(range(len(word)+1), range(len(word)+1)):
                if i < j:
                    new_word = word_class(word[:i] + instance[0] + word[i:j] 
                                          + instance[1] + word[j:], 
                                          double_occurrence=double_occurrence)
                else:
                    new_word = word_class(word[:j] + instance[1] + word[j:i]
                                          + instance[0] + word[i:], 
                                          double_occurrence=double_occurrence)
                insertions.add(new_word)

        return insertions