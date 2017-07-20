"""
Produces a list of all DOWs up to a user-specified size 
and saves the list to a (user-specified) text file.
"""

from random import shuffle
from itertools import permutations, combinations, product

#from objects import Word
from word_graph import Word
from ascending_order import convert_to_ascending_order


def main():
    max_size = int(input("\nUp to what size words do you wish to retrieve? "))
    file_name = input("Enter a text file name for the output list: ")
    word_list = get_dow_list(max_size)
    # time_diff = (time_end - time_start).total_seconds()
    # print(time_diff)
    with open(file_name, "w") as output_file:
        for word in word_list:
            print(word, file=output_file)


def get_all_words(max_length, alphabet_size):
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

    return word_list


def get_all_dows(max_size):
    word_list = set()
    for s in range(1, max_size+1):
        print("generating words of size " + str(s) + "...")
        letters = [str(i) for i in range(1, min(s,9)+1)]
        if s > 9:
            letters.extend(chr(i+96) for i in range(10, s+1))
        words = set(Word("".join(tupl)) for tupl in permutations(letters+letters))
        word_list |= words
        
    word_list = list(word_list)
    word_list.sort(key=lambda word: len(word))
    return word_list


def get_all_dows_noeq(max_size):
    word_list = set()
    letter_list = [str(i) for i in range(1, min(max_size,9)+1)]
    if max_size > 9:
        letter_list.extend(chr(i+96) for i in range(10, max_size+1))
    for s in range(1, max_size+1):
        print("generating words of size " + str(s) + "...")
        for letters in combinations(letter_list, s):
            words = set(Word("".join(tupl)) for tupl in permutations(letters+letters))
            word_list |= words

    word_list = list(word_list)
    word_list.sort()
    word_list.sort(key=lambda word: len(word))
    return word_list


def get_random_sample(word_list):
    random_sample = []
    for word in word_list:
        letters = [letter for letter in word]
        shuffle(letters)
        random_word = Word(convert_to_ascending_order("".join(letters)))
        random_sample.append(random_word)

    return random_sample


if __name__ == '__main__':
    main()
