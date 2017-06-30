"""
Produces a list of all DOWs up to a user-specified size 
and saves the list to a (user-specified) text file.
"""

from random import shuffle
from itertools import permutations, combinations

#from objects import Word
from word_graph import Word, convert_to_ascending_order


def main():
    size = int(input("\nUp to what size words do you wish to retrieve? "))
    file_name = input("Enter a text file name for the output list: ")
    word_list = get_word_list(size)
    # time_diff = (time_end - time_start).total_seconds()
    # print(time_diff)
    with open(file_name, "w") as output_file:
        for word in word_list:
            print(word, file=output_file)


def get_word_list(size):
    word_list = set()
    for s in range(1, size+1):
        print("generating words of size " + str(s) + "...")
        letters = [str(i) for i in range(1, min(s,9)+1)]
        if s > 9:
            letters.extend(chr(i+96) for i in range(10, s+1))
        words = set(Word("".join(tupl)) for tupl in permutations(letters+letters))
        word_list |= words
        
    word_list = list(word_list)
    word_list.sort(key=lambda word: len(word))
    return word_list


def get_word_list_noeq(size):
    word_list = set()
    letter_list = [str(i) for i in range(1, min(size,9)+1)]
    if size > 9:
        letter_list.extend(chr(i+96) for i in range(10, size+1))
    for s in range(1, size+1):
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
