"""
Input/output for operations API.

Functions:

    output_flipping_results
"""

from word_explorer.io import store_data, retrieve_data


def output_flipping_results(paths, path_types, counterexamples):
    for word in counterexamples:
        print("Counterexample found:", word)
    print("\nStart word:", "'" + start_word + "'")
    print("Paths:", len(paths))
    print("End words:", len(path_types))
    print("Counterexamples:", len(counterexamples))