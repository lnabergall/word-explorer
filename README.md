word-explorer 

A collection of modules for working with words, in particular double occurrence words in ascending order, calculating pattern indices and word distances, and constructing and analyzing word graphs and their homology.

The library is organized into five packages:

objects - Contains the implementations of words, patterns, and pattern indices as classes, as well as scripts for generating word lists and converting words into ascending order.

operations - Contains functions for generating insertions and deletions of pattern instances from a word and manipulating the resulting reductions and reverse reductions. 

pattern_indices - Contains a GUI script for calculating the pattern index of a word or the distance between two words, as well as functions for storing and retrieving patterns and pattern indices and calculating some simple statistics of a batch of pattern index computations. 

word_graph - Contains classes and functions for generating word graphs, where each insertion of a pattern instance into a word defines an edge between two vertices (words), and locating special subgraphs within word graphs. Both pure Python and CUDA implementations are provided for trading off readability vs. speed, with the latter implemented using numba and a restricted version of Python.