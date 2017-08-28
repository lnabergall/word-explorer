"""
Produces a graph with vertices representing words and directed edges 
representing the deletion of a repeat or return word.
"""

from time import time

from numpy import array

from word_explorer.objects import is_equivalent, Word as Word_eq
from word_explorer.operations.insertions import generate_insertions
from word_explorer.objects.ascending_order import convert_to_ascending_order
#from word_graph_gpu import find_adjacent_vertices as find_adjacent_vertices_gpu
#from word_graph_gpu2 import find_adjacent_vertices as find_adjacent_vertices_gpu
from word_graph_gpu_cputest import find_adjacent_vertices as find_adjacent_vertices_gpu


WORDS_LEQ2_FILE = "words_up_to_size_2_all.txt"
WORDS_LEQ3_FILE = "words_up_to_size_3_all.txt"
WORDS_LEQ4_FILE = "words_up_to_size_4_all.txt"
WORDS_LEQ5_FILE = "words_up_to_size_5_all.txt"

AOWORDS_LEQ2_FILE = "words_up_to_size_2.txt"
AOWORDS_LEQ3_FILE = "words_up_to_size_3.txt"
AOWORDS_LEQ4_FILE = "words_up_to_size_4.txt"
AOWORDS_LEQ5_FILE = "words_up_to_size_5.txt"
AOWORDS_LEQ6_FILE = "words_up_to_size_6.txt"
AOWORDS_LEQ7_FILE = "words_up_to_size_7.txt"

REPEAT_WORD = (
    "1...1", "12...12", "21...21", "123...123", "213...213", 
    "132...132", "321...321", "231...231", "312...312", "1234...1234",
    "1243...1243", "1324...1324", "1342...1342", "1423...1423", 
    "1432...1432", "2143...2143", "2134...2134", "2341...2341",
    "2314...2314", "2431...2431", "2413...2413", "3124...3124",
    "3142...3142", "3214...3214", "3241...3241", "3412...3412", 
    "3421...3421", "4132...4132", "4123...4123", "4231...4231", 
    "4213...4213", "4321...4321", "4312...4312",
)
RETURN_WORD = (
    "1...1", "12...21", "21...12", "123...321", "213...312", 
    "132...231", "321...123", "231...132", "312...213", "1234...4321",
    "1243...3421", "1324...4231", "1342...2431", "1423...3241", 
    "1432...2341", "2143...3412", "2134...4312", "2341...1432",
    "2314...4132", "2431...1342", "2413...3142", "3124...4213",
    "3142...2413", "3214...4123", "3241...1423", "3412...2143", 
    "3421...1243", "4132...2314", "4123...3214", "4231...1324", 
    "4213...3124", "4321...1234", "4312...2134",
)
REPEAT_WORD_AO = (
    "1...1", "12...12", "123...123", "1234...1234", "12345...12345",
    "123456...123456", "1234567...1234567", "12345678...12345678"
)
RETURN_WORD_AO = (
    "1...1", "12...21", "123...321", "1234...4321", "12345...54321",
    "123456...654321", "1234567...7654321", "12345678...87654321"
)


class Word(Word_eq):

    def __eq__(self, other):
        return str(self) == str(other)

    def __hash__(self):
        return hash(str(self))


def load_words(file_name):
    with open(file_name, "r") as word_file:
        word_list = word_file.readlines()
        word_list.sort()
        if "" not in word_list:
            word_list = [""] + word_list
        return [Word_eq(word.strip()) for word in word_list]


class WordGraph:

    def __init__(self, word_list, size_limit=None, 
                 ascending_order=False, use_gpu=False):
        self.vertices = word_list
        self.size_limit = size_limit
        self.word_class = Word if not ascending_order else Word_eq
        self.use_gpu = use_gpu
        self.ascending_order = ascending_order
        self.directed_neighborhoods = self.compute_neighborhoods()
        if self.ascending_order:
            print("Converting to ascending order...")
            self.directed_neighborhoods = convert_to_ascending_order(
                self.directed_neighborhoods)
            print("Finished!")
        self.vertex_count = len(self.vertices)
        self.edge_count = 0
        for vertex in self.directed_neighborhoods:
            self.edge_count += len(self.directed_neighborhoods[vertex])

    def compute_neighborhoods(self):
        neighborhoods = {}
        if self.use_gpu:
                neighbors_list = find_adjacent_vertices_gpu(
                    self.vertices, self.size_limit, self.ascending_order)
                for word, neighbors in zip(self.vertices, neighbors_list):
                    neighborhoods[word] = neighbors
        else:
            for word in self.vertices:
                neighbors = self.find_adjacent_vertices(word)
                neighborhoods[word] = neighbors

        return neighborhoods

    def find_adjacent_vertices(self, word):
        neighbors = set()
        patterns = (REPEAT_WORD + RETURN_WORD if self.word_class == Word 
                    else REPEAT_WORD_AO + RETURN_WORD_AO)
        for pattern_instance in patterns:
            if len(word)//2 + (len(pattern_instance) - 3)//2 <= self.size_limit:
                some_neighbors = self.generate_insertions(word, pattern_instance)
                neighbors |= some_neighbors

        return neighbors

    def generate_insertions(self, word, pattern_instance):
        return generate_insertions(
            word, pattern_instance, self.size_limit, self.word_class, 
            ascending_order=self.ascending_order)


if __name__ == '__main__':
    words = load_words(AOWORDS_LEQ6_FILE)
    start_time = time()
    word_graph = WordGraph(words, size_limit=6, ascending_order=True, use_gpu=False)
    end_time = time()
    print("Total time:", end_time - start_time)
    with open("aoword_graph_size6_fast.txt", "w") as output_file:
        print("Vertex count: " + str(word_graph.vertex_count), file=output_file)
        print("Edge count: " + str(word_graph.edge_count) + "\n\n", file=output_file)
        words = list(word_graph.directed_neighborhoods.keys())
        words.sort()
        for word in words:
            if word_graph.directed_neighborhoods[word]:
                neighborhood = word + ": " + str(
                    word_graph.directed_neighborhoods[word])
                print(neighborhood, file=output_file)
    with open("aoword_graph_size6_fast.txt", "r") as output_file:
        text = output_file.read()
        text = text.replace("\'", "")
    with open("aoword_graph_size6_fast.txt", "w") as output_file:
        output_file.write(text)
