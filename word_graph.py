"""
Produces a graph with vertices representing words and directed edges 
representing the deletion of a repeat or return word.
"""

from itertools import product, permutations

from objects import is_equivalent, Word as Word_eq
from word_graph_gpu import find_adjacent_vertices as find_adjacent_vertices_gpu


WORDS_LEQ2_FILE = "words_up_to_size_2_all.txt"
WORDS_LEQ3_FILE = "words_up_to_size_3_all.txt"
WORDS_LEQ4_FILE = "words_up_to_size_4_all.txt"

AOWORDS_LEQ2_FILE = "words_up_to_size_2.txt"
AOWORDS_LEQ3_FILE = "words_up_to_size_3.txt"
AOWORDS_LEQ4_FILE = "words_up_to_size_4.txt"
AOWORDS_LEQ5_FILE = "words_up_to_size_5.txt"
AOWORDS_LEQ6_FILE = "words_up_to_size_6.txt"

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
        self.directed_neighborhoods = self.compute_neighborhoods()
        if ascending_order:
            self.directed_neighborhoods = convert_to_ascending_order(
                self.directed_neighborhoods)
        self.vertex_count = len(self.vertices)
        self.edge_count = 0
        for vertex in self.directed_neighborhoods:
            self.edge_count += len(self.directed_neighborhoods[vertex])

    def compute_neighborhoods(self):
        neighborhoods = {}
        for word in self.vertices:
            if self.use_gpu:
                neighbors = find_adjacent_vertices_gpu(word, self.size_limit)
            else:
                neighbors = WordGraph.find_adjacent_vertices(
                    word, self.size_limit, self.word_class)
            neighborhoods[word] = neighbors

        return neighborhoods

    @staticmethod
    def find_adjacent_vertices(word, size_limit, word_class):
        neighbors = set()
        patterns = (REPEAT_WORD + RETURN_WORD if word_class == Word 
                    else REPEAT_WORD_AO + RETURN_WORD_AO)
        for pattern_instance in patterns:
            if len(word)//2 + (len(pattern_instance) - 3)//2 <= size_limit:
                some_neighbors = WordGraph.generate_insertions(
                    word, pattern_instance, size_limit, word_class)
                neighbors |= some_neighbors

        return neighbors

    @staticmethod
    def generate_insertions(word, pattern_instance, size_limit, word_class):
        new_letters = list(range(1, size_limit + 1))
        new_letters = [str(letter) for letter in new_letters 
                       if str(letter) not in word]
        instance_letters = list(set(pattern_instance.replace("...", "")))
        instance_letters.sort()
        pattern_parts = pattern_instance.split("...")
        return_word = pattern_parts[0] == pattern_parts[1][::-1]
        insertions = set()
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
                    new_word = word_class(word[:i] + instance[0] + word[i:j] 
                                          + instance[1] + word[j:])
                else:
                    new_word = word_class(word[:j] + instance[1] + word[j:i]
                                          + instance[0] + word[i:])
                insertions.add(new_word)

        return insertions


def convert_to_ascending_order(word_collection):
    collection_type = type(word_collection)
    if collection_type == dict:
        converted_collection = {}
        for key in word_collection:
            if type(key) == Word_eq:
                key = convert_to_ascending_order(key)
            converted_value = convert_to_ascending_order(word_collection[key])
            converted_collection[key] = converted_value
        return converted_collection
    elif collection_type == list:
        for i, element in enumerate(word_collection):
            word_collection[i] = convert_to_ascending_order(element)
        return word_collection
    elif collection_type == tuple:
        return tuple(convert_to_ascending_order(list(word_collection)))
    elif collection_type == set:
        return set(convert_to_ascending_order(list(word_collection)))
    elif collection_type == Word_eq:
        if word_collection.size >= 10:
            raise NotImplementedError()
        translation = {}
        new_word = ""
        for char in word_collection:
            try:
                new_letter = translation[char]
            except KeyError:
                new_letter = str(len(translation)+1)
                translation[char] = new_letter
            finally:
                new_word += new_letter
        return Word_eq(new_word)
    else:
        raise TypeError("Invalid word collection type!")


if __name__ == '__main__':
    words = load_words(WORDS_LEQ4_FILE)
    word_graph = WordGraph(words, size_limit=4, ascending_order=False, use_gpu=True)
    with open("word_graph_size4_gpucodetest.txt", "w") as output_file:
        print("Vertex count: " + str(word_graph.vertex_count), file=output_file)
        print("Edge count: " + str(word_graph.edge_count) + "\n\n", file=output_file)
        words = list(word_graph.directed_neighborhoods.keys())
        words.sort()
        for word in words:
            if word_graph.directed_neighborhoods[word]:
                neighborhood = word + ": " + str(
                    word_graph.directed_neighborhoods[word])
                print(neighborhood, file=output_file)
    with open("word_graph_size4_gpucodetest.txt", "r") as output_file:
        text = output_file.read()
        text = text.replace("\'", "")
    with open("word_graph_size4_gpucodetest.txt", "w") as output_file:
        output_file.write(text)
