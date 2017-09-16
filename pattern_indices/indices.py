"""
Reduces a double occurrence word using user-defined patterns 
and associated reductions. Outputs the associated pattern index of the 
word. This module also contains a simple command-line interface; a more 
powerful and flexible GUI is implemented in interface.py.

Usage:
	
	$ python indices.py

Classes:

	Calculator

Functions:

	output_instructions, output_choices, get_pattern_examples, 
	get_user_input, filter_equivalent_reductions, 
	contains_no_complete_reductions
"""

from word_explorer.objects import (Pattern, Word, PatternIndex, 
								   PatternExample, is_equivalent)
from .storage import StorageHandler


def output_instructions():
	# Print usage instructions
	print("\nReduces a double occurrence word using arbitrary user-defined\n" 
		  + "patterns and outputs the associated pattern index of the word.\n")
	print("Input the word, then input the patterns to be used for reduction.\n")
	print("For each pattern, input an example instance of each possible size.\n"  
		  + "Use '...' to represent a gap of arbitary length.\n"
		  + "e.g. for an example repeat word of size 3, input '123...123'.")
	print("Enter a blank line when completed.")


def output_choices(storage_handler):
	"""
	Args:
		storage_handler: An instance of StorageHandler.
	Prints:
		Pattern and pattern index names.
	Returns:
		A dictionary containing pattern and pattern index names as values 
		for retrieval of the chosen patterns and indices.
	"""
	# Output indices
	index_names = storage_handler.get_index_names()
	if bool(index_names):
		print("\nPattern Indices:")
		for index_name in index_names:
			print(index_name)

	# Output patterns
	pattern_names = storage_handler.get_pattern_names()
	if bool(pattern_names):
		print("\nPatterns:")
		for pattern_name in pattern_names:
			print(pattern_name)

	if not (bool(index_names) or bool(pattern_names)):
		return None

	# Later add reduction operations...

	return {"I": index_names, "P": pattern_names}


def get_pattern_examples(start_size):
	"""
	A function for querying a user via the command-line for 
	pattern examples of size start_size or greater.

	Args:
		start_size: Integer.
	Returns:
		A list of instances of PatternExample. 
	"""
	pattern_examples = []

	# First example
	pattern_example = input("Input an example of size " 
							+ str(start_size) + ": ")

	# Collect larger examples until user enters blank line
	while not pattern_example == "":
		while True:
			try:
				pattern_example = PatternExample(pattern_example)
				if pattern_example.size != start_size:
					raise ValueError()
			except ValueError:
				print("That is not a valid pattern example of size " 
					  + str(start_size) + ". Try again...")
				pattern_example = input("Input an example of size " 
										+ str(start_size) + ": ")
			else:
				pattern_examples.append(pattern_example)
				break
			
		start_size += 1
		pattern_example = input("Input an example of size " 
								+ str(start_size) + ": ")

	return pattern_examples


def get_user_input():
	output_instructions()

	# Input and validate word
	word = input("\nA word (use letters for words of size >= 10): ")
	word = Word(word)
	while word is None:
		print("Not a valid word. Try again...")
		word = input("A word (use letters for words of size >= 10): ")
		word = Word(word)

	# Output pattern, reduction, and index choices
	storage_handler = StorageHandler()
	choices = output_choices(storage_handler)

	# Get user choices and make sure the chosen patterns have sufficiently
	# large examples provided. 
	if choices is not None:
		index_choice = input("\nPattern index to use (leave blank for none): ")
		if index_choice != "":
			pattern_index = storage_handler.get_index(index_choice.strip())
			while pattern_index is None:
				print("Sorry, that choice does not exist, try again...")
				index_choice = input("Pattern index to use: ")
				pattern_index = storage_handler.get_index(index_choice.strip())
			for pattern in pattern_index.patterns:
				if len(pattern) < word.size:
					define_more_sizes = input(
						"The pattern " + pattern.name + 
						" is only defined for sizes up to " + str(len(pattern)) 
						+ ". Define greater sizes ('y' or 'n')? ")
					if define_more_sizes.lower().strip().startswith("y"):
						examples = get_pattern_examples(len(pattern)+1)
						for pattern_example in examples:
							pattern.append(pattern_example)
							storage_handler.store_pattern_example(
								pattern_example, pattern)
			return word, pattern_index
		else:
			patterns = []
			pattern_choice = input("\nPattern to use: ")
			while pattern_choice != "":
				pattern = storage_handler.get_pattern(pattern_choice)
				while pattern is None:
					print("Sorry, that choice does not exist, try again...")
					pattern_choice = input("Pattern to use: ")
					pattern = storage_handler.get_pattern(pattern_choice)
				if len(pattern) < word.size:
					define_more_sizes = input(
						"This pattern is only defined for sizes up to " 
						+ str(len(pattern)) + ". Define greater sizes "
						+ "('y' or 'n')? ")
					if define_more_sizes.lower().strip().startswith("y"):
						examples = get_pattern_examples(len(pattern)+1)
						for pattern_example in examples:
							pattern.append(pattern_example)
							storage_handler.store_pattern_example(
								pattern_example, pattern)
				patterns.append(pattern)
				pattern_choice = input("Pattern to use: ")

	# Ask user if they need to use more patterns and, if so, how many.
	if "patterns" not in locals():
		patterns = []
	else:
		more_patterns_needed = input("Do you require extra new patterns" 
									 + " ('y' or 'n')? ")
		if more_patterns_needed.lower().strip().startswith("n"):
			return word, patterns

	while True:
		try:
			pattern_count = int(input("How many patterns will you use? "))
		except ValueError:
			print("Invalid count. Try again...")
			pattern_count = int(input("How many patterns will you use? "))
		else:
			break

	# Input and validate patterns
	for i in range(pattern_count):
		example_size = 1
		pattern = Pattern()
		print("\nInputting pattern " + str(i+1) + "...")
		pattern_name = input("Pattern name: ")
		pattern.name = pattern_name

		pattern_examples = get_pattern_examples(1)
		for pattern_example in pattern_examples:
			pattern.append(pattern_example)

		storage_handler.store_pattern(pattern)
		patterns.append(pattern)

	# Define a new index
	define_new_index = input("Does this set of patterns define "
							 + "a new pattern index ('y' or 'n')? ")
	if define_new_index.lower().strip().startswith("y"):
		index_name = input("Pattern index name: ")
		pattern_index = PatternIndex(index_name.strip(), patterns)
		storage_handler.store_index(pattern_index)
		return word, pattern_index

	return word, patterns


def filter_equivalent_reductions(reductions):
	"""
	Args:
		reductions: List of reductions, where each reduction is 
			a list of words. Assumes that each reduction is of the same length.
	Returns: 
		A filtered copy of reductions that does not contain two reductions 
		with the same last word.
	"""
	# Remove identical words
	reductions_equivalence_classes = {}
	non_equivalent_reductions = set()
	for i, reduction in enumerate(reductions):
		new_value = reductions_equivalence_classes.get(reduction[-1], [])
		new_value.append(i)
		reductions_equivalence_classes[reduction[-1]] = new_value
		non_equivalent_reductions.add(reduction[-1])

	unique_reductions = []
	for word in non_equivalent_reductions:
		reduction_index = reductions_equivalence_classes[word][-1]
		unique_reductions.append(reductions[reduction_index])

	return unique_reductions

def contains_no_complete_reductions(reductions):
	"""Checks whether reductions contains a reduction with the empty word."""
	for reduction in reductions:
		if reduction[-1] == "":
			return False
	return True

class Calculator():
	"""
	A class for handling the calculation of the pattern index of a given word.
	Uses an instance method, stop, to add the option to halt 
	the computation prematurely.

	Methods:

		stop, calculate_pattern_index
	"""

	def __init__(self):
		self.stop = False

	def stop(self):
		self.stop = True

	def calculate_pattern_index(self, word, patterns):
		# Perform initial reductions.
		letter_removal_used = False
		reductions = []
		for pattern in patterns:
			if pattern.is_instance(["1", "1"], 1) and len(pattern) == 1:
				letter_removal_used = True
				for letter in set(word):
					reduced_word = word.delete_letter(letter)
					reductions.append([reduced_word])
			else:
				instances = word.find_instances(pattern)
				for instance in instances:
					# If equivalent to letter removal
					if len(instance) == 2 and letter_removal_used:	
						continue
					reduced_word = word.perform_reduction(instance)
					reductions.append([reduced_word])

		# Iteratively reduce until a reduction achieves the empty word.
		while contains_no_complete_reductions(reductions):
			if self.stop == True:
				return
			# Remove all but 1 from each "equivalence class" of reductions.
			# Works since all reductions here are of the same size.
			reductions = filter_equivalent_reductions(reductions)
			if self.stop == True:
				return

			# Now continue finding new reductions
			reductions_current = reductions[:]
			for i, reduction in enumerate(reductions_current):
				initial_reduction_size = len(reduction)
				for pattern in patterns:
					if pattern.is_instance(["1", "1"], 1) \
					and len(pattern) == 1:
						for j, letter in enumerate(set(reduction[-1])):
							if self.stop == True:
								return
							reduced_word = reduction[-1].delete_letter(letter)
							new_reduction = reduction[:]
							new_reduction.append(reduced_word)
							if len(reductions[i]) == initial_reduction_size:
								reductions[i] = new_reduction
							else:
								reductions.append(new_reduction)
					else:
						instances = reduction[-1].find_instances(pattern)
						for j, instance in enumerate(instances):
							if self.stop == True:
								return
							# If equivalent to letter removal
							if len(instance) == 2 and letter_removal_used:	
								continue
							reduced_word = reduction[-1].perform_reduction(instance)
							new_reduction = reduction[:]
							new_reduction.append(reduced_word)
							if len(reductions[i]) == initial_reduction_size:
								reductions[i] = new_reduction
							else:
								reductions.append(new_reduction)

		return min(map(len, reductions))


if __name__ == '__main__':
	word, index_or_patterns = get_user_input()
	storage_handler = StorageHandler()
	if isinstance(index_or_patterns, PatternIndex):
		index_value = storage_handler.get_word_index(word, 
			index=index_or_patterns)
		if index_value is None:
			print("Calculating...")
			calc = Calculator()
			index_value = calc.calculate_pattern_index(word, 
				index_or_patterns.patterns)
		print("The", index_or_patterns.name.lower(), "of", word, "is", index_value)
		storage_handler.store_word_index(word, 
			index_value, index=index_or_patterns)
	else:
		index_value = storage_handler.get_word_index(word, 
			patterns=index_or_patterns)
		if index_value is None:
			print("Calculating...")
			calc = Calculator()
			index_value = calc.calculate_pattern_index(word, index_or_patterns)
		print("The pattern index of", word, "is", index_value)
		storage_handler.store_word_index(word, 
			index_value, patterns=index_or_patterns)


# IMPORTANT!
# Implement second reduction type choice, deleting all "maximal", 
# disjoint instances of a pattern in one reduction.

# EVENTUALLY!
# Need to give patterns an understanding of the sizes of their examples,
# but can't assume that a pattern is defined for every natural number 
# size. So either convert to dictionary or create PatternExample class
# with a size attribute.