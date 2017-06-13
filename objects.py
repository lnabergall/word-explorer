import re
from itertools import combinations
from collections import Counter, OrderedDict


def is_equivalent(seq1, seq2):
	# Checks for existence of a bijection between input sequences.
	letters1 = set(seq1)
	letters2 = set(seq2)
	distinct_mappings = set(zip(seq1, seq2))
	return (len(letters1) == len(letters2) == len(distinct_mappings)
			and len(seq1) == len(seq2))


def is_equivalent_ascending(seq1, seq2):
	return str(seq1) == str(seq2)


class PatternExample(str):

	def __new__(cls, content):
		if not PatternExample.is_pattern_example(content):
			raise ValueError("Not a valid pattern example!")
		else:
			return str.__new__(cls, content)

	def __init__(self, content):
		self.size = self.get_size()

	def __len__(self):
		return len(self.replace("...", ""))

	@staticmethod
	def is_pattern_example(pattern_example):
		return bool(re.fullmatch(r"(?:[0-9a-zA-Z]+\.\.\.|[0-9a-zA-Z]+)+", 
							 	 pattern_example))

	def get_size(self):
		return len(set(self.replace("...", "")))


class Pattern(list):
	"""
	A Pattern is defined by a list of instance examples, one for each possible
	size; "..." is used to denote a gap in a pattern of arbitrary length. 
	A name can also be optionally provided. 

	The pattern can also be defined inductively by providing a base, an example 
	of size 1 (either "11" or "1...1"), and an inductive step, a 2-tuple 
	containing 2 lists. The first list contains two indices specifying where to 
	insert the two occurrences of the second letter to form an example of size 
	2, and the second list contains two indices specifying where to insert the 
	two occurrences of the nth letter relative to the two occurrences of the 
	(n-1)th letter, respectively, for n > 2. When calculating indices, if there 
	is a gap, the gap is included for the first list; for the second list, it 
	is assumed that the there will be an occurrence of each letter on either 
	side of the gap. Indexing is assumed to start at 1. 

	Note: Inductive definitions are up to size 22, and currently there is no
		  method to extend the pattern to larger sizes. 

	Note: For reduction purposes, expects a pattern example of every size up 
		  to the maximum possible instance size (usually 1 less than the size 
		  of the word). This is not enforced in this class, but rather should 
		  be enforced in any corresponding interface. 

	Examples: 
		repeat_word = Pattern(["1...1", "12...12", "123...123"], 
							  name="Repeat word")
		tangled_cord = Pattern(["11", "1212", "121323", "12132434"])
		letter_removal = Pattern(["1...1"], name="Letter removal")
		loop_sequence = Pattern(["11", "1122", "112233", "11223344"])

	Custom Methods:
		is_instance -- Checks if given sequence is an instance of pattern; if 
					   so, returns True, otherwise returns False. 

		find_size_dependencies -- Creates a dictionary mapping pattern sizes
								  to lists of those smaller sizes their 
								  existence is dependent upon.
	"""

	def __init__(self, *args, name=None, base=None, inductive_step=None):
		if name is None:
			raise ValueError("Name argument not provided!")
		list.__init__(self, *args)
		if base is not None and (len(inductive_step) != 2 or 
				len(inductive_step[0]) != 2 or len(inductive_step[1]) != 2 
				or (len(base) != 2 and len(base) != 5) or 
				inductive_step[0][0] > inductive_step[0][1]):
			raise ValueError("Invalid inductive arguments!")
		if base is not None and list(self) == []:
			# Define inductively (note that user indices start at 1):
			self.append(PatternExample(base))
			next_example = list(base)
			next_index1 = inductive_step[0][0]-1
			next_index2 = inductive_step[0][1]
			next_example.insert(next_index1, "2")
			next_example.insert(next_index2, "2")
			self.append(PatternExample("".join(next_example)))

			for i in range(17):
				if inductive_step[1][0] > 0 and inductive_step[1][1] > 0:
					next_index1 += inductive_step[1][0]
					next_index2 += inductive_step[1][1]
				elif inductive_step[1][0] > 0 and inductive_step[1][1] < 0:
					next_index1 += inductive_step[1][0]
					next_index2 += inductive_step[1][1]+1
				elif inductive_step[1][0] < 0 and inductive_step[1][1] > 0:
					next_index1 += inductive_step[1][0]+1
					next_index2 += inductive_step[1][1]
				elif inductive_step[1][0] < 0 and inductive_step[1][1] < 0:
					next_index1 += inductive_step[1][0]+1
					next_index2 += inductive_step[1][1]+1
				else:
					raise ValueError("Invalid index!")

				# Transform indices
				if next_index2 >= next_index1:
					next_index2 += 1
				else:
					next_index1 += 1

				# Check that new indices don't intersect any gap
				gap_start = "".join(next_example).find("...")
				if gap_start != -1 and (next_index1 in 
				[gap_start+1, gap_start+2] or next_index2 in 
				[gap_start+1, gap_start+2]):
					raise ValueError("Invalid index encountered!")

				# Construct and append pattern example
				if i < 7:
					next_example.insert(next_index1, str(i+3))
					next_example.insert(next_index2, str(i+3))
				else:
					next_example.insert(next_index1, chr(i+90))
					next_example.insert(next_index2, chr(i+90))
				self.append(PatternExample("".join(next_example)))

		self.name = name
		self.size_dependencies = self.find_size_dependencies()
		if base is not None:
			self.base = PatternExample(base)
		self.inductive_step = inductive_step

	def append(self, *args):
		if hasattr(self, "base"):
			if self.base is not None:
				raise TypeError("This pattern is already defined inductively.")
		list.append(self, *args)
		self.size_dependencies = self.find_size_dependencies()

	def __eq__(self, other):
		return self.name == other.name

	def __ne__(self, other):
		return not self.__eq__(other)

	def __hash__(self):
		return hash((self.name, ))

	def find_size_dependencies(self):
		""" 
		Returns: Dictionary with pattern sizes as keys and lists of smaller
		sizes as values, where size i is in the list value of size j if a 
		size j instance of this pattern contains a size i instance of this 
		pattern as a 'subsequence'. 
		"""
		# larger_example is currently of size str, change so it is converting
		# examples to PatternExample objects.
		size_dependencies = {}
		gap_counts = [Counter(example)["."] for example in self]

		for i, example in enumerate(self):
			for j, larger_example in enumerate(self[i+1:], start=i+1):
				if gap_counts[i] > gap_counts[j]:	# more gaps in smaller
					continue
				non_gap_indices = set(i for i in range(len(larger_example) 
					+ gap_counts[j]) if not larger_example[i] == ".")
				gap_indices = set(i for i in range(len(larger_example)
					+ gap_counts[j]) if i not in non_gap_indices)
				possible_start_indices = combinations(non_gap_indices, 
					len(example.split("...")))
				for index_list in possible_start_indices:
					valid_starts = True
					for k, part in zip(index_list, example.split("...")):
						if bool(set(range(k, k+len(part))) & gap_indices):
							valid_starts = False
					if not valid_starts:
						continue
					else:
						sequence = [larger_example[k:k+len(part)] for k, part 
									in zip(index_list, example.split("..."))]
						if self.is_instance(sequence, i+1):
							dependencies = size_dependencies.get(j+1, set())
							dependencies.add(i+1)
							size_dependencies[j+1] = dependencies

		return size_dependencies

	def is_instance(self, sequence, pattern_size):
		pattern_joined = "".join(self[pattern_size-1].split("..."))
		possible_instance = "".join(sequence)
		if is_equivalent(pattern_joined, possible_instance):
			return True
		else:
			return False


class Word(str):

	def __new__(cls, content):
		if content != "" and not Word.is_double_occurrence_word(content):
			return None
		else:
			return str.__new__(cls, content)

	def __eq__(self, other):
		return str(self) == str(other)
		# if self is None:
		# 	return other is None
		# elif other is None:
		# 	return self is None
		# else:
		# 	return is_equivalent(self, other)

	def __ne__(self, other):
		return not self.__eq__(other)

	def __hash__(self):
		letter_indices = OrderedDict()
		for i, letter in enumerate(self):
			letter_indices.setdefault(letter, [])
			letter_indices[letter].append(i)
		letter_indices_list = []
		for index_list in letter_indices.values():
			index_list = tuple(index_list)
			letter_indices_list.append(index_list)
		return hash((tuple(letter_indices_list), ))

	def __init__(self, content):
		self.size = len(self) // 2
		# self.irreducible = Word.is_irreducible(content)
		# self.strongly_irreducible = self.is_strongly_irreducible(content)

	@staticmethod
	def is_double_occurrence_word(word_string):
		letter_counts = set(Counter(word_string).values())
		if len(letter_counts) == 1 and letter_counts.pop() == 2:
			return True
		else:
			return False

	@staticmethod
	def is_irreducible(word_string):
		for i in range(1, len(word_string)-1):
			if (Word.is_double_occurrence_word(word_string[:i]) 
			and Word.is_double_occurrence_word(word_string[i:])):
				return False
		return True

	def is_strongly_irreducible(self, word_string):
		if self.irreducible == False:
			return False
		for i in range(len(word_string)):
			for j in range(i+2, len(word_string)):
				if Word.is_double_occurrence_word(word_string[i:j]):
					return False
		return True

	def delete_letter(self, letter):
		reduced_word = "".join(self.split(str(letter)))
		return Word(reduced_word)

	def find_instances(self, pattern):
		"""
		Input: An instance of Pattern.
		Returns: List of lists of indices of the instances of the 
				 input pattern in this word.
		"""

		pattern_parts = [pattern[i].split("...") 
						 for i in range(len(pattern))]
		pattern_parts_sizes = ([len(part) for part in pattern_parts[i]] 
							   for i in range(len(pattern_parts)))
		instances = []
		sizes_found = set()
		for i, sizes in enumerate(pattern_parts_sizes):
			# First test if there is a size dependency and that an 
			# instance of that size was not found already.
			if i >= 2:
				skip = False
				for j in range(2,i+1):
					if j in pattern.size_dependencies.get(i+1, set()) \
					and j not in sizes_found:
						skip = True
						break
				if skip:
					continue

			start_index_lists = combinations(range(len(self)), len(sizes))
			for indices in start_index_lists:
				# First check if these indices are spread wide 
				# enough for this pattern example.
				valid_starts = True
				for j, size in enumerate(sizes):
					try:
						if not indices[j+1]-indices[j] >= size:
							valid_starts = False
							break
					except IndexError:
						break
				if not len(self)-indices[-1] >= sizes[-1]:
					valid_starts = False
				if not valid_starts:
					continue
				# Indices have passed, so compare the sequence defined 
				# by these indices against the pattern. 
				else:
					sequence = []
					sequence_indices = []
					for size, index in zip(sizes, indices):
						sequence.append(self[index:index+size])
						sequence_indices.extend(range(index, index+size))
					if pattern.is_instance(sequence, i+1):
						instances.append(sequence_indices)
						sizes_found.add(i+1)

		return instances

	def perform_reduction(self, pattern_instance_indices):
		return Word("".join(self[i] for i in range(len(self)) 
							if i not in pattern_instance_indices))


class PatternIndex():

	def __init__(self, name, patterns, reductions=None):
		self.name = name
		self.patterns = patterns
		self.reductions = reductions


