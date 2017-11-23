"""
Contains classes for storing and retrieving patterns, reductions operations, 
pattern indices, and the pattern index values of specific words. 
There are two interfaces that have been implemented: StorageHandler
and SQLStorageHandler---the former uses a text-file-based system, while
the latter uses a local SQLite database and SQLAlchemy ORM. 

Classes:

	StorageHandler, SQLStorageHandler, StoredPattern, StoredPatternIndex, 
	StoredWord, Value
"""

import re

from word_explorer.objects import (Pattern, PatternExample, 
								   PatternIndex, is_equivalent)

PATTERN_STORE = "pattern_indices/data/patterns.txt"
REDUCTION_STORE = "pattern_indices/data/reduction_operations.txt"
INDEX_STORE = "pattern_indices/data/pattern_indices.txt"
WORD_STORE = "pattern_indices/data/word_indices.txt"


class StorageHandler():
	"""
	Handles the storage and retrieval of all patterns, reduction 
	operations, pattern indices, and any words for which indices have 
	been found. By default, expects the existence of global strings 
	PATTERN_STORE, REDUCTION_STORE, INDEX_STORE, and WORD_STORE specifying 
	the names of the pattern, reduction operation, pattern index, and 
	word storage text files.

	Methods:
		store_pattern, store_pattern_example, store_reduction
		store_index, store_word_index, get_pattern, get_reduction
		get_index, get_word_index, get_pattern_names, get_reduction_names
		get_index_names
	"""

	def __init__(self, pattern_store=PATTERN_STORE, 
				 reduction_store=REDUCTION_STORE,
				 index_store=INDEX_STORE, word_store=WORD_STORE):
		self.pattern_store = pattern_store
		self.reduction_store = reduction_store
		self.index_store = index_store
		self.word_store = word_store

	def store_pattern(self, pattern):
		with open(self.pattern_store, "a") as pattern_store:
			pattern_store.write("\n\nName: " + pattern.name + "\n")
			for size_example in pattern:
				pattern_store.write(size_example + "\n")

	def store_pattern_example(self, pattern_example, pattern):
		with open(self.pattern_store, "r") as pattern_store:
			pattern_store_text = pattern_store.read()
			pattern_store_lines = pattern_store_text.splitlines()

		pattern_exists = False
		for i, line in enumerate(pattern_store_lines):
			if line.find(pattern.name) != -1:
				line_num = i
				pattern_exists = True
				break
		if not pattern_exists:
			raise EOFError("Pattern was not found!")
		else:
			for j, line in enumerate(pattern_store_lines[line_num+1:],
									 start=line_num+1):
				if line.strip() == pattern_example:
					break
				if line == "" and pattern_store_lines[j+1] == "":
					line = line
					new_line_num = j
					break
		try:
			pattern_store_lines.insert(new_line_num, pattern_example)
		except NameError:
			pattern_store_lines.append(pattern_example)

		with open(self.pattern_store, "w") as pattern_store:
			for line in pattern_store_lines:
				print(line, file=pattern_store)

	def store_reduction(self, reduction_operation):
		pass 	# Add later

	def store_index(self, index):
		with open(self.index_store, "a") as index_store:
			index_store.write("\n\nName: " + index.name + "\n")
			for i, pattern in enumerate(index.patterns):
				if i == 0:
					index_store.write("Patterns: " + pattern.name)
				else:
					index_store.write(", " + pattern.name)
			if index.reductions is not None:
				pass	# Add later

	def store_word_index(self, word, index_value, patterns=None, 
				   reductions=None, index=None):
		if bool(patterns) == bool(index):
			raise ValueError("Requires exactly one keyword \
							 argument among pattern and index.")
		with open(self.word_store, "r") as word_store:
			word_store_text = word_store.read()
			word_store_lines = word_store_text.splitlines()

		# Check first if word is already stored.
		new_word = True
		for j, line in enumerate(word_store_lines):
			if line.strip().replace("Word: ", "") == word:
				new_word = False
				line_num = j
				break
		if new_word:
			word_store_lines.append("\nWord: " + word)
		# If using unnamed index
		if patterns is not None:
			for i, pattern in enumerate(patterns):
				if i == 0:
					output = "Patterns: " + pattern.name
				else:
					output += ", " + pattern.name
			if reductions is not None:
				pass 	# Add later
		else:	# If using named index
			output = "Index: " + index.name
		output += "; " + str(index_value)
		# Check for repetition
		if not new_word:
			for k, line in enumerate(word_store_lines[line_num+1:], 
				start=line_num+1):
				if line == "":
					new_line_num = k
					break
				if line.lower().find(output.lower()) != -1:
					return
		try:
			word_store_lines.insert(new_line_num, output)
		except NameError:
			word_store_lines.append(output)

		with open(self.word_store, "w") as word_store:
			for line in word_store_lines:
				print(line, file=word_store)

	def get_pattern(self, pattern_name):
		if pattern_name == "":
			return None
		size_examples = []
		found = False
		try:
			with open(self.pattern_store, "r") as pattern_store:
				for line in pattern_store:
					if found and line.strip() == "":
						break
					if found:
						size_examples.append(PatternExample(
							line.strip().replace("\n", "")))
					if line.lower().find(pattern_name.lower()) != -1:
						found = True
		except FileNotFoundError:
			pass

		if not found:
			return None
		else:
			return Pattern(size_examples, name=pattern_name)

	def get_reduction(self, reduction_name):
		pass 	# Add later

	def get_index(self, index_name):
		if index_name == "":
			return None
		patterns = []
		found = False
		try:
			with open(self.index_store, "r") as index_store:
				for line in index_store:
					if found and line.startswith("Patterns: "):
						pattern_names = [name.strip() for name
										 in line[10:].split(", ")]
						for pattern_name in pattern_names:
							pattern = self.get_pattern(pattern_name)
							patterns.append(pattern)
						break
					if line.strip().startswith("Name: ") and \
					line.lower().find(index_name.lower()) != -1:
						found = True
		except FileNotFoundError:
			pass

		if not found:
			return None
		else:
			return PatternIndex(index_name, patterns)

	def get_word_index(self, word, patterns=None, 
					   reductions=None, index=None):
		if bool(patterns) == bool(index):
			raise ValueError("Requires exactly one keyword " 
							 + "argument among pattern and index.")
		found = False
		try:
			with open(self.word_store, "r") as word_store:
				for line in word_store:
					if line.find(word) != -1:
						found = True
						break
				if not found:
					return None
				else:
					line = "fill"
					while line != "\n" and line != "":
						line = word_store.readline()
						if index is not None:
							if line.startswith("Index: " + index.name):
								return int(line.rstrip()[-2:])
						else:
							if line.strip().startswith("Patterns: "):
								match = True
								for pattern in patterns:
									if line.lower().find(
									pattern.name.lower()) != -1:
										line = line.lower().replace(
											pattern.name.lower(), "")
									else:
										match = False
										break
								if not match:
									continue
								if not bool(re.search(r"[a-zA-Z]", 
											line.replace("patterns: ", ""))):
									return int(line.rstrip()[-2:])
							if reductions is not None:
								pass 	# Add later
		except FileNotFoundError:
			pass

		return None

	def get_pattern_names(self):
		pattern_names = []
		try:
			with open(self.pattern_store, "r") as pattern_store:
				for line in pattern_store:
					if line.startswith("Name: "):
						pattern_names.append(line.strip()[6:])
		except FileNotFoundError:
			pass

		return pattern_names

	def get_reduction_names(self):
		pass 	# Add later

	def get_index_names(self):
		index_names = []
		try:
			with open(self.index_store, "r") as index_store:
				for line in index_store:
					if line.startswith("Name: "):
						index_names.append(line.strip()[6:])
		except FileNotFoundError:
			pass

		return index_names


#####################################################################

# SQLite storage

from sqlalchemy import (create_engine, Table, Column, 
						Integer, String, ForeignKey, func)
from sqlalchemy.orm import relationship, backref, sessionmaker
from sqlalchemy.ext.declarative import declarative_base

DATABASE = "sqlite:///pattern_index.db"
Base = declarative_base()


class SQLStorageHandler():

	def __init__(self, database=DATABASE):
		engine = create_engine(DATABASE, echo=False)
		Base.metadata.create_all(engine)
		Base.metadata.bind = engine
		self.session = sessionmaker(bind=engine)()

	def store_pattern(self, pattern):
		if pattern.base is None:
			new_pattern = StoredPattern(
				name=pattern.name.strip(), 
				examples=str(pattern),
			)
		else:
			new_pattern = StoredPattern(
				name=pattern.name.strip(), 
				examples=str(pattern),
				base=pattern.base,
				inductive_indices=str(pattern.inductive_step),
			)
		try:
			self.session.add(new_pattern)
			self.session.commit()
		except:
			self.session.rollback()
			return

	def store_pattern_example(self, pattern_example, pattern):
		try:
			stored_pattern = self.session.query(StoredPattern).filter(
				StoredPattern.name.ilike("%" + pattern.name + "%")).one()
			examples = eval(stored_pattern.examples)
			examples.append(pattern_example)
			stored_pattern.examples = str(examples)
			self.session.commit()
		except:
			self.session.rollback()
			return

	def store_reduction(self, reduction_operation):
		pass 	# Add later

	def store_index(self, index):
		try:
			patterns = self.session.query(StoredPattern).filter(
				func.lower(StoredPattern.name).in_(
					[pattern.name.lower() for pattern in index.patterns]))
			new_index = StoredPatternIndex(
				name=index.name.strip(),
				patterns=patterns,
			)
			self.session.add(new_index)
			self.session.commit()
		except:
			self.session.rollback()
			return

	def store_word_index(self, word, index_value, patterns=None, 
				   		 reductions=None, index=None):
		if patterns is not None and index is not None:
			raise ValueError("Requires exactly one keyword \
							 argument among pattern and index.")
		try:
			stored_word = self.session.query(StoredWord).filter(
				StoredWord.word == word).one()
			if patterns is not None:
				stored_patterns = self.session.query(StoredPattern).filter(
					func.lower(StoredPattern.name).in_(
						[pattern.name.lower() for pattern in patterns]))
				new_computed_value = Value(
					value=index_value,
					word=stored_word,
					patterns=stored_patterns,
				)
			else:
				stored_index = self.session.query(StoredPatternIndex).filter(
					StoredPatternIndex.name.ilike("%" + index.name + "%")).one()
				new_computed_value = Value(
					value=index_value,
					word=stored_word,
					pattern_index=stored_index,
				)
			self.session.add(new_computed_value)
			self.session.commit()
		except:
			self.session.rollback()
			return

	def delete_pattern(self, pattern):
		try:
			stored_pattern = self.session.query(StoredPattern).filter(
				StoredPattern.name.ilike("%" + pattern.name + "%")).one()
			self.session.delete(stored_pattern)
			self.session.commit()
		except:
			self.session.rollback()
			return

	def delete_index(self, index):
		try:
			stored_index = self.session.query(StoredPatternIndex).filter(
				StoredPatternIndex.name.ilike("%" + index.name + "%")).one()
			self.session.delete(stored_index)
			self.session.commit()
		except:
			self.session.rollback()
			return

	def delete_reduction(self, reduction_operation):
		pass 	# Add later

	def get_pattern(self, pattern_name):
		if pattern_name == "":
			return None
		try:
			stored_pattern = self.session.query(StoredPattern).filter(
				StoredPattern.name.ilike("%" + pattern_name + "%")).one()
		except:
			return None
		stored_pattern_examples = [PatternExample(example) for example 
									   in eval(stored_pattern.examples)]
		if stored_pattern.base is None:
			return Pattern(stored_pattern_examples, 
						   name=stored_pattern.name)
		else:
			return Pattern(
				stored_pattern_examples,
				name=stored_pattern.name, 
				base=stored_pattern.base, 
				inductive_step=eval(stored_pattern.inductive_indices),
			)

	def get_reduction(self, reduction_name):
		pass 	# Add later

	def get_index(self, index_name):
		if index_name == "":
			return None
		try:
			stored_index = self.session.query(StoredPatternIndex).filter(
				StoredPatternIndex.name.ilike("%" + index_name + "%")).one()
		except:
			return None
		patterns = []
		for stored_pattern in stored_index.patterns:
			stored_pattern_examples = [PatternExample(example) for example 
									   in eval(stored_pattern.examples)]
			if stored_pattern.base is None:
				patterns.append(Pattern(stored_pattern_examples, 
							            name=stored_pattern.name))
			else:
				patterns.append(Pattern(
					stored_pattern_examples,
					name=stored_pattern.name, 
					base=stored_pattern.base, 
					inductive_step=eval(stored_pattern.inductive_indices),
				))
		return PatternIndex(stored_index.name, patterns)

	def get_word_index(self, word, patterns=None, 
					   reductions=None, index=None):
		if patterns is not None and index is not None:
			raise ValueError("Requires exactly one keyword " 
							 + "argument among pattern and index.")
		try:
			if patterns is not None:
				stored_value_query = self.session.query(Value).filter(
					Value.word == word)
				for pattern in patterns:
					stored_value_query = stored_value_query.filter(
						Value.patterns.any(
						StoredPattern.name.ilike("%" + pattern.name + "%")))
			else:
				stored_value_query = self.session.query(Value).filter(
					Value.word == word).filter(
						Value.index.name.ilike("%" + index.name + "%"))
			stored_index_value = stored_value_query.one()
			return stored_index_value.value
		except:
			return None

	def get_pattern_names(self):
		try:
			patterns = self.session.query(StoredPattern).all()
		except:
			return []

		return [pattern.name for pattern in patterns]

	def get_reduction_names(self):
		pass 	# Add later

	def get_index_names(self):
		try:
			indices = self.session.query(StoredPatternIndex).all()
		except:
			return []

		return [index.name for index in indices]


class StoredPattern(Base):
	__tablename__ = "patterns"
	id = Column(Integer, primary_key=True)
	name = Column(String, unique=True)
	examples = Column(String)	# To be eval'd
	base = Column(String)
	inductive_indices = Column(String)	# To be eval'd


index_patterns = Table("index_patterns", Base.metadata,
	Column("index_id", Integer, ForeignKey("pattern_indices.id")),
	Column("pattern_id", Integer, ForeignKey("patterns.id"))
)


class StoredPatternIndex(Base):
	__tablename__ = "pattern_indices"
	id = Column(Integer, primary_key=True)
	name = Column(String, unique=True)
	patterns = relationship("StoredPattern", secondary=index_patterns, 
							backref="pattern_indices")


class StoredWord(Base):
	__tablename__ = "words"
	id = Column(Integer, primary_key=True)
	word = Column(String, unique=True)


value_patterns = Table("value_patterns", Base.metadata,
	Column("value_id", Integer, ForeignKey("values.id")),
	Column("pattern_id", Integer, ForeignKey("patterns.id"))
)


class Value(Base):
	__tablename__ = "values"
	id = Column(Integer, primary_key=True)
	value = Column(Integer)
	word_id = Column(Integer, ForeignKey("words.id"))
	word = relationship("StoredWord", backref="index_values")
	pattern_index_id = Column(Integer, ForeignKey("pattern_indices.id"))
	pattern_index = relationship("StoredPatternIndex", 
		foreign_keys="Value.pattern_index_id", backref="calculated_values")
	patterns = relationship("StoredPattern", secondary=value_patterns,
							backref="calculated_index_values")
