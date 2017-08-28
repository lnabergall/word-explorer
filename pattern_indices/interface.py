"""
A tkinter GUI for the pattern index calculator (that is, the class
Calculator) implemented in main.py.

Usage:

	$ python interface.py

Classes:

	PatternIndexApp, CalculatingThread, CalculatingDialog, Controller,
	ReductionOptionsView, ReductionSelectionsView, SizedButton, 
	OutputView, PatternDialog, IndexDialog
"""

import threading
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import font
from tkinter.filedialog import asksaveasfile, askopenfile

from main import Calculator
from storage import StorageHandler, SQLStorageHandler
from word_explorer.objects import Pattern, Word, PatternIndex, PatternExample


class PatternIndexApp(ttk.Frame):

	def __init__(self, master=None):
		self.master = master
		ttk.Frame.__init__(self, master, width=750, height=600, padding=10)
		self.pack(expand=True, fill=tk.BOTH)

		# Define base layout
		self.input_pane = ttk.PanedWindow(self, orient=tk.VERTICAL)
		self.input_pane.grid(column=0, row=0, padx=10, pady=10)
		self.output_pane = ttk.PanedWindow(self, width=375, 
			height=600, orient=tk.VERTICAL)
		self.output_pane.grid(column=1, row=0, padx=10, pady=10)

		# Change default global fonts
		default_font = font.nametofont("TkDefaultFont")
		default_font.configure(size=10)
		text_font = font.nametofont("TkTextFont")
		text_font.configure(size=10)

		# Controller for interfacing with core main.py functionality
		self.controller = Controller()

		# Create widgets
		self.display_widgets()
		self.create_menu()

	def create_menu(self):
		self.top = self.winfo_toplevel()
		self.menu_bar = tk.Menu(self.top, tearoff=0)
		self.top["menu"] = self.menu_bar

		self.file_menu = tk.Menu(self.menu_bar, tearoff=0)
		self.menu_bar.add_cascade(label="File", menu=self.file_menu)
		file_menu_button = ttk.Menubutton(self.file_menu, takefocus=True)
		self.file_menu.add_command(label="Batch Process Words", 
								   command=self.batch_process_words)
		self.file_menu.add_command(label="Save Output", 
								   command=self.output_view.save)
		self.file_menu.add_command(label="Quit", command=self.quit)

	def display_widgets(self):
		self.create_word_input()
		self.display_choices()
		self.display_selections()
		self.add_main_buttons()
		self.display_output()

	def create_word_input(self):
		self.word_input_frame = ttk.Frame(self, height=70, width=265, padding=5)
		self.word_input_frame.grid_propagate(0)
		self.input_pane.add(self.word_input_frame)

		validate_word_function = self.register(self.validate_word)
		invalid_response_function = self.register(self.invalid_word_response)
		self.word_input = ttk.Entry(self.word_input_frame, justify=tk.LEFT, 
			exportselection=0, width=30, validate="focusout", 
			validatecommand=(validate_word_function, "%P"), 
			invalidcommand=invalid_response_function)
		self.word_input.grid(row=1, column=1)
		self.word_input_label = ttk.Label(self.word_input_frame, text="Word: ")
		self.word_input_label.grid(row=1, column=0)
		self.word_error_message = ttk.Label(self.word_input_frame, 
			text="             ", foreground="#ff0000")
		self.word_error_message.grid(row=0, column=1)

	def display_choices(self):
		self.choices_display = ReductionOptionsView(self.input_pane, 
			self.controller, padding=5)
		self.input_pane.add(self.choices_display)

		def add_selections():
			self.selections_display.add_selections()

		self.choices_add_button_frame = ttk.Frame(self.input_pane)
		self.input_pane.add(self.choices_add_button_frame)
		self.choices_add_button = SizedButton(self.choices_add_button_frame, 
			height=28, width=55, text="Add", 
			command=add_selections)
		self.choices_add_button.pack()

	def display_selections(self):
		self.selections_display = ReductionSelectionsView(self.input_pane, 
			self.controller, self.choices_display, padding=5)
		self.input_pane.add(self.selections_display)

		self.selections_delete_button_frame = ttk.Frame(self.input_pane)
		self.input_pane.add(self.selections_delete_button_frame)
		self.selections_delete_button = SizedButton(
			self.selections_delete_button_frame, height=28, width=75, 
			text="Remove", command=self.selections_display.delete_selection)
		self.selections_delete_button.pack()

	def add_main_buttons(self):
		self.main_buttons_frame = ttk.Frame(self.input_pane, padding=15)
		self.input_pane.add(self.main_buttons_frame)

		self.define_pattern_button = SizedButton(self.main_buttons_frame, 
			height=28, width=120, text="Define Pattern", 
			command=self.define_pattern)
		self.define_pattern_button.pack(side=tk.LEFT)
		self.calculate_button = SizedButton(self.main_buttons_frame,
			height=28, width=80, text="Calculate", 
			command=self.validate_for_calculation)
		self.calculate_button.pack(side=tk.RIGHT)

	def display_output(self):
		self.output_view = OutputView(self.output_pane)
		self.output_pane.add(self.output_view)

	def validate_word(self, word_string):
		word = self.controller.create_word(word_string)
		words = self.controller.create_two_words(word_string)
		if word is None and words is None:
			return False
		elif word is None and len(words) != 2:
			return False
		elif word is not None:
			if not word.irreducible or word == "":
				self.word_error_message["text"] = "             "
			elif word.irreducible and not word.strongly_irreducible:
				self.word_error_message["text"] = "Irreducible! "
			else:
				self.word_error_message["text"] = "Strongly irreducible!"
			self.calculate_button.set_state("enable")
			return True
		else:
			self.word_error_message["text"] = "             "
			self.calculate_button.set_state("enable")
			return True

	def invalid_word_response(self):
		self.calculate_button.set_state("disable")
		self.word_error_message["text"] = "Invalid word!"

	def define_pattern(self):
		pattern_dialog = PatternDialog(self, self.controller, 
			self.choices_display, pattern=None)

	def batch_process_words(self):
		try:
			with askopenfile(title="Open Word List") as open_file:
				stripped_lines = [line.strip() for line 
								  in open_file.read().splitlines()]
				self.controller.create_words(stripped_lines)
		except AttributeError:
			pass

	def validate_for_calculation(self):
		self.past_stored_indices = self.controller.get_index_list()
		self.selected_indices, self.selected_patterns = \
			self.selections_display.get_selections()

		size_conflicted_patterns = self.controller.get_size_conflicts(
			self.selected_indices, self.selected_patterns)
		if (size_conflicted_patterns != [] 
		and self.controller.get_words() is None):
			add_examples_dialog = PatternDialog(self, self.controller,
				self.choices_display, 
				size_conflicted_patterns=size_conflicted_patterns, 
				word_size=self.controller.get_word().size)
			return
		elif (size_conflicted_patterns != [] 
		and self.controller.get_words() is not None):
			max_size = max(self.controller.get_words(), 
						   key=lambda word: word.size).size
			add_examples_dialog = PatternDialog(self, self.controller,
				self.choices_display, 
				size_conflicted_patterns=size_conflicted_patterns, 
				word_size=max_size)
			return
		
		self.check_for_patterns()

	def check_for_patterns(self):
		if self.selected_patterns != []:
			new_index_dialog = IndexDialog(self, self.controller, 
				self.choices_display, self.selected_patterns)
			return

		self.calculate_index()

	def calculate_index(self):
		current_stored_indices = self.controller.get_index_list()
		if self.past_stored_indices != current_stored_indices:
			created_index = set(current_stored_indices) \
							- set(self.past_stored_indices)
			self.selected_indices.append(created_index.pop())

		if "created_index" in locals() or self.selected_patterns == []:
			calculating_dialog = CalculatingDialog(self.controller, 
				self.output_view, indices=self.selected_indices)
		elif self.selected_indices == []:
			calculating_dialog = CalculatingDialog(self.controller,
				self.output_view, patterns=self.selected_patterns)
		else:
			calculating_dialog = CalculatingDialog(self.controller,
				self.output_view, indices=self.selected_indices, 
				patterns=self.selected_patterns)

	def quit(self):
		self.master.destroy()


class CalculatingThread(threading.Thread):

	def __init__(self, calculating_dialog, controller, output_view, 
				 *args, indices=None, patterns=None, **kwargs):
		super(CalculatingThread, self).__init__(*args, daemon=True, **kwargs)
		self.calculating_dialog = calculating_dialog
		self.controller = controller
		self.output_view = output_view
		self.indices = indices
		self.patterns = patterns

	def run(self):
		if self.controller.get_words() is not None:
			for word in self.controller.get_words():
				self.controller._word = word
				self.calculate()
			self.controller._words = None
			self.calculating_dialog.destroy()
			self.output_view.save()
		else:
			self.calculate()
			self.calculating_dialog.destroy()

	def calculate(self):
		pattern_index_value_list = None
		index_values = None
		if self.patterns is not None and self.patterns != []:
			pattern_index_value_list = self.controller.calculate_index(
				patterns=self.patterns)
			if self.controller.calc.stop == True:
				return
			self.output_view.output_result(self.controller.get_word(), 
				pattern_names=self.patterns, 
				pattern_index_value=pattern_index_value_list)
		if self.indices is not None and self.indices != []:
			index_values = self.controller.calculate_index(
				indices=self.indices)
			if self.controller.calc.stop == True:
				return
			self.output_view.output_result(self.controller.get_word(), 
				index_names=self.indices, index_values=index_values)


	def stop(self):
		self.controller.calc.stop = True
		self.calculating_dialog.destroy()


class CalculatingDialog(tk.Toplevel):

	def __init__(self, controller, output_view, patterns=None, indices=None):
		tk.Toplevel.__init__(self)
		self.title("")
		self._frame = ttk.Frame(self, padding=20)
		self._frame.pack(expand=True)
		self._controller = controller
		self.output_view = output_view
		self.patterns = patterns
		self.indices = indices
		self.calculate()

	def calculate(self):
		progress_frame = ttk.Frame(self._frame, 
			height=75, width=250, padding=10)
		progress_frame.pack_propagate(0)
		progress_frame.pack(side=tk.TOP, anchor=tk.N)
		label = ttk.Label(progress_frame, text="Calculating...")
		label.pack(side=tk.TOP)
		progress_bar = ttk.Progressbar(progress_frame, mode="indeterminate",
			orient=tk.HORIZONTAL, length=175)
		progress_bar.pack(side=tk.BOTTOM)

		button_frame = ttk.Frame(self._frame, height=80, width=200)
		button_frame.pack_propagate(0)
		button_frame.pack(side=tk.BOTTOM)
		cancel_button = SizedButton(button_frame, height=28, width=80,
			text="Cancel", command=self.stop_calculation)
		cancel_button.pack(side=tk.BOTTOM)

		self.calculate_thread = CalculatingThread(self, self._controller, 
			self.output_view, indices=self.indices, patterns=self.patterns)
		progress_bar.start(20)
		self.calculate_thread.start()

	def stop_calculation(self):
		self.calculate_thread.stop()


class Controller():

	def __init__(self):
		self._storage_handler = StorageHandler()

	def create_word(self, word_string):
		self._word = Word(word_string)
		return self._word

	def create_two_words(self, words_string):
		if words_string.count(", ") != 1:
			return None
		words = [Word(word_string) for word_string in words_string.split(", ")]
		self._words = [word for word in words if word is not None]
		return self._words

	def create_words(self, words_string):
		words = [Word(word_string) for word_string in words_string]
		self._words = [word for word in words if word is not None]
		return self._words

	def create_pattern_example(self, example_string):
		try:
			example = PatternExample(example_string)
			return example
		except ValueError:
			return None

	def define_pattern(self, name, pattern_examples=None, base=None, 
					   first_indices=None, induction_indices=None):
		if base is not None:
			first_indices = first_indices.split(",")
			first_index = int(first_indices[0].strip())
			second_index = int(first_indices[1].strip())

			induction_indices = induction_indices.split(",")
			first_induct_index = int(induction_indices[0].strip())
			second_induct_index = int(induction_indices[1].strip())
			inductive_step = ([first_index, second_index], 
							  [first_induct_index, second_induct_index])

			new_pattern = Pattern(name=name, base=base, 
								  inductive_step=inductive_step)
			self._storage_handler.store_pattern(new_pattern)
		else:
			new_pattern = Pattern(pattern_examples, name=name)
			self._storage_handler.store_pattern(new_pattern)

	def define_index(self, name, pattern_names):
		patterns = []
		for pattern_name in pattern_names:
			pattern = self._storage_handler.get_pattern(pattern_name)
			patterns.append(pattern)
		new_index = PatternIndex(name, patterns)
		self._storage_handler.store_index(new_index)

	def add_examples(self, pattern_examples, pattern):
		for example in pattern_examples:
			pattern_example = PatternExample(example)
			self._storage_handler.store_pattern_example(
				pattern_example, pattern)

	def get_word(self):
		return self._word

	def get_words(self):
		try:
			return self._words
		except (NameError, AttributeError):
			return None

	def get_pattern(self, pattern_name):
		return self._storage_handler.get_pattern(pattern_name)

	def get_index(self, index_name):
		return self._storage_handler.get_index(index_name)

	def get_pattern_list(self):
		return self._storage_handler.get_pattern_names()

	def get_index_list(self):
		return self._storage_handler.get_index_names()

	def get_size_conflicts(self, index_names, pattern_names):
		patterns = set(self.get_pattern(pattern) for pattern in pattern_names)
		index_patterns = set()
		for index_name in index_names:
			index = self._storage_handler.get_index(index_name)
			index_patterns.update(index.patterns)
		patterns |= index_patterns
		if hasattr(self, "_words") and self._words is not None:
			max_size = max(self._words, key=lambda word: word.size).size
			size_conflicted_patterns = [pattern for pattern in patterns
										if len(pattern) < max_size]
		else:
			size_conflicted_patterns = [pattern for pattern in patterns 
										if len(pattern) < self._word.size]
		return size_conflicted_patterns

	def calculate_index(self, patterns=None, indices=None):
		self.calc = Calculator()
		if patterns is not None and indices is not None:
			raise ValueError("Too many arguments!")
		elif patterns is not None:
			patterns = [self._storage_handler.get_pattern(pattern) 
						for pattern in patterns]
			index_value = self._storage_handler.get_word_index(self._word, 
				patterns=patterns)
			if index_value is None:
				index_value = self.calc.calculate_pattern_index(
					self._word, patterns)
				if self.calc.stop == True:
					return
				try:
					self._storage_handler.store_word_index(self._word, 
						index_value, patterns=patterns)
				except OSError:
					pass
			return [index_value]
		elif indices is not None:
			indices = [self._storage_handler.get_index(index) 
				   	   for index in indices]
			index_values = []
			for index in indices:
				index_value = self._storage_handler.get_word_index(self._word,
					index=index)
				if index_value is None:
					index_value = self.calc.calculate_pattern_index(self._word, 
						index.patterns)
					if self.calc.stop == True:
						return
					try:
						self._storage_handler.store_word_index(self._word, 
							index_value, index=index)
					except OSError:
						pass
				index_values.append(index_value)
			return index_values


class ReductionOptionsView(ttk.Frame):

	def __init__(self, master, controller, **kwargs):
		self._master = master
		self._controller = controller
		ttk.Frame.__init__(self, master, **kwargs)
		self.pack(fill=tk.X)

		self._view = ttk.Treeview(self, selectmode="extended", show="tree")
		self._view.grid(row=0, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
		self._yscrollbar = ttk.Scrollbar(self, orient="vertical", 
								   command=self._view.yview)
		self._yscrollbar.grid(row=0, column=1, sticky=tk.N+tk.S+tk.E)
		self._view.configure(yscrollcommand=self._yscrollbar.set)

		self.display_choices()

	def display_choices(self):
		self.index_names = self._controller.get_index_list()
		self.pattern_names = self._controller.get_pattern_list()

		self._view.insert("", 0, iid="#Indices", 
						  open=True, text="Pattern Indices")
		for index in self.index_names:
			self._view.insert("#Indices", "end", iid=index, 
							  open=True, text=index)

		self._view.insert("", 0, iid="#Patterns", open=True, text="Patterns")
		for pattern in self.pattern_names:
			self._view.insert("#Patterns", "end", iid=pattern, 
							  open=True, text=pattern)

	def update_choices(self, index_name=None, pattern_name=None):
		if index_name is not None:
			self._view.insert("#Indices", "end", iid=index_name, 
						  	  open=True, text=index_name)
		if pattern_name is not None:
			self._view.insert("#Patterns", "end", iid=pattern_name, 
						  	  open=True, text=pattern_name)

		self.index_names = self._controller.get_index_list()
		self.pattern_names = self._controller.get_pattern_list()

	def get_selection(self):
		index_names = set(self.index_names)
		pattern_names = set(self.pattern_names)
		selections = self._view.selection()
		contains_pattern = False
		contains_index = False
		for selection in selections:
			if selection in index_names:
				contains_index = True
			elif selection in pattern_names:
				contains_pattern = True
			else:
				return []

		return selections

	def deselect(self):
		self._view.selection_remove(self._view.selection())


class ReductionSelectionsView(ttk.Frame):

	def __init__(self, master, controller, choices_display, **kwargs):
		self._master = master
		self._controller = controller
		self._choices_display = choices_display
		ttk.Frame.__init__(self, master, **kwargs)
		self.pack(fill=tk.X)

		self._view = ttk.Treeview(self, selectmode="extended", show="tree")
		self._view.grid(row=0, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
		self._yscrollbar = ttk.Scrollbar(self, orient="vertical", 
								   command=self._view.yview)
		self._yscrollbar.grid(row=0, column=1, sticky=tk.N+tk.S+tk.E)
		self._view.configure(yscrollcommand=self._yscrollbar.set)

	def delete_selection(self):
		selections = self._view.selection()
		for selection in selections:
			self._view.delete(selection)

	def add_selections(self):
		selections = self._choices_display.get_selection()
		if selections == []:
			return
		if selections != []:
			self._choices_display.deselect()
		for selection in selections:
			self._view.insert("", "end", iid=selection, 
				open=True, text=selection)

	def get_selections(self):
		index_names = set(self._controller.get_index_list())
		pattern_names = set(self._controller.get_pattern_list())
		selected_indices = []
		selected_patterns = []
		for selection in self._view.get_children():
			if selection in index_names:
				selected_indices.append(selection)
			elif selection in pattern_names:
				selected_patterns.append(selection)
		return selected_indices, selected_patterns


class SizedButton(ttk.Frame):

	def __init__(self, master, height=None, width=None, 
				 text="", command=None, style=None):
		self._master = master
		ttk.Frame.__init__(self, master, height=height, width=width)
		#self.pack()
		self.pack_propagate(0)

		self._button = ttk.Button(self, text=text, command=command, style=style)
		self._button.pack(expand=True, fill=tk.BOTH)

	def button_configure(self, *args, **kwargs):
		self._button.configure(*args, **kwargs)

	def set_state(self, string):
		if string == "disable":
			self._button.state(["disabled"])
		elif string == "enable":
			self._button.state(["!disabled"])
		else:
			raise ValueError("Invalid argument!")


class OutputView(ttk.LabelFrame):

	def __init__(self, master, text="Output", labelanchor="n", padding=10):
		self.master = master
		ttk.LabelFrame.__init__(self, master, text=text, style="TLabelframe",
								labelanchor=labelanchor, padding=10)
		style = ttk.Style()
		style.configure("TLabelframe.Label", foreground="black")
		self.output = ttk.Label(self, text="", foreground="blue",
								justify=tk.LEFT, anchor=tk.W)
		self.output.pack(side=tk.TOP, anchor=tk.W)

	def output_result(self, word, index_names=None, pattern_names=None, 
					  index_values=None, pattern_index_value=None):
		if self.output["text"] != "":
			self.output["text"] += "\n\n"
		self.output["text"] += word

		if index_names is not None:
			for index, index_value in zip(index_names, index_values):
				self.output["text"] += "\n" + index + ": " + str(index_value)
		if pattern_names is not None:
			for i, pattern in enumerate(pattern_names):
				if len(self.output["text"].splitlines()[-1] + pattern) >= 55:
					self.output["text"] += "\n"
				if len(pattern_names) == 1:
					self.output["text"] += "\nPattern Index with " \
										   + pattern + ": "
					break 
				if i == 0:
					self.output["text"] += "\nPattern Index with " \
										   + pattern + ", "
				elif len(pattern_names) - 1 == i:
					self.output["text"] += pattern + ": "
				else:
					self.output["text"] += pattern + ", "
			self.output["text"] += str(pattern_index_value[0])

	def save(self):
		try:
			with asksaveasfile(title="Save Output", 
			defaultextension=".txt") as output:
				output.write(self.output["text"])
		except AttributeError:
			pass


class PatternDialog(tk.Toplevel):

	def __init__(self, master, controller, options_display, 
				 pattern=None, size_conflicted_patterns=None, word_size=None):
		tk.Toplevel.__init__(self)
		self.transient()
		self.master = master
		self._frame = ttk.Frame(self, padding=10)
		self._frame.pack(expand=True)
		self._controller = controller
		self._choices_display = options_display
		self.pattern = pattern
		self.conflicted_patterns = size_conflicted_patterns
		self.word_size = word_size
		if self.conflicted_patterns is not None:
			self.title("Possible Size Conflict: " 
					   + self.conflicted_patterns[-1].name)
			self.ask_to_add()
		elif self.pattern is None:
			self.title("Define a Pattern")
			self.make_new_pattern()

	def make_new_pattern(self):
		self.name_input_frame = ttk.Frame(self._frame, height=35, 
										  width=250, padding=10)
		self.name_input_frame.grid_propagate(0)
		self.name_input_frame.grid(row=0, column=0, columnspan=2)

		self.name_input = ttk.Entry(self.name_input_frame, justify=tk.LEFT, 
			exportselection=0, width=25)
		self.name_input.grid(row=1, column=1, sticky=tk.SW)
		self.name_input_label = ttk.Label(self.name_input_frame, text="Name: ")
		self.name_input_label.grid(row=1, column=0, sticky=tk.SE)

		self.inputs_frame = ttk.Frame(self._frame)
		self.inputs_frame.grid(row=2, column=0, columnspan=2)

		self.base_input = ttk.Entry(self.inputs_frame, justify=tk.LEFT, 
			exportselection=0, width=10)
		self.base_input.grid(row=1, column=1, sticky=tk.W)
		base_input_label = ttk.Label(self.inputs_frame, text="Base: ")
		base_input_label.grid(row=1, column=0, sticky=tk.E)
		self.error_message = ttk.Label(self.inputs_frame, 
			text="                           ", foreground="#ff0000")
		self.error_message.grid(row=0, column=0, columnspan=2, pady=10)

		self.first_indices_input = ttk.Entry(self.inputs_frame, 
			justify=tk.LEFT, exportselection=0, width=10)
		self.first_indices_input.grid(row=3, column=1, sticky=tk.W)
		first_indices_input_label = ttk.Label(self.inputs_frame, 
			text="First Insertion Indices: ")
		first_indices_input_label.grid(row=3, column=0, sticky=tk.E)
		unused_first_error_message = ttk.Label(self.inputs_frame, 
			text="             ", foreground="#ff0000")
		unused_first_error_message.grid(row=2, column=1)

		self.inductive_input = ttk.Entry(self.inputs_frame, 
			justify=tk.LEFT, exportselection=0, width=10)
		self.inductive_input.grid(row=5, column=1, sticky=tk.W)
		self.inductive_input_label = ttk.Label(self.inputs_frame, 
			text="Inductive Indices: ")
		self.inductive_input_label.grid(row=5, column=0, sticky=tk.E)
		unused_inductive_error_message = ttk.Label(self.inputs_frame, 
			text="             ", foreground="#ff0000")
		unused_inductive_error_message.grid(row=4, column=1)

		self.buttons_frame = ttk.Frame(self._frame, padding=20)
		self.buttons_frame.grid(row=6, column=0, columnspan=2)

		self.cancel_button = SizedButton(self.buttons_frame, height=28, width=80,
			text="Cancel", command=self.destroy)
		self.cancel_button.grid(row=0, column=0, padx=20, pady=10)
		self.submit_button = SizedButton(self.buttons_frame, height=28, width=60,
			text="OK", command=self.submit_via_induction)
		self.submit_button.grid(row=0, column=1, padx=20, pady=15)
		self.example_button =  SizedButton(self.buttons_frame, height=28, 
			width=146, text="Define via Examples", command=self.make_via_example)
		self.example_button.grid(row=1, column=0, columnspan=2, padx=10, pady=10)

	def submit_via_induction(self):
		pattern_name = self.name_input.get()
		if pattern_name.strip() == "":
			return
		base = self.base_input.get().strip()
		first_insertion_indices = self.first_indices_input.get().strip()
		induction_indices = self.inductive_input.get().strip()
		try:
			self._controller.define_pattern(pattern_name, base=base, 
				first_indices=first_insertion_indices, 
				induction_indices=induction_indices)
		except (ValueError, IndexError, TypeError):
			self.error_message["text"] = "Invalid input! Try again..."
			return
		self._choices_display.update_choices(pattern_name=pattern_name)
		self.destroy()

	def make_via_example(self):
		self.name_input_frame.destroy()
		self.inputs_frame.destroy()
		self.buttons_frame.destroy()

		# Name the pattern
		self.name_input_frame = ttk.Frame(self._frame,	
			height=45, width=280, padding=15)
		self.name_input_frame.grid_propagate(0)
		self.name_input_frame.pack(side=tk.TOP)

		self.name_input = ttk.Entry(self.name_input_frame, justify=tk.LEFT, 
			exportselection=0, width=30)
		self.name_input.grid(row=0, column=1)
		self.name_input_label = ttk.Label(self.name_input_frame, text="Name: ")
		self.name_input_label.grid(row=0, column=0)

		self.example_inputs = []

		def add_example_input():
			input_frame = ttk.Frame(self._frame, 
				height=50, width=410, padding=5)
			input_frame.grid_propagate(0)
			input_frame.pack(side=tk.TOP)

			def validate_pattern_example(example_string):
				example = self._controller.create_pattern_example(example_string)
				if example is None:
					return False
				else:
					error_message["text"] = "                "
					self.submit_button.set_state("enable")
					return True

			def invalid_example_response():
				error_message["text"] = "Invalid example!"
				self.submit_button.set_state("disable")

			validate_example_function = self.register(
				validate_pattern_example)
			invalid_response_function = self.register(
				invalid_example_response)

			example_input = ttk.Entry(input_frame, justify=tk.LEFT, 
				exportselection=0, width=40, validate="focusout", 
				validatecommand=(validate_example_function, "%P"), 
				invalidcommand=invalid_response_function)
			example_input.grid(row=1, column=1)
			word_input_label = ttk.Label(input_frame, 
				text="Example of size "+str(len(self.example_inputs)+1)+": ")
			word_input_label.grid(row=1, column=0, sticky=tk.W)
			error_message = ttk.Label(input_frame, 
				text="             ", foreground="#ff0000")
			error_message.grid(row=0, column=1)

			self.example_inputs.append(example_input)

		add_example_input()

		buttons_frame = ttk.Frame(self._frame, height=110, width=200)
		buttons_frame.pack_propagate(0)
		buttons_frame.pack(side=tk.BOTTOM, anchor=tk.S)
		self.add_button = SizedButton(buttons_frame, height=28, width=45,
			text="+", command=add_example_input)
		self.add_button.pack(side=tk.TOP, pady=10)
		self.cancel_button = SizedButton(buttons_frame, height=28, width=80,
			text="Cancel", command=self.destroy)
		self.cancel_button.pack(side=tk.LEFT, pady=10, anchor=tk.S)
		self.submit_button = SizedButton(buttons_frame, height=28, width=60,
			text="OK", command=self.submit_new_pattern)
		self.submit_button.pack(side=tk.RIGHT, pady=10, anchor=tk.S)

	def submit_new_pattern(self):
		pattern_name = self.name_input.get()
		if pattern_name.strip() == "":
			return
		pattern_examples = []
		for example_input in self.example_inputs:
			if example_input.get() == "":
				return
			pattern_examples.append(example_input.get())
		self._controller.define_pattern(pattern_name, 
			pattern_examples=pattern_examples)
		self._choices_display.update_choices(pattern_name=pattern_name)
		self.destroy()

	def ask_to_add(self):
		self.pattern = self.conflicted_patterns.pop()

		message_text = "Your word has size " + str(self.word_size) \
					   + ", yet this pattern is only defined up to size " \
					   + str(len(self.pattern)) + "." \
					   + "\n\n\t\t       Define more sizes?"
		self.message = ttk.Label(self._frame, text=message_text)
		self.message.pack(side=tk.TOP)

		self.buttons_frame = ttk.Frame(self._frame, height=65, width=200, padding=10)
		self.buttons_frame.pack_propagate(0)
		self.buttons_frame.pack(side=tk.BOTTOM, anchor=tk.S)
		submit_button = SizedButton(self.buttons_frame, height=28, width=60,
			text="Yes", command=self.add_to_pattern)
		submit_button.pack(side=tk.RIGHT, anchor=tk.S, padx=15)
		cancel_button = SizedButton(self.buttons_frame, height=28, width=60,
			text="No", command=self.ask_to_add_again)
		if self.conflicted_patterns == []:
			cancel_button.button_configure(command=self.close)
		cancel_button.pack(side=tk.LEFT, anchor=tk.S, padx=15)

	def ask_to_add_again(self):
		self.message.destroy()
		self.buttons_frame.destroy()
		self.ask_to_add()
		self.title("Possible Size Conflict: " 
				   + self.pattern.name)

	def add_to_pattern(self):
		self.message.destroy()
		self.buttons_frame.destroy()

		self.title("Extend Definition: " + self.pattern.name)
		self.example_inputs = []

		def add_example_input():
			input_frame = ttk.Frame(self._frame, 
				height=50, width=410, padding=5)
			input_frame.grid_propagate(0)
			input_frame.pack(side=tk.TOP)

			def validate_pattern_example(example_string):
				example = self._controller.create_pattern_example(example_string)
				if example is None:
					return False
				else:
					error_message["text"] = "                "
					self.submit_button.set_state("enable")
					return True

			def invalid_example_response():
				error_message["text"] = "Invalid example!"
				self.submit_button.set_state("disable")

			validate_example_function = self.register(
				validate_pattern_example)
			invalid_response_function = self.register(
				invalid_example_response)

			example_input = ttk.Entry(input_frame, justify=tk.LEFT, 
				exportselection=0, width=40, validate="focusout", 
				validatecommand=(validate_example_function, "%P"), 
				invalidcommand=invalid_response_function)
			example_input.grid(row=1, column=1)
			word_input_label = ttk.Label(input_frame, 
				text="Example of size "+str(len(self.pattern)\
					+len(self.example_inputs)+1)+": ")
			word_input_label.grid(row=1, column=0, sticky=tk.W)
			error_message = ttk.Label(input_frame, 
				text="             ", foreground="#ff0000")
			error_message.grid(row=0, column=1)

			self.example_inputs.append(example_input)

		add_example_input()

		buttons_frame = ttk.Frame(self._frame, height=90, width=200)
		buttons_frame.pack_propagate(0)
		buttons_frame.pack(side=tk.BOTTOM)
		self.add_button = SizedButton(buttons_frame, height=28, width=45,
			text="+", command=add_example_input)
		self.add_button.pack(side=tk.TOP, pady=10)
		self.cancel_button = SizedButton(buttons_frame, height=28, width=80,
			text="Cancel", command=self.close)
		self.cancel_button.pack(side=tk.LEFT, anchor=tk.S)
		self.submit_button = SizedButton(buttons_frame, height=28, width=60,
			text="OK", command=self.submit_pattern_examples)
		self.submit_button.pack(side=tk.RIGHT, anchor=tk.S)

	def submit_pattern_examples(self):
		pattern_examples = []
		for example_input in self.example_inputs:
			if example_input.get() == "":
				return
			pattern_examples.append(example_input.get())

		self._controller.add_examples(pattern_examples, self.pattern)
		if self.conflicted_patterns == []:
			self.close()
		else:
			self.ask_to_add_again()

	def close(self):
		self.master.check_for_patterns()
		self.destroy()


class IndexDialog(tk.Toplevel):

	def __init__(self, master, controller, choices_display, patterns):
		tk.Toplevel.__init__(self)
		self.master = master
		self._frame = ttk.Frame(self, padding=20)
		self._frame.pack(expand=True)
		self._controller = controller
		self._choices_display = choices_display
		self.patterns = patterns
		self.ask_to_add()

	def ask_to_add(self):
		message_text = "Would you like to define a new " \
					   + "index from the selected patterns?"
		self.message = ttk.Label(self._frame, text=message_text)
		self.message.pack(side=tk.TOP)

		self.buttons_frame = ttk.Frame(self._frame, height=60, width=180)
		self.buttons_frame.pack_propagate(0)
		self.buttons_frame.pack(side=tk.BOTTOM, anchor=tk.S)
		self.submit_button = SizedButton(self.buttons_frame, height=28, width=60,
			text="Yes", command=self.make_new_index)
		self.submit_button.pack(side=tk.RIGHT, anchor=tk.S, pady=10)
		self.cancel_button = SizedButton(self.buttons_frame, height=28, width=60,
			text="No", command=self.close)
		self.cancel_button.pack(side=tk.LEFT, anchor=tk.S, pady=10)

	def make_new_index(self):
		# Remove ask label and buttons
		self.message.pack_forget()
		self.buttons_frame.pack_forget()

		# Name the pattern
		self.name_input_frame = ttk.Frame(self._frame,	
			height=45, width=320, padding=5)
		self.name_input_frame.grid_propagate(0)
		self.name_input_frame.pack(side=tk.TOP)

		self.name_input = ttk.Entry(self.name_input_frame, justify=tk.LEFT, 
			exportselection=0, width=30)
		self.name_input.grid(row=0, column=1)
		self.name_input_label = ttk.Label(self.name_input_frame, 
										  text="Index Name: ")
		self.name_input_label.grid(row=0, column=0)

		buttons_frame = ttk.Frame(self._frame, height=60, width=180)
		buttons_frame.pack_propagate(0)
		buttons_frame.pack(side=tk.BOTTOM, anchor=tk.S)
		self.cancel_button = SizedButton(buttons_frame, height=28, width=80,
			text="Cancel", command=self.close)
		self.cancel_button.pack(side=tk.LEFT, anchor=tk.S)
		self.submit_button = SizedButton(buttons_frame, height=28, width=60,
			text="OK", command=self.submit_new_index)
		self.submit_button.pack(side=tk.RIGHT, anchor=tk.S)

	def submit_new_index(self):
		index_name = self.name_input.get().strip()
		if index_name == "":
			return
		self._controller.define_index(index_name, self.patterns)
		self._choices_display.update_choices(index_name=index_name)
		self.close()

	def close(self):
		self.master.calculate_index()
		self.destroy()


if __name__ == '__main__':
	root = tk.Tk()
	root.title("Pattern Index Calculator")
	root.iconbitmap(default="blank.ico")
	pattern_index_app = PatternIndexApp(master=root)
	pattern_index_app.mainloop()


# POSSIBLE EXTENSIONS:

# Implement use of inductive definition to extend pattern examples into 
# more sizes when necessary. (1)

# Ability to delete patterns and indices, and check for repetition
# of names (at least) when saving. Convert data storage to SQLite 
# database. (2)

# Back buttons on all dialogs (maybe save state too?). (5)

# Help + documentation/explanations in-built (both menu options
# and maybe clickable pop-ups). (4)

# More extensive validation and error handling (what if they 
# input and save a pattern incorrectly defined? Deleting options
# already mentioned probably suffice). (3)

# Make graphical layout look better. (6)

# Make a button which when clicked inputs a random double occurrence 
# word of some size (possibly user specified). (7)