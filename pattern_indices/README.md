Python scripts, including a tkinter GUI, to calculate arbitrary pattern indices and word distances of double occurrence words.

To start the GUI, run interface.py in the command-line. Words are input via a text entry box or from a text file via the File menu for batch processing. 
The calculations are output in the Output box and can be saved by selecting "Save Output" and inputting a full text file name (including extension).
To calculate the distance between two words, enter the two words separated by a comma and a space in the text entry box (e.g. "121323, 1212"). 

**indices** - Provides the main class, Calculator, used for calculating pattern indices, as well as a basic text interface.

**storage** - Defines two classes, StorageHandler and SQLStorageHandler, for handling the storage of patterns, pattern indices, and the pattern index values of specific words. The former uses a simple text file based storage system, while the latter uses a SQLite database controlled via SQLAlchemy.

**interface** - A collection of Tkinter classes that collectively define a GUI that allows a user to calculate pattern indices of a word or batch of words and distances between two words.

**output_processing** - Contains functions for processing output from the GUI in pattern_indices.interface and computing and plotting various statistics.

**io** - Input/output utilities for the pattern_indices API.