"""
Some basic utility functions for input and output.

Functions:

    store_data, retrieve_data
"""

import os


def format_filename(file_name):
    if os.path.exists("output"):
        file_name = os.path.join("output", file_name)
    elif os.path.exists(os.path.join(os.pardir, "output")):
        file_name = os.path.join(os.path.join(os.pardir, "output"), file_name)
    return file_name


def store_data(data_list, file_name, append=False, add_output_dir=True):
    if add_output_dir:
        file_name = format_filename(file_name)
    mode = "w" if not append else "a"
    with open(file_name, mode) as output_file:
        for data in data_list:
            print(data, file=output_file)


def retrieve_data(file_name, add_output_dir=True):
    if add_output_dir:
        file_name = format_filename(file_name)
    with open(file_name, "r") as input_file:
        for line in input_file:
            yield line


