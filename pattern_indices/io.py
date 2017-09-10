"""
Some auxiliary input/output functions for pattern_indices API.

Functions:

    process_multibatch_output_file, output_statistics_batchwise
"""

import os

from word_explorer.io import store_data, retrieve_data


def process_multibatch_output_file(output_file_name):
    # Expects file corresponding to output_file_name is in 
    # the folder pattern_indices/data/
    output_file_name = os.path.join(
        "pattern_indices", os.path.join("data", output_file_name))
    experiments = []
    experiment = {"words": {}}
    previous_word = "a"*100
    words_calculated = 0
    for line in retrieve_data(output_file_name):
        if re.fullmatch(r"[0-9a-z]+", line.strip()):
            word = line.strip()
            if len(previous_word) < len(word):
                experiment["size"] = words_calculated
                experiments.append(experiment)
                words_calculated = 0
                experiment = {"words": {}}
            previous_word = word
            experiment["words"][word] = {}
            words_calculated += 1
        if ":" in line:
            pattern_index, value = line.strip().split(":")
            value = int(value.strip())
            experiment[pattern_index] = experiment.get(pattern_index, 0) + value
            experiment["words"][word][pattern_index] = value
    if not experiments:
        experiment["size"] = words_calculated
        experiments.append(experiment)
    
    return experiments


def output_statistics_batchwise(statistics):
    """
    Args:
        statistics: A dictionary of the form
            {"per_batch": {...}, 
             <statistic_name1>: <float1>, 
             <statistic_name2>: <float2>, 
             ...}
    Prints: 
        The name and value for each statistic in statistics.
    """
    for statistic, values in statistics.items():
        if statistic != "per_batch":
            print("\n" + statistic + ":", values)