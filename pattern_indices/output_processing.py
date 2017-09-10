"""
Provides functions for parsing an output text file (for a batch calculation) 
from the pattern index calculator implemented in interface.py and provides 
functions for calculating various statistics.

Functions:

	process_data, get_statistics, process_data_output, 
	compute_statistics_batchwise, compute_statistics_sizewise, 
	plot_statistics_batchwise, plot_statistics_sizewise
"""
import re

import numpy as np
import matplotlib
from matplotlib import pyplot as plotter

from .io import process_multibatch_output_file


font = {'family' : 'normal',
        'size'   : 28}
matplotlib.rc('font', **font)

PRI = "Pattern Recurrence Index"
TI = "Tangled Index"
TPRI = "Tangled Pattern Recurrence Index"


def process_data(data_file_name, random_file_name, batchwise=False, sizewise=False):
	"""
	Input the names of two files, random_file_name and data_file_name, 
	that contain the output from a batch calculation of the pattern
	indices of a sequence of randomly sampled words and a sequence
	of words from data. 

	It then computes various statistics of these batches depending on 
	the argument values. If batchwise = True, it first computes 
	the mean pattern index values for each index and each random sample, 
	and then calculates statistics of this population: mean, median, variance, 
	standard deviation, upper quartile, lower quartile, minimum, and maximum. 
	If sizewise = True, for each word size in the data it computes 
	these statistics for all the pattern indices of every word of that size. 

	Args:
		data_file_name: String.
		random_file_name: String.
		batchwise: Boolean, defaults to False.
		sizewise: Boolean, defaults to False.
	"""
	random_experiments = process_multibatch_output_file(random_file_name)
	data_output = process_multibatch_output_file(data_file_name)
	if batchwise:
		random_sample_statistics_batchwise = compute_statistics_batchwise(
			random_experiments)
		plot_statistics_batchwise(random_sample_statistics_batchwise, data_output)
	if sizewise:
		data_processed = process_data_output(data_output)
		random_sample_statistics_sizewise = compute_statistics_sizewise(
			random_experiments)
		plot_statistics_sizewise(random_sample_statistics_sizewise, data_processed)


def get_statistics(*sequences):
	"""
	Args:
		sequences: List or Tuple of lists of floats or integers.
	Returns:
		A dictionary with statistic names as keys and statistic values
		as values. 
	"""
	means = [np.mean(sequence) for sequence in sequences]
	standard_deviations = [np.std(sequence) for sequence in sequences]
	medians = [np.percentile(sequence, 50) for sequence in sequences]
	lower_quartiles = [np.percentile(sequence, 25) for sequence in sequences]
	upper_quartiles = [np.percentile(sequence, 75) for sequence in sequences]
	minimums = [np.percentile(sequence, 0) for sequence in sequences]
	maximums = [np.percentile(sequence, 100) for sequence in sequences]

	return {
		"mean": means, 
		"standard_deviation": standard_deviations, 
		"median": medians, 
		"lower_quartile": lower_quartiles, 
		"upper_quartile": upper_quartiles, 
		"minimum": minimums, 
		"maximum": maximums
	}


def process_data_output(data_output):
	words_by_size = {}
	for word in data_output[0]["words"]:
		words_by_size[len(word)//2] = words_by_size.get(len(word)//2, []) + [word]

	data_processed = {}
	for index in [PRI, TI, TPRI]:
		data_processed[index] = []
		for size, words in words_by_size.items():
			data_processed[index].append(
				(size, np.mean([data_output[0]["words"][word][index] 
								for word in words])))

	return data_processed


def compute_statistics_batchwise(experiments):
	statistics = {}
	statistics["per_batch"] = experiments
	pri_means = [experiment[PRI]/experiment["size"] 
				 for experiment in statistics["per_batch"]]
	ti_means = [experiment[TI]/experiment["size"] 
				for experiment in statistics["per_batch"]]
	tpri_means = [experiment[TPRI]/experiment["size"] 
				  for experiment in statistics["per_batch"]]
	stats_dictionary = get_statistics(pri_means, ti_means, tpri_means)
	index_map = [PRI, TI, TPRI]
	for stat in stats_dictionary:
		statistics[stat] = {index_map[i]: value for i, value 
							in enumerate(stats_dictionary[stat])}

	return statistics


def compute_statistics_sizewise(experiments):
	words_by_size = {}
	for experiment in experiments:
		for word in experiment["words"]:
			words_by_size[len(word)//2] = words_by_size.get(len(word)//2, []) + [word]

	statistics = {}
	statistics["per_batch"] = experiments
	statistics["per_size"] = {}
	for size, words in words_by_size.items():
		pri_means = [np.mean([experiment["words"][word][PRI] 
							  for word in words if word in experiment["words"]]) 
					 for experiment in experiments]
		ti_means = [np.mean([experiment["words"][word][TI] 
							 for word in words if word in experiment["words"]]) 
					for experiment in experiments]
		tpri_means = [np.mean([experiment["words"][word][TPRI] 
							   for word in words if word in experiment["words"]]) 
					  for experiment in experiments]
		stats_dictionary = get_statistics(pri_means, ti_means, tpri_means)
		index_map = [PRI, TI, TPRI]
		statistics["per_size"][size] = {}
		for stat in stats_dictionary:
			statistics["per_size"][size][stat] = {
				index_map[i]: mean for i, mean 
				in enumerate(stats_dictionary[stat])}

	return statistics


def plot_statistics_batchwise(statistics, processed_data):
	boxplot_sequences = {}
	pri_means = [experiment[PRI]/experiment["size"] 
				 for experiment in statistics["per_batch"]]
	ti_means = [experiment[TI]/experiment["size"] 
				for experiment in statistics["per_batch"]]
	tpri_means = [experiment[TPRI]/experiment["size"] 
				  for experiment in statistics["per_batch"]]
	indices = [PRI, TI, TPRI]
	labels = ["RR", "T", "RRT"]
	index_means = [pri_means, ti_means, tpri_means]
	plotter.figure()
	plotter.boxplot(index_means, showfliers=False, labels=labels)
	for i, index in enumerate(indices):
		data_index_value = processed_data[0][index]/processed_data[0]["size"]
		plotter.plot(i+1, data_index_value, "or")

	plotter.xlabel("Pattern Index", labelpad=18)
	plotter.show()


def plot_statistics_sizewise(statistics, processed_data):
	plot_tracks = {PRI: {}, TI: {}, TPRI: {}}
	for index in plot_tracks:
		plot_tracks[index]["mean"] = []
		plot_tracks[index]["error"] = []
		for size in statistics["per_size"]:
			plot_tracks[index]["mean"].append(
				(size, statistics["per_size"][size]["mean"][index]))
			plot_tracks[index]["error"].append(
				(size, statistics["per_size"][size]["standard_deviation"][index]))
	for index in plot_tracks:
		# Sort to ensure proper plotting
		plot_tracks[index]["mean"].sort()
		processed_data[index].sort()
		plotter.figure()
		plotter.errorbar(
			[size for size, _ in plot_tracks[index]["mean"]],
			[mean for _, mean in plot_tracks[index]["mean"]],
			yerr=[error for _, error in plot_tracks[index]["error"]],
			label="Random Samples"
		)
		plotter.plot(
			[size for size, _ in processed_data[index]], 
			[value for _, value in processed_data[index]],
			label="22 Highly Scrambled Cases"
		)
		plotter.xlabel("Word Size")
		plotter.ylabel("Average " + index)
		plotter.legend(loc="upper left")

	plotter.show()