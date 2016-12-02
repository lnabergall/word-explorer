"""
Parses an output text file (for a batch calculation) of the pattern 
index calculator and returns the average of the value for all 
calculated indices.
"""


def main():
	output_file_name = input("input a file name (including the extension): ")
	with open(output_file_name, "r") as output_file:
		metric_list = get_metrics(output_file)
		print(metric_list)


def get_metrics(output_file):
	lines = output_file.readlines()
	lines_between_blanks = 0
	first_blank = 0
	for i, line in enumerate(lines):
		if first_blank != 0 and (line == "\n" or line == ""):
			second_blank = i
			lines_between_blanks = second_blank - first_blank - 1
			break
		if line == "\n" or line == "":
			first_blank = i

	metric_list = []
	for i in range(lines_between_blanks-1):
		value_sum = 0
		line_count = 0
		for line in lines[i+1::lines_between_blanks+1]:
			value_sum += int(line.strip()[-1])
			line_count += 1
		metric_list.append(value_sum/line_count)

	return metric_list

if __name__ == '__main__':
	main()