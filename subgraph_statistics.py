"""
Contains functions to assemble word subgraph statistics 
from text files containing computed subgraphs. 
"""

import re

BASE = "aoword_graph_size"
EXT = ".txt"
PARALLEL_STRUCTURE = "((0, 1), (1, 2), (3, 2), (0, 3))"


def get_integer(string):
    integers = re.findall(r"\d+", string)
    if not integers:
        raise NotImplementedError
    else:
        return int(integers[-1])


def get_subgraph_statistics(sizes):
    statistics = {}
    for size in sizes:
        statistics[size] = {}

        # Get word graph stats
        with open(BASE + str(size) + EXT, "r") as word_graph_file:
            for line in word_graph_file:
                if line.startswith("Vertex count"):
                    vertex_count = get_integer(line)
                elif line.startswith("Edge count"):
                    edge_count = get_integer(line)
            statistics[size]["vertices"] = vertex_count
            statistics[size]["edges"] = edge_count

        # Get 3-path stats
        try:
            with open(BASE + str(size) + "_3paths" + EXT) as length3_paths_file:
                for line in length3_paths_file:
                    if line.startswith("3-path subgraph count"):
                        statistics[size]["3-paths"] = get_integer(line)
        except FileNotFoundError:
            print("File", BASE + str(size) + "_3paths" + EXT, "not found")

        # Get 4-path stats
        try:
            with open(BASE + str(size) + "_4paths" + EXT) as length4_paths_file:
                for line in length4_paths_file:
                    if line.startswith("4-path subgraph count"):
                        statistics[size]["4-paths"] = get_integer(line)
        except FileNotFoundError:
            print("File", BASE + str(size) + "_4paths" + EXT, "not found")

        # Get triangle stats
        try:
            with open(BASE + str(size) + "_triangles" + EXT) as triangles_file:
                for line in triangles_file:
                    if line.startswith("Triangle subgraph count"):
                        statistics[size]["triangles"] = get_integer(line)
        except FileNotFoundError:
            print("File", BASE + str(size) + "_triangles" + EXT, "not found")

        # Get square stats
        statistics[size]["squares"] = {}
        try:
            with open(BASE + str(size) + "_squares_sorted" + EXT, "r") as sorted_squares_file:
                for line in sorted_squares_file:
                    if line.startswith("Square subgraph count"):
                        statistics[size]["squares"]["total"] = get_integer(line)
                    elif line.startswith("((") and "count" in line:
                        statistics[size]["squares"][line[:line.find("))")+2]] = (
                            get_integer(line))
        except FileNotFoundError:
            print("File", BASE + str(size) + "_squares_sorted" + EXT, "not found")
            try:
                with open(BASE + str(size) + "_squares_parallel" + EXT) as parallel_squares_file:
                    for line in parallel_squares_file:
                        if line.startswith("Square subgraph count"):
                            statistics[size]["squares"][PARALLEL_STRUCTURE] = (
                                get_integer(line))
            except FileNotFoundError:
                print("File", BASE + str(size) + "_squares_parallel" + EXT, "not found")

        # Get cube stats
        statistics[size]["cubes"] = {}
        try:
            with open(BASE + str(size) + "_cubes" + EXT) as cubes_file:
                for line in cubes_file:
                    if line.startswith("Cube subgraph count"):
                        statistics[size]["cubes"]["total"] = get_integer(line)
                    elif line.startswith("Linearly Ordered cube subgraph count:"):
                        statistics[size]["cubes"]["linear"] = get_integer(line)
                    elif line.startswith("Not Linearly Ordered cube subgraph count:"):
                        statistics[size]["cubes"]["nonlinear"] = get_integer(line)
        except FileNotFoundError:
            print("File", BASE + str(size) + "_cubes" + EXT, "not found")
            try:
                with open(BASE + str(size) + "_cubes_parallel" + EXT) as cubes_file:
                    for line in cubes_file:
                        if line.startswith("Cube subgraph count"):
                            statistics[size]["cubes"]["total"] = get_integer(line)
                        elif line.startswith("Linearly Ordered cube subgraph count:"):
                            statistics[size]["cubes"]["linear"] = get_integer(line)
                        elif line.startswith("Not Linearly Ordered cube subgraph count:"):
                            statistics[size]["cubes"]["nonlinear"] = get_integer(line)
            except FileNotFoundError:
                print("File", BASE + str(size) + "_cubes_parallel" + EXT, "not found")

    return statistics


def save_subgraph_statistics(subgraph_statistics):
    with open("aoword_graph_subgraph_statistics.txt", "w") as statistics_file:
        print("\nAscending Order Double Occurrence Word Subgraph Statistics", 
              file=statistics_file)
        for i, size in enumerate(subgraph_statistics):
            print("\n\n---------------- Words up to size", size, 
                  "----------------", file=statistics_file)

            # Graph stats
            print("\nVertices:", subgraph_statistics[size]["vertices"], 
                  file=statistics_file)
            print("Edges:", subgraph_statistics[size]["edges"], 
                  file=statistics_file)

            # 3-path stats
            try:
                print("\n3-paths:", subgraph_statistics[size]["3-paths"], 
                      file=statistics_file)
            except KeyError:
                pass

            # 4-path stats
            try:
                print("\n4-paths:", subgraph_statistics[size]["4-paths"], 
                      file=statistics_file)
            except KeyError:
                pass

            # Triangle stats
            try:
                print("\nTriangles:", subgraph_statistics[size]["triangles"], 
                      file=statistics_file)
            except KeyError:
                pass

            # Square stats
            if subgraph_statistics[size]["squares"]:
                try:
                    print("\nSquares:", subgraph_statistics[size]["squares"]["total"], 
                          file=statistics_file)
                except KeyError:
                    print("\n", file=statistics_file)
                for directed_structure in subgraph_statistics[size]["squares"]:
                    count = subgraph_statistics[size]["squares"][directed_structure]
                    if directed_structure != "total":
                        print("Squares of type", directed_structure + ":", count, 
                              file=statistics_file)

            # Cube stats
            if subgraph_statistics[size]["cubes"]:
                print("\nCubes:", subgraph_statistics[size]["cubes"]["total"], 
                      file=statistics_file)
                print("Linear cubes:", subgraph_statistics[size]["cubes"]["linear"],
                      file=statistics_file)
                print("Nonlinear cubes:", subgraph_statistics[size]["cubes"]["nonlinear"],
                      file=statistics_file)


if __name__ == '__main__':
    print("\nThis program assembles subgraph statistics into a single file.\n")
    user_input = "none"
    sizes = []
    size = 0
    while user_input != "":
        if user_input != "none":
            sizes.append(int(user_input))
        user_input = input("Type a size (or hit 'enter' to assembly): ").strip()
    subgraph_statistics = get_subgraph_statistics(sizes)
    save_subgraph_statistics(subgraph_statistics)
    print(subgraph_statistics)