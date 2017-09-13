"""
Functions for computing the homology of simplicial complexes 
corresponding to word graphs.

Functions:

    recursion_handler, calculate_homology_groups
"""

import os
import sys
import threading

import numpy as np
from simplex import SimplicialComplex

from .io import retrieve_maximal_simplices, store_homology_data


def recursion_handler(function):
    def function_wrapper(*args, **kwargs):
        threading.stack_size(67108864) # 64MB stack
        sys.setrecursionlimit(2**22) # something very big, 64MB limit hit first

        # only new threads get the redefined stack size
        thread = threading.Thread(target=function, args=args, kwargs=kwargs)
        thread.start()

    return function_wrapper


@recursion_handler
def calculate_homology_groups(ascending_order, size, max_homology_group=3,
                              max_simplex_types=["triangle"],
                              name_base="word_graph_size", name_suffix="",
                              output_boundary_matrices=False):
    maximal_simplices = retrieve_maximal_simplices(
        ascending_order, size, name_base, name_suffix, max_simplex_types)
    maximal_1simplices_count = len([simplex for simplex in maximal_simplices
                                    if len(simplex) == 2])
    simplicial_complex = SimplicialComplex(maximal_simplices)

    homology_group_data = []
    for i in range(max_homology_group+1):
        homology_group_data.append([simplicial_complex.betti_number(i, 2)])
        if output_boundary_matrices:
            boundary_matrix = simplicial_complex.get_boundary_matrix(i, 2)
            reduced_matrix = simplicial_complex.reduce_matrix(boundary_matrix)[0]
            homology_group_data[-1].extend([boundary_matrix, reduced_matrix])
        homology_group_data[-1] = tuple(homology_group_data[-1])

    store_homology_data(ascending_order, size, homology_group_data, 
                        max_simplex_types, name_base, name_suffix)
