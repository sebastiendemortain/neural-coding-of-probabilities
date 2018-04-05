import numpy as np

def get_dimension_list(matrix):
    dims = []
    while isinstance(matrix, list) and matrix is not None:
        dims.append(len(matrix))
        matrix = matrix[0]
    number_of_dimensions = len(dims)
    return number_of_dimensions