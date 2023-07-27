import numpy as np
from itertools import product

def generate_binary_combinations(n):
    return list(product([0, 1], repeat=n))[1:-1]

n_bits = 4 

result_lists = generate_binary_combinations(n_bits)
list_matrices = []

for res in result_lists:
    matrix = np.zeros((n_bits,n_bits))
    matrix1 = np.zeros((n_bits,n_bits))
    for i in range(len(res)):
        if res[i] == 1:
            for j in range(len(res)):
                matrix[i][j] = 1
                matrix1[j][i] = 1
    list_matrices.append(matrix)
    list_matrices.append(matrix1)
    
print(len(list_matrices))
