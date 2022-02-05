import random
import numpy as np

def add(mat1, mat2):
	result = arr_like(mat1)

	for i in range(len(mat1)):
		for j in range(len(mat1[0])):
			result[i][j] = mat1[i][j] + mat2[i][j]

	return result

def sub(mat1, mat2):
	return add(mat1, con(mat2, -1))

def mul(mat1, mat2):

	return np.dot(mat1, mat2).tolist()

def bias(mat, b):
	result = arr_like(mat)

	for i in range(len(mat)):
		for j in range(len(mat[0])):
			result[i][j] = mat[i][j] + b[0][j]

	return result

def con(mat, c):
	result = arr_like(mat)

	for i in range(len(mat)):
		for j in range(len(mat[0])):
			result[i][j] = mat[i][j]*c

	return result

def T(mat):
	return [[mat[j][i] for j in range(len(mat))] for i in range(len(mat[0]))]

def arr_like(mat):
	return [[0 for j in range(len(mat[0]))] for i in range(len(mat))]

if __name__ == '__main__':
	print(T([[1,2,3],[1,2,3]]))