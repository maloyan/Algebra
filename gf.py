import numpy as np

# q - степень многочлена
#
# res - матрицу соответствия между десятичным представлением и
# степенным представлением ненулевых элементов поля по стандартному
# примитивному элементу α
def gen_pow_matrix(primpoly):
	q = np.floor(np.log2(primpoly)).astype(int)

	res = np.zeros((2 ** q - 1, 2), dtype=np.int)
	alpha = 2
	for i in range(1, res.shape[0] + 1):
		res[alpha - 1, 0] = i
		res[i - 1, 1] = alpha
		alpha <<=  1
		if alpha > res.shape[0]:
			alpha = alpha ^ primpoly
	return res

# поэлементный xor
def add(X, Y):
	return X^Y

# xor всех векторов по оси x или y
def sum(X, axis=0):
	if axis == 0:
		res = np.zeros((X.shape[1]), dtype=np.int)
		for i in range(X.shape[0]):
			res = add(res, X[i, :])
	if axis == 1:
		res = np.zeros((X.shape[0]), dtype=np.int)
		for j in range(X.shape[1]):
			res = add(res, X[:, j])
	return res

def prod(X, Y, pm):
	X = X.astype(int)
	Y = Y.astype(int)
	pow_X = pm[X - 1, 0]
	pow_Y = pm[Y - 1, 0]
	pow_hadamard_prod = (pow_X + pow_Y) % (pm.shape[0])
	hadamard_prod = pm[pow_hadamard_prod - 1, 1]
	hadamard_prod[np.logical_or(X == 0, Y == 0)] = 0
	return hadamard_prod.astype(int)

#def divide(X, Y, pm):

#def linsolve(A, b, pm):

def main():
	X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
	Y = np.array([[7, 8, 9], [4, 5, 6], [1, 2, 3]])
	print(prod(X, Y, gen_pow_matrix(11)))

if __name__ == "__main__":
	main()