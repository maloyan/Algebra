import numpy as np
"""
Малоян Нарек 327
Пишу на питоне впервые. Я слышал, что это удобный язык, но чтобы настолько.
Cпасибо, что открыли глаза.
"""

def gen_pow_matrix(primpoly):
	# q - степень многочлена. Чтобы узнать, надо определить количество
	# элементов в двоичном представлении числа primpoly
	q = np.floor(np.log2(primpoly)).astype(int)

	# res - матрица соответствия, размера (2^q - 1)x2
	res = np.zeros((2 ** q - 1, 2), dtype=np.int)

	# alpha - примитивный элемент
	alpha = 2

	for i in range(1, res.shape[0] + 1):
		# соответствие десятичного числа со степенью элемента
		res[alpha - 1, 0] = i
		# соответвие степени элемента с десятичным числом
		res[i - 1, 1] = alpha
		# берем следующее число
		alpha = alpha * 2
		# проверяем, что мы не выходим за рамки q 
		if alpha > res.shape[0]:
			alpha = alpha ^ primpoly
	return res

# поэлементный xor
def add(X, Y):
	return X^Y

# xor всех векторов по оси x или y
def sum(X, axis=0):
	# матрица результата, заполненная предварительно нулями
	res = np.zeros((X.shape[abs(axis-1)]), dtype=np.int)
	# пробегаем по всем векторам
	for i in range(X.shape[axis]):
		# складываем вектора с помощью уже написанной функции add()
		if axis == 0:
			res = add(res, X[i, :])
		if axis == 1:
			res = add(res, X[:, i])
	return res

def prod(X, Y, pm):
	# по десятичным числам из матрицы X опеределяем степень многочлена
	# с помощью матрицы соответствий десятичных чисел и степеней pm
	k1_arr = pm[X - 1, 0]
	# тоже самое для Y
	k2_arr = pm[Y - 1, 0]
	# произведение двух элементов поля a^k1 и a^k2 равно
	# a^((k1 + k2) mod (2^q - 1))
	# в res_pow записываем (k1 + k2) mod (2^q - 1)
	res_pow = (k1_arr + k2_arr) % (pm.shape[0])
	# по матрице pm переводим степень многочлена в десятичное число
	res = pm[res_pow - 1, 1]
	# для нулевых значений результат также нулевой
	res[X == 0] = 0
	res[Y == 0] = 0
	return res

def divide(X, Y, pm):
	# по десятичным числам из матрицы X опеределяем степень многочлена
	# с помощью матрицы соответствий десятичных чисел и степеней pm
	k1_arr = pm[X - 1, 0]
	# тоже самое для Y
	k2_arr = pm[Y - 1, 0]
	# частное двух элементов поля a^k1 и a^k2 равно
	# a^((k1 - k2) mod (2^q - 1))
	# в res_pow записываем (k1 - k2) mod (2^q - 1)
	res_pow = (k1_arr - k2_arr) % (pm.shape[0])
	# по матрице pm переводим степень многочлена в десятичное число
	res = pm[res_pow - 1, 1]
	# для нулевых делимых результат также нулевой
	res[X == 0] = 0
	return res

def linsolve(A, b, pm):
    A = np.copy(A)
    b = np.copy(b)
    # Gauss-like method
    # Forward:
    for j in range(A.shape[0]):
        nz_col_idx = np.nonzero(A[j:, j])[0] + j
        if nz_col_idx.size == 0:
            return np.nan
        # Division:
        b[nz_col_idx] = divide(b[nz_col_idx], A[nz_col_idx, j], pm)
        A[nz_col_idx, :] = divide(A[nz_col_idx, :], 
                                  np.tile(A[nz_col_idx, j].reshape(-1, 1), A.shape[1]), pm)
        # Subtracting:
        for i in nz_col_idx[1:]:
            A[i, :] = add(A[i, :], A[nz_col_idx[0], :])
            b[i] = add(b[i], b[nz_col_idx[0]])
        # Swapping:
        if nz_col_idx[0] > j:
            tmp_row = A[j, :].copy()
            A[j, :] = A[nz_col_idx[0], :].copy()
            A[nz_col_idx[0], :] = tmp_row.copy()
            tmp = b[j].copy()
            b[j] = b[nz_col_idx[0]].copy()
            b[nz_col_idx[0]] = tmp.copy()
    # Backward:
    x = np.zeros(b.size, dtype=int)
    for i in range(b.size - 1, -1, -1):
        x[i] = add(b[i], sum(prod(x[(i + 1):], A[i, (i + 1):], pm).reshape(-1, 1))).copy()
    return x.astype(int)



def main():
	X = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
	b = np.array([1, 1, 1])
	print(linsolve(X, b, gen_pow_matrix(11)))

if __name__ == "__main__":
	main()