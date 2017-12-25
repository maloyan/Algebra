import numpy as np


def gen_pow_matrix(primpoly):
    # q - степень многочлена. Чтобы узнать, надо определить количество
    # элементов в двоичном представлении числа primpoly
    q = np.floor(np.log2(primpoly)).astype(int)

    # res - матрица соответствия, размера (2^q - 1)x2
    res = np.zeros((2 ** q - 1, 2), dtype = np.int)

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
    return X ^ Y


# xor всех векторов по оси x или y
def sum(X, axis = 0):
    # матрица результата, заполненная предварительно нулями
    res = np.zeros((X.shape[abs(axis - 1)]), dtype = np.int)
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
    # размеры матрицы A|b
    n = A.shape[0]
    m = A.shape[1] + 1
    # получаем матрицу A|b путем присоединения вектора b к матрице A
    Ab = np.concatenate((A, b.reshape(-1, 1)), axis = 1)
    # метод Гаусса
    # прямой ход
    for i in range(n):
        nonzero_ind = np.nonzero(Ab[i:, i])[0] + i
        if nonzero_ind.size == 0:
            return np.nan
        Ab[nonzero_ind, :] = divide(Ab[nonzero_ind, :], np.tile(Ab[nonzero_ind, i].reshape(-1, 1), m), pm)
        for j in nonzero_ind[1:]:
            Ab[j, :] = add(Ab[j, :], Ab[nonzero_ind[0], :])
        if nonzero_ind[0] > i:
            Ab[i, :], Ab[nonzero_ind[0], :] = Ab[nonzero_ind[0], :].copy(), Ab[i, :].copy()
    res = np.zeros(n, dtype = np.int)
    # обратный ход
    for i in range(n - 1, -1, -1):
        res[i] = add(Ab[i, m - 1], sum(prod(res[(i + 1):], Ab[i, (i + 1):n], pm).reshape(-1, 1))).copy()
    return res

def minpoly(x, pm):
    # корни - это все элементы из x; все элементы x, помноженные на себя, итд
    s = set()
    for i in range(x.size):
        s |= {x[i]}
        tmp = prod(np.array([x[i]]), np.array([x[i]]), pm)
        while True:
            if (tmp[0] == x[i]) or (tmp[0] in s):
                break
            s |= {tmp[0]}
            tmp = prod(tmp, tmp, pm)
    # нам надо вернуть np.array, поэтому приводим set к np.array
    roots = np.array(list(s))
    # m - минимальный полином
    m = polyprod(np.array([1, roots[0]]), np.array([1, roots[1]]), pm)
    for i in range(2, roots.size):
        m = polyprod(m, np.array([1, roots[i]]), pm)

    return m, roots

def polyval(p, x, pm):
    # схема Горнера
    res = np.zeros(x.size)
    for i in range(p.size):
        if i == 0:
            res = p[0]
        else:
            res = add(prod(res, x, pm), np.array(p[i]))
    return res

def polyprod(p1, p2, pm):
    if p1.size < p2.size:
        p1, p2 = p2.copy(), p1.copy()
    # заводим вектор для результата
    res = np.zeros(p1.size + p2.size - 1, dtype=np.int)

    # временный вектор для удобства промежуточных подсчетов
    tmp = np.zeros(p1.size, dtype=np.int)

    for i in range(-1, -p2.shape[0] - 1, -1):
        # заполняем весь вектор tmp значением из p2
        tmp[:] = p2[i]
        # умножаем на вектор p1
        tmp = prod(tmp, p1, pm)
        # записываем результат в вектор результата
        if i == -1:
            # на первом проходе просто записываем значения
            res[-tmp.size:] = tmp
        else:
            # на последующих суммируем с предыдущими значениями
            res[-tmp.size + i + 1: i + 1] ^= tmp
    return res

def polydiv(p1, p2, pm):
    # q - частное
    # r - остаток
    # если делимое меньше делителя, то частное = 0, остаток = делителю
    if p1.size < p2.size:
        q = np.array([0], dtype=np.int)
        r = p1
    else:
        # нулевой шаг(инициализация переменных)
        q = np.empty(p1.size - p2.size + 1, dtype=np.int)
        r = p1
        for i in range(q.size):
            # находим частное
            q[i] = divide(r[0: 1], p2[0: 1], pm)
            # умножаем частное на делитель
            tmp = polyprod(p2, np.array([q[i]]), pm)
            # находим остаток
            r = add(r, np.append(tmp, np.zeros(r.size - tmp.size, dtype=np.int)))[1:]
    # избавляемся от лишних нулей
    while r.size > 1 and r[0] == 0:
        r = r[1:]
    return q, r
"""
def euclid(p1, p2, pm, max_deg = 0): 
"""