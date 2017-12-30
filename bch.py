import numpy as np
import gf
import matplotlib.pyplot as plt
import time

class BCH():
    def __init__(self, n, t):
        primpoly = [7, 11, 19, 37, 67, 131, 285, 529, 1033, 2053, 4179, 8219, 16427, 32771, 65581]

        q = int(np.log2(n + 1))

        for i in range(len(primpoly)):
            if primpoly[i] >= 2 ** q:
                prim_poly = primpoly[i]
                break

        self.pm = gf.gen_pow_matrix(prim_poly)

        arr = [2]
        for i in range(2, 2 * t + 1):
            arr = arr + [self.pm[i % self.pm.shape[0] - 1, 1]]
        self.R = np.array(arr)

        self.g = gf.minpoly(self.R, self.pm)[0]
        self.n = n
        self.t = t

    def encode(self, U):
        # res = x^m * u(x) - (x^m * u(x)) mod g(x)
        #       |________|    |________|
        #           ^             ^
        #           |             |
        #           s      +      s       mod g(x)
        message_number, k = U.shape
        m = self.g.size - 1
        x_m = np.zeros(m + 1, dtype=int)
        x_m[0] = 1
        res = np.zeros((message_number, k + m), dtype=int)
        for i in range(message_number):
            s = gf.polyprod(x_m, U[i, :], self.pm)
            mod = gf.polydiv(s, self.g, self.pm)[1]
            tmp = gf.polyadd(s, mod)
            res[i, -tmp.size:] = tmp.copy()
        return res

    def decode(self, W, method='euclid'):
        message_number, n = W.shape
        # res - результат
        res = np.zeros(W.shape).astype('int')

        # PGZ
        for i in range(message_number):
            # s - синдром
            # для принятоо слова W вычислим синдром
            s = gf.polyval(W[i, :], self.R, self.pm)
            # если все s = 0, то возвращаем W в качестве ответа
            if np.all(s == 0):
                res[i] = W[i].copy()
                continue
            # вычислим коэффициенты полинома локаторов ошибок путем решения СЛАУ
            if method == 'pgz':
                for j in range(self.t, 0, -1):
                    # составим матрицу A для СЛАУ
                    A = np.zeros((j, j)).astype('int')
                    for k in range(j):
                        A[k, :] = s[k: k + j]
                    b = s[j: 2 * j]
                    # решаем СЛАУ
                    Lambda = gf.linsolve(A, b, self.pm)
                    if not np.any(np.isnan(Lambda)):
                        break
                if np.any(np.isnan(Lambda)):
                    res[i] = np.nan
                    continue
                Lambda = np.append(Lambda, np.array([1]))

            elif method == 'euclid':
                s = np.append(s[::-1], np.array([1]))
                # z^(2t + 1)
                z = np.zeros(((self.R.size + 1) + 1), dtype=np.int)
                z[0] = 1
                # алгоритм евклида
                # z^(2t + 1) * A(z) + S(z)L(z) = r(z)
                # находим L(z)
                Lambda = gf.euclid(z, s, self.pm, max_deg=self.t)[2]

            # получаем позиции ошибок
            roots = gf.polyval(Lambda, self.pm[:, 1], self.pm)
            pos_error = np.nonzero(roots.reshape(-1) == 0)[0]

            # инвертируем биты в позициях ошибок
            tmp = W[i].copy()
            tmp[pos_error] = 1 - tmp[pos_error].astype(np.int)
            res[i] = tmp


            s = gf.polyval(res[i].astype(int), self.R, self.pm)
            if not np.all(s == 0):
                res[i, :] = np.ones(self.n) * np.nan
                continue
        return res

    def dist(self):
        k = self.n - self.g.size + 1
        V = self.encode(np.eye(k))
        dist = self.n + 1
        for num in range(1, 2 ** k):
            # получаем массив из бинарного представления числа
            bin_num = list(bin(num)[2:])
            # растягиваем этот массив до размера k, путем добавления нулей в начало
            bin_num = np.array(((k - len(bin_num)) * [0] + bin_num), dtype=int).reshape(k, 1)
            # выбираем из массива V нужные на этом этапе строки
            cur_rows = V * bin_num
            # суммируем по столбцам и берем по модулю 2, получаем вектор
            sum_rows = np.sum(cur_rows, axis=0) % 2
            # суммируем значения в этом векторе, это и есть Хеммингов вес
            tmp = np.sum(sum_rows)
            # нужен минимальный вес
            if tmp < dist:
                dist = tmp
        return dist


def generate_U(msg_cnt, k):
    U = np.zeros((msg_cnt, k)).astype('int')
    inds = np.arange(0, k)
    for i in range(msg_cnt):
        try:
            tmp = np.random.choice(inds, k)
            ind = np.unique(tmp)
        except:
            print(np.random.choice(inds, k))
        U[i, ind] = 1

    return U


def error_W(V, r):
    W = np.zeros(V.shape).astype('int')
    msg_cnt, n = V.shape
    inds = np.arange(0, n)
    for i in range(msg_cnt):
        error = np.zeros(n).astype('int')
        ind = np.random.choice(inds, r, replace=False)
        error[ind] = 1
        W[i, :] = V[i, :] ^ error

    return W

def main():
    """
    U = np.array([[1, 0, 1, 1, 1, 0],
                  [1, 1, 1, 0, 0, 1],
                  [0, 0, 1, 1, 1, 1],
                  [1, 0, 1, 0, 1, 0]])
    #print(x.encode(U))


    W = np.array([[1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0],
                  [1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                  [0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0],
                  [0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0]])
    x = BCH(15, 2)
    U = np.array([[0, 1, 1, 0, 1],
                  [1, 0, 1, 0, 1],
                  [0, 1, 0, 0, 1],
                  [1, 1, 0, 1, 0],
                  [0, 1, 0, 1, 0]])
    """
    n_val = [7, 15, 31, 63, 127]
    t_val = [1, 2, 4, 8, 11]
    msg_val = [500, 500, 500, 500, 500]
    column = ["n", "t", "Euclid decoder, time", "PGZ decoder, time"]
    for i in range(len(n_val)):
        msg_cnt = msg_val[i]
        n = n_val[i]
        t = t_val[i]
        row = [n, t]
        bch_code = BCH(n, t)
        k = n - bch_code.g.size + 1
        U = generate_U(msg_cnt, k)
        V = bch_code.encode(U)
        W = error_W(V, 1)

        start = time.process_time()
        V_hat = bch_code.decode(W, method='euclid')
        end = time.process_time()
        print('euclid', n, t)
        print(end - start)

        start = time.process_time()
        V_hat = bch_code.decode(W, method='pgz')
        end = time.process_time()
        print('pgz', n, t)
        print(end - start)


        #table_time = table_time.append(pd.Series(row, index=table_time.columns),ignore_index=True)
    """
    start = time.time()
    x = BCH(31, 9)
    print(x.decode())
    end = time.time()
    print(end - start)
    """
    """
    r_q_t = []
    for q in range(2, 11): #11
        n = 2 ** q - 1
        r_q = []
        for t in range(1, min(511, (n - 1) // 2 + 5)):
            x = BCH(n, t)#, prim_list)
            r_q += [(n - len(x.g) + 1) / n]
        r_q_t += [r_q]

    print(len(r_q_t))

    q = 3
    for i in range(1, len(r_q_t[:-1])):
        plt.figure()
        n = 2 ** q - 1
        t = np.arange(1, min(511, (n - 1) // 2 + 5))
        plt.xlabel("Number of errors, t")
        plt.ylabel("Code rate, r = k/n")
        plt.plot(t, r_q_t[i], lw=1)
        q += 1
    plt.show()
    """
if __name__ == "__main__":
    main()
