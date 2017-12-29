import numpy as np
import gf

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
            v = gf.polyadd(s, mod)
            res[i, -v.size:] = v.copy()
        return res

    def decode(self, W, method='euclid'):
        message_number, n = W.shape
        # s - синдром
        # s = np.zeros((message_number, 2 * self.t)).astype('int')
        # err - ошибки
        err = []
        # res - результат
        res = np.zeros((W.shape)).astype('int')

        # PGZ
        for i in range(message_number):
            # для принятоо слова W вычислим синдром
            s = gf.polyval(W[i, :], self.R, self.pm)
            # если все s = 0, то возвращаем W в качестве ответа
            if np.all(s == 0):
                res[i, :] = W[i, :]
                continue
            # Вычислим коэффициенты лямбда
            if method == 'pgz':
                for j in range(self.t, 0, -1):

                    # matrix for linear solve
                    A = np.zeros((j, j)).astype('int')
                    for k in range(j):
                        A[k, :] = s[k: k + j]
                    b = s[j: 2 * j]
                    Lambda = gf.linsolve(A, b, self.pm)
                    if np.all(np.isnan(Lambda)):
                        continue
                    else:
                        break

                # decode error
                if (j == 1) and np.all(np.isnan(Lambda)):
                    res[i, :] = np.ones(self.n) * np.nan
                    continue

                # coef of locator polynoms
                loc_poly_coef = np.zeros(Lambda.shape[0] + 1).astype('int')
                loc_poly_coef[-1] = 1
                loc_poly_coef[: -1] = Lambda
            elif method == 'euclid':
                s = np.append(s[::-1], np.array([1]))
                # z^(2t + 1)
                z = np.zeros(((self.R.size + 1) + 1), dtype=np.int)
                z[0] = 1
                # euclid algorithm
                loc_poly_coef = gf.euclid(z, s, self.pm, max_deg=self.t)[2]

            # find root
            locator_val = gf.polyval(loc_poly_coef, self.pm[:, 1], self.pm)
            roots = self.pm[np.where(locator_val == 0)[0], 1]
            pos_error = (-self.pm[roots - 1, 0]) % self.pm.shape[0]
            pos_error = self.n - pos_error - 1
            # error polynom
            error_poly = np.zeros(self.n).astype('int')
            error_poly[pos_error] = 1

            # decode
            v_hat = W[i, :] ^ error_poly
            s_v_hat = gf.polyval(v_hat, self.R, self.pm)

            if not np.all(s_v_hat == 0):
                res[i, :] = np.ones(self.n) * np.nan
                continue

            if (roots.shape[0] != loc_poly_coef.shape[0] - 1):
                res[i, :] = np.ones(self.n) * np.nan
                continue
            res[i, :] = v_hat

        return res

    def dist(self):
        k = self.n - self.g.size + 1
        U = np.eye(k)
        V = self.encode(U)
        min_dist = self.n + 1
        for num in range(1, 2 ** k):
            num_list = list(bin(num)[2:])
            lin_comb_coefs = np.array(((k - len(num_list)) * [0] + num_list), dtype=int).reshape(k, 1)
            new_dist = np.sum(np.sum(V * lin_comb_coefs, axis=0) % 2)
            if new_dist < min_dist:
                min_dist = new_dist
        return min_dist

def main():
    x = BCH(31, 7)
    U = np.array([[1, 0, 1, 1, 1, 0],
                  [1, 1, 1, 0, 0, 1],
                  [0, 0, 1, 1, 1, 1],
                  [1, 0, 1, 0, 1, 0]])
    print(x.encode(U))
    W = np.array([[1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0],
                 [1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0],
                 [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                 [0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0],
                 [0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0]])
    """
    [[0 1 1 0 1 1 1 0 0 0 0 1 0 1 0]
     [0 0 1 0 1 0 0 1 1 0 1 1 1 0 0]
     [0 0 0 0 0 0 1 1 1 0 1 0 0 0 1]
     [0 1 1 1 0 0 0 0 1 0 1 0 0 1 1]
     [0 0 0 0 0 1 1 1 0 1 0 0 0 1 0]]
    """
    x = BCH(15, 3)
    U = np.array([[10, 2, 3], [11, 8, 5]])
    print(x.dist())
if __name__ == "__main__":
    main()
