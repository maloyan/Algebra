import numpy as np
import gf

class BCH():
    def __init__(self, n, t):
        with open('primpoly.txt') as f:
            primpoly = f.readlines()[0].split(', ')

        primpoly = [int(poly) for poly in primpoly]
        prim_poly = primpoly[np.where(np.log2(np.array(primpoly)).astype('int') >= int(np.log2(n + 1)))[0][0]]

        self.pm = gf.gen_pow_matrix(prim_poly)

        self.R = np.array([self.pm[i % self.pm.shape[0] - 1, 1] for i in range(1, 2 * t + 1)])

        self.g, r = gf.minpoly(self.R, self.pm)
        # self.m = self.g.shape[0] - 1

        self.n = n

    def encode(self, U):
        message_number, k = U.shape
        res = np.zeros((message_number, self.n)).astype('int')
        g = np.zeros(self.g.size).astype('int')
        g[0] = 1

        for i in range(message_number):

            s = gf.polyprod(g, U[i, :], self.pm)
            div, mod = gf.polydivmod(s, self.g, self.pm)
            res_vec = gf.polyadd(s, mod)
            res[i, (self.n - res_vec.size):] = res_vec

        return res

    #def decode(self, W, method=’euclid’):

    product(A, B) returns the same as ((x,y) for x in A for y in B).
    itertools.product([0, 1], repeat=self.n - self.m)
    def dist(self):
        a = list((x, y) for x in [0, 1] for y in repeat=(self.n - self.g.size - 1))
        U = np.array(a, dtype=int)[1:]
        V = self.encode(U)
        res = np.min(np.sum(V, axis=1))

        return res

def main():
    x = BCH(2, 2)
    print(gf.gen_pow_matrix(7))
    #print(euclid(np.array([2, 3, 6]), np.array([1, 0, 1]), gen_pow_matrix(19), 0))
    #(array([13]), array([13, 12]), array([15,  7]))
if __name__ == "__main__":
    main()
