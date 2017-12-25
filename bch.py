import numpy as np
import gf

class BCH():
    def __init__(self, n, t):
        with open('primpoly.txt') as f:
            primpoly = f.readlines()[0].split(', ')

        primpoly = [int(poly) for poly in primpoly]
        prim_poly = primpoly[np.where(np.log2(np.array(primpoly)).astype('int') >= int(np.log2(n + 1)))[0][0]]

        self.n = n
        self.t = t

        self.pm = gf.gen_pow_matrix(prim_poly)

        self.R = [self.pm[i % self.pm.shape[0] - 1, 1] for i in range(1, 2 * self.t + 1)]
        self.R = np.array(self.R)

        self.g, _ = gf.minpoly(self.R, self.pm)
        self.m = self.g.shape[0] - 1

def main():
    print(gen_pow_matrix(7))
    #print(euclid(np.array([2, 3, 6]), np.array([1, 0, 1]), gen_pow_matrix(19), 0))
    #(array([13]), array([13, 12]), array([15,  7]))
if __name__ == "__main__":
    main()
