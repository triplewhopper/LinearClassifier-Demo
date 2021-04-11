import numpy as np
import random as rd
from typing import Dict, Tuple
import pickle
from sklearn import datasets


def gen(n=1000) -> Tuple[np.ndarray, np.ndarray]:
    x, y = datasets.make_classification(n_samples=n, n_features=2,
                                                n_informative=2, n_redundant=0,
                                                n_repeated=0, n_classes=4,
                                                n_clusters_per_class=1)
    return x.T, y

    p = np.random.rand(2, n) * 5

    ans = [0] * n

    for i in range(n):
        x, y = p[0][i], p[1][i]
        ans[i] = int(2 * (x - 2) >= y - 2) << 1 | int(-3 * (x - 1) >= y - 4)
    return p, np.array(ans)


def write():
    rd.seed()
    x, y = gen()

    for name, obj in {'x': x, 'y': y}.items():
        with open(name + '.pkl', 'wb') as f:
            pickle.dump(obj, file=f)


if __name__ == '__main__':
    write()
