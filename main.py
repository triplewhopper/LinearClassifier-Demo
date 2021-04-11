import numpy as np
import random as rd
from typing import Dict, Tuple, Callable, Iterable
import pickle
import matplotlib.pyplot as plt
import dataMaker
import linearDiscriminateFunction as classifier


def read_from(filename: str):
    if not filename.endswith('.pkl'): filename += '.pkl'
    with open(filename, 'rb') as f:
        obj = pickle.load(f)

    return obj


def save_as(filename: str, obj):
    if not filename.endswith('.pkl'): filename += '.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


if __name__ == '__main__':
    x, y = read_from('x'), read_from('y')
    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)
    print('x.shape=', x.shape)
    print('y.shape=', y.shape)

    while (jouken := input('继续训练？(y/n)测试？(t)')) not in ('y', 'n', 't'):
        ...
    if jouken == 'y' or jouken == 'n':
        c = object()
        if jouken == 'y':
            c = read_from('c')
        elif jouken == 'n':
            c = classifier.LDF(2, 4)
        assert isinstance(c, classifier.LDF)
        c.fit(x, y)
        np.set_printoptions(formatter={'float': '{:.3f}'.format})

        print('{:.2%}'.format(c.accuracy(x, y)))
        save_as('c', c)

    elif jouken == 't':
        c = read_from('c')
        assert isinstance(c, classifier.LDF)
        x, y = dataMaker.gen(1000)
        print(f'{c.accuracy(x, y):.2%}')
