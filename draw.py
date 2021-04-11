import linearDiscriminateFunction as classifier
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import dataMaker as dm
import matplotlib
import pickle


def read_from(filename: str):
    if not filename.endswith('.pkl'): filename += '.pkl'
    with open(filename, 'rb') as f:
        obj = pickle.load(f)

    return obj


def draw(x, y, c):
    x_min, x_max = x[0, :].min() - 1, x[0, :].max() + 1
    y_min, y_max = x[1, :].min() - 1, x[1, :].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    z = c.predict(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    plt.figure()
    pc=plt.contourf(xx,yy, z)
    plt.scatter(x[0, :], x[1, :], marker='x', c=y)
    plt.title('線形判別関数による4種類分類器の分類結果（1116201017賈書瑞）')
    plt.show()

if __name__ == '__main__':
    import matplotlib.font_manager as fm

    # fmgr = fm.FontManager()
    # mat_fonts = sorted(set(f.name for f in fmgr.ttflist))
    # print(mat_fonts)
    plt.rcParams['font.sans-serif'] = ['Hiragino Mincho ProN']
    c = read_from('c')
    x, y = read_from('x'), read_from('y')
    draw(x, y, c)
