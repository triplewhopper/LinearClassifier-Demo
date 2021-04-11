import numpy as np

eps = 1e-6


class LDF:
    def __init__(self, data_dimension, label_number):
        self.W: np.ndarray = np.random.random((label_number, data_dimension))
        self.W = np.c_[self.W, np.zeros((self.W.shape[0], 1))]
        self.epoch = 0

    def reset(self):
        assert isinstance(self.W, np.ndarray)
        self.W = np.random.random(self.W.shape)
        self.W = np.c_[self.W, np.zeros((self.W.shape[0], 1))]
        self.epoch = 0

    def fit(self, training_data: np.ndarray, labels: np.ndarray, max_epoch=60000):
        assert training_data.ndim == 2
        assert labels.ndim == 1
        assert training_data.shape[1] == labels.shape[0]
        assert training_data.shape[0] + 1 == self.W.shape[1]
        assert set(labels).issubset(set(range(self.W.shape[0])))
        self.__fit(np.r_[training_data, np.ones((1, training_data.shape[1]))],
                   labels, max_epoch)

    def __fit(self, training_data, labels, max_epoch):
        epoch, alpha = 0, 1
        x, y = training_data, labels
        gradient = lambda W: self.softmax_loss_grad(W, x, y)  # grad(lambda W: self.softmax_loss(W, x, y))
        loss = self.softmax_loss(self.W, x, y)
        gg = gradient(self.W)
        # print(gg - self.softmax_loss_grad(self.W, x, y))
        while epoch < max_epoch:
            assert loss == loss
            if epoch % 1000 == 0:
                print(f'epoch={self.epoch}, accuracy={AccuracyOf(self.W, x, y):.2%}')
                # print(gg - self.softmax_loss_grad(self.W, x, y))
            epoch += 1
            self.epoch += 1
            if alpha < eps or -eps < loss < eps:
                return

            while (loss_ww := self.softmax_loss(ww := self.W - alpha * gg, x, y)) >= loss:
                if alpha < eps or -eps < loss - loss_ww < eps:
                    return
                assert ww.shape == self.W.shape
                alpha /= 2
            else:
                assert ww.shape == self.W.shape
                self.W, loss = ww, loss_ww
                gg = gradient(self.W)
                alpha *= 2

    def softmax_loss(self, W, x, y):  # 4*3, 3*n, 1*n
        scores = W.dot(x)  # 4*n
        if scores.max() > 709:
            scores = scores - scores.max() + 709
        expscores: np.ndarray = np.exp(scores)  # 4*n

        q = expscores[y, np.arange(expscores.shape[1])]  # 1*n
        p = np.sum(expscores, axis=0)  # 1*n
        return np.average(-np.log(q / p))

    def softmax_loss_grad(self, W, x, y):
        def softmax(a) -> np.ndarray:
            exp_of_a = np.exp(a)
            column_wise_sum = np.sum(exp_of_a, axis=0, keepdims=True)
            return exp_of_a / column_wise_sum

        s = softmax(self.W.dot(x))
        assert y.ndim == 1
        assert s.ndim == 2
        for i in range(len(y)):
            s[y[i]][i] -= 1
        return s.dot(x.T) / x.shape[1]

    def predict(self, test_data):
        assert isinstance(test_data, np.ndarray)
        assert test_data.shape[0] + 1 == self.W.shape[1]
        test_data = np.r_[test_data, np.ones((1, test_data.shape[1]))]
        return np.argmax(self.W.dot(test_data), axis=0)

    def accuracy(self, test_data, labels):
        assert isinstance(test_data, np.ndarray)
        assert test_data.shape[0] + 1 == self.W.shape[1]
        assert test_data.shape[1] == labels.shape[0]
        test_data = np.r_[test_data, np.ones((1, test_data.shape[1]))]
        return AccuracyOf(self.W, test_data, labels)


def Pr(W, x):
    scores = W.dot(x)
    expscores = np.exp(scores)
    s = np.tile(np.sum(expscores, axis=0), (expscores.shape[0], 1))
    return expscores / s


def AccuracyOf(W, x, y):
    estimated = np.argmax(Pr(W, x), axis=0)
    assert len(estimated) == len(y)
    return sum(int(estimated[i] == y[i]) for i in range(len(y))) / len(y)


def grad(f):
    def _g(x):
        h = 1e-7
        assert isinstance(x, np.ndarray)
        res = np.zeros(x.shape)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                t = x[i][j]
                x[i][j] += h
                res[i][j] = f(x)
                x[i][j] = t
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                t = x[i][j]
                x[i][j] -= h
                res[i][j] -= f(x)
                x[i][j] = t
        return res / (2 * h)

    return _g
