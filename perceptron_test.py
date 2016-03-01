import numpy as np

class perceptron(object):
    """docstring for """
    def __init__(self, n_epoch=10, eta=.10):
        self.n_epoch = n_epoch
        self.eta = eta

    def fit(self, X, y):
        self.w_ = np.zeros(X.shape[1] + 1)
        self.errors_ = []

        for _ in np.arange(self.n_epoch):
            error = 0
            for (xi, outcome) in zip(X, y):
                pred = self.predict(xi)
                print(xi, outcome, pred, self.w_)
                self.w_[1:] += self.eta * (outcome - pred) * xi
                self.w_[0] += self.eta * (outcome - pred)
                error += int(outcome != pred)
                print(xi, outcome, pred, self.w_)
            self.errors_.append(error)
        return self

    def predict(self, X):
        return np.where((np.dot(X, self.w_[1:]) + self.w_[0]) >= 0,
                        1,
                        -1)

if __name__ == "__main__":
    import operator
    from functools import reduce
    clf = perceptron(50, .1)
    X = np.array([(x, y) for x in [0, 1] for y in [0, 1]])
    y = np.apply_along_axis(lambda x: reduce(operator.mul, x, 1),
                            axis=1, arr=X)
    y = y * 2 - 1
    print(X, y)
    clf.fit(X, y)
    print(clf.predict(X))
    print(clf.errors_)
