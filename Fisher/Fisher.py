from __future__ import annotations
import numpy as np
import os
import sys
import typing
import plotly.graph_objects as go
from plotly.offline import plot
base_folder = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".."))
if base_folder not in sys.path:
    sys.path.append(base_folder)
if True:
    from Datasets.Datasets import Dataset


class Fisher:
    @staticmethod
    def _calculate(x: np.ndarray[np.float64]) -> tuple[np.ndarray[np.float64], np.ndarray[np.float64]]:
        n, m = x.shape
        mu = x.sum(axis=0) / n
        S = np.zeros((m, m))
        for i in range(n):
            d = (x[i] - mu).reshape(m, 1)
            S += d @ d.T
        return mu, S

    def __init__(self, x1: np.ndarray[np.float64], x2: np.ndarray[np.float64], method: str = 'mid') -> None:
        mu1, S1 = self._calculate(x1)
        mu2, S2 = self._calculate(x2)
        Sw = S1 + S2
        w = (np.linalg.inv(Sw) @ (mu1 - mu2))
        y1, y2 = float(w.T @ mu1), float(w.T @ mu2)

        def f(x: np.ndarray[np.float64]) -> float:
            if method == 'mid':
                y0 = (y1 + y2) / 2
                y = float(w.T @ x)
                return (y - y0) / (y1 - y2)
            elif method == 'weighted':
                N1, N2 = x1.shape[1], x2.shape[1]
                y0 = (N1 * y1 + N2 * y2) / (N1 + N2)
                y = float((w.T @ x))
                if y >= y0:
                    return (y - y0) / (y1 - y0)
                else:
                    return (y - y0) / (y0 - y2)
            else:
                raise TypeError("Illegal argument passed to Fisher.__init__")
        self._f = f

    def __call__(self, x: np.ndarray[np.float64]) -> float:
        return self._f(x)


def test(dataset: Dataset, method: str = 'mid', print_tmp: bool = False, plot_figure: bool = False) -> float:
    name = dataset.__class__.__name__
    acc = 0
    if plot_figure:
        validation_record, fisher_record, acc_record = None, None, 0
    C, K = dataset.C, dataset.train.K
    for k in range(K):
        train, validation = dataset.train.split()
        cnt = 0

        '''if C == 2:
            positive, negative = [], []
            for x, y in train:
                if y[0, 0] == 1:
                    positive.append(x)
                else:
                    negative.append(x)
            positive = np.concatenate(positive, axis=1)
            negative = np.concatenate(negative, axis=1)
            fisher = Fisher(positive, negative, method=method)
            for x, y in validation:
                if (fisher(x) > 0 and y[0, 0] == 1) or (fisher(x) < 0 and y[0, 0] == 0):
                    cnt += 1
        else:'''
        fisher = []
        positive, negative = [[] for c in range(C)], [[] for c in range(C)]
        for x, y in train:
            index = np.argmax(y.squeeze())
            positive[index].append(x)
            for c in range(C):
                if c != index:
                    negative[c].append(x)
        positive = [np.stack(j) for j in positive]
        negative = [np.stack(j) for j in negative]
        for c in range(C):
            fisher.append(Fisher(positive[c], negative[c], method=method))

        def f(x: np.ndarray[np.float64]) -> np.ndarray[np.float64]:
            eye = np.eye(C)
            y = [0 for _ in range(C)]
            for c in range(C):
                y[c] = fisher[c](x)
            c = np.argmax(np.array(y))
            return eye[c]
        for x, y in validation:
            if (f(x) == y).all():
                cnt += 1

        acc_cur = cnt / len(validation)
        if plot_figure and acc_cur > acc_record:
            validation_record, fisher_record, acc_record = validation, fisher, acc_cur
        acc += acc_cur
        acc_record = acc_cur
        if print_tmp:
            print('test{} on dataset {} with method \'{}\': accuracy = {}'.format(
                k, name, method, acc_cur))
        dataset.train.reshuffle()
    acc /= K

    if plot_figure:
        positive, negative = [[] for c in range(C)], [[] for c in range(C)]
        fig = go.Figure()
        for x, y in validation_record:
            index = np.argmax(y.squeeze())
            positive[index].append(x)
            for c in range(C):
                if c != index:
                    negative[c].append(x)
        for c in range(C):
            p, n = np.array([fisher_record[c](x) for x in positive[c]]), \
                np.array([fisher_record[c](x) for x in negative[c]])
            fig.add_trace(
                go.Scatter(
                    name="C{}".format(c),
                    x=p,
                    y=np.ones_like(p) * c,
                    mode="markers"
                )
            )
            fig.add_trace(
                go.Scatter(
                    name="Not C{}".format(c),
                    x=n,
                    y=np.ones_like(p) * c,
                    mode="markers"
                )
            )

        fig.add_trace(
            go.Scatter(
                name="Decision Boundary",
                x=[0, 0],
                y=[-1, C],
                mode="lines"
            )
        )
        fig.update_layout(
            title="Splitting",
            width=1280,
            height=720,
            xaxis_title=r"$y_{i}$",
            yaxis_title="c"
        )
        plot(fig, filename="Fisher_Classifier_for_{}.html".format(
            name), include_mathjax='cdn')

    return acc


if __name__ == "__main__":
    from Datasets.Datasets import Iris, Sonar
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    fig, acc_record = go.Figure(), []
    for d in range(1, 5):
        acc = test(Iris(split_point=1, D=d),
                   method='weighted', print_tmp=False, plot_figure=(d == 4))
        print('D = {}, total accuracy = {}'.format(d, acc))
        acc_record.append(acc)
    fig.add_trace(
        go.Scatter(
            x=list(range(1, 5)),
            y=acc_record,
            mode="markers+lines",
        )
    )
    fig.update_layout(
        title="Accuracy",
        width=1280,
        height=720,
        xaxis_title="D",
        yaxis_title="Accuracy",
    )
    plot(fig, filename="D_dimensional_Fisher_Classifier_for_Iris.html")

    fig, acc_record = go.Figure(), []
    for d in range(1, 61):
        acc = test(Sonar(split_point=1, D=d),
                   method='weighted', print_tmp=False, plot_figure=(d == 60))
        print('D = {}, total accuracy = {}'.format(
            d, acc))
        acc_record.append(acc)
    fig.add_trace(
        go.Scatter(
            x=list(range(1, 61)),
            y=acc_record,
            mode="markers+lines",
        )
    )
    fig.update_layout(
        title="Accuracy",
        width=1280,
        height=720,
        xaxis_title="D",
        yaxis_title="Accuracy",
    )
    plot(fig, filename="D_dimensional_Fisher_Classifier_for_Sonar.html")
