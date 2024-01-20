from __future__ import annotations
import torch
from torch.nn.functional import softmax
import random
import numpy as np
import os
import sys
import time
from typing import Iterable
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import plot
base_folder = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".."))
if base_folder not in sys.path:
    sys.path.append(base_folder)
if True:
    from Datasets.Datasets import Dataset


class KNN:
    device = torch.device("cuda")\
        if torch.cuda.is_available() else torch.device("cpu")

    def __init__(self, data: Dataset.Subset):
        # data = data.toTensor()
        self.C = data.C
        self.feature = torch.from_numpy(
            np.stack(data.feature).astype(np.float32)).to(self.device)
        self.SCALE = (torch.max(self.feature) - torch.min(self.feature)) / \
            torch.prod(torch.as_tensor(
                self.feature.shape[1:]), dim=0).to(self.device)
        self.LEN = int(self.feature.shape[0])
        self.label = torch.from_numpy(
            np.stack(data.label).astype(np.float32)).to(self.device)

    def __call__(self, x: np.ndarray, K: int = 4, measure="Euclidean", weighted: bool = True, batch_size: int = 16, print_tmps=True) -> np.ndarray[np.float32]:
        tX = torch.from_numpy(x.astype(np.float32)).to(self.device)
        N = int(x.shape[0])
        label = torch.zeros(self.C, N, device=self.device)
        for i in range(0, N, batch_size):
            end = min(i+batch_size, N)
            actual_batch_size = end - i
            tx = tX[i:end].unsqueeze(1)
            # calculating distance between tx & self.feature
            if measure == "Euclidean":
                distance = (self.feature-tx)**2
            elif measure == "Manhatton":
                distance = torch.abs((self.feature-tx))
            distance = torch.sum(distance,
                                 dim=torch.Size(np.arange(2, distance.dim())))
            # selecting indexes with K minimum distances
            idx = torch.topk(distance, K, dim=1,
                             largest=False, sorted=False)[1]
            if weighted:
                # calculating weights according to distance
                weights = torch.stack([distance[n].index_select(0, idx[n])
                                       for n in range(actual_batch_size)])
                weights = softmax(-(weights.T / torch.max(weights, dim=1)[0]).T,
                                  dim=1).unsqueeze(1)
                # collecting votes from each selected index
                votes = torch.stack([self.label.index_select(0, idx[n])
                                    for n in range(actual_batch_size)])
                # generating labels
                label[:, i:end] = softmax((weights @ votes).squeeze(1),
                                          dim=1).T
            else:
                label[:, i:end] = torch.sum(torch.stack([self.label.index_select(0, idx[n])
                                                         for n in range(actual_batch_size)]), dim=1).T
            if print_tmps:
                print('current batch: {}, samples: {}'.format(i // batch_size, i))
        return label.cpu().numpy()


def run_test(knn: KNN, test: Dataset.Subset, K: int, measure="Euclidean", weighted: bool = True, batch_size: int = 16, save=False, print_tmps=True) -> tuple(int, float):
    start = time.time()
    result = knn(np.stack([i for i in test.feature]), K=K, measure=measure, weighted=weighted,
                 batch_size=batch_size, print_tmps=print_tmps)
    label = np.zeros_like(result)
    actual = np.stack([i.squeeze() for i in test.label]).T
    label[np.argmax(result, axis=0), np.arange(len(test))] = 1
    acc = 1 - np.sum(np.abs(label - actual)) / (2 * len(test))
    if print_tmps:
        print('K = {}, accuracy = {}'.format(K, acc))
    end = time.time()
    if print_tmps:
        print("Time: {}s, len(train): {}, len(test): {}".format(
            str(end-start), len(knn.label), len(test)))
    if save:
        np.savetxt('result.csv', result, delimiter=',')
        np.savetxt('label.csv', label)
        np.savetxt('actual.csv', actual)
    return K, acc, end-start


def run_test_on_iris():
    dataset = Iris(K=1, split_point=0.8)
    train, test = dataset.train, dataset.test
    rec, rec_w = [], []
    knn = KNN(train)
    for K in range(1, int(len(train)**0.5)):
        rec.append(run_test(knn, test, K, weighted=False, batch_size=120))
        rec_w.append(run_test(knn, test, K, batch_size=120))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            name="equal",
            x=[t[0] for t in rec],
            y=[t[1] for t in rec],
            mode="lines+markers",
        )
    )
    fig.add_trace(
        go.Scatter(
            name="weighted",
            x=[t[0] for t in rec_w],
            y=[t[1] for t in rec_w],
            mode="lines+markers",
        )
    )
    fig.update_layout(
        title="KNN on Iris",
        width=640,
        height=720,
        xaxis_title="K",
        yaxis_title="Accuracy",
    )
    plot(fig, filename="KNN on Iris.html")


def run_test_on_mnist():
    dataset = Mnist(K=1)
    train, test = dataset.train, dataset.test
    knn = KNN(train)

    def K_Test(knn: KNN) -> None:
        rec, rec_w = [], []
        for K in (2, 4, 8, 16):
            rec.append(run_test(knn, test, K, batch_size=16, weighted=False))
            rec_w.append(run_test(knn, test, K, batch_size=16))
        with open("K.txt", "w") as f:
            f.writelines((str(rec)+'\n', str(rec_w)+'\n'))
        fig = make_subplots(1, 2, subplot_titles=["Accuracy", "Time"])
        fig.add_trace(
            go.Scatter(
                name="equal",
                x=[t[0] for t in rec],
                y=[t[1] for t in rec],
                mode="lines+markers",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                name="equal",
                x=[t[0] for t in rec],
                y=[t[2] for t in rec],
                mode="lines+markers",
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                name="weighted",
                x=[t[0] for t in rec_w],
                y=[t[1] for t in rec_w],
                mode="lines+markers",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                name="weighted",
                x=[t[0] for t in rec_w],
                y=[t[2] for t in rec_w],
                mode="lines+markers",
            ),
            row=1,
            col=2,
        )
        fig.update_xaxes(title_text="K", row=1, col=1)
        fig.update_xaxes(title_text="K", row=1, col=2)
        fig.update_yaxes(title_text="Accuracy", row=1, col=1)
        fig.update_yaxes(title_text="Time", row=1, col=2)
        fig.update_layout(
            title="K Test on Mnist",
            width=1280,
            height=720,
        )
        # plot(fig, filename="K Test on Mnist.html")

    def Measure_Test(knn: KNN) -> None:
        rec, rec_m = [], []
        for K in (2, 4, 8, 16):
            rec.append(
                run_test(knn, test, K, measure="Euclidean", batch_size=16))
            rec_m.append(
                run_test(knn, test, K, measure="Manhatton", batch_size=16))
        with open("Measure.txt", "w") as f:
            f.writelines((str(rec)+'\n', str(rec_m)+'\n'))
        fig = make_subplots(1, 2, subplot_titles=("Accuracy", "Time"))
        fig.add_trace(
            go.Scatter(
                name="Euclidean",
                x=[t[0] for t in rec],
                y=[t[1] for t in rec],
                mode="lines+markers",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                name="Euclidean",
                x=[t[0] for t in rec],
                y=[t[2] for t in rec],
                mode="lines+markers",
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                name="Manhatton",
                x=[t[0] for t in rec_m],
                y=[t[1] for t in rec_m],
                mode="lines+markers",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                name="Manhatton",
                x=[t[0] for t in rec_m],
                y=[t[2] for t in rec_m],
                mode="lines+markers",
            ),
            row=1,
            col=2,
        )
        fig.update_xaxes(title_text="K", row=1, col=1)
        fig.update_xaxes(title_text="K", row=1, col=2)
        fig.update_yaxes(title_text="Accuracy", row=1, col=1)
        fig.update_yaxes(title_text="Time", row=1, col=2)
        fig.update_layout(
            title="Measure Test on Mnist",
            width=1280,
            height=720,
        )
        plot(fig, filename="Measure Test on Mnist.html")

    def BatchSize_Test(knn: KNN) -> None:
        rec = []
        for bs in (2, 4, 8, 16):
            rec.append((bs,) + run_test(knn, test, 4, batch_size=bs)[1:])
        with open("BatchSize.txt", "w") as f:
            f.writelines((str(rec)+"\n",))
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                name="time(s)",
                x=[t[0] for t in rec],
                y=[t[2] for t in rec],
                mode="lines+markers",
            )
        )
        fig.update_layout(
            title="Batch Size Test on Mnist",
            width=640,
            height=720,
            xaxis_title="BatchSize",
            yaxis_title="Time",
        )
        plot(fig, filename="Batch Size Test on Mnist.html")

    K_Test(knn)
    Measure_Test(knn)
    BatchSize_Test(knn)


if __name__ == "__main__":
    from Datasets.Datasets import Iris, Sonar, Mnist
    os.chdir(os.path.abspath(os.path.dirname(__file__)))
    # run_test_on_iris()
    # run_test_on_mnist()

    dataset = Sonar(K=1, split_point=0.8)
    train, test = dataset.train, dataset.test
    knn = KNN(train)
    for k in (1, 2, 4, 8):
        print(run_test(knn, test, k, batch_size=256, weighted=False))

pass
