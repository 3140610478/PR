from __future__ import annotations
import torch
import numpy as np
import os
import sys
base_folder = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../.."))
if base_folder not in sys.path:
    sys.path.append(base_folder)
if True:
    from Datasets.Datasets import Dataset


class FCM:
    device = torch.device("cuda")\
        if torch.cuda.is_available() else torch.device("cpu")

    def distance(self, tX: torch.Tensor, tY: torch.Tensor, batch_size: int = 65536, print_tmps: bool = False) -> torch.Tensor:
        N = int(tX.shape[0])
        distance = torch.zeros((N, int(tY.shape[0])), device=self.device)
        for start in range(0, N, batch_size):
            end = min(start+batch_size, N)
            tx = tX[start:end].unsqueeze(1)
            # calculating distance between tx & tY
            distance_of_batch = (tY-tx)**2
            distance[start:end] = torch.sum(distance_of_batch,
                                            dim=torch.Size(torch.arange(2, distance_of_batch.dim())))
            if print_tmps:
                print('current batch: {}, samples: {}'.format(
                    start // batch_size, end))
        return distance

    def _Kmeanspp(self, K: int) -> torch.Tensor:
        centers = torch.zeros(
            (K, *self.feature[0].shape), device=self.device)
        centers[0] = self.feature[torch.randint(0, self.LEN, (1,))]
        distance = torch.ones((self.LEN,), device=self.device) * torch.inf
        for i in range(1, K):
            distance = torch.minimum(distance,
                                     self.distance(self.feature, centers[i-1:i]).squeeze())
            centers[i] = self.feature[torch.multinomial(distance, 1)]
        return centers

    def __init__(self, data: Dataset.Subset | Dataset.TensorSubset, b: int, method="K-Means++"):
        self.C = data.C
        self.b = b
        self.EYE = torch.eye(self.C, device=self.device)
        self.LEN = len(data)
        if isinstance(data, Dataset.Subset):
            data = data.toTensor()
        self.feature, self.label_actual = data.feature, data.label
        if method == "K-Means++":
            self.m = self._Kmeanspp(self.C)
        elif method == "Random":
            self.m = self.feature[torch.randperm(self.LEN)[:self.C]]

    def __call__(self, batch_size: int = 65536, update: bool = True, print_tmps=False):
        feature_shape = self.feature[0].shape
        feature = self.feature.flatten(start_dim=1)

        distance = self.distance(
            self.feature, self.m, batch_size=batch_size, print_tmps=print_tmps)
        eps = torch.topk(distance.flatten(), self.C+1, largest=False)[0][-1]/10
        distance_pow = torch.pow(distance+eps, -1/(self.b-1))
        mu = (distance_pow.T / torch.sum(distance_pow, dim=1)).T

        mu_pow = torch.pow(mu, self.b)
        m = (mu_pow.T @ feature)
        m = (m.T / torch.sum(mu_pow, dim=0)).T
        m = m.reshape(self.C, *feature_shape)

        if update:
            self.mu, self.m = mu, m
            self.label = self.EYE[torch.argmax(self.mu, dim=1)]

        return mu

    def purity(self) -> torch.Tensor:
        idx = self.label.T.bool()
        purity = torch.stack(
            [torch.max(torch.sum((self.label_actual[i]), dim=0)) for i in idx])
        length = torch.sum(self.label, dim=0)
        purity /= torch.max(length, torch.ones_like(length,
                            device=self.device))
        mean_purity = float((length @ purity) / torch.sum(length))
        return mean_purity

    def rand_index(self) -> torch.Tensor:
        l = torch.argmax(self.label, dim=1).to(dtype=torch.uint8)
        la = torch.argmax(self.label_actual, dim=1).to(dtype=torch.uint8)
        l, la = l.unsqueeze(1), la.unsqueeze(1)
        l, la = ((l ^ l.T) == 0), ((la ^ la.T) == 0)
        ri = float((~l ^ la).count_nonzero() - self.LEN) / \
            float(self.LEN * (self.LEN - 1))
        return ri

    def iterate(self, n: int, batch_size: int = 65536, print_tmps=False) -> tuple:
        result = []
        for i in range(n):
            mu = self(batch_size=batch_size,
                      update=True, print_tmps=print_tmps)
            result.append((i, self.purity(), self.rand_index(),
                          "b = {}".format(self.b),))
            print("iteration {0}, {3}, purity = {1}, rand index = {2}".format(
                *(result[-1])))
        return result


if __name__ == '__main__':
    from Datasets.Datasets import Iris, Mnist
    os.chdir(os.path.abspath(os.path.dirname(__file__)))
    iris = Iris(K=1, split_point=1)
    mnist = Mnist(K=1)
    fcm_iris = FCM(iris.train, 2)
    fcm_iris.iterate(10)
    fcm_mnist = FCM(mnist.test, 5)
    fcm_mnist.iterate(16)
