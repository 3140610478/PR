from __future__ import annotations
import torch
import numpy as np
import os
import sys
# from sklearn.cluster import kmeans_plusplus
base_folder = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../.."))
if base_folder not in sys.path:
    sys.path.append(base_folder)
if True:
    from Datasets.Datasets import Dataset


class Kmeans:
    device = torch.device("cuda")\
        if torch.cuda.is_available() else torch.device("cpu")

    def distance(self, tX: torch.Tensor, centers: torch.Tensor, batch_size: int = 65536, print_tmps: bool = False) -> torch.Tensor:
        N = int(tX.shape[0])
        distance = torch.zeros((N, int(centers.shape[0])), device=self.device)
        for start in range(0, N, batch_size):
            end = min(start+batch_size, N)
            tx = tX[start:end].unsqueeze(1)
            # calculating distance between tx & center
            if self.measure == "Euclidean":
                distance_of_batch = (centers-tx)**2
            elif self.measure == "Manhatton":
                distance_of_batch = torch.abs(centers-tx)
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

    def __init__(self, data: Dataset.Subset | Dataset.TensorSubset, measure="Euclidean", method="K-Means++"):
        self.C = data.C
        self.LEN = len(data)
        if isinstance(data, Dataset.Subset):
            data = data.toTensor()
        self.feature, self.label_actual = data.feature, data.label
        self.label = torch.zeros_like(self.label_actual).to(self.device)
        self.EYE = torch.eye(self.C, device=self.device)
        self.measure = measure
        if method == "K-Means++":
            self.centers = self._Kmeanspp(self.C)
        elif method == "Random":
            self.centers = self.feature[torch.randperm(self.LEN)[:self.C]]

    def __call__(self, batch_size: int = 65536, update: bool = True, print_tmps=False) -> np.ndarray[np.float32] | torch.Tensor:
        distance = self.distance(
            self.feature, self.centers, batch_size=batch_size, print_tmps=print_tmps)

        # assigning to nearest cluster
        idx = torch.argmin(distance, dim=1)
        label = self.EYE[idx]

        if update:
            self.centers = torch.stack([torch.mean(self.feature[idx == i], dim=0)
                                        for i in range(self.C)])
            self.label = label
        return label

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
        ri = float((~(l ^ la)).count_nonzero() - self.LEN) / \
            float(self.LEN * (self.LEN - 1))
        return ri

    def iterate(self, n: int, batch_size: int = 65536, print_tmps=False) -> None:
        result = []
        for i in range(n):
            label = self(batch_size=batch_size,
                         update=True, print_tmps=print_tmps)
            result.append((i, self.purity(), self.rand_index(),
                          "measured = "+self.measure,))
            print("iteration {0}, {3}, purity = {1}, rand index = {2}".format(
                *(result[-1])))
        return result


if __name__ == '__main__':
    from Datasets.Datasets import Iris, Mnist
    os.chdir(os.path.abspath(os.path.dirname(__file__)))
    iris = Iris(K=1, split_point=1)
    mnist = Mnist(K=1)
    # kmeans_iris = Kmeans(iris.train)
    # kmeans_iris.iterate(10)
    kmeans_mnist = Kmeans(mnist.test)
    kmeans_mnist.iterate(10)
