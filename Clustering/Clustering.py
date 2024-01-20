from Clustering.Kmeans import Kmeans
from Clustering.FCM import FCM
from Clustering.MnistDR import MnistDR
from typing import Iterable
import plotly.graph_objects as go
from plotly.offline import plot
from plotly.subplots import make_subplots
import sys
import os
base_folder = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".."))
if base_folder not in sys.path:
    sys.path.append(base_folder)
if True:    
    from Datasets.Datasets import Dataset


def add_subplot(fig: go.Figure, result: list, col: int) -> None:
    for rec in result:
        name = rec[0][-1]
        fig.add_trace(
            go.Scatter(
                name=name,
                x=[t[0] for t in rec],
                y=[t[1] for t in rec],
                mode="lines+markers",
            ),
            row=1,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                name=name,
                x=[t[0] for t in rec],
                y=[t[2] for t in rec],
                mode="lines+markers",
            ),
            row=2,
            col=col,
        )


def plot_result(result: list, title: str, subplot_titles: list[str]):
    fig = make_subplots(rows=2, cols=len(subplot_titles),
                        shared_xaxes=True, shared_yaxes=True, subplot_titles=subplot_titles,
                        horizontal_spacing=0.02, vertical_spacing=0.02)
    for i in range(len(subplot_titles)):
        add_subplot(fig, result[i], i+1)
    fig.update_layout(
        width=1200,
        height=1200,
        title=title,
    )
    fig.update_yaxes(title_text="Purity", row=1, col=1)
    fig.update_yaxes(title_text="Rand Index", row=2, col=1)
    plot(fig, filename="./"+title+".html")
    pass


def run_test_kmeans(data: Dataset.Subset, n: int = 16, title: str = "") -> None:
    result = [[Kmeans(data, measure="Euclidean").iterate(n, print_tmps=False),
              Kmeans(data, measure="Manhatton").iterate(n, print_tmps=False), ],
              [Kmeans(data, measure="Euclidean", method="Random").iterate(n, print_tmps=False),
              Kmeans(data, measure="Manhatton", method="Random").iterate(n, print_tmps=False), ],]
    plot_result(result, title, ["K-Means++", "Random Initializing",])


def run_test_fcm(data: Dataset.Subset | Dataset.TensorSubset, n: int = 16, list_b: Iterable | None = None, title: str = "") -> None:
    result = [[FCM(data, b).iterate(n, print_tmps=False) for b in list_b],
              [FCM(data, b, method="Random").iterate(n, print_tmps=False) for b in list_b],]
    plot_result(result, title, ["K-Means++", "Random Initializing",])


if __name__ == "__main__":
    from Datasets.Datasets import Iris, Sonar, Mnist
    os.chdir(os.path.abspath(os.path.dirname(__file__)))
    iris = Iris(K=1, split_point=1)
    sonar = Sonar(K=1, split_point=1)
    mnist = Mnist(K=1).test
    mnist_PCA = mnist.toTensor().PCA(100)
    mnist_DR = MnistDR()(mnist)
    
    run_test_kmeans(iris.train, 16, title="kmeans on iris")
    run_test_kmeans(sonar.train, 32, title="kmeans on sonar")
    run_test_kmeans(sonar.train.toTensor().PCA(10), 32,
                    title="kmeans on sonar with PCA")
    run_test_kmeans(mnist, 32, title="kmeans on mnist")
    run_test_kmeans(mnist_PCA, 32, title="kmeans on mnist with PCA")
    run_test_fcm(iris.train, 16,
                 list_b=[i/2 for i in range(3, 8)], title="fcm on iris")
    run_test_fcm(sonar.train, 32,
                 list_b=[i/2 for i in range(3, 8)], title="fcm on sonar")
    run_test_fcm(sonar.train.toTensor().PCA(10), 32,
                 list_b=[i/2 for i in range(3, 8)], title="fcm on sonar with PCA")
    run_test_fcm(mnist, 32,
                 list_b=[i for i in (1.125, 1.25, 1.375, 1.5, 2, 3, 4, 5, 6, 7)], title="fcm on mnist")
    run_test_fcm(mnist_PCA, 32,
                 list_b=[i for i in (1.125, 1.25, 1.375, 1.5, 2, 3, 4, 5, 6, 7)], title="fcm on mnist with PCA")
    run_test_fcm(mnist_DR, 32,
                 list_b=[i for i in (1.125, 1.25, 1.375, 1.5, 2, 3, 4, 5, 6, 7)], title="fcm on mnist with Dimensionality Reduction")
    run_test_fcm(mnist, 32,
                 list_b=[i for i in (1.18, 1.181, 1.182, 1.183, 1.184, 1.185,)], title="fcm b tuning on mnist")
    run_test_fcm(mnist_PCA, 32,
                 list_b=[i for i in (1.18, 1.181, 1.182, 1.183, 1.184, 1.185,)], title="fcm b tuning on mnist with PCA")
    run_test_fcm(mnist_DR, 32,
                 list_b=[i for i in (1.18, 1.181, 1.182, 1.183, 1.184, 1.185,)], title="fcm b tuning on mnist with Dimensionality Reduction")
