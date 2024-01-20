import plotly.graph_objects as go
from plotly.offline import plot
from plotly.subplots import make_subplots
from sklearn import svm
import sys
import os
import numpy as np
base_folder = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".."))
if base_folder not in sys.path:
    sys.path.append(base_folder)
if True:
    from Datasets.Datasets import Dataset

def test(data: Dataset, notation=""):
    name = data.__class__.__name__
    print(f"Test on {name}:")
    kfold, test = data.train, data.test
    acc_best, C_best, kernel_best, classifier_best = 0, 1, None, None
    fig = make_subplots(rows=1, cols=2, subplot_titles=("kernel test", "C test"),
                        shared_yaxes=True, horizontal_spacing=0.02)
    for kernel in ("linear", "poly", "rbf", "sigmoid"):
        print(f"kernel: {kernel}")
        acc_rec = []
        for _ in range(kfold.K):
            train, validation = kfold.split()
            kfold.reshuffle()
            classifier = svm.SVC(kernel=kernel).fit(
                train.feature, np.argmax(train.label, axis=1))
            predict = np.rint(classifier.predict(
                validation.feature)).astype(np.int32)
            actual = np.argmax(validation.label, axis=1)
            acc = 1 - np.count_nonzero(predict - actual) / len(validation)
            acc_rec.append(acc)
            print(acc)
        acc = np.mean(acc_rec)
        if (acc > acc_best):
            acc_best, kernel_best, classifier_best = acc, kernel, classifier
        fig.add_trace(
            go.Scatter(
                name=f"{kernel} kernel",
                x=list(range(kfold.K)),
                y=acc_rec,
                mode="lines+markers",
            ),
            row=1,
            col=1,
        )
    print(f"best kernel: {kernel_best}, best accuracy: {acc_best}\n")
    for C in (0.25, 0.5, 1, 2, 4):
        print(f"C = {C}")
        acc_rec = []
        for _ in range(kfold.K):
            train, validation = kfold.split()
            kfold.reshuffle()
            classifier = svm.SVC(C=C, kernel=kernel_best).fit(
                train.feature, np.argmax(train.label, axis=1))
            predict = np.rint(classifier.predict(
                validation.feature)).astype(np.int32)
            actual = np.argmax(validation.label, axis=1)
            acc = 1 - np.count_nonzero(predict - actual) / len(validation)
            acc_rec.append(acc)
            print(acc)
        acc = np.mean(acc_rec)
        if (acc > acc_best):
            acc_best, C_best, classifier_best = acc, C, classifier
        fig.add_trace(
            go.Scatter(
                name=f"C = {C}",
                x=list(range(kfold.K)),
                y=acc_rec,
                mode="lines+markers",
            ),
            row=1,
            col=2,
        )
    print(f"best C: {C_best}, best accuracy: {acc_best}\n")

    predict_final = np.rint(classifier_best.predict(
        test.feature)).astype(np.int32)
    actual_final = np.argmax(test.label, axis=1)
    acc_final = 1 - np.count_nonzero(predict_final - actual_final) / len(test)
    fig.update_layout(
        title=f"SVM on {name} ({notation}), best kernel: {kernel_best}, best C = {C_best}, accuracy on test = {acc_final}",
        width=1280,
        height=720,
    )
    plot(fig, filename=f"SVM_Classifier_for_{name}_{notation}.html")


if __name__ == "__main__":
    from Datasets.Datasets import Iris, Sonar, Mnist
    os.chdir(os.path.abspath(os.path.dirname(__file__)))
    iris_ = Iris(K=6, split_point=0.8)
    sonar_ = Sonar(K=6, split_point=0.8)
    iris = Iris(K=6, split_point=0.8, normalize=True)
    sonar = Sonar(K=6, split_point=0.8, normalize=True)
    test(iris_, "not_normalized")
    test(sonar_, "not_normalized")
    test(iris)
    test(sonar)
