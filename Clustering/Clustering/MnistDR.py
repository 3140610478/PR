from plotly.offline import plot
import plotly.graph_objects as go
import os
import sys
from collections import OrderedDict
import numpy as np
import torch
from PIL import Image
from torch import nn
base_folder = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../.."))
if base_folder not in sys.path:
    sys.path.append(base_folder)
if True:
    from Datasets.Datasets import Mnist, Dataset


mnist = Mnist(K=1).test
train, test = Dataset.KFold(mnist.feature, mnist.label, mnist.C, 5).split()


class MnistDR():
    class ScalarMul(nn.Module):
        device = torch.device("cuda")\
            if torch.cuda.is_available() else torch.device("cpu")

        def __init__(self, C):
            super().__init__()
            self.C = torch.Tensor((C,)).to(self.device)
            if self.C.shape != torch.Size((1,)):
                raise ValueError("C for ScalarMul must be a scalar.")

        def forward(self, x: torch.Tensor):
            return self.C * x

    class View(nn.Module):
        def __init__(self, shape):
            super().__init__()
            self.SHAPE = torch.Size(shape)

        def forward(self, x: torch.Tensor):
            return x.view(x.shape[:1]+self.SHAPE)

    class MnistDataset(torch.utils.data.Dataset):
        def __init__(self, data):
            super().__init__()
            self.LEN = data.LEN
            self.data = data.feature

        def __len__(self):
            return self.LEN

        def __getitem__(self, index: int):
            x = self.data[index].astype(np.float32)
            return x.copy(), x.copy()
    device = torch.device("cuda")\
        if torch.cuda.is_available() else torch.device("cpu")
    path = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "./MnistDR/"))
    extractor_path = os.path.abspath(
        os.path.join(path, "./extractor.model"))
    reconstructor_path = os.path.abspath(
        os.path.join(path, "./reconstructor.model"))

    def __init__(self, train=train, test=test, shape: tuple[int] = (1, 28, 28), batch_size: int = 64, new_model: bool = False):
        if not new_model and os.path.exists(self.extractor_path) and os.path.exists(self.reconstructor_path):
            self.extractor = torch.load(self.extractor_path)
            self.reconstructor = torch.load(self.reconstructor_path)
        else:
            train_loader = torch.utils.data.DataLoader(MnistDR.MnistDataset(train),
                                                       batch_size=batch_size, shuffle=True, pin_memory=True)
            test_loader = torch.utils.data.DataLoader(MnistDR.MnistDataset(test),
                                                      batch_size=batch_size, shuffle=True, pin_memory=True)

            extractor = nn.Sequential(
                MnistDR.View(shape),
                nn.Conv2d(1, 16, kernel_size=3, stride=1, padding="same"),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(16, 64, kernel_size=3, stride=1, padding="same"),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Flatten(),
                nn.Linear(64*7*7, 1024),
                nn.ReLU(),
                nn.Linear(1024, 256),
                nn.ReLU(),
                nn.Linear(256, 64),
            )
            reconstructor = nn.Sequential(
                nn.Linear(64, 256),
                nn.ReLU(),
                nn.Linear(256, 1024),
                nn.ReLU(),
                nn.Linear(1024, 28*28),
                nn.Sigmoid(),
                MnistDR.ScalarMul(255),
            )
            model = nn.Sequential(OrderedDict([
                ("extractor", extractor),
                ("reconstructor", reconstructor),
            ]))
            model = model.to(self.device)

            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)

            message = '[{:0>4d}] train loss :{:.3f},\ttest loss :{:.3f}'
            for epoch in range(32):
                train_loss, test_loss = [], []
                for sample in train_loader:
                    x, y = sample
                    x, y = x.to(self.device), y.to(self.device)

                    optimizer.zero_grad()

                    h = model(x)
                    loss = criterion(h, y)
                    loss.backward()
                    optimizer.step()
                    train_loss.append(float(loss))
                for sample in test_loader:
                    x, y = sample
                    x, y = x.to(self.device), y.to(self.device)

                    h = model(x)
                    loss = criterion(h, y)
                    test_loss.append(float(loss))
                print(message.format(
                    epoch+1, np.mean(train_loss), np.mean(test_loss)))

            print('Finished Training')
            self.extractor, self.reconstructor = extractor, reconstructor
            if not os.path.exists(self.path):
                os.mkdir(self.path)
            torch.save(extractor, self.extractor_path)
            torch.save(reconstructor, self.reconstructor_path)
            # # showing feature map
            # for sample in test_loader:
            #     x, y = sample
            #     x, y = x.to(self.device), y.to(self.device)

            #     dr = self.extractor(x)
            #     h = self.reconstructor(dr)
            #     dr, h, y = dr[0], h[0], y[0]
            #     pic_dr = Image.fromarray(dr.detach().cpu().numpy().astype("uint8").reshape((8, 8)), "L")
            #     pic_h = Image.fromarray(h.detach().cpu().numpy().astype("uint8").reshape((28, 28)), "L")
            #     pic_y = Image.fromarray(y.detach().cpu().numpy().astype("uint8").reshape((28, 28)), "L")
            #     pic_dr.show()
            #     pic_h.show()
            #     pic_y.show()

    def __call__(self, data: Dataset.Subset):
        feature = data.feature
        feature = torch.stack([torch.Tensor(i) for i in feature]).to(self.device)
        feature = self.extractor(feature)
        feature = [i.detach().cpu().numpy() for i in feature]
        return Dataset.Subset(feature, data.label, data.C)


if __name__ == "__main__":
    os.chdir(os.path.abspath(os.path.dirname(__file__)))
    mnist = Mnist().test
    DR = MnistDR(new_model=True)
    a = DR(mnist)
