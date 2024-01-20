import os
import random
import numpy as np
import torch
from typing import Iterable, overload
from abc import ABC, abstractmethod


# ensuring labels to be columns of an eye matrix
class Dataset(ABC):
    class Subset():
        def __init__(self,
                     feature: Iterable[np.ndarray[np.float64]],
                     label: Iterable[np.ndarray[np.float64]],
                     C: int | None = None):
            self.feature = [i.flatten() for i in feature]
            self.label = [i.flatten() for i in label]
            self.LEN = len(self.label)
            self.C = C if C is not None else self.label[0].shape[0]
            pass

        def __len__(self) -> int:
            return self.LEN

        def __getitem__(self, key: int):
            return self.feature[key], self.label[key]

        def __iter__(self):
            self.iterater_index = 0
            return self

        def __next__(self):
            if self.iterater_index < self.LEN:
                x, y = self.__getitem__(self.iterater_index)
                self.iterater_index += 1
                return x, y
            else:
                del self.iterater_index
                raise StopIteration

        def reshuffle(self) -> None:
            tmp = list(zip(self.feature, self.label))
            random.shuffle(tmp)
            self.feature, self.label = tuple(zip(*tmp))

        def __str__(self):
            feature = [i.squeeze() for i in self.feature]
            label = [i.squeeze() for i in self.label]
            l = list(zip(feature, label))
            return str(l)
        
        def toTensor(self):
            return Dataset.TensorSubset(self)

    class KFold(Subset):
        def __init__(self,
                     feature: Iterable[np.ndarray[np.float64]],
                     label: Iterable[np.ndarray[np.float64]],
                     C: int | None = None,
                     K: int = 4):
            super().__init__(feature, label)
            C = self.C
            self.K = K
            self._current = 0
            self._group = np.zeros((self.LEN), dtype=np.int64)

            r = np.zeros((C), dtype=np.int64)
            for i in range(self.LEN):
                index = np.argmax(self.label[i].squeeze())
                self._group[i] = r[index] % K
                r[index] = r[index] + 1

        def split(self):
            train_feature = [self.feature[i] for i in range(self.LEN)
                             if self._group[i] != self._current]
            train_label = [self.label[i] for i in range(self.LEN)
                           if self._group[i] != self._current]
            validation_feature = [self.feature[i] for i in range(self.LEN)
                                  if self._group[i] == self._current]
            validation_label = [self.label[i] for i in range(self.LEN)
                                if self._group[i] == self._current]
            train = Dataset.Subset(train_feature, train_label)
            validation = Dataset.Subset(validation_feature, validation_label)
            return train, validation

        def reshuffle(self) -> None:
            tmp = list(zip(self.feature, self.label, self._group))
            random.shuffle(tmp)
            self.feature, self.label, self._group = tuple(zip(*tmp))
            self._current = (self._current + 1) % self.K
            
    class TensorSubset:        
        device = torch.device("cuda") \
            if torch.cuda.is_available() else torch.device("cpu")
        def __init__(self, subset):
            if isinstance(subset, Dataset.Subset):
                self.LEN = len(subset)
                self.C = subset.C
                self.feature = torch.from_numpy(
                    np.stack(subset.feature).astype(np.float32)).to(self.device)
                SCALE = (torch.max(self.feature) - torch.min(self.feature)) / \
                    torch.prod(torch.as_tensor(
                        self.feature.shape[1:]), dim=0).to(self.device)
                self.feature /= SCALE
                self.label = torch.from_numpy(
                    np.stack(subset.label).astype(np.float32)).to(self.device)
                pass
            elif isinstance(subset, Dataset.TensorSubset):
                self.LEN, self.C = subset.LEN, subset.C
                self.feature, self.label = subset.feature.clone(), subset.label.clone()
            
        def __len__(self) -> int:
            return self.LEN
        
        def PCA(self, d: int):
            PCA = Dataset.TensorSubset(self)
            U, S, V = torch.pca_lowrank(self.feature, d)
            PCA.feature = self.feature @  V
            return PCA

    @overload
    def __init__(self, train: KFold, test: Subset):
        pass

    @overload
    def __init__(self,
                 v_feature: Iterable[np.ndarray],
                 v_label: Iterable[np.ndarray],
                 K: int | None = 4,
                 split_point: float | None = 0.75):
        pass

    @abstractmethod
    def __init__(self,
                 train: KFold | Subset | None = None,
                 test: Subset | None = None,
                 v_feature: Iterable[np.ndarray] | None = None,
                 v_label: Iterable[np.ndarray] | None = None,
                 K: int | None = None,
                 split_point: float | None = None,
                 normalize: bool = False):
        if v_feature != None and v_label != None and train == None and test == None:
            if K == None:
                K = 4
            if split_point == None:
                split_point = 0.75
            v_feature, v_label = list(v_feature), list(v_label)
            tmp = list(zip(v_feature, v_label))
            np.random.shuffle(tmp)
            v_feature, v_label = tuple(zip(*tmp))
                
            C = v_label[0].size
            cnt = np.zeros((C), dtype=np.float64)
            for i in range(len(v_label)):
                index = np.argmax(v_label[i], axis=0)
                cnt[index] += 1
            cnt = (cnt * split_point).astype(np.int64)
            train_feature, test_feature, train_label, test_label = [], [], [], []
            for i in range(len(v_label)):
                index = np.argmax(v_label[i], axis=0)
                if cnt[index]:
                    cnt[index] -= 1
                    train_feature.append(v_feature[i])
                    train_label.append(v_label[i])
                else:
                    test_feature.append(v_feature[i])
                    test_label.append(v_label[i])
            if K != 1:
                train = self.KFold(train_feature, train_label, C, K)
            else:
                train = self.Subset(train_feature, train_label, C)
            test = self.Subset(test_feature, test_label, C)
        if isinstance(train, self.Subset) and isinstance(test, self.Subset):
            self.train = train
            self.test = test
            self._len = train.LEN + test.LEN
            self.C = self.train.C
        else:
            raise ValueError("Invalid argument for __init__ in Dataset.Subset")
        
        if normalize:
            mf, sf = np.mean(self.train.feature),  np.std(self.train.feature)
            self.MEAN, self.STD = mf, sf
            self.train.feature = [(f - mf) / sf for f in self.train.feature]
            self.test.feature = [(f - mf) / sf for f in self.test.feature]
        else:
            self.MEAN, self.STD = np.nan, np.nan

    def __str__(self):
        s = 'Dataset {0} of size {1}:\n'.format(
            self.__class__.__name__, self._len)
        for X, y in self:
            s += 'feature = ' + str(X) + ',\nlabel = ' + str(y) + '\n'
        return s
    
    def normalize(self, feature: list[np.ndarray]):
        return [(f - self.MEAN) / self.STD for f in feature]

            

class Iris(Dataset):
    Classes = {'Iris-setosa': np.array([[1, 0, 0]]).T,
               'Iris-versicolor': np.array([[0, 1, 0]]).T,
               'Iris-virginica': np.array([[0, 0, 1]]).T}

    def __init__(self, K: int | None = None, split_point: float | None = None, D: int | None = None, normalize: bool = False):
        dir = os.getcwd()
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        with open("Iris/iris.data") as f:
            v = [i.split(',') for i in f.readlines()]
            D_total = len(v[0]) - 1
            if D == None:
                D = D_total
            v_feature = [np.array(tuple(map(float, e[:D:]))
                                  ).reshape((D, 1)) for e in v]
            v_label = [self.Classes[e[-1].rstrip('\n')] for e in v]
        super().__init__(v_feature=v_feature, v_label=v_label, K=K, split_point=split_point, normalize=normalize)
        os.chdir(dir)


class Sonar(Dataset):
    # Rocks, Mines
    Classes = {'R': np.array([[1], [0]]), 'M': np.array([[0], [1]])}

    def __init__(self, K: int | None = None, split_point: float | None = None, D: int | None = None, normalize: bool = False):
        dir = os.getcwd()
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        with open("Sonar/sonar.csv") as f:
            v = [i.split(',') for i in f.readlines()]
            D_total = len(v[0]) - 1
            if D == None:
                D = D_total
            v_feature = [np.array(tuple(map(float, e[:D:]))
                                  ).reshape((D, 1)) for e in v]
            v_label = [self.Classes[e[-1].rstrip('\n')] for e in v]
        super().__init__(v_feature=v_feature, v_label=v_label, K=K, split_point=split_point, normalize=normalize)
        os.chdir(dir)


class Mnist(Dataset):
    Classes = {i: np.eye(10)[i].reshape(10, 1) for i in range(10)}

    @staticmethod
    def _LoadInt(x: tuple, size=4):
        return sum((x[i] << ((size - i - 1) * 8)) for i in range(len(x)))

    def __init__(self, K: int | None = None, normalize: bool = False):
        if K == None:
            K = 4
        dir = os.getcwd()
        os.chdir(os.path.join(os.path.dirname(
            os.path.abspath(__file__)), 'Mnist'))
        filenames = ['train-images.idx3-ubyte',
                     'train-labels.idx1-ubyte',
                     't10k-images.idx3-ubyte',
                     't10k-labels.idx1-ubyte',]
        l = []
        for fn in filenames:
            with open(fn, 'rb') as f:
                content = tuple(f.read())
                d = content[3]
                shape = tuple(self._LoadInt(content[i: i+4:])
                              for i in range(4, 4*d+1, 4))
                l.append(list(np.array(content[4*(d+1)::]).reshape(shape)))
        if K != 1:
            train = self.KFold(l[0], [self.Classes[i] for i in l[1]], K=K)
        else: 
            train = self.Subset(l[0], [self.Classes[i] for i in l[1]])
        test = self.Subset(l[2], [self.Classes[i] for i in l[3]])
        super().__init__(train=train, test=test, K=K, normalize=normalize)
        os.chdir(dir)


if __name__ == '__main__':
    os.chdir(os.path.abspath(os.path.dirname(__file__)))
    Iris()
    Sonar()
    Mnist(K=1).train.toTensor().PCA()
