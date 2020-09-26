from torch import nn
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn,optim
from torch.utils.data import Dataset, DataLoader


class Data(Dataset):
    """This class simply loads the previously stored data according to the specified parameters, creating a Pytorch
    Dataset."""
    # Constructor
    def __init__(self, tickers, start='2014-01-01', end='2018-01-01', interval='1d', n_series=20, T_pred=10, n_cols=40, n_rows=30, T_space=10, train=True):

        self.folder = './' + ''.join(tickers) + '_start' + start + '_end' + end + '_int' + interval + \
                      '/case' + str(n_series) + '_' + str(T_pred) + '_' + str(n_cols) + '_' + str(n_rows) + '_' + str(T_space)

        if train:
            self.x = np.load(self.folder + '/Xtrain.npy')
            self.y = np.load(self.folder + '/Ytrain.npy')
        else:
            self.x = np.load(self.folder + '/Xtest.npy')
            self.y = np.load(self.folder + '/Ytest.npy')

        # Shape of X: (Number of datasamples, Number of tickers, Number of rows, Number of columns)
        # Shape of Y: (Number of datasamples, Number of tickers)
        self.len = self.x.shape[0]

    # Getter
    def __getitem__(self, index):
        return self.x[index], self.y[index]

    # Get Length
    def __len__(self):
        return self.len


class CNN(nn.Module):
    # Contructor
    def __init__(self, n_cols, n_rows, tickers, out_1=16, out_2=32):
        """This class simply generates the CNN in charge of modelling the predicted stock price. Special care is taken
        in order to ensure that no matter what the parameters in the binarization of the plot are, the Convolutional
        NN is capable of properly operating."""
        super(CNN, self).__init__()

        kernel_1 = tuple(map(int, [n_rows / 5, n_cols / 5]))
        padd_1, size1 = self.find_padding(kernel_1, (1, 1), (1, 1), (n_rows, n_cols))
        self.cnn1 = nn.Conv2d(in_channels=len(tickers), out_channels=out_1, kernel_size=kernel_1, padding=padd_1)
        self.relu1 = nn.ReLU()

        kernel_2 = (2, 2)
        padd_2, size2 = self.find_padding(kernel_2, (0, 0), (1, 1), size1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=kernel_2)

        kernel_3 = tuple(map(int, [size2[0] / 5, size2[1] / 5]))
        padd_3, size3 = self.find_padding(kernel_3, (0, 0), (1, 1), size2)
        self.cnn2 = nn.Conv2d(in_channels=out_1, out_channels=out_2, kernel_size=kernel_3, padding=padd_3)
        self.relu2 = nn.ReLU()

        kernel_4 = (2, 2)
        padd_4, size4 = self.find_padding(kernel_4, (0, 0), (1, 1), size3)
        self.maxpool2 = nn.MaxPool2d(kernel_size=kernel_4)
        self.fc1 = nn.Linear(out_2 * np.prod(size4), 2)

    # Prediction
    def forward(self, x):
        out = self.cnn1(x)
        out = self.relu1(out)
        out = self.maxpool1(out)
        out = self.cnn2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        return out

    @staticmethod
    def find_padding(kernel, padd0, stride, input):
        padd = []
        output = []
        for i, k in enumerate(kernel):
            o = 1.34
            p = padd0[i] - 1
            while int(o) != o:
                p += 1
                o = (input[i] + 2 * padd0[i] - k) / stride[i] + 1
            padd.append(p)
            output.append(o)
        return tuple(padd), tuple(output)

class Trainer:
    def __init__(self, tickers, start='2014-01-01', end='2018-01-01', interval='1d', n_series=20, T_pred=10, n_cols=40, n_rows=30, T_space=10):
        self.tickers = tickers

        self.train_data = Data(tickers, start, end, interval, n_series, T_pred, n_cols, n_rows, T_space)
        self.test_data = Data(tickers, start, end, interval, n_series, T_pred, n_cols, n_rows, T_space, train=False)

        self.model = CNN(n_cols, n_rows, tickers)

        self.criterion = nn.CrossEntropyLoss()

        self.learning_rate = 0.1
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_data, batch_size=75, shuffle=True)
        validation_loader = torch.utils.data.DataLoader(dataset=self.test_data, batch_size=40, shuffle=True)
