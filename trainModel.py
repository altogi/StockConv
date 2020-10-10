from torch import nn
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from LoadData import LoadData
import pandas as pd

# This script here contains on the one hand the class in charge of defining a PyTorch dataset based on data stored with
# LoadData.py, the class defining the CNN which interprets the data in order to carry out a prognosis, and a Trainer class
# which carries out the training the CNN based on the specified dataset.

class Data(Dataset):
    """This class simply loads the previously stored data according to the specified parameters, creating a Pytorch
    Dataset."""
    # Constructor
    def __init__(self, tickers, start='2014-01-01', end='2018-01-01', interval='1d', n_series=20, T_pred=10, n_cols=30, n_rows=30, T_space=10, train=True):

        self.folder = './' + ''.join(tickers) + '_start' + start + '_end' + end + '_int' + interval + \
                      '/case' + str(n_series) + '_' + str(T_pred) + '_' + str(n_cols) + '_' + str(n_rows) + '_' + str(T_space)

        try:
            self.original = np.load(self.folder + '/original.npy')
            if train:
                self.x = np.load(self.folder + '/Xtrain.npy')
                self.y = np.load(self.folder + '/Ytrain.npy')
            else:
                self.x = np.load(self.folder + '/Xtest.npy')
                self.y = np.load(self.folder + '/Ytest.npy')
        except:
            ld = LoadData(tickers, start, end, interval)
            try:
                ld.unprocessed = pd.read_csv('./' + ''.join(tickers) + '_start' + start + '_end' + end + '_int' + interval + '/UnprocessedData.csv')
            except:
                print('DOWNLOADING DATA')
                ld.download()
            print('PROCESSING DATA')
            ld.process(n_series, T_pred, n_cols, n_rows, T_space, plot=True)
            ld.cut_and_shuffle()

            if train:
                self.x = ld.Xtrain
                self.y = ld.Ytrain
            else:
                self.x = ld.Xtest
                self.y = ld.Ytest
            self.original = ld.original

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
        padd_1, size1, kernel_1 = self.find_padding(kernel_1, (1, 1), (1, 1), (n_rows, n_cols))
        self.cnn1 = nn.Conv2d(in_channels=len(tickers), out_channels=out_1, kernel_size=kernel_1, padding=padd_1)
        self.relu1 = nn.ReLU()

        kernel_2 = (2, 2)
        padd_2, size2, kernel_2 = self.find_padding(kernel_2, (0, 0), kernel_2, size1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=kernel_2)

        kernel_3 = tuple(map(int, [size2[0] / 4, size2[1] / 4]))
        padd_3, size3, kernel_3 = self.find_padding(kernel_3, (0, 0), (1, 1), size2)
        self.cnn2 = nn.Conv2d(in_channels=out_1, out_channels=out_2, kernel_size=kernel_3, padding=padd_3)
        self.relu2 = nn.ReLU()

        kernel_4 = (2, 2)
        padd_4, size4, kernel_4 = self.find_padding(kernel_4, (0, 0), kernel_4, size3)
        self.maxpool2 = nn.MaxPool2d(kernel_size=kernel_4)
        self.fc1 = nn.Linear(int(out_2 * np.prod(size4)), 2)

        self.size = [size1, size2, size3, size4]

    # Prediction
    def forward(self, x):
        x = x.float()
        self.actual_size = []
        out = self.cnn1(x)
        self.actual_size.append(tuple(out.size()))
        out = self.relu1(out)
        out = self.maxpool1(out)
        self.actual_size.append(tuple(out.size()))
        out = self.cnn2(out)
        self.actual_size.append(tuple(out.size()))
        out = self.relu2(out)
        out = self.maxpool2(out)
        self.actual_size.append(tuple(out.size()))
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        return out

    @staticmethod
    def find_padding(kernel, padd0, stride, input):
        padd = []
        output = []
        kernel = list(kernel)
        stride = list(stride)
        for i, k in enumerate(kernel):
            o = 1.34
            p = padd0[i] - 1
            while int(o) != o:
                p += 1
                o = ((input[i] + 2 * p - k) / (stride[i])) + 1

                if p > 5:
                    k += 1
                    stride[i] = k
                    p = padd0[i] - 1
            padd.append(p)
            output.append(o)
            kernel[i] = k
        return tuple(padd), tuple(output), tuple(kernel)

class Trainer:
    """This class containes the bulk of the model's training, since it creates the dataset and the CNN and iterates through
    the specified epochs in order to train the model."""
    def __init__(self, tickers, predict=8, start='2014-01-01', end='2018-01-01', interval='1d', n_series=30, T_pred=15, n_cols=30, n_rows=30, T_space=15):
        self.tickers = tickers
        self.predict = predict #Predict indexes tickers, indicating which of the securities to model/predict
        self.n_series = n_series
        self.T_pred = T_pred

        self.folder = './' + ''.join(tickers) + '_start' + start + '_end' + end + '_int' + interval + \
                      '/case' + str(n_series) + '_' + str(T_pred) + '_' + str(n_cols) + '_' + str(n_rows) + '_' + str(
            T_space)

        self.train_data = Data(tickers, start, end, interval, n_series, T_pred, n_cols, n_rows, T_space)
        self.test_data = Data(tickers, start, end, interval, n_series, T_pred, n_cols, n_rows, T_space, train=False)

        self.model = CNN(n_cols, n_rows, tickers)

        self.criterion = nn.CrossEntropyLoss()

        self.learning_rate = 0.1
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_data, batch_size=75, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(dataset=self.test_data, batch_size=500, shuffle=True)

        self.loss = []
        self.accuracy = []

    def epoch(self):
        for x, y in self.train_loader:
            y = y[:, self.predict]
            self.optimizer.zero_grad()
            z = self.model(x)
            loss = self.criterion(z, y.long())
            loss.backward()
            self.optimizer.step()

        correct = 0
        # perform a prediction on the test  data
        for x_test, y_test in self.test_loader:
            y_test = y_test[:, self.predict]
            z = self.model(x_test)
            _, yhat = torch.max(z.data, 1)
            correct += (yhat == y_test).sum().item()
        accuracy = correct / len(self.test_data)
        self.accuracy.append(accuracy)
        self.loss.append(loss.item())

    def train(self, epochs=1000, plot=True):
        print('MODEL TRAINING')
        self.model = self.model.float()
        for e in range(epochs):
            if e % 10 == 1:
                print('Epoch #' + str(e) + ' - Loss = ' + str(round(self.loss[-1], 2)) + '; Model Accuracy = ' + str(round(self.accuracy[-1], 2)) + ';')
            self.epoch()

        if plot:
            fig, ax = plt.subplots(1, 2)
            fig.set_size_inches(16, 8)
            # fig.tight_layout()

            ax[0].set_xlabel('Epochs [-]', fontsize=24)
            ax[0].grid(True)
            ax[0].set_ylabel('Loss [-]', fontsize=20)
            ax[0].tick_params(axis='both', labelsize=18)
            ax[0].set_title('Evolution of Model Loss with Training', fontsize=24)
            ax[0].plot(self.loss)

            ax[1].set_xlabel('Epochs [-]', fontsize=24)
            ax[1].grid(True)
            ax[1].set_ylabel('Model Accuracy [-]', fontsize=20)
            ax[1].tick_params(axis='both', labelsize=18)
            ax[1].set_title('Evolution of Model Accuracy with Training', fontsize=24)
            ax[1].plot(self.accuracy)
            fig.savefig(self.folder + '/Training.jpg')

        torch.save(self.model.state_dict(), self.folder + '/TrainedModel')

    def visualize_execution(self):
        data = np.zeros((1, 2))
        correct = np.zeros((1, 2))
        incorrect = np.zeros((1, 2))
        self.loader = torch.utils.data.DataLoader(dataset=self.test_data, batch_size=1, shuffle=False)
        for i, r in enumerate(self.loader):
            x = r[0]
            y = r[1]
            y = y[:, self.predict]
            z = self.model(x)
            _, yhat = torch.max(z.data, 1)
            series = self.test_data.original[i][:, self.predict]

            index = np.arange(len(series)) + data[-1, 0]

            newD = np.vstack((index, series))
            data = np.vstack((data, np.transpose(newD)))

            if yhat == y:
                correct = np.vstack((correct, np.array([index[-1], series[-1]])))
            else:
                incorrect = np.vstack((incorrect, np.array([index[-1], series[-1]])))

        fig, ax = plt.subplots()
        fig.set_size_inches(30, 8)
        # fig.tight_layout()

        ax.set_xlabel('t [-]', fontsize=24)
        ax.grid(True)
        ax.set_ylabel(self.tickers[self.predict] + ' Price [-]', fontsize=20)
        ax.tick_params(axis='both', labelsize=18)
        ax.set_title('Visualization of Model Performance', fontsize=24)
        ax.plot(data[:, 0], data[:, 1])
        ax.scatter(correct[:, 0], correct[:, 1], marker='*', c='green', s=75)
        ax.scatter(incorrect[:, 0], incorrect[:, 1], marker='X', c='red', s=75)


# t = Trainer(['CL=F', 'GC=F', '^GSPC', '^IXIC', '^FTSE', '^TNX'], start='2000-01-01', end='2010-01-01', n_series=20, T_pred=20, T_space=10, predict=5)
tickers = ['GOOG', 'MSFT', 'AAPL', 'AMZN', 'MA', 'V', 'TSLA', 'BABA', 'JD', 'NTES', 'NVDA', 'ZLDSF', 'CRM', 'AMGN', 'HON', 'AMD', 'KL', 'SHOP', 'RNG']
start = '2020-08-05'
end = '2020-10-03'
interval = '5m'
t = Trainer(tickers, 2, start, end, interval)
t.train()
t.visualize_execution()
plt.show()