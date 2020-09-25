import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import os

class LoadData:
    """This class downloads and processes all the specified stock values, outputting a csv file with the unprocessed
    price time series as well as with the time series converted into a binary mask with which to operate with a Conv2D NN"""
    def __init__(self, tickers, start='2014-01-01', end='2018-01-01', interval='1d'):
        self.tickers = tickers
        self.start = start
        self.end = end
        self.interval = interval

        self.folder = './' + ''.join(self.tickers) + '_start' + self.start + '_end' + self.end + '_int' + self.interval
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

    def download(self):
        """The main goal of this function is to download all stock price data of the specified tickers for the specified
        time frame with the specified interval, and save it properly in a CSV file"""
        self.unprocessed = yf.download(self.tickers, start=self.start, end=self.end, interval=self.interval)['Adj Close']
        #each row is represents a time instant, each column a ticker

        self.unprocessed.to_csv(path_or_buf=self.folder + '/UnprocessedData.csv', header=True)

    def process(self, n_series=20, T_pred=10, n_cols=40, n_rows=30, T_space=10, plot=True):
        """This is the bulk of the processing task. Here, the time series of each evaluated stock value is processed,
        transformed into datasamples made up of binary images of the input price series as well as an output boolean
        indicating if the price rises or falls after the prediction time"""
        self.time = self.unprocessed.index.tolist()
        self.Xordered = np.zeros((1, len(self.tickers), n_rows, n_cols))
        self.Yordered = np.zeros((1, len(self.tickers)))

        T = 0
        while max(T, T + n_series + T_pred) < len(self.time):
            data = self.unprocessed.iloc[T:T + n_series]
            pred = self.unprocessed.iloc[T + n_series + T_pred]

            x_i = np.zeros((1, len(self.tickers), n_rows, n_cols))
            y_i = np.zeros((1, len(self.tickers)))
            for i, tick in enumerate(self.tickers):
                prices = data[tick].values
                image = self.binary_image(prices, n_cols, n_rows)
                x_i[:, i, :, :] = image
                self.Xordered = np.concatenate((self.Xordered, x_i), axis=0)

                y = (pred[tick] > prices[-1]) * 1
                y_i[:, i] = y
                self.Yordered = np.concatenate((self.Yordered, y_i), axis=0)

                if plot and i == 0:
                    self.plot_image(prices, image)
                    plot = False
            T += T_space

        self.Xordered = self.Xordered[1:]
        self.Yordered = self.Yordered[1:]
        np.save(self.folder + '/Xordered.npy', self.Xordered)
        np.save(self.folder + '/Yordered.npy', self.Yordered)


    @staticmethod
    def binary_image(data, n_cols, n_rows):
        """This simple function is in charge of the conversion of a time series of data into its binary image equivalent"""
        X = np.linspace(0, len(data), n_cols)
        Y = np.linspace(np.max(data), np.min(data), n_rows)
        image = np.zeros((1, 1, n_rows, n_cols))

        for t in range(len(data)):
            i = np.argmin(np.abs(X - t))
            j = np.argmin(np.abs(Y - data[t]))
            image[:, :, j, i] = 1
        return image

    @staticmethod
    def plot_image(data, image):
        fig, ax = plt.subplots(1, 2)
        fig.set_size_inches(16, 8)
        # fig.tight_layout()

        ax[0].set_xlabel('t [-]', fontsize=24)
        ax[0].grid(True)
        ax[0].set_ylabel('Closing Stock Price [$]', fontsize=20)
        ax[0].tick_params(axis='both', labelsize=18)
        ax[0].set_title('Unprocessed Stock Price', fontsize=24)
        t = np.arange(len(data)) + 1
        ax[0].plot(t, data)

        ax[1].imshow(image[0, 0, :, :], cmap=plt.cm.binary)
        ax[1].set_title('Resulting Binary Image', fontsize=24)
        ax[1].set_xlabel('t [-]', fontsize=24)
        ax[1].tick_params(axis='both', labelsize=18)

    def cut_and_shuffle(self):
        """This simple function separates the original ordered dataset into a training and a testing datasets, after having
        shuffled it appropriately."""
        index = np.random.permutation(self.Xordered.shape[0])
        self.Xtrain = self.Xordered[index[:int(len(index) * 0.8)]]
        self.Ytrain = self.Yordered[index[:int(len(index) * 0.8)]]
        self.Xtest = self.Xordered[index[int(len(index) * 0.8):]]
        self.Ytest = self.Yordered[index[int(len(index) * 0.8):]]

        print(str(int(len(index) * 0.8)) + ' data samples in training dataset, ' + str(len(index) - int(len(index) * 0.8)) + ' data samples in test dataset.')
        np.save(self.folder + '/Xtrain.npy', self.Xtrain)
        np.save(self.folder + '/Ytrain.npy', self.Ytrain)
        np.save(self.folder + '/Xtest.npy', self.Xtest)
        np.save(self.folder + '/Ytest.npy', self.Ytest)


ld = LoadData(['AAPL', 'AMZN'])
ld.download()
ld.process()
ld.cut_and_shuffle()
plt.show()

