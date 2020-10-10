import numpy as np
import matplotlib.pyplot as plt
from trainModel import Trainer

class ParametricStudy:
    def __init__(self, tickers, predict, start, end, interval, n_series=20, T_pred=20, n_cols=30, n_rows=30, T_space=10):
        self.parameters = {'tickers': tickers, 'predict': predict, 'start': start, 'end': end, 'interval': interval,
                           'n_series': n_series, 'T_pred': T_pred, 'n_cols': n_cols, 'n_rows': n_rows, 'T_space': T_space}
        self.tochange = []
        self.trainers = {}
        for k, v in self.parameters.items():
            if isinstance(v, list) and k != 'tickers' and k != 'n_rows':
                self.tochange.append(k)
                self.trainers[k] = []

        for t in self.tochange:
            self.vary_parameter(t)

    def vary_parameter(self, key, plot=True):
        values = self.parameters[key]
        for i, v in enumerate(values):
            case = self.parameters.copy()
            case[key] = v
            if key == 'n_cols':
                case['n_rows'] = v
            t = Trainer(**case)
            t.train(plot=False)
            self.trainers[key].append(t)

        if plot:
            self.plot_evolution(key)

    def plot_evolution(self, key):
        fig, ax = plt.subplots(1, 2)
        fig.set_size_inches(24, 8)
        # fig.tight_layout()

        ax[0].set_xlabel('Epochs [-]', fontsize=24)
        ax[0].grid(True)
        ax[0].set_ylabel('Loss [-]', fontsize=20)
        ax[0].tick_params(axis='both', labelsize=18)
        ax[0].set_title('Evolution of Training Loss with ' + key, fontsize=24)
        # ax[0].plot(self.loss)

        ax[1].set_xlabel('Epochs [-]', fontsize=24)
        ax[1].grid(True)
        ax[1].set_ylabel('Model Accuracy [-]', fontsize=20)
        ax[1].tick_params(axis='both', labelsize=18)
        ax[1].set_title('Evolution of Model Accuracy with ' + key, fontsize=24)
        # ax[1].plot(self.accuracy)

        for i, t in enumerate(self.trainers[key]):
            lab = key + ' = ' + str(self.parameters[key][i])
            ax[0].plot(t.loss, label=lab)
            ax[1].plot(t.accuracy, label=lab)

        ax[0].legend(fontsize=20)
        ax[1].legend(fontsize=20, loc=3)



# ps = ParametricStudy(['CL=F', 'GC=F', '^GSPC', '^IXIC', '^FTSE', '^TNX'], predict=5, start='2000-01-01', end='2010-01-01', interval='1d')
tickers = ['GOOG', 'MSFT', 'AAPL', 'AMZN', 'MA', 'V', 'TSLA', 'BABA', 'JD', 'NTES', 'NVDA', 'ZLDSF', 'CRM', 'AMGN', 'HON', 'AMD', 'KL', 'SHOP', 'RNG']
start = '2020-08-05'
end = '2020-10-03'
interval = '5m'
ps = ParametricStudy(tickers, predict=[i for i in range(len(tickers))], start=start, end=end, interval=interval)
plt.show()