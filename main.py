import itertools

import numpy as np
from scipy.stats import norm
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.graphics.tsaplots as sgt
from sklearn.model_selection import train_test_split
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
import pmdarima as pm
import statsmodels.api as sm

import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras.layers import Bidirectional, Dropout, Activation, Dense, LSTM

import yfinance as yf


def black_scholes_call(S, K, T, r, sigma):
    '''

    :param S: Asset price
    :param K: Strike price
    :param T: Time to maturity
    :param r: risk-free rate
    :param sigma: volatility
    :return: call price
    '''

    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    call = S * norm.cdf(d1) - norm.cdf(d2) * K * np.exp(-r * T)
    return call

def black_scholes_put(S, K, T, r, sigma):
    '''

    :param S: Asset price
    :param K: Strike price
    :param T: Time to maturity
    :param r: risk-free rate
    :param sigma: volatility
    :return: put price
    '''

    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    put = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(d1)
    return put

class timeseries():
    def __init__(self):
        try:
            self.msft = yf.Ticker("AAPL")
            self.df = self.msft.history(period='3y')
            self.df.dropna(inplace=True)
            self.api = True


        except:
            csvroute = input("Error API accebility please enter the neme f a csv file")

            self.df = pd.read_csv(csvroute, sep=",")
            self.df.dropna(inplace=True)






    def plotCallPutt(self):

        optdatte = self.msft.options[0]
        self.opt = self.msft.option_chain(optdatte)

        self.call = self.opt.calls
        self.put = self.opt.puts


        print(self.call.head(5))
        print(self.put.head(5))

    def printSerie(self):
        print(self.df)
        print(self.df.describe())
        self.df.info()

    def setDateformat(self):
        if not self.api:
            self.df["Date"] = pd.to_datetime(self.df["Date"])
            self.df.sort_values(by="Date", inplace=True)
            self.df.set_index("Date", inplace=True)

    def featureselection(self,feature=['Open', 'High', 'Low', 'Close']):
        self.df = self.df[feature]

    def ressampletimeseire(self,parameter="W"):
        self.df = self.df.resample(parameter).mean().dropna()

    def difference(self,row = "Open"):
        self.df['Diff'] = self.df[row].diff()

    def plotrow(self,row='Open',xlabel='Year',ylabel='Price',title="graph"):
        self.df[row].plot(figsize=(15, 5))
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.show()

    def logreturn(self,row = "Close"):
        self.df['Logreturn'] =  np.log(self.df[row] / self.df[row].shift(1)).dropna()

    def volatility(self,row = "Close"):
        self.df['volatility'] = self.df[row].rolling(window=2).std()

    def calculpvalue(self,row='Diff'):
        print("p-value",row, ": ",adfuller(self.df[row])[1])

    def plotPacf(self,row = "Diff",zeroValue=False,alphaValue= 0.05,title="PACF Plot for First Order Differenced Data" ):
        sgt.plot_pacf(self.df[row], lags=np.arange((len(self.df) - 1)/2), zero=zeroValue, alpha=alphaValue)
        plt.title(title)
        plt.show()

    def plotAcf(self,row = "Diff",zeroValue=False,alphaValue= 0.05,title = "ACF Plot for First Order Differenced Data"):
        sgt.plot_acf(self.df[row], lags=np.arange(len(self.df) - 1), zero=zeroValue, alpha=alphaValue)
        plt.title(title)
        plt.show()



    def splitTrainTest(self,test_size=0.2,row="Close"):
        self.train, self.test = train_test_split(self.df[row], test_size=test_size, shuffle=False)


    def AutoArimaPredict(self,row = "Close"):
        self.auto_arima = pm.auto_arima(self.train, stepwise=False, seasonal=True)
        print(self.auto_arima.summary())

        test = self.auto_arima.predict(n_periods=len(self.test))
        print( 'coucou')

        self.df['autoArima'] = [None] * (len(self.train))+ list(test)


        plt.plot(self.df[row], label=row)
        plt.plot(self.df['autoArima'], label="autoArima")
        plt.legend()
        plt.show()
        print('If this is a straight line its mean that auto arima think this a random step')




    def ArimaPredict(self,row = "Close"): # TODO reparer

        p = range(0, 6)
        d = range(1, 5)
        q = range(0, 7)
        pdq = list(itertools.product(p, d, q))
        least_MSE = 10000000000000000

        for param in pdq:
            try:
                    mod = ARIMA(self.train,
                                  order=param,
                                  )
                    results = mod.fit()

                    if (results.mse < least_MSE):
                        best_result = results
                        least_MSE = results.mse

            except:
                    break
                    continue


        print(best_result.summary())

        forecast_test = best_result.forecast(len(self.test))
        self.df['Arima'] = [None] * len(self.train) + list(forecast_test)

        plt.plot(self.df[row], label=row)
        plt.plot(self.df['Arima'], label="Arima")
        plt.legend()
        plt.show()

    def SarimaPredict(self,row = "Close"):
        # TODO calculer parametre

        p = range(0, 3)
        d = range(1, 2)
        q = range(0, 3)
        pdq = list(itertools.product(p, d, q))
        seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
        least_AIC = 10000000000000000

        for param in pdq:
            for param_seasonal in seasonal_pdq:
                try:
                    mod = SARIMAX(self.train,
                                                    order=param,
                                                    seasonal_order=param_seasonal,
                                                    enforce_stationarity=False,
                                                    enforce_invertibility=False)
                    results = mod.fit()
                    if(results.aic<least_AIC):
                        best_result = results
                        least_AIC = results.aic
                except:
                    continue


        print(best_result.summary())

        forecast_test = best_result.forecast(len(self.test))
        self.df['Sarima'] = [None] * len(self.train) + list(forecast_test)

        plt.plot(self.df[row], label=row)
        plt.plot(self.df['Sarima'], label="Sarima")
        plt.legend()
        plt.show()

    def LSTMpredict(self,row="Close"):
        # TODO REFACTO
        scaler = MinMaxScaler()

        close_price = self.df[row].values.reshape(-1, 1)
        scaled_close = scaler.fit_transform(close_price)

        seq_len = 10
        x_train, y_train, x_test, y_test = get_train_test_sets(scaled_close, seq_len, train_frac=0.9)

        # fraction of the input to drop; helps prevent overfitting
        dropout = 0.2
        window_size = seq_len - 1

        # build a 3-layer LSTM RNN
        model = keras.Sequential()

        model.add(
            LSTM(window_size, return_sequences=True,
                 input_shape=(window_size, x_train.shape[-1]))
        )

        model.add(Dropout(rate=dropout))
        # Bidirectional allows for training of sequence data forwards and backwards
        model.add(
            Bidirectional(LSTM((window_size * 2), return_sequences=True)
                          ))

        model.add(Dropout(rate=dropout))
        model.add(
            Bidirectional(LSTM(window_size, return_sequences=False))
        )

        model.add(Dense(units=1))
        # linear activation function: activation is proportional to the input
        model.add(Activation('linear'))


        model.compile(
            loss='mean_squared_error',
            optimizer='adam'
        )

        history = model.fit(
            x_train,
            y_train,
            epochs=10,
            batch_size=16,
            shuffle=False,
            validation_split=0.2
        )

        y_pred = model.predict(x_test)

        # invert the scaler to get the absolute price data
        y_test_orig = scaler.inverse_transform(y_test)
        y_pred_orig = scaler.inverse_transform(y_pred)

        offset = y_test_orig[0]-y_pred_orig[0]
        y_pred_orig = y_pred_orig+offset

        # plots of prediction against actual data
        plt.plot(y_test_orig, label='Actual Price', color='orange')
        plt.plot(y_pred_orig, label='Predicted Price', color='green')

        plt.title(' Price Prediction')
        plt.xlabel('Days')
        plt.ylabel('Price')
        plt.legend(loc='best')

        plt.show()

        plt.plot(np.arange(0, len(y_train)), scaler.inverse_transform(y_train), color='brown', label='Historical Price')
        plt.plot(np.arange(len(y_train), len(y_train) + len(y_test_orig)), y_test_orig, color='orange',
                 label='Actual Price')
        plt.plot(np.arange(len(y_train), len(y_train) + len(y_pred_orig)), y_pred_orig, color='green',
                 label='Predicted Price')

        plt.title(' Prices')
        plt.xlabel('Days')
        plt.ylabel('Price ')
        plt.legend()
        plt.show();



def get_train_test_sets(data, seq_len, train_frac):
            n_seq = len(data) - seq_len + 1
            sequences = np.array([data[i:(i + seq_len)] for i in range(n_seq)])
            n_train = int(sequences.shape[0] * train_frac)
            x_train = sequences[:n_train, :-1, :]
            y_train = sequences[:n_train, -1, :]
            x_test = sequences[n_train:, :-1, :]
            y_test = sequences[n_train:, -1, :]
            return x_train, y_train, x_test, y_test


def mainProjet():
    projetPricing = timeseries()
    projetPricing.setDateformat()


    projetPricing.featureselection(['Close'])
    projetPricing.difference('Close')
    projetPricing.logreturn()
    projetPricing.volatility()
    projetPricing.printSerie()
    projetPricing.ressampletimeseire()
    projetPricing.printSerie()
    projetPricing.calculpvalue('Close')
    projetPricing.calculpvalue()
    projetPricing.plotrow('Close','Time',title='Close price evolution over time')
    projetPricing.plotrow('Diff','Time',title='Diff of close price evolution over time')
    projetPricing.plotAcf("Close",title="ACF Plot for Data")
    projetPricing.plotPacf("Close",title="PACF Plot for Data")

    projetPricing.LSTMpredict()




    projetPricing.plotAcf()
    projetPricing.plotPacf()

    projetPricing.splitTrainTest()

    #projetPricing.AutoArimaPredict()

    #projetPricing.ArimaPredict()

    #projetPricing.SarimaPredict()
    ''' 
    
     '''



if __name__ == '__main__':
    mainProjet()
'''
    msft = yf.Ticker("AAPL")

    # get all stock info (slow)
    print('a')
    print(msft.history(period="3y"))
    print('2')
    print(msft.dividends)
    print('3')
    print(msft.actions)
    print('4')
    print(msft.splits)'''
