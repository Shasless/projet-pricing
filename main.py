import itertools
import math
from plotly.subplots import make_subplots

import numpy as np
from scipy.stats import norm
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.graphics.tsaplots as sgt
from sklearn.model_selection import train_test_split
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, pacf, acf
import pmdarima as pm
import statsmodels.api as sm

import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras.layers import Bidirectional, Dropout, Activation, Dense, LSTM

import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go


def black_scholes_call(S, K, T, r, sigma):
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    Nd1 = norm.cdf(d1)
    Nd2 = norm.cdf(d2)
    call_price = S * Nd1 - K * math.exp(-r * T) * Nd2
    return call_price


def black_scholes_put(S, K, T, r, sigma):
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    Nd1 = norm.cdf(-d1)
    Nd2 = norm.cdf(-d2)
    put_price = K * math.exp(-r * T) * Nd2 - S * Nd1
    return put_price


def implied_volatility(S, K, T, r, price, option_type='call'):
    """
    S: stock price
    K: strike price
    T: time to maturity in years
    r: risk-free interest rate
    price: option price
    option_type: 'call' or 'put'
    """
    epsilon = 0.0001
    sigma = 0.3
    max_iter = 5000
    mini = 1000000000000000000
    defaultSigma = -1
    for i in range(max_iter):
        if option_type == 'call':
            option_price = black_scholes_call(S, K, T, r, sigma)
        elif option_type == 'put':
            option_price = black_scholes_put(S, K, T, r, sigma)
        diff = option_price - price

        if abs(diff) < epsilon:
            return sigma

        elif abs(diff) < mini:
            mini = abs(diff)
            defaultSigma = sigma
        vega = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))

        sigma -= diff / (vega + 1e-8)
    return defaultSigma


def implied_volatility_slow(S, K, T, r, price, option_type='call'):
    volatility_candidates = np.arange(0.01, 5, 0.0001).tolist() + np.arange(5, 100, 0.25).tolist()
    price_differences = np.zeros_like(volatility_candidates)

    for i in range(len(volatility_candidates)):
        candidate = volatility_candidates[i]
        if option_type == 'call':
            price_differences[i] = price - black_scholes_call(S, K, T, r, candidate)

        elif option_type == 'put':
            price_differences[i] = price - black_scholes_put(S, K, T, r, candidate)

    idx = np.argmin(abs(price_differences))

    return volatility_candidates[idx]


class timeseries():
    def __init__(self):
        try:
            self.msft = yf.Ticker("AAPL")
            self.df = self.msft.history(period='3y')
            self.df.dropna(inplace=True)
            self.api = True


        except:
            csvroute = input(
                "Error API accebility please enter the neme f a csv file but fonctioality like call put will be impossible")

            self.df = pd.read_csv(csvroute, sep=",")
            self.df.dropna(inplace=True)

    def candleStickChart(self):

        fig = go.Figure(data=[go.Candlestick(x=self.df.index,
                                             open=self.df['Open'],
                                             high=self.df['High'],
                                             low=self.df['Low'],
                                             close=self.df['Close'])])

        fig.show()

    def plotCallPutt(self):

        df = pd.DataFrame({'lastPrice': [], 'strike': [], 'type': [], 'date': []})

        for i in self.msft.options:
            optdatte = i

            opt = self.msft.option_chain(optdatte)

            calls = opt.calls[['strike', 'lastPrice']]
            puts = opt.puts[['strike', 'lastPrice']]
            concatenated_df = pd.concat([calls.assign(type='calls'), puts.assign(type='puts')])

            concatenated_df['date'] = i

            df = pd.concat([concatenated_df, df], axis=0)

        # Make 0th trace visible

        fig = px.scatter(df, x="strike", y="lastPrice", animation_frame="date", color="type")

        fig["layout"].pop("updatemenus")  # optional, drop animation buttons

        fig.show()

    def plotimpliedvolatibility(self, mode='slow', date='2023-06-16', r=0.04):
        implied_volatilities_calls = []
        try:
            options = self.msft.option_chain(date)
        except:
            date = self.msft.options[0]

            options = self.msft.option_chain(date)

        calls = options.calls
        T = (pd.to_datetime(date) - pd.Timestamp.now()) / pd.Timedelta(days=365)
        regular_price = self.msft.info['regularMarketPrice']

        if (mode == 'slow'):
            for i in range(len(calls)):
                implied_volatilities_calls.append(
                    implied_volatility_slow(regular_price, calls['strike'][i], T, r, calls['lastPrice'][i]))
        elif (mode == 'fast'):
            for i in range(len(calls)):
                implied_volatilities_calls.append(
                    implied_volatility(regular_price, calls['strike'][i], T, r, calls['lastPrice'][i]))
        else:
            for i in range(len(calls)):
                implied_volatilities_calls.append(calls['impliedVolatility'][i])

        fig = go.Figure()

        # Add scatter plot for calls
        fig.add_trace(go.Scatter(
            x=calls['strike'],
            y=calls['lastPrice'],
            mode='markers',
            marker=dict(color='red'),
            name='Calls'
        ))

        # Add line plot for implied volatilities of calls
        fig.add_trace(go.Scatter(
            x=calls['strike'],
            y=implied_volatilities_calls,
            mode='lines',
            line=dict(color='blue'),
            name='Implied Volatility call',
            yaxis='y2'
        ))

        # Set axis labels
        fig.update_layout(
            xaxis_title='Strike Price',
            yaxis=dict(title='Option Price'),
            yaxis2=dict(title='Implied Volatility', overlaying='y', side='right'),
        )

        # Add legend
        fig.update_layout(
            legend=dict(
                x=0,
                y=1,
                traceorder='normal',
                font=dict(size=10),
                bgcolor='LightSteelBlue',
                bordercolor='Black',
                borderwidth=1
            )
        )

        fig.show()

    def printSerie(self):
        print(self.df)
        print(self.df.describe())
        self.df.info()

    def setDateformat(self):
        if not self.api:
            self.df["Date"] = pd.to_datetime(self.df["Date"])
            self.df.sort_values(by="Date", inplace=True)
            self.df.set_index("Date", inplace=True)

    def featureselection(self, feature=['Open', 'High', 'Low', 'Close']):
        self.df = self.df[feature]

    def ressampletimeseire(self, parameter="W"):
        self.df = self.df.resample(parameter).mean().dropna()

    def difference(self, row="Open"):
        self.df['Diff'] = self.df[row].diff().dropna()

    def plotColumn(self, row='Open', title="graph"):

        fig = px.line(self.df, x=self.df.index, y=row, title=title)
        fig.show()

    def logreturn(self, row="Close"):
        self.df['Logreturn'] = np.log(self.df[row] / self.df[row].shift(1))

    def volatility(self, row="Close"):
        self.df['volatility'] = self.df[row].rolling(window=2).std()

    def calculpvalue(self, row='Diff'):
        print("p-value", row, ": ", adfuller(self.df[row])[1])

    def plotPacf(self, row="Diff", zeroValue=False, alphaValue=0.05,
                 title="PACF Plot for First Order Differenced Data"):
        pacf_values, confint = sgt.pacf(self.df[row], nlags=int((len(self.df) - 1) / 2) - 2, alpha=alphaValue)
        lags = [i for i in range(int(len(pacf_values)))]
        fig = make_subplots()
        fig.add_trace(go.Scatter(x=lags, y=pacf_values, mode='markers+lines', name="PACF"), row=1, col=1)
        fig.add_trace(go.Scatter(x=lags, y=confint[:, 0], line=dict(dash='dash'), showlegend=False,
                                 name=f"Confidence Interval ({alphaValue})"), row=1, col=1)
        fig.add_trace(go.Scatter(x=lags, y=confint[:, 1], line=dict(dash='dash'), showlegend=False), row=1, col=1)
        fig.update_layout(title=title, xaxis_title="Lag", yaxis_title="Partial Autocorrelation")
        fig.show()

    def plotAcf(self, row="Diff", zeroValue=False, alphaValue=0.05, title="ACF Plot for First Order Differenced Data"):
        acf_values, confint = sgt.acf(self.df[row], nlags=len(self.df) - 1, alpha=alphaValue)
        lags = [i for i in range(int(len(acf_values)))]
        fig = make_subplots()
        fig.add_trace(go.Scatter(x=lags, y=acf_values, mode='markers+lines', name="ACF"), row=1, col=1)
        fig.add_trace(go.Scatter(x=lags, y=confint[:, 0], line=dict(dash='dash'), showlegend=False,
                                 name=f"Confidence Interval ({alphaValue})"), row=1, col=1)
        fig.add_trace(go.Scatter(x=lags, y=confint[:, 1], line=dict(dash='dash'), showlegend=False), row=1, col=1)
        fig.update_layout(title=title, xaxis_title="Lag", yaxis_title="Autocorrelation")
        fig.show()

    def plotPacfPlt(self, row="Diff", zeroValue=False, alphaValue=0.05,
                    title="PACF Plot for First Order Differenced Data"):
        sgt.plot_pacf(self.df[row], lags=np.arange((len(self.df) - 1) / 2), zero=zeroValue, alpha=alphaValue)
        plt.title(title)
        plt.show()

    def plotAcfPlt(self, row="Diff", zeroValue=False, alphaValue=0.05,
                   title="ACF Plot for First Order Differenced Data"):
        sgt.plot_acf(self.df[row], lags=np.arange(len(self.df) - 1), zero=zeroValue, alpha=alphaValue)
        plt.title(title)
        plt.show()

    def splitTrainTest(self, test_size=0.2, row="Close"):
        self.train, self.test = train_test_split(self.df[row], test_size=test_size, shuffle=False)

    def AutoArimaPredict(self, row="Close"):
        self.auto_arima = pm.auto_arima(self.train, stepwise=False, seasonal=True)
        print(self.auto_arima.summary())

        test = self.auto_arima.predict(n_periods=len(self.test))
        print('coucou')

        self.df['autoArima'] = [None] * (len(self.train)) + list(test)

        fig = px.line(self.df, x=self.df.index, y=[row, 'autoArima'])
        fig.show()

        print('If this is a straight line its mean that auto arima think this a random step')

    def ArimaPredict(self, row="Close"):  # TODO reparer si jamais c'est possible

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
                continue

        print(best_result.summary())

        forecast_test = best_result.forecast(len(self.test))
        self.df['Arima'] = [None] * len(self.train) + list(forecast_test)

        fig = px.line(self.df, x=self.df.index, y=[row, 'Arima'])
        fig.show()

    def SarimaPredict(self, row="Close"):

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
                    if (results.aic < least_AIC):
                        best_result = results
                        least_AIC = results.aic
                except:
                    continue

        print(best_result.summary())

        forecast_test = best_result.forecast(len(self.test))
        self.df['Sarima'] = [None] * len(self.train) + list(forecast_test)

        fig = px.line(self.df, x=self.df.index, y=[row, 'Sarima'])
        fig.show()

    def LSTMpredict(self, row="Close"):

        scaler = MinMaxScaler()

        close_price = self.df[row].values.reshape(-1, 1)
        scaled_close = scaler.fit_transform(close_price)

        seq_len = 10
        # initialisation train and test
        n_seq = len(scaled_close) - seq_len + 1
        sequences = np.array([scaled_close[i:(i + seq_len)] for i in range(n_seq)])
        n_train = int(sequences.shape[0] * 0.9)
        x_train = sequences[:n_train, :-1, :]
        y_train = sequences[:n_train, -1, :]
        x_test = sequences[n_train:, :-1, :]
        y_test = sequences[n_train:, -1, :]

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

        model.fit(
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

        offset = y_test_orig[0] - y_pred_orig[0]
        y_pred_orig = y_pred_orig + offset

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.df.index[:- len(y_test_orig)],
                                 y=scaler.inverse_transform(y_train).reshape(-1),
                                 name='Historical Price',
                                 line=dict(color='brown')
                                 ))
        fig.add_trace(go.Scatter(x=self.df.index[len(y_train):],
                                 y=y_test_orig.reshape(-1),
                                 name='Actual Price',
                                 line=dict(color='orange')
                                 ))
        fig.add_trace(go.Scatter(x=self.df.index[len(y_train):],
                                 y=y_pred_orig.reshape(-1),
                                 name='Predicted Price LSTM',
                                 line=dict(color='green')
                                 ))

        fig.update_layout(title='Prices',
                          xaxis_title='Days',
                          yaxis_title='Price',
                          legend=dict(x=0.7, y=0.9))
        fig.show()


def mainProjet():
    projetPricing = timeseries()
    projetPricing.setDateformat()

    projetPricing.featureselection(['Close'])
    projetPricing.difference('Close')
    projetPricing.plotColumn('Close', title='Close price evolution over time')
    projetPricing.plotColumn('Diff', title='Diff of close price evolution over time')
    # projetPricing.logreturn()
    # projetPricing.volatility()
    # projetPricing.printSerie()
    # projetPricing.LSTMpredict()

    projetPricing.ressampletimeseire()
    projetPricing.printSerie()
    # projetPricing.calculpvalue('Close')
    # projetPricing.calculpvalue()

    projetPricing.plotAcfPlt("Close", title="ACF Plot for Close")
    projetPricing.plotPacfPlt("Close", title="PACF Plot for Close")
    projetPricing.plotAcfPlt("Diff", title="ACF Plot for Diff")
    projetPricing.plotPacfPlt("Diff", title="PACF Plot for Diff")

    #
    # projetPricing.plotAcf()
    # projetPricing.plotPacf()
    #
    # projetPricing.splitTrainTest()
    #
    # projetPricing.AutoArimaPredict()
    #
    # projetPricing.ArimaPredict()
    #
    # projetPricing.SarimaPredict()
    # projetPricing.plotCallPutt()
    # projetPricing.plotimpliedvolatibility()
    # projetPricing.plotimpliedvolatibility(mode='dfgdfg')


if __name__ == '__main__':
    #mainProjet()
    projetPricing = timeseries()
    projetPricing.setDateformat()
    projetPricing.candleStickChart()
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
