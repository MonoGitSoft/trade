from _ast import slice

import numpy as np
import pandas as pd
from flask_sqlalchemy import xrange

from numpy import linalg as LA

from gradients import *


def binomial(n, k):
    """
    A fast way to calculate binomial coefficients by Andrew Dalke.
    See http://stackoverflow.com/questions/3025162/statistics-combinations-in-python
    """
    if 0 <= k <= n:
        ntok = 1
        ktok = 1
        for t in xrange(1, min(k, n - k) + 1):
            ntok *= n
            ktok *= t
            n -= 1
        return ntok // ktok
    else:
        return 0

class Candles:
    def __init__(self, data):
        #self.data = pd.read_csv(file_location, index_col=0, parse_dates=True)
        self.data = data
        self.bidOpens= data['BidOpen'].values
        self.bidCloses = data['BidClose'].values
        self.bidHighes = data['BidHigh'].values
        self.bidLowes = data['BidLow'].values
        self.askOpens = data['AskOpen'].values
        self.askCloses = data['AskLow'].values
        self.askHighes = data['AskHigh'].values
        self.askLowes = data['AskLow'].values
        self.closeMid = data[['AskClose','BidClose']].mean(axis = 1).values
        self.candle_nums = len(self.askCloses)
        self.data = data
        self.data_gradients = []
        self.data_sma = []
        self.data_mix_sma_grad = []
        self.window_size = 0
        self.data_for_sim = []

    def get_candle(self, iter):
        candle = self.data[iter:iter + 1].values
        return candle

    def setSMAToSimulation(self):
        self.data_for_sim = self.data_sma

    def setGradToSimulation(self):
        self.data_for_sim = self.data_gradients

    def setMIXToSimulation(self):
        self.data_for_sim = np.append(self.data_gradients,self.data_sma,axis=1)

    def get_gradients(self, iter):
        return self.data_gradients[iter,:] * (1 / LA.norm(self.data_gradients[iter,:]))

    def norm_by_column_sma(self):
        for row in range(np.size(self.data_sma, 0)):
            self.data_sma[row, :] = self.data_sma[row, :] * 1 / LA.norm(self.data_sma[row, :])

    def norm_by_column_grad(self):
        for row in range(np.size(self.data_gradients, 0)):
            self.data_gradients[row, :] = self.data_gradients[row, :] * 1 / LA.norm(self.data_gradients[row, :])

    def get_sma(self, iter): #slide window avarage you knowwww
        return self.data_sma[iter, :] * (1 / LA.norm(self.data_sma[iter, :]))

    def get_mix_sma_gradients(self, iter):
        grad = self.data_gradients[iter, :]
        sma = self.data_sma[iter, :]
        return np.append(self.data_gradients[iter, :],self.data_sma[iter, :])

    def calc_gradients(self, window_sizes):
        self.data_gradients = np.zeros((self.candle_nums ,len(window_sizes)), dtype=float)
        column = 0
        for win_size in window_sizes:
            result = gradient_linreg_slidewindow(self.closeMid, win_size)
            self.data_gradients[:,column] = result['gradiens']
            column = column + 1

    def calc_sma_seq(self,window_sizes):
        self.data_sma = np.zeros((self.candle_nums, len(window_sizes)), dtype=float)
        column = 0
        for win_size in window_sizes:
            result = slide_window_filter(self.closeMid, win_size)
            self.data_sma[:, column] = result
            column = column + 1

        column = 0
        max_column = len(window_sizes) - 1
        tmp = np.zeros((self.candle_nums, max_column), dtype=float)
        tmp_counter = 0
        print(max_column)
        for i in range(max_column):
            tmp[:, tmp_counter] = np.subtract(self.data_sma[:, i], self.data_sma[:, i + 1])
            a = tmp[:, tmp_counter]
            print(np.var(a))
            tmp_counter = tmp_counter + 1
        self.data_sma = np.copy(tmp)

    def calc_sma(self, window_sizes):
        self.data_sma = np.zeros((self.candle_nums ,len(window_sizes)), dtype=float)
        column = 0
        for win_size in window_sizes:
            result = slide_window_filter(self.closeMid, win_size)
            self.data_sma[:,column] = result
            column = column + 1
            print("asd")

        column = 0
        max_column = len(window_sizes)
        tmp = np.zeros((self.candle_nums , binomial(max_column, 2)), dtype=float)
        print(binomial(max_column, 2))
        tmp_counter = 0
        for i in range(max_column - 1):
            print("elso cilkus" + str(i))
            for j in range((i + 1),max_column):
                tmp[:,tmp_counter] = np.subtract(self.data_sma[:,i],self.data_sma[:,j])
                tmp_counter = tmp_counter + 1
        self.data_sma = np.copy(tmp)