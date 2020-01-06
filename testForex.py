from numpy.testing import clear_and_catch_warnings

from forex import FOREX
import candle
import FXCMDataLoader as ld
import matplotlib as plt
plt.use("TkAgg")
from matplotlib import pyplot as plt





file_location = 'data/1.csv'

startDate = {"year": 2018, "week": 1}
instrument = 'EURUSD'


data = ld.load(ld.Interval.HOURE, instrument, startDate, 1)

candles = candle.Candles(data)

#for i in range(len(candles.data_sma_deviation[0,:])):
#    plt.plot(candles.data_sma_deviation[:,i])

plt.plot(candles.closeMid, 'b')
plt.plot(candles.askCloses, 'r')
plt.plot(candles.bidCloses, 'g')
plt.show()
