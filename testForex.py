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


data = ld.load(ld.Interval.MINUT, instrument, startDate, 1)

candles = candle.Candles(data)
candles.calc_sma([89, 144, 233])

candles.norm_by_column_sma_dev()
candles.norm_by_column_sma()

#for i in range(len(candles.data_sma_deviation[0,:])):
#    plt.plot(candles.data_sma_deviation[:,i])

for i in range(len(candles.data_sma[0,:])):
    plt.plot(candles.data_sma[:,i])
plt.show()


plt.show()