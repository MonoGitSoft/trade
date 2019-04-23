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


data = ld.load(ld.Interval.HOURE, instrument, startDate, 30)

candles = candle.Candles(data)
candles.calc_sma_seq([10,20,40,80,160])
candles.calc_gradients([5,10,20,50,100])


plt.plot(candles.closeMid)

#for i in range(len(candles.data_gradients[0,:])):
#    plt.plot(candles.data_gradients[:,i])
#plt.show()



plt.show()