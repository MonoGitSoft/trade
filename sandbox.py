import candle
import FXCMDataLoader as ld
import matplotlib as plt
plt.use("TkAgg")
from matplotlib import pyplot as plt
import numpy as np
import math

import urllib
try :
    url = "https://www.google.com"
    urllib.urlopen(url)
    status = "Connected"
except :
    status = "Not connect"
print(status)

startDate = {"year": 2015, "week": 1}
instrument = 'EURUSD'


chart_data = {"start_date": {"year": 2017, "week": 5}, "instrument" : "EURUSD", "length" : 54}

data = ld.load(ld.Interval.HOURE, instrument, startDate, 20)

candles = candle.Candles(data)
candles_slow = candle.Candles(data)

candles.calc_sma_seq([3,5])


#candles.norm_by_variance()
#candles.norm_by_column_sma()

candles.norm_by_column_sma()
candles_slow.norm_by_column_sma()
for i in range(len(candles.data_sma[0,:])):
    plt.plot(candles.data_sma[:,i])


plt.show()




