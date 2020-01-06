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
instrument_1 = 'EURGBP'

chart_data = {"start_date": {"year": 2017, "week": 5}, "instrument" : "EURUSD", "length" : 54}

data = ld.load(ld.Interval.HOURE, instrument, startDate, 20)

startDate = startDate = {"year": 2012, "week": 1}

data_con = ld.load_and_concat(ld.Interval.HOURE, instrument, instrument_1, startDate, 54*7)

data = ld.load(ld.Interval.HOURE, instrument, startDate, 10)
print("rohadt nyomorek kurbva anyadat hogy bazdmeg " + str(data_con['AskClose'].values[0]))

candles = candle.Candles(data)
candles_slow = candle.Candles(data_con)
candles_slow.calc_sma_seq([50,100,200,300,400])

candles.calc_sma_seq([50,100,200,300,400])

print("calc ochl")
#candles.create_ochl()
#candles.norm_ochl()
#candles.norm_by_variance()
#candles.norm_by_column_sma()


#for i in range(len(candles_slow.data_sma[0,:])):
#    plt.plot(candles_slow.data_sma[:,i])


plt.plot(candles_slow.bidOpens)
plt.plot(candles_slow.askOpens)
plt.plot(candles_slow.bidHighes)
plt.plot(candles_slow.bidLowes)

plt.show()




