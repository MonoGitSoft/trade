import pandas as pd
import matplotlib as plt
import numpy as np
plt.use("TkAgg")
from matplotlib import pyplot as plt
import candle
import FXCMDataLoader as ld
print("gello")
from scipy.fftpack import fft
import datetime



#https://github.com/fxcm/MarketData
#file_location = 'http://hilpisch.com/eurusd.csv'
file_location  = 'data/1.csv'
data = pd.read_csv(file_location, index_col=0, parse_dates=True)
data.info()

#data['OCDiff'].plot()
#data['HLDiff'].plot()

file_location  = 'data/1.csv'
http_location = 'https://candledata.fxcorporate.com/D1/EURUSD/2017.csv.gz'

#AUDCAD, AUDCHF, AUDJPY, AUDNZD, CADCHF, EURAUD,
#EURCHF, EURGBP, EURJPY, EURUSD, GBPCHF, GBPJPY,
#GBPNZD, GBPUSD, NZDCAD, NZDCHF, NZDJPY, NZDUSD,
#USDCAD, USDCHF, USDJPY, AUDUSD, CADJPY, GBPCAD,
#USDTRY, EURNZD

startDate = {"year": 2012, "week": 1}
instrument = 'EURUSD'

data = ld.load(ld.Interval.HOURE, instrument, startDate, 10)
candles = candle.Candles(data)




#candles.calc_gradients([50,100,150,200])

#candles.calc_sma_seq([10,30,50,70,90,110,130,150,170,190,210])
candles.calc_candle_baee()

plt.plot(candles.data_candle[:,1])



plt.show()

#for i in range(len(candles.data_gradients[0,:])):
#   plt.plot(candles.data_gradients[:,i])








