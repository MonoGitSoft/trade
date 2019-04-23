import pandas as pd
import matplotlib as plt
plt.use("TkAgg")
from matplotlib import pyplot as plt
import candle
import FXCMDataLoader as ld
print("gello")
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




startDate = {"year": 2018, "week": 1}
instrument = 'EURUSD'


data = ld.load(ld.Interval.MINUT, instrument, startDate, 10)


candles = candle.Candles(data)


candles.calc_gradients([5, 10, 20, 40, 100])
candles.calc_sma([360,480,600])

#candles.calc_sma([3,4,5,6])

#for i in list(range(2,14)):
#    result = gradient_linreg_slidewindow(closeAsk_y, window_size)
#    plt.plot(result['gradiens'])
#    window_size = window_size + 1
#plt.plot(closeAsk_y)


candles.norm_by_column_grad()

#for i in range(len(candles.data_gradients[0,:])):
#    plt.plot(candles.data_gradients[:,i])

plt.plot(candles.closeMid)

plt.show()

#result = gradient_linreg_slidewindow(closeAsk_y, 10)
#plt.plot(result['gradiens'])
#result = gradient_linreg_slidewindow(closeAsk_y, 20)
#plt.plot(result['gradiens'])



#plt.show()






