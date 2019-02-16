import pandas as pd
import matplotlib as plt
import numpy as np
plt.use("TkAgg")
from matplotlib import pyplot as plt
from IPython.core.pylabtools import figsize
import candle
from gradients import *

print("gello")

#file_location = 'http://hilpisch.com/eurusd.csv'
file_location  = 'data/1.csv'
data = pd.read_csv(file_location, index_col=0, parse_dates=True)
data.info()

#data['OCDiff'].plot()
#data['HLDiff'].plot()

file_location  = 'data/1.csv'

candles = candle.Candles(file_location)

candles.calc_sma([360,480,600])

#for i in list(range(2,14)):
#    result = gradient_linreg_slidewindow(closeAsk_y, window_size)
#    plt.plot(result['gradiens'])
#    window_size = window_size + 1
#plt.plot(closeAsk_y)


#candles.calc_gradients(range(60,360,60))


plt.plot(candles.closeMid)

for i in range(len(candles.data_sma[0,:])):
    plt.plot(candles.data_sma[:,i])
    print(i)

plt.show()

#result = gradient_linreg_slidewindow(closeAsk_y, 10)
#plt.plot(result['gradiens'])
#result = gradient_linreg_slidewindow(closeAsk_y, 20)
#plt.plot(result['gradiens'])



#plt.show()






