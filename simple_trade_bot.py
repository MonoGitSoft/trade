from forex import FOREX
import candle
import numpy as np
import FXCMDataLoader as ld
import matplotlib as plt
plt.use("TkAgg")
from matplotlib import pyplot as plt

startDate = {"year": 2012, "week": 1}
instrument = 'EURUSD'

chart_data = {"start_date": {"year": 2017, "week": 5}, "instrument" : "EURUSD", "length" : 54}

data = ld.load(ld.Interval.HOURE, instrument, startDate, 54*1)

candles = candle.Candles(data)
# ez eddif
#candles.calc_sma_seq([3,5,8,10,20,30,40,50,60,70,80,90,100,120,140,160,180,200,240,280,320,370,430,480,530,580,630,680,730,780,830,880])
candles.calc_sma_seq([80,100])

candles.norm_by_column_sma()

candles.setSMAToSimulation()
env = FOREX(candles)

sell = 0
buy = 1
idle = 2

cross = 0

terminal = False;
while not terminal:
    if cross < 0:
        state, terminal, rew = env.execute(sell)
    else:
        state, terminal, rew = env.execute(buy)
    cross = state[0]


for i in range(len(candles.data_sma[0,:])):
    plt.plot(candles.data_sma[:,i])
plt.show()