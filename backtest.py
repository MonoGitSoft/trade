from backtesting import Strategy
from backtesting.lib import crossover
import pandas as pd
import FXCMDataLoader as ld
from backtesting import Backtest
from tensorforce.agents import PPOAgent
from forex import FOREX
from backtesting.test import SMA, GOOG

startDate = {"year": 2015, "week": 1}
instrument = 'EURUSD'

import candle as cn

EU_USD = ld.load(ld.Interval.HOURE, instrument, startDate, 100)

raw_data = EU_USD.copy()

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

EU_USD.rename(index=str, columns={"BidOpen": "Open", "BidHigh": "High", "BidLow": "Low", "BidClose": "Close"},inplace=True)

EU_USD.drop('AskOpen', axis=1, inplace=True)
EU_USD.drop('AskClose', axis=1, inplace=True)
EU_USD.drop('AskHigh', axis=1, inplace=True)
EU_USD.drop('AskLow', axis=1, inplace=True)
EU_USD.index = pd.to_datetime(EU_USD.index)




candles = cn.Candles(raw_data)
candles.calc_sma_seq([5,10,20,30,40,50,60,70,80,90,100,120,140,160,180,200,220,240,280,300])
candles.norm_by_column_sma()
candles.setSMAToSimulation()
env = FOREX(candles)


dense_lstm_net = [
    dict(type='dense', size=32),
    dict(type='internal_lstm', size=10)
]


dense_net = [
    dict(type='dense', size=32),
    dict(type='dense', size=64),
    dict(type='dense', size=64),
    dict(type='dense', size=32)
]


states = env.states,
actions = env.actions,
network = dense_lstm_net


agent = PPOAgent(
    states=env.states,
    actions=env.actions,
    network=dense_net,
    update_mode=dict(
        unit='episodes',
        batch_size=30
    ),
    memory = dict(
        type='latest',
        include_next_states=False,
        capacity=( 164 * 30 * 54 * 4)
    ),
    step_optimizer=dict(type='adam', learning_rate=1e-3)
)


f = open("longlong/checkpoint", "r")
lines = f.readlines()
split = lines[30].split()
model_path = split[1]
print(model_path[1:len(model_path) - 1])
real_model_path = model_path[1:len(model_path) - 1]

agent.restore_model(directory='longlong', file = real_model_path)


base_currency = 1
pair_currency = 0



buy_counter = 0

sim_tick = 0;

class RLStrategy(Strategy):
    def init(self):
        Close = self.data.Close
        self.ma1 = self.I(SMA, Close, 10)
        self.ma2 = self.I(SMA, Close, 20)




    def next(self):

        global base_currency
        global pair_currency
        global buy_counter
        global sim_tick

        action = agent.act(candles.data_for_sim[sim_tick,:], deterministic=True, independent=True);
        sim_tick = sim_tick + 1


        if action == 0:
            if base_currency != 0:
                self.sell()
                pair_currency = 1
                base_currency = 0

        if action == 1:
            if pair_currency != 0:
                print("buy as")
                self.buy()
                print("buy as asd " + str(buy_counter))
                base_currency = 1
                pair_currency = 0
                buy_counter = buy_counter + 1




bt = Backtest(EU_USD, RLStrategy, cash=10000, commission=.0015)

print(str(buy_counter) + " NUmber of buy")

print( bt.run())

bt.plot()


