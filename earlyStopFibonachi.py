from tensorforce.agents import PPOAgent
from tensorforce.execution import Runner
from forex import FOREX
import candle
import numpy as np
import FXCMDataLoader as ld
import matplotlib as plt
plt.use("TkAgg")
from matplotlib import pyplot as plt
import json

startDate = {"year": 2019, "week": 1}
instrument = 'EURUSD'






data = ld.load(ld.Interval.HOURE, instrument, startDate, 40)



candles = candle.Candles(data)
#candles.calc_gradients([10,20,30,50,100,150,200,250,300])
candles.calc_sma_seq([3,5,8,10,20,30,40,50,60,70,80,90,100,120,140,160,180,200,240,280,320,370,430,480,530,580])

candles.norm_by_column_sma()
candles.setSMAToSimulation()
env = FOREX(candles)

dense_lstm_net = [
    dict(type='dense', size=32),
    dict(type='internal_lstm', size=32)
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
        batch_size=10
    ),
    memory = dict(
        type='latest',
        include_next_states=False,
        capacity=434500
    ),
    step_optimizer=dict(type='adam', learning_rate=1e-3),
    # PGModel
)






def episode_finished_train(r):
    print("Trained mother: " + str(r.episode_rewards[-1]))

    train_reward.append(r.episode_rewards[-1])
    plt.plot(train_reward, 'r+')
    plt.pause(0.01)
    return True


f = open("sma_lstm_fucking_long/checkpoint", "r")

lines = f.readlines()

train_reward = list();
validator_reward = list();


for i in range(1,len(lines)-1):
    print(i)
    split = lines[i].split()
    model_path = split[1]
    print(model_path[1:len(model_path)-1])
    real_model_path = model_path[1:len(model_path) - 1]
    print(real_model_path)
    agent.restore_model(directory='sma_lstm_fucking_long', file = real_model_path)
    train_runner = Runner(agent=agent, environment=env)
    train_runner.run(episodes=1, max_episode_timesteps=(candles.candle_nums + 100),episode_finished=episode_finished_train, deterministic=True)



plt.savefig("32_32_lstm_validation.png")