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

startDate = {"year": 2018, "week": 1}
instrument = 'EURUSD'






traindata = ld.load(ld.Interval.HOURE, instrument, startDate, 50)



candles = candle.Candles(traindata)
candles.calc_sma([2,3,4,5,6,7,8,9,10])
candles.setSMAToSimulation();
train_env = FOREX(candles)

mid = candles.closeMid

print("asdasdasd")
print(mid[-1]/mid[0])

dense_lstm_net = [
    dict(type='dense', size=32),
    dict(type='internal_lstm', size=64)
]

dense_net = [
    dict(type='dense', size=32),
    dict(type='dense', size=64),
    dict(type='dense', size=16)
]

states = train_env.states,
actions = train_env.actions,
network = dense_lstm_net


train_agent = PPOAgent(
    states=train_env.states,
    actions=train_env.actions,
    network=dense_lstm_net,
    update_mode=dict(
        unit='episodes',
        batch_size=35
    ),
    memory = dict(
        type='latest',
        include_next_states=False,
        capacity=( 164 * 35 * 54 * 4)
    ),
    step_optimizer=dict(type='adam', learning_rate=1e-4)
)





def episode_finished_train(r):
    print("Trained mother: " + str(r.episode_rewards[-1]))

    train_reward.append(r.episode_rewards[-1])
    plt.plot(train_reward, 'r+')
    plt.pause(0.01)
    return True

f = open("smaLSTM/checkpoint", "r")

lines = f.readlines()

train_reward = list();
validator_reward = list();

for i in range(1,60):
    split = lines[i].split()
    model_path = split[1]
    print(model_path[1:len(model_path)-1])
    real_model_path = model_path[1:len(model_path) - 1]
    train_agent.restore_model(directory='smaLSTM', file = real_model_path)
    train_runner = Runner(agent=train_agent, environment=train_env)
    train_runner.run(episodes=1, max_episode_timesteps=(candles.candle_nums + 100),episode_finished=episode_finished_train, deterministic=True)



