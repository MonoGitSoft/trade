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

startDate = {"year": 2017, "week": 1}
instrument = 'EURUSD'


validatordata = ld.load(ld.Interval.HOURE, instrument, startDate, 50)


train_candles = candle.Candles(traindata)
train_candles.calc_gradients([3,4,5,6,7,8,9,10,11])
train_candles.calc_sma_seq([3,4,5,7,9,11])
train_candles.norm_by_column()
train_candles.norm_by_column_grad()
train_candles.setGradToSimulation()
train_env = FOREX(train_candles)

validator_candles = candle.Candles(validatordata)
validator_candles.calc_gradients([3,4,5,6,7,8,9,10,11])
validator_candles.calc_sma_seq([3,4,5,7,9,11])
validator_candles.norm_by_column()
validator_candles.norm_by_column_grad()
validator_env = FOREX(train_candles)


dense_lstm_net = [
    dict(type='dense', size=32),
    dict(type='internal_lstm', size=64)
]

dense_net = [
    dict(type='dense', size=32),
    dict(type='dense', size=64),
    dict(type='dense', size=16)
]

train_agent = PPOAgent(
    states=train_env.states,
    actions=train_env.actions,
    network=dense_net,
    update_mode=dict(
        unit='episodes',
        batch_size=30
    ),
    memory = dict(
        type='latest',
        include_next_states=False,
        capacity=( 164 * 30 * 50)
    ),
    step_optimizer=dict(type='adam', learning_rate=1e-3)
)

train_agent.restore_model(directory = 'forex_models_gradient_2')

validator_agent = PPOAgent(
    states=validator_env.states,
    actions=validator_env.actions,
    network=dense_net,
    update_mode=dict(
        unit='episodes',
        batch_size=30
    ),
    memory = dict(
        type='latest',
        include_next_states=False,
        capacity=( 164 * 30 * 50)
    ),
    step_optimizer=dict(type='adam', learning_rate=1e-3)
)


def episode_finished_train(r):
    print("Trained mother: " + str(r.episode_rewards[-1]))

    train_reward.append(r.episode_rewards[-1])
    plt.plot(train_reward, 'r+')
    plt.pause(0.01)
    return True

def episode_finished_validator(r):
    print("validator " + str(r.episode_rewards[-1]))
    validator_reward.append(r.episode_rewards[-1])
    plt.plot(validator_reward, 'b+')
    plt.pause(0.01)

    return True

f = open("forex_models_gradient_2/checkpoint", "r")

lines = f.readlines()

train_reward = list();
validator_reward = list();

for i in range(1,40):
    split = lines[i].split()
    model_path = split[1]
    print(model_path[1:len(model_path)-1])
    real_model_path = model_path[1:len(model_path) - 1]
    train_agent.restore_model(directory='forex_models_gradient_2', file = real_model_path)
    train_runner = Runner(agent=train_agent, environment=train_env)
    train_runner.run(episodes=1, max_episode_timesteps=(validator_candles.candle_nums + 100),episode_finished=episode_finished_train, deterministic=True)




for i in range(1,40):
    split = lines[i].split()
    model_path = split[1]
    print(model_path[1:len(model_path)-1])
    real_model_path = model_path[1:len(model_path) - 1]
    validator_runner = Runner(agent=validator_agent, environment=validator_env)
    validator_runner.run(episodes=1, max_episode_timesteps=(validator_candles.candle_nums + 100),episode_finished=episode_finished_validator ,deterministic=True)
    validator_agent.restore_model(directory='forex_models_gradient_2', file = real_model_path)