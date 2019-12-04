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

iter = 1

file_location = 'data/1.csv'
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

iter = 1

file_location = 'data/1.csv'


startDate = {"year": 2018, "week": 1}
instrument = 'EURUSD'

chart_data = {"start_date": {"year": 2017, "week": 5}, "instrument" : "EURUSD", "length" : 54}

data = ld.load(ld.Interval.MINUT, instrument, startDate, 3)

candles = candle.Candles(data)

candles.calc_sma([144, 233, 377, 610, 987])

candles.norm_by_column_sma_dev()
candles.norm_by_column_sma()

candles.setSMAToSimulation()
env = FOREX(candles)

dense_lstm_net = [
    dict(type='dense', size=16),
    dict(type='internal_lstm', size=128)
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
        capacity=candles.candle_nums
    ),
    step_optimizer=dict(type='adam', learning_rate=1e-3),
    #entropy_regularization=0.01,
    #likelihood_ratio_clipping=0.15,
    #PGModel
   #baseline_mode='states',
    baseline=dict(
        type='mlp',
        sizes=[16, 16]
    )
)

#agent.restore_model(directory = 'sma_lstm_minuttteee_short')

# Create the runner
runner = Runner(agent=agent, environment=env)

lofasz = 0

# Callback function printing episode statistics

t = list()
rew = list()

modelSaves = 1

def episode_finished(r):
    print("Finished episode {ep} after {ts} timesteps (reward: {reward})".format(ep=r.episode, ts=r.episode_timestep,
                                                                             reward=r.episode_rewards[-1]))
    plt.plot(r.episode_rewards, 'r+')
    global iter
    global modelSaves
    plt.pause(0.01)


    if(iter == 10):
        iter = 0
        agent.save_model('sma_lstm_minuttteee_short/dense_mix')
        modelSaves = modelSaves + 1
    else:
        iter = iter + 1

    return True


# Start learning
runner.run(episodes=7000, max_episode_timesteps=(candles.candle_nums + 100), episode_finished=episode_finished)

#runner.run(episodes=1, max_episode_timesteps=(candles.candle_nums + 100), episode_finished=episode_finished, deterministic=True)



# Print statistics
print("Learning finished. Total episodes: {ep}. Average reward of last 100 episodes: {ar}.".format(
    ep=runner.episode,
    ar=np.mean(runner.episode_rewards[-100:]))
)


print(env.pair_currency)
print(env.base_currency)

runner.close()


