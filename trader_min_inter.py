from py._path.svnwc import cache
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


startDate = {"year": 2012, "week": 1}
instrument = 'EURUSD'

chart_data = {"start_date": {"year": 2017, "week": 5}, "instrument" : "EURUSD", "length" : 54}

data = ld.load(ld.Interval.HOURE, instrument, startDate, 54*7)

candles = candle.Candles(data)
# ez eddif
#candles.calc_sma_seq([3,5,8,10,20,30,40,50,60,70,80,90,100,120,140,160,180,200,240,280,320,370,430,480,530,580,630,680,730,780,830,880])
candles.calc_sma_seq([3,5,8,10,20,30,40,50,60,70,80,90,100,120,140,160,180,200,240,280])

candles.norm_by_column_sma()

candles.setSMAToSimulation()
env = FOREX(candles)

dense_lstm_net = [
    dict(type='internal_lstm', size=30, dropout=0.2) #dropout=0.4
]

dense_net = [
    dict(type='linear', size=32),
    dict(type='tf_layer', layer='batch_normalization'),
    dict(type='nonlinearity', name='relu'),
    dict(type='linear', size=128),
    dict(type='tf_layer', layer='batch_normalization'),
    dict(type='nonlinearity', name='relu'),
    dict(type='linear', size=128),
    dict(type='tf_layer', layer='batch_normalization'),
    dict(type='nonlinearity', name='relu'),
    dict(type='dense', size=32)
]

states = env.states,
actions = env.actions,
network = dense_lstm_net

batch_size = 12


agent = PPOAgent(
    states=env.states,
    actions=env.actions,
    network=dense_lstm_net,
    update_mode=dict(
        unit='episodes',
        batch_size=batch_size
    ),
    memory = dict(
        type='latest',
        include_next_states=False,
        capacity=candles.candle_nums*batch_size
    ),
    step_optimizer=dict(type='adam', learning_rate=1e-3),
    #entropy_regularization=5e-5,
    likelihood_ratio_clipping=0.1,
    #PGModel
    #baseline_mode='states',
    #baseline=dict(
    #    type='mlp',
    #    sizes=[32, 5]
    #)
)

#agent.restore_model(directory = 'lstm_26_0.09_5e-5')

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


    if(iter == batch_size):
        iter = 0
        agent.save_model('lstm_30_0_01/dense_mix')
        modelSaves = modelSaves + 1
    else:        iter = iter + 1

    return True


# Start learning
runner.run(episodes=7000, max_episode_timesteps=(candles.candle_nums + 100), episode_finished=episode_finished)

runner.run(episodes=1, max_episode_timesteps=(candles.candle_nums + 100), episode_finished=episode_finished, deterministic=True)



# Print statistics
print("Learning finished. Total episodes: {ep}. Average reward of last 100 episodes: {ar}.".format(
    ep=runner.episode,
    ar=np.mean(runner.episode_rewards[-100:]))
)


print(env.pair_currency)
print(env.base_currency)

runner.close()


