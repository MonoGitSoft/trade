from tensorforce.agents import DQNAgent
from tensorforce.execution import Runner
from forex import FOREX
import candle
import numpy as np
import FXCMDataLoader as ld
import matplotlib as plt
plt.use("TkAgg")
from matplotlib import pyplot as plt




file_location = 'data/1.csv'


startDate = {"year": 2018, "week": 15}
instrument = 'EURUSD'

data = ld.load(ld.Interval.MINUT, instrument, startDate, 1)

candles = candle.Candles(data)
candles.calc_sma([2,5,10,20,30,40])
env = FOREX(candles)

dense_lstm_net = [
    dict(type='dense', size=64),
    dict(type='internal_lstm', size=128)
]

agent = DQNAgent(
    states=env.states,
    actions=env.actions,
    network=dense_lstm_net,
    update_mode=dict(unit='timesteps', batch_size=1000)
)

#agent.save_model('forex_agent_sma/')

#agent.restore_model('forex_DQNAgent/')

# Create the runner
runner = Runner(agent=agent, environment=env)

lofasz = 0

# Callback function printing episode statistics

t = list()
rew = list()

def episode_finished(r):
    print("Finished episode {ep} after {ts} timesteps (reward: {reward})".format(ep=r.episode, ts=r.episode_timestep,
                                                                             reward=r.episode_rewards[-1]))
    plt.plot(r.episode_rewards, 'r+')
    plt.pause(0.01)
    agent.save_model('forex_DQNAgent/')
    return True


# Start learning
runner.run(episodes=7000, max_episode_timesteps=(candles.candle_nums + 100), episode_finished=episode_finished)

agent.save_model('forex_agent_sma/')

# Print statistics
print("Learning finished. Total episodes: {ep}. Average reward of last 100 episodes: {ar}.".format(
    ep=runner.episode,
    ar=np.mean(runner.episode_rewards[-100:]))
)


print(env.pair_currency)
print(env.base_currency)

runner.close()
