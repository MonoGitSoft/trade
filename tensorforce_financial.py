from tensorforce.agents import PPOAgent
from tensorforce.execution import Runner
from tensorforce.environments.environment import Environment
from forex import FOREX
import candle
import numpy as np
from tensorforce.agents import DQNAgent

file_location  = 'data/1.csv'

candles = candle.Candles(file_location)
candles.calc_gradients(range(60,500,60))
candles.calc_sma([2,30,60,90,120])
candles.window_size = 20
env = FOREX(candles)

print(dict(type='float', shape=(10,)))
print(env.states)
# Instantiate a Tensorforce agent

agent = PPOAgent(
    states=env.states,
    actions=env.actions,
    network=[
        dict(type='dense', size=64),
        dict(type='dense', size=64)
    ],
    step_optimizer=dict(type='adam', learning_rate=1e-3),

)


#agent.save_model()

agent.restore_model('forex_agent_sma/')

# Create the runner
runner = Runner(agent=agent, environment=env)


# Callback function printing episode statistics
def episode_finished(r):
    print("Finished episode {ep} after {ts} timesteps (reward: {reward})".format(ep=r.episode, ts=r.episode_timestep,
                                                                                 reward=r.episode_rewards[-1]))
    return True


# Start learning
runner.run(episodes=10000, max_episode_timesteps=(candles.candle_nums + 100), episode_finished=episode_finished)

runner.close()

agent.save_model()

# Print statistics
print("Learning finished. Total episodes: {ep}. Average reward of last 100 episodes: {ar}.".format(
    ep=runner.episode,
    ar=np.mean(runner.episode_rewards[-100:]))
)

print(env.pair_currency)
print(env.base_currency)