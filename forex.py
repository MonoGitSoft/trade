# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from itertools import product

import candle
import math


import numpy as np


from tensorforce.environments import Environment

ACTION_NAMES = ["Sell","Buy","Idle"]

SELL = 0
BUY = 1
IDLE = 2

class FOREX(Environment):

    def __init__(self, data):
        """
        Initialize FOREX env.
        """

        self.data = data
        self.max_lose = 0.0
        self.start_currency = 5000
        self.base_currency = self.start_currency #EUR/USD arfolyamnal az EUR a base es a USD a pair
        self.pair_currency = 0
        self.canldeIter = 0
        self.lastCandle = False
        self.sell_price= 0
        self.buy_price = 0
        self.idle_punishment = 0
        self.profit = 0
        self.create_current_state()
        self.number_of_buy = 0
        self.number_of_sell = 0
        self.lastCandle = False
        self.reward_mean = 0
        self.worst_profit = 100000
        self.worst_trade = 10000
        self.best_trade = 0
        self.punish = 50
        self.petting = 50

    def sell(self):
        if self.base_currency == 0:# no currency to sell
            return 1
        self.sell_price = self.base_currency
        self.pair_currency = self.base_currency * self.data.bidOpens[self.canldeIter]
        self.base_currency = 0
        self.idle_punishment = 0
        self.number_of_sell += 1
        if self.buy_price == 0:
            return 1
        profit = self.pair_currency / self.buy_price
        return profit

    def buy(self):
        if self.pair_currency == 0:
            return 1
        self.buy_price = self.pair_currency
        self.base_currency = self.pair_currency * 1 / self.data.askOpens[self.canldeIter]
        self.pair_currency = 0

        self.idle_punishment = 0

        if (self.base_currency/self.start_currency) < self.max_lose:
            self.lastCandle = True

        profit = self.base_currency / self.sell_price

        if( profit < self.worst_trade):
            self.worst_trade = profit

        if (self.best_trade < profit):
            self.best_trade = profit

        self.reward_mean = self.reward_mean * self.number_of_buy / (self.number_of_buy + 1) + profit/(self.number_of_buy + 1)

        self.number_of_buy += 1
        return profit

    def __str__(self):
        return 'FOREX'

    def close(self):
        self.data = None

    def create_current_state(self):
        sum_cur = self.base_currency + self.pair_currency
        base_cur_norm = self.base_currency / sum_cur

        cur_distribution = np.array([base_cur_norm])
        if self.base_currency == 0:
            cur_distribution = 0.5
        else:
            cur_distribution = -0.5

        #self.forex_current_state = np.append(self.data.get_mix_sma_gradients(self.canldeIter), cur_distribution)
        if self.pair_currency == 0:
            pair_currency = self.base_currency * self.data.bidOpens[self.canldeIter]
            if self.buy_price == 0:
                profit = 0
            else:
                profit = pair_currency / self.buy_price - 1
        else:
            base_currency = self.pair_currency * 1 / self.data.askOpens[self.canldeIter]
            if self.sell_price == 0:
                profit = 0
            else:
                profit = base_currency / self.buy_price - 1



        #self.forex_current_state = self.data.get_mix_sma_gradients(self.canldeIter)
        self.forex_current_state = self.data.data_for_sim[self.canldeIter, :]
        self.forex_current_state = np.append(self.data.data_for_sim[self.canldeIter, :], cur_distribution)

        #print("during asd profit" + str(profit))

        #self.forex_current_state = np.append(self.forex_current_state, self.data.data_sma_deviation[self.canldeIter, :])
        self.canldeIter = self.canldeIter + 1

    def reset(self):
        print("buy sell money")

        print(self.number_of_sell)
        print(self.base_currency / self.start_currency)
        self.base_currency = self.start_currency #EUR/USD arfolyamnal az EUR a base es a eur a pair
        self.pair_currency = 0
        self.lastCandle = False
        self.canldeIter = 0
        self.idle_punishment = 0
        self.sell_price = 0
        self.selled_currency = 0
        self.number_of_buy = 0
        self.number_of_sell = 0
        self.create_current_state()
        self.reward_mean = 0
        self.worst_profit = 100000
        self.buy_price = 0
        self.worst_trade = 10000
        self.best_trade = 0

        return self.forex_current_state

    def execute(self, action):
        rew = 0
        if action == 0:
            rew = self.calc_rew(self.sell())

        if action == 1:
            rew = self.calc_rew(self.buy())#math.log(self.buy())*10000

        if action == 2:
            rew = 0
        # Get reward and process terminal & next state.
        #rew = 0
        if self.canldeIter == (self.data.candle_nums - 2):
            self.lastCandle = True

         # TTTEEESSST csak a utcsi eredmenyel tanulhat
        if self.lastCandle:
            if self.pair_currency != 0:
                self.buy()
            #rew = math.log(self.worst_profit)*1000
            rew = self.base_currency/self.start_currency
            lofasz = rew
            print("rew last" + str(rew))
            rew = self.calc_rew(rew)
            print("rew after emphasisese " + str(lofasz ) + " " + str(rew))



        self.create_current_state()

        if self.lastCandle:
            print("Last candle")
            print(self.number_of_buy)
            print(self.base_currency/self.start_currency)
            print("worat profit: " + str(self.base_currency/self.start_currency))
            print("worst :" + str(self.worst_trade))
            print("best: " + str(self.best_trade))


        terminal = self.lastCandle
        state_tp1 = self.forex_current_state
        return state_tp1, terminal, rew

    def calc_rew(self, raw):

        #return math.tan(raw - 1)
        #if raw <= 1:
        #    return math.log(raw)
        #else:
        #    return math.exp(raw) - math.e
        return raw - 1



    @property
    def states(self):
        return dict(shape=(self.forex_current_state.shape), type='float')

    @property
    def actions(self):
        return dict(num_actions=3, type='int')

    @property
    def current_state(self):
        return np.copy(self.forex_current_state)

    @property
    def is_terminal(self):
        return self.lastCandle

    @property
    def action_names(self):
        return np.asarray(ACTION_NAMES)
