from enum import Enum
import pandas as pd
import datetime

class Interval(Enum):
    MINUT = 'm1'
    HOURE = 'H1'
    DAY = 'D1'
    def __str__(self):
        return self.value

http = 'https://candledata.fxcorporate.com'
file_formate = '.csv.gz'


def load(interval : Interval, instrument : str, startDate : dict, numOfSample : int) -> pd.DataFrame:
    yr = startDate['year']
    wk = startDate['week']
    first = True
    for i in range(numOfSample):
        web_location = http + '/' + str(interval) + '/' + instrument + '/' + str(yr) + '/' + str(wk) + file_formate
        wk = wk + 1
        if wk > 52:
            wk = 1
            yr = yr + 1
        print('Load...')
        print(web_location)
        data_loaded = pd.read_csv(web_location, index_col=0, parse_dates=True)
        if first:
            first = False
            data = data_loaded
        else:
            data = pd.concat([data, data_loaded])
    return data
