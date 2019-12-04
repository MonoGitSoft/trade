from enum import Enum
import pandas as pd
import datetime
import urllib

class Interval(Enum):
    MINUT = 'm1'
    HOURE = 'H1'
    DAY = 'D1'
    MINUT_30 = 'm30'
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
        if wk > 54:
            wk = 1
            yr = yr + 1
        print('Load...')
        print(web_location)
        try:
            data_loaded = pd.read_csv(web_location, index_col=0, parse_dates=True)
        except urllib.error.HTTPError as err:
            print("Error :(")
            continue

        if first:
            first = False
            data = data_loaded
        else:
            data = pd.concat([data, data_loaded])
        
    return data
