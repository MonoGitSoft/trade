from enum import Enum
import pandas as pd
import datetime
import urllib
import os.path
from os import path


class Interval(Enum):
    MINUT = 'm1'
    HOURE = 'H1'
    DAY = 'D1'
    MINUT_30 = 'm30'
    def __str__(self):
        return self.value

http = 'https://candledata.fxcorporate.com'
file_formate = '.csv.gz'


def load_last(interval : Interval, instrument : str, startDate : dict, numOfSample : int):
    yr = startDate['year']
    wk = startDate['week']
    save_name = instrument + "_" + str(interval) + '_' + str(yr)  + '_' + str(wk) + '_' + str(numOfSample) + '.csv'
    return pd.read_csv(save_name)

def load(interval : Interval, instrument : str, startDate : dict, numOfSample : int) -> pd.DataFrame:
    yr = startDate['year']
    wk = startDate['week']
    first = True

    save_name = instrument + "_" + str(interval) + '_' + str(yr)  + '_' + str(wk) + '_' + str(numOfSample) + '.csv'


    if path.exists(save_name):
        print(save_name + " is already loaded")
        return pd.read_csv(save_name)

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
    data.to_csv(save_name)
    return data


#self.bidOpens = data['BidOpen'].values
#self.bidCloses = data['BidClose'].values
#self.bidHighes = data['BidHigh'].values
#self.bidLowes = data['BidLow'].values
#self.askOpens = data['AskOpen'].values
#self.askCloses = data['AskClose'].values
#self.askHighes = data['AskHigh'].values
#self.askLowes = data['AskLow'].values
#'AskClose','BidClose'

def load_and_concat(interval: Interval, instrument : str, instrument_1 : str, startDate : dict, numOfSample : int) -> pd.DataFrame:
    data_1 = load(interval, instrument, startDate, numOfSample)
    data_2 = load(interval, instrument_1, startDate, numOfSample)
    last_1 = data_1.index.stop
    data_2['AskClose'] = data_2['AskClose'] + (data_1['AskClose'].values[-1] - data_2['AskClose'].values[0])
    data_2['BidClose'] = data_2['BidClose'] + (data_1['BidClose'].values[-1] - data_2['BidClose'].values[0])
    data_2['AskOpen'] = data_2['AskOpen'] + (data_1['AskOpen'].values[-1] - data_2['AskOpen'].values[0])
    data_2['BidOpen'] = data_2['BidOpen'] + (data_1['BidOpen'].values[-1] - data_2['BidOpen'].values[0])
    print(data_1.tail())
    print(data_2.head())

    print("new index" + str(data_2.index))
    data_con = pd.concat([data_1, data_2])
    print("size" + str(data_con.size))
    return data_con