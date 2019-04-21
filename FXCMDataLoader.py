from enum import Enum
import pandas as pd

class Interval(Enum):
    MINUT = 'm1'
    HOURE = 'H1'
    DAY = 'D1'

http = 'https://candledata.fxcorporate.com'
file_formate = '.csv.gz'

class FXCMDataLoader:
    def load(self, interval, instrument, startDate, finishDate):
        for yr in range(startDate["year"], finishDate["year"] + 1):
            for wk in range(startDate["week"], finishDate["week"] + 1):
                web_location = http + '/' + interval + '/' + instrument + '/' + yr + '/' + wk + file_formate
                data = pd.read_csv(web_location, index_col=0, parse_dates=True)
