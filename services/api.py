from builtins import print

from alpha_vantage.timeseries import TimeSeries
import matplotlib.pyplot as plt

# you can get apiKey from AlphaVantage.co
from services.KEY import getApiKey

class API_Stock:
    data = []
    meta_data = []
    def getByStockSymbolDaily(self, symbolStock):
        ts = TimeSeries(key=getApiKey(), output_format='csv')
        data, meta_data = ts.get_daily(symbol=symbolStock, outputsize="full")
        # print(data)
        # print(meta_data)
        return data, meta_data
