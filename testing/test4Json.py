from pprint import pprint
from builtins import print

from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
from alpha_vantage.sectorperformance import SectorPerformances
from alpha_vantage.cryptocurrencies import CryptoCurrencies
import matplotlib
import matplotlib.pyplot as plt
import os

# you can get apiKey from AlphaVantage.co
from services.KEY import getApiKey

def main():
    ts = TimeSeries(key=getApiKey(),output_format="pandas")

    data, meta_data = ts.get_daily('ADHI.JKT',outputsize="full")
    # print(data)
    # print(meta_data)
    # pprint(data.head)
    data.describe()
if __name__ == '__main__':
    main()