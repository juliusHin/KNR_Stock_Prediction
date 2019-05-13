import math
import numpy as np
import pandas as pd
from scipy.spatial import distance
from sklearn.neighbors import KNeighborsRegressor
from sklearn import metrics
from alpha_vantage.timeseries import TimeSeries
from services.KEY import getApiKey
import plotly
import plotly.plotly as py
import plotly.graph_objs as go


plotly.tools.set_credentials_file(username='junhin04', api_key='c0YKlYKB4vqlUGfaEDNO')

