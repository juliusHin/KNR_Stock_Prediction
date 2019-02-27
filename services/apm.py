# Add APMonitor toolbox available from
# http://apmonitor.com/wiki/index.php/Main/PythonApp
from apm import *

# server and application
s = 'http://byu.apmonitor.com'
a = 'regression'

# clear any prior application
apm(s,a,'clear all')