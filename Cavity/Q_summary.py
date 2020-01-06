import numpy as np
from matplotlib import pyplot as plt
from matplotlib import style
import datetime
from matplotlib.dates import (YEARLY, DateFormatter,
                              rrulewrapper, RRuleLocator, drange)
style.use('seaborn-paper')

#High power
fname = 'Z:\Old Server Data\GROUP\Shared\Projects\Cavity\Data\\Q vs date 2col.txt'
data = np.genfromtxt(fname,skip_footer = 0, skip_header =0, delimiter = '')

Q = data[:,1]
date = []
date = np.append(date, datetime.date(2015,6,5))
date = np.append(date, datetime.date(2015,7,10))
date = np.append(date, datetime.date(2015,7,22))
date = np.append(date, datetime.date(2015,9,9))
date = np.append(date, datetime.date(2015,9,24))
date = np.append(date, datetime.date(2015,10,5))
date = np.append(date, datetime.date(2015,10,20))
date = np.append(date, datetime.date(2015,10,21))
date = np.append(date, datetime.date(2015,10,29))
date = np.append(date, datetime.date(2015,11,4))
date = np.append(date, datetime.date(2015,11,12))
formatter = DateFormatter('%m/%d/%y')
fig, ax = plt.subplots(figsize = [9,6])
plt.plot_date(date, Q*1e-6, marker='s', markersize = 10)

#low power
date = []
date = np.append(date, datetime.date(2015,10,5))
date = np.append(date, datetime.date(2015,10,20))
date = np.append(date, datetime.date(2015,10,29))
date = np.append(date, datetime.date(2015,11,4))
date = np.append(date, datetime.date(2015,11,12))
Q = np.array([1.4e7,8.33e6,7.72e6, 7.27e6, 8e6])
formatter = DateFormatter('%m/%d/%y')
plt.plot_date(date, Q*1e-6, marker='o', markersize = 10)
ax.xaxis.set_major_formatter(formatter)
ax.xaxis.set_tick_params(rotation=30, labelsize=14)
ax.yaxis.set_tick_params(labelsize=16)
plt.yscale('log')
plt.show()

