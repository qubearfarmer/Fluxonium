import numpy as np
from matplotlib import pyplot as plt
from matplotlib import style

style.use('seaborn-paper')

antenna_length = np.array([1, 0.8, 0.7, 0.6, 0.5, 0.4])
C_shunt = np.array([])
g = np.array([])/2
#with small pad
C_shunt_sp = np.array([])
g_sp = np.array([])/2