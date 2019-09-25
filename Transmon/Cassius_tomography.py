import numpy as np
from matplotlib import pyplot as plt


#Input
signal_gnd = #=beta_I*I + beta_Z*Z
signal_etd = #=beta_I*I - beta_Z*Z
signal_I = #=beta_I*I + beta_Z*Z
signal_X2p = #=beta_I*I + beta_Z*Y
signal_Y2m = #=beta_I*I + beta_Z*X

#Get beta coefficients
beta_I = 0.5*(signal_gnd+signal_etd)
beta_Z = 0.5*(signal_gnd-signal_etd)


#Compute and reconstruct tomography
#rho = 0.5(a b // c d)
#sigma_Z = 1 for gnd state, -1 for etd state




