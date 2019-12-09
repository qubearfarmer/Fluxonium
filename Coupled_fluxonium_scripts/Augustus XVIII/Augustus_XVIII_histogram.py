import numpy as np
import sys
sys.path.append('C:\Program Files (x86)\Labber\Script')
import Labber
from matplotlib import pyplot as plt
from qutip import*
from scipy.optimize import curve_fit

#constants
kB = 1.38e-23
h = 6.626e-34

# f = Labber.LogFile('C:\Data\Projects\Fluxonium\Data\Augustus 18\\2019\\11\Data_1120\Histogram.hdf5')
# guess1D = np.array([435, -480, 715, -305, 1300, -78, 1400, 86, 60])
f = Labber.LogFile('C:\Data\Projects\Fluxonium\Data\Augustus 18\\2019\\12\Data_1205\Histogram_heralding_check.hdf5')
guess1D = np.array([1435, -800, 715, -605, 2300, -150, 2400, 86, 60])
# d = f.getEntry(0)
# for (channel, value) in d.items():
#     print(channel, ":", value)

def gaussian (x,a,x0,sigma):
    return a*np.exp(-(x-x0)**2/sigma**2)
def gaussian4(x,a1,x1,a2,x2,a3,x3,a4,x4,sigma):
    return a1*np.exp(-(x-x1)**2/sigma**2) + a2*np.exp(-(x-x2)**2/sigma**2) + \
           a3*np.exp(-(x-x3)**2/sigma**2) + a4*np.exp(-(x-x4)**2/sigma**2)
def gaussian2d(coord,a,x0,y0,sigma):
    return a*np.exp((-(coord[0]-x0)**2-(coord[1]-y0)**2)/sigma**2)


signal = f.getData('AlazarTech Signal Demodulator - Channel A - Demodulated values')
# delay_time = f.getData('Multi-Qubit Pulse Generator - Delay after heralding')[:,0]
repetition = 10
signal = signal[0]
sReal = np.real(signal)*1e6
sImag = np.imag(signal)*1e6
H, xedges, yedges = np.histogram2d(sReal,sImag, bins = [100,100])
H = H.T
##########################################################
#2D fit
# plt.figure(1)
# X,Y = np.meshgrid(xedges, yedges)
# plt.pcolormesh(X,Y,H, cmap = 'GnBu')
# axis_nice_x = np.linspace(xedges[0], xedges[-1], 1001)
# axis_nice_y = np.linspace(yedges[0], yedges[-1], 1001)
# X,Y = np.meshgrid(axis_nice_x, axis_nice_y)
# #mean vector and covariance matrix
# mu = np.array([86, -442])
# sigma = np.array([[60 , 60], [60,  60]])
#
# # Pack X and Y into a single 3-dimensional array
# pos = np.empty(X.shape + (2,))
# pos[:, :, 0] = X
# pos[:, :, 1] = Y
#
# def multivariate_gaussian(pos, mu, Sigma):
#     """Return the multivariate Gaussian distribution on array pos.
#
#     pos is an array constructed by packing the meshed arrays of variables
#     x_1, x_2, x_3, ..., x_k into its _last_ dimension.
#
#     """
#
#     n = mu.shape[0]
#     Sigma_det = np.linalg.det(Sigma)
#     Sigma_inv = np.linalg.inv(Sigma)
#     N = np.sqrt((2*np.pi)**n * Sigma_det)
#     # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
#     # way across all the input variables.
#     fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)
#
#     return np.exp(-fac / 2) / N

# The distribution on the variables X, Y packed into pos.
# Z = multivariate_gaussian(pos, mu, Sigma)
# plt.plot(X,Y,Z)
# Create a surface plot and projected filled contour plot under it.
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot_surface(X, Y, Z, rstride=3, cstride=3, linewidth=1, antialiased=True,
#                 cmap=cm.viridis)
#
# cset = ax.contourf(X, Y, Z, zdir='z', offset=-0.15, cmap=cm.viridis)
#
# plt.pcolormesh(X,Y,Z)
##########################################################
#1D fit
plt.figure(2)
counts, edges = np.histogram(sReal, bins = 100)
plt.plot(edges[:-1], counts)
opt,cov = curve_fit(gaussian4, edges[:-1],counts, guess1D)
axis_nice = np.linspace(edges[0], edges[-1], 1001)
plt.plot(axis_nice, gaussian4(axis_nice,*opt))
a1,x1,a2,x2,a3,x3,a4,x4,sigma = opt
plt.plot(axis_nice, gaussian(axis_nice,a1,x1,sigma))
plt.plot(axis_nice, gaussian(axis_nice,a2,x2,sigma))
plt.plot(axis_nice, gaussian(axis_nice,a3,x3,sigma))
plt.plot(axis_nice, gaussian(axis_nice,a4,x4,sigma))
pgg = a3/(a1+a2+a3+a4)
peg = a4/(a1+a2+a3+a4)
pge = a2/(a1+a2+a3+a4)
pee = a1/(a1+a2+a3+a4)
f_a = 72.5e6
f_b = 136.5e6
T_a = h*f_a/(-kB*np.log((pee+peg)/(pgg+pge)))
T_b = h*f_b/(-kB*np.log((pee+pge)/(pgg+peg)))
populations = dict({'P_gg=': round(pgg,3), 'P_eg=': round(peg,3), 'P_ge=': round(pge,3), 'P_ee=': round(pee,3)})
temperatures = dict({'T_A(mK)=': round(T_a*1e3,3), 'T_B(mK)=':round(T_b*1e3,3)})
textstr =""
for key in temperatures:
    textstr = textstr+key+str(temperatures[key])+'\n'
plt.text(-1000,1000,textstr)

plt.show()