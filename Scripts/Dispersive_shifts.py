from Fluxonium_hamiltonians.Single_small_junction import charge_dispersive_shift as nChi
from Fluxonium_hamiltonians.Single_small_junction import flux_dispersive_shift as pChi
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider

directory = "C:\Data\Fluxonium #10 simulations"
simulation = "Dispersive_shift"
path = directory + "\\" + simulation

N = 50
E_l = 0.5
E_c = 3
E_j = 2
level_num = 15
g = 0.2
path = path+'_E_l='+str(E_l)+'E_c='+str(E_c)+'E_j='+str(E_j)
iState = 0
fState = 1
phi_ext = np.linspace(0,1,100)
wr = np.linspace(2,22,40)
chi = np.zeros((len(phi_ext),len(wr)))

'''
for idy,w in enumerate(wr):
    for idx, phi in enumerate(phi_ext):
        chi[idx,idy]= nChi(N, level_num, E_l, E_c, E_j, phi*2*np.pi, iState, fState, w, g)
np.savetxt(path+'.txt',chi)
'''

#Plotting
chi = np.genfromtxt(path+'.txt')
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)
l, = plt.plot(phi_ext,chi[:,0],'b.')
axcolor = 'lightgoldenrodyellow'
axCav = plt.axes([0.25, 0.1, 0.65, 0.03], axisbg=axcolor)
sCav= Slider(axCav, 'Cavity index', 0 , len(wr)-1, valinit = 0, valfmt='%0.00f')
def update(cav_index):
    cav_index = sCav.val
    l.set_ydata(chi[:,cav_index])
    plt.title('Cav res='+str(wr[cav_index]))

sCav.on_changed(update)
plt.xlabel('External flux')
plt.show()