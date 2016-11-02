from Fluxonium_hamiltonians.Single_small_junction import charge_matrix_element as nem
from Fluxonium_hamiltonians.Single_small_junction import phase_matrix_element as pem
from Fluxonium_hamiltonians.Single_small_junction import qp_matrix_element as qpem
import numpy as np
from matplotlib import pyplot as plt

N = 50
E_l = 0.1
E_c = 10
E_j = 10
iState = 0
fState = 1

phi_ext = np.linspace(0,0.5,100)
element = np.zeros(len(phi_ext))
for idx, phi in enumerate(phi_ext):
    element[idx]=abs(nem(N, E_l, E_c, E_j, phi*2*np.pi, iState, fState))
fig1 = plt.figure(1)
plt.plot(phi_ext, element)

phi_ext = np.linspace(0,0.5,100)
element = np.zeros(len(phi_ext))
for idx, phi in enumerate(phi_ext):
    element[idx]=abs(pem(N, E_l, E_c, E_j, phi*2*np.pi, iState, fState))
fig1 = plt.figure(1)
plt.plot(phi_ext, element)


plt.show()
