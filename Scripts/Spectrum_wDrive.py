from Fluxonium_hamiltonians.Single_small_junction import bare_hamiltonian
from Fluxonium_hamiltonians.Single_small_junction import charge_matrix_element as nem
import numpy as np
from matplotlib import pyplot as plt
from qutip import*

#Qubit and computation parameters
N = 60
E_l = 0.5
E_c = 2.5
E_j = 10
level_num = 10
# wd = 8.89
g = 0.2

phi_ext = 0
#Compute eigennergies and matrix elements
energies = np.zeros(level_num)
nElement = np.zeros((level_num,level_num))
H = bare_hamiltonian(N, E_l, E_c, E_j, phi_ext*2*np.pi)
for idx in range(level_num):
    energies[idx]=H.eigenenergies()[idx]
for idx in range (level_num):
    for idy in range (level_num):
        nElement[idx,idy] = nem(N, E_l, E_c, E_j, phi_ext, idx, idy)

wd = energies[1] - energies[0]
#Interaction Hamiltonian
state = basis(level_num,0)
H=state*state.dag()*energies[0]
for idx in range(1,level_num):
    state = basis(level_num,idx)
    H=H+state*state.dag()*(energies[idx]-wd)
for idx in range(level_num):
    for idy in range(level_num):
        drive_term = basis(level_num,idx)*basis(level_num,idy).dag()*nElement[idy,idx]
        H = H + drive_term*g

#Time dynamics
time_list = np.linspace(0,1000,1001)
psi0 = basis(level_num,0)
collapse_ops = []
ope1 = basis(level_num,0)*basis(level_num,0).dag()
ope2 = basis(level_num,1)*basis(level_num,1).dag()
quantum_operators = [ope1, ope2]
output = mesolve(H, psi0, time_list, collapse_ops, quantum_operators)
st0 = output.expect[0]
st1 = output.expect[1]
plt.plot(time_list, st0, time_list, st1)
plt.show()
