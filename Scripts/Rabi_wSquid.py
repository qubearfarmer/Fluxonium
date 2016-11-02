from Fluxonium_hamiltonians.Squid_small_junctions import bare_hamiltonian
from Fluxonium_hamiltonians.Squid_small_junctions import charge_matrix_element as nem
import numpy as np
from matplotlib import pyplot as plt
from qutip import*

#Qubit and computation parameters
#Define constants
e = 1.602e-19    #Fundamental charge
h = 6.62e-34    #Placnk's constant
phi_o = h/(2*e) #Flux quantum

#Qubit and computation parameters
N = 50
E_l = 0.746959655208
E_c = 0.547943694372
E_j_sum = 21.9627179709
level_num = 10
B_coeff = 60
A_j = 3.80888914574e-12
A_c = 1.49982268962e-10
beta_squid = 0.00378012644185
beta_ext = 0.341308382441
d=0.0996032153487
current = np.linspace(0.03,0.045,1501)
energies = np.zeros((len(current),level_num))
shift = (41.6713-41.6413)

#Compute eigenenergies
for idx, curr in enumerate(current):
    curr = curr + shift * 1e-3
    flux_squid = curr*B_coeff*A_j*1e-4
    flux_ext = curr*B_coeff*A_c*1e-4
    H = bare_hamiltonian(N, E_l, E_c, E_j_sum, d, 2*np.pi*(flux_squid/phi_o - beta_squid),
                         2 * np.pi * (flux_ext / phi_o - beta_ext))
    for idy in range(level_num):
        energies[idx,idy] = H.eigenenergies()[idy]

wd = 9.5#energies[2] - energies[0]
g = 0.05

#Interaction Hamiltonian
H=0
for idx in range(1,level_num):
    state = basis(level_num,idx)
    H=H + state*state.dag()*(energies[idx]-energies[0]-wd)
for idx in range(level_num):
    for idy in range(level_num):
        drive_term = basis(level_num,idx)*basis(level_num,idy).dag()*nElement[idy,idx]
        H = H + drive_term*g

#Time dynamics
time_list = np.linspace(0,100,101)
psi0 = basis(level_num,0)
collapse_ops = []
ope0 = basis(level_num,0)*basis(level_num,0).dag()
ope1 = basis(level_num,1)*basis(level_num,1).dag()
ope2 = basis(level_num,2)*basis(level_num,2).dag()
ope3 = basis(level_num,3)*basis(level_num,3).dag()
quantum_operators = [ope0, ope1, ope2, ope3]
output = mesolve(H, psi0, time_list, collapse_ops, quantum_operators)
st0 = output.expect[0]
st1 = output.expect[1]
st2 = output.expect[2]
st3 = output.expect[3]
plt.plot(time_list, st1, time_list, st2, time_list, st3)
plt.show()
