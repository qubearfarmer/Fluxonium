import numpy as np
from qutip import *
from scipy.special import kv

#Define constants
e = 1.602e-19    #Fundamental charge
h = 6.626e-34    #Placnk's constant
phi_o = h/(2*e) #Flux quantum

def bare_hamiltonian(N, scale):
    a = tensor(destroy(N))
    x = (a+a.dag())/2
    p = 1j*(a.dag()-a) / 2
    H = 0.5*scale * (x**2 + p**2) - ((2*np.sqrt(np.pi)*x).cosm() + (2*np.sqrt(np.pi)*p).cosm())
    return H

def x_matrix_element(N, scale):
    a = tensor(destroy(N))
    x = (a + a.dag()) / 2.0
    p = 1j * (a.dag() - a) / 2.0
    H = 0.5 * scale * (x ** 2.0 + p ** 2.0) - ((2.0 * np.sqrt(np.pi) * x).cosm() + (2.0 * np.sqrt(np.pi) * p).cosm())

    eigen_energies, eigen_states = H.eigenstates()
    element = x.matrix_element(eigen_states[0],eigen_states[1])
    return element

def p_matrix_element(N, scale):
    a = tensor(destroy(N))
    x = (a + a.dag()) / 2
    p = 1j * (a.dag() - a) / 2
    H = 0.5 * scale * (x ** 2 + p ** 2) - ((2 * np.sqrt(np.pi) * x).cosm() + (2 * np.sqrt(np.pi) * p).cosm())

    eigen_energies, eigen_states = H.eigenstates()
    element = p.matrix_element(eigen_states[0],eigen_states[1])
    return element

scale = 1
print(x_matrix_element(100,scale=scale))
print(p_matrix_element(100,scale=scale))