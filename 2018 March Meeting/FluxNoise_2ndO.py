import numpy as np

sensitivity = np.array([650, 3196, 324, 147, 1270, 61, 1740])
e = 1.602e-19  # Fundamental charge
h = 6.62e-34  # Placnk's constant
phi_o = h / (2 * e)  # Flux quantum
gamma = 2 * np.pi * sensitivity * 1e9 * (2.0e-6) ** 2
t2 = 1.0 / gamma
print(t2 * 1e3)
