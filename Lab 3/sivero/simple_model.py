import numpy as np


muabo = np.genfromtxt("./muabo.txt", delimiter=",")
muabd = np.genfromtxt("./muabd.txt", delimiter=",")

red_wavelength = 600
green_wavelength = 520
blue_wavelength = 460

wavelength = np.array([red_wavelength, green_wavelength, blue_wavelength])

def mua_blood_oxy(x): return np.interp(x, muabo[:, 0], muabo[:, 1])
def mua_blood_deoxy(x): return np.interp(x, muabd[:, 0], muabd[:, 1])

bvf = 0.01 # Blood volume fraction, average blood amount in tissue
oxy = 0.8 # Blood oxygenation

# Absorption coefficient ($\mu_a$ in lab text)
# Units: 1/m
mua_other = 25 # Background absorption due to collagen, et cetera
mua_blood = (mua_blood_oxy(wavelength)*oxy # Absorption due to
            + mua_blood_deoxy(wavelength)*(1-oxy)) # pure blood
mua = mua_blood*bvf + mua_other

# reduced scattering coefficient ($\mu_s^\prime$ in lab text)
# the numerical constants are thanks to N. Bashkatov, E. A. Genina and
# V. V. Tuchin. Optical properties of skin, subcutaneous and muscle
# tissues: A review. In: J. Innov. Opt. Health Sci., 4(1):9-38, 2011.
# Units: 1/m
musr = 100 * (17.6*(wavelength/500)**-4 + 18.78*(wavelength/500)**-0.22)

# mua and musr are now available as shape (3,) arrays
# Red, green and blue correspond to indexes 0, 1 and 2, respectively

# TODO calculate penetration depth

delta = 1 / np.sqrt(3*mua*(mua + musr))

print("Absorption coefficient (1/m):", mua)
print("Reduced scattering coefficient (1/m):", musr)
print("Penetration depth (mm):", delta*1e3)

d = 14e-3
T = np.exp(-d/delta)
print("Transmission percentage:", T*100)


R = np.exp(-2*d/delta)
print("Reflection percentage:", R*100)


K_R = np.abs(15.23-82.60) /82.60
print(K_R)

K_G = np.abs(0.13 -69.70) / 69.70
print(K_G)

K_B = np.abs(0.003 - 60.46) / 60.46
print(K_B)