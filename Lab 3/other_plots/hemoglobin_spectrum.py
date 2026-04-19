import numpy as np
import matplotlib.pyplot as plt

lambda_nm, hemoglobin_o2, hemoglobin = np.loadtxt("hemoglobin_data.csv", skiprows=2, unpack=True)

# Convert from cm^-1/M to M^-1 m^-1 (1 cm^-1 = 100 m^-1)
hb_o2_m = hemoglobin_o2 * 100.0
hb_m = hemoglobin * 100.0

fig, ax = plt.subplots(figsize=(8, 6))
ax.semilogy(lambda_nm, hb_o2_m, linewidth=1.5, label="Hemoglobin, oxygenated")
ax.semilogy(lambda_nm, hb_m, linewidth=1.5, linestyle="--", label="Hemoglobin, deoxygenated")

ax.set_xlabel("Wavelength λ [nm]")
ax.set_ylabel(r"Molar extinction Coefficient $\mu_a$ [M$^{-1}$ m$^{-1}$]")
ax.set_xlim(250, 1000)
ax.set_ylim(1e2, 1e8)
ax.grid()
ax.legend()

fig.savefig("hemoglobin_plot.png")
plt.show()
