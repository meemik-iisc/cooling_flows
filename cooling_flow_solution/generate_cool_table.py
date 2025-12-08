import numpy as np
import matplotlib.pyplot as plt
# Simulated Sutherland & Dopita 1993-like base cooling table with temp (K) and cooling rate (erg cm^3 s^-1)
# This is a rough representative table; for actual data, real tables should be used or Cloudy tables.
T_Solar = np.array([
    1e4, 2e4, 5e4, 1e5, 2e5, 5e5, 1e6, 2e6, 5e6, 1e7, 2e7, 5e7, 1e8])

Lambda_Solar = np.array([
    1e-23, 3e-22, 1e-21, 2.5e-22, 1e-22, 5e-23, 2e-23, 5e-24, 1e-24, 5e-25, 2e-25, 1e-25, 5e-26
])

# Convert to code units:
# kpc = 3.086e21 cm, km/s = 1e5 cm/s, proton mass = 1.6726e-24 g
# Energy unit (code) = mp * v^2 = 1.6726e-14 erg
# Time unit = length / velocity = 3.086e16 s
# Volume unit = (3.086e21)^3 cm^3

length_unit_cm = 3.086e21
speed_unit_cm_s = 1e5
mass_unit_g = 1.6726e-24
energy_unit_erg = mass_unit_g * speed_unit_cm_s**2
time_unit_s = length_unit_cm / speed_unit_cm_s
volume_unit_cm3 = length_unit_cm**3

Lambda_code = Lambda_Solar * time_unit_s / energy_unit_erg / volume_unit_cm3

# Since Temperature code unit = 1 K, T_Solar is already in code units
T_code = T_Solar

# Cutoff cooling below 10^4 K exactly
mask = T_code < 1e4
Lambda_code[mask] = 0

# Format cooling table data for PLUTO
with open('cooltable_sd1993.dat', 'w') as f:
    for T, L in zip(T_code, Lambda_code):
        f.write(f"{T:.6e}   {L:.6e}\n")

"Sutherland & Dopita 1993 rough cooling table converted to code units and saved as 'cooltable_sd1993.dat'"

plt.plot(T_code, Lambda_code)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Temperature (K)')
plt.ylabel('Cooling Rate (erg cm^-3 s^-1)')
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
plt.savefig(os.path.join(script_dir, "cooling_table1.png"))
plt.show()
