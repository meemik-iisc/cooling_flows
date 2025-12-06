import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
import os

# ============================================================================
# CONSTANTS
# ============================================================================
kB = 1.38e-16          # erg/K
mp = 1.67e-24          # g
gamma = 5.0/3.0
mu = 1.0
pc = 3.086e18          # cm

# ============================================================================
# PARAMETERS
# ============================================================================
zc = -1.0 * pc
zh = 1.0 * pc
delz = zh - zc
Tc = 1e4               # K
Th = 1e6               # K
T0 = np.sqrt(Th * Tc)
P0 = 1e3 * kB          # erg/cm^3
Kt = 1e4               # ergcm^-1s^-1K^-1

# Dimensionless parameters
zc_tilde = zc / delz
zh_tilde = zh / delz
Tc_tilde = Tc / T0
Th_tilde = Th / T0

# ============================================================================
# COOLING TABLE
# ============================================================================
data = np.loadtxt('cooltable.dat')
T_tab = data[:, 0]
Lambda_tab = data[:, 1]
Lambda_interp = interp1d(T_tab, Lambda_tab, kind='linear', 
                         bounds_error=False, fill_value='extrapolate')

def cool_lambda(T):
    """Safe cooling function with clamping"""
    T_safe = np.clip(T, T_tab.min() * 0.95, T_tab.max() * 1.05)
    return Lambda_interp(T_safe)


work_dir = os.path.dirname(os.path.abspath(__file__))
Kt_str = f"{Kt:.0e}".replace("+0", "").replace("+", "")
if Kt == 0:
    Kt_str = "0"
filename = f"temp_profile_Kt_{Kt_str}"
output_dir = os.path.join(work_dir, "energy_balance")

#Get data
df = pd.read_csv(os.path.join(work_dir,f"{filename}.csv"))

M_dot = df['M_dot'][0]
# M_dot = 1e-23
print("M_dot = ",M_dot)


def energy_balance(T_tilde,z_tilde):
    dT_tildedz_tilde = np.gradient(T_tilde,z_tilde)
    d2T_tildedz_tilde2 = np.gradient(dT_tildedz_tilde,z_tilde)
    heat_flux = Kt*T0*d2T_tildedz_tilde2/delz**2
    advection = (M_dot*gamma*kB*T0*dT_tildedz_tilde)/((gamma-1)*mu*mp*delz)
    cooling = (((P0/kB)**2)*cool_lambda(T_tilde*T0))/((T_tilde*T0)**2)
    return heat_flux, advection, cooling


z_tilde = df['z_pc']*pc/delz
T_tilde = df['T_tilde']

heat_flux,advection,cooling = energy_balance(T_tilde,z_tilde)
print(heat_flux[0],advection[0],cooling[0])
energy_norm = 1e-25

fig1,ax1 = plt.subplots(figsize=(8,6))
ax1.plot(z_tilde*delz/pc, heat_flux/energy_norm, 'o-', linewidth=2, markersize=2,label="Heat Flux")
ax1.plot(z_tilde*delz/pc, advection/energy_norm, 'o-', linewidth=2, markersize=2,label="Advection")
ax1.plot(z_tilde*delz/pc, cooling/energy_norm, 'o-', linewidth=2, markersize=2,label="Cooling")
ax1.plot(z_tilde*delz/pc, (heat_flux+advection-cooling)/energy_norm, 'o-', linewidth=2, markersize=2,label="Heat Flux + Advection - Cooling")
ax1.set_xlabel('z[pc]')
ax1.set_ylabel('Energy Balance ($10^{-25} erg cm^{-3} s^{-1}$)')
ax1.grid()
ax1.legend()
save_filename1 = os.path.join(output_dir,f"energy_balance_Kt_{Kt_str}_vs_z.png")
plt.savefig(save_filename1)
plt.close(fig1)

fig2,ax2 = plt.subplots(figsize=(8,6))
ax2.plot(np.log10(T_tilde*T0),heat_flux/energy_norm, 'o-', linewidth=2, markersize=2,label="Heat Flux")
ax2.plot(np.log10(T_tilde*T0),advection/energy_norm, 'o-', linewidth=2, markersize=2,label="Advection")
ax2.plot(np.log10(T_tilde*T0),cooling/energy_norm, 'o-', linewidth=2, markersize=2,label="Cooling")
ax2.plot(np.log10(T_tilde*T0), (heat_flux+advection-cooling)/energy_norm,'o-', linewidth=2, markersize=4, label="Heat Flux + Advection - Cooling")
ax2.set_xlabel('$log_{10}T  [K]$')
ax2.set_ylabel('Energy Balance ($10^{-25} erg cm^{-3} s^{-1}$)')
ax2.grid()
ax2.legend()
save_filename2 = os.path.join(output_dir,f"energy_balance_Kt_{Kt_str}_vs_T.png")
plt.savefig(save_filename2)
plt.close(fig2)