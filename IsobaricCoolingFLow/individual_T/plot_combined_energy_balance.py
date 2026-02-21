import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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
# Kt = 1e8               # ergcm^-1s^-1K^-1

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
# Kt_str = f"{Kt:.0e}".replace("+0", "").replace("+", "")
# if Kt == 0:
#     Kt_str = "0"
# filename = f"temp_profile_Kt_{Kt_str}"
output_dir = os.path.join(work_dir, "energy_balance")

#Get data
df_1e4 = pd.read_csv(os.path.join(work_dir,"temp_profile_Kt_1e4.csv"))
df_1e8 = pd.read_csv(os.path.join(work_dir,"temp_profile_Kt_1e8.csv"))

M_dot_1e4 = df_1e4['M_dot'][0]
M_dot_1e8 = df_1e8['M_dot'][0]
# M_dot = 1e-23
print("M_dot for Kt = 1e4 = ",M_dot_1e4)
print("M_dot for Kt = 1e8 = ",M_dot_1e8)


def energy_balance(T_tilde,z_tilde, M_dot, Kt):
    dT_tildedz_tilde = np.gradient(T_tilde,z_tilde)
    d2T_tildedz_tilde2 = np.gradient(dT_tildedz_tilde,z_tilde)
    heat_flux = Kt*T0*d2T_tildedz_tilde2/delz**2
    advection = (M_dot*gamma*kB*T0*dT_tildedz_tilde)/((gamma-1)*mu*mp*delz)
    cooling = (((P0/kB)**2)*cool_lambda(T_tilde*T0))/((T_tilde*T0)**2)
    return heat_flux, advection, cooling


z_tilde_1e4 = df_1e4['z_pc']*pc/delz
T_tilde_1e4 = df_1e4['T_tilde']

z_tilde_1e8 = df_1e8['z_pc']*pc/delz
T_tilde_1e8 = df_1e8['T_tilde']

heat_flux1,advection1,cooling1 = energy_balance(T_tilde_1e4,z_tilde_1e4, M_dot_1e4, Kt=1e4)
heat_flux2,advection2,cooling2 = energy_balance(T_tilde_1e8,z_tilde_1e8, M_dot_1e8, Kt=1e8)
# print(heat_flux[0],advection[0],cooling[0])
energy_norm = 1e-25

fig1,ax1 = plt.subplots(figsize=(8,6))
ax1.plot(np.log10(T_tilde_1e4*T0), np.abs(heat_flux1),  linewidth=2, markersize=2, color='red')
ax1.plot(np.log10(T_tilde_1e4*T0), np.abs(advection1),  linewidth=2, markersize=2, color='green') 
ax1.plot(np.log10(T_tilde_1e4*T0), np.abs(cooling1),  linewidth=2, markersize=2, color='blue')
ax1.plot(np.log10(T_tilde_1e4*T0), np.abs(heat_flux2),  linewidth=2, markersize=2, color='red',    linestyle='dotted')
ax1.plot(np.log10(T_tilde_1e4*T0), np.abs(advection2),  linewidth=2, markersize=2, color='green',  linestyle='dotted')
ax1.plot(np.log10(T_tilde_1e4*T0), np.abs(cooling2),  linewidth=2, markersize=2, color='blue',   linestyle='dotted')



# ax1.plot(np.log10(T_tilde*T0), np.abs(heat_flux+advection-cooling),'o-', linewidth=2, markersize=4, label="Heat Flux + Advection - Cooling")
ax1.set_xlabel('$log_{10}T$ [K]', fontsize = 14)
ax1.set_ylabel(r'($erg\,cm^{-3}\,s^{-1}$)', fontsize = 14)
ax1.set_yscale("log")
ax1.set_ylim(1e-30,1e-21)
legend1 = ax1.legend(
    handles=[
        Line2D([0], [0], color='red',    lw=2, label='Thermal Conduction'),
        Line2D([0], [0], color='green',  lw=2, label='Advection'), 
        Line2D([0], [0], color='blue',   lw=2, label='Cooling')
    ],
    loc='lower left', fontsize=12, frameon=True, fancybox=True, shadow=True
)

# Legend 2: Linestyles (Cases) - top-right  
legend2 = ax1.legend(
    handles=[
        Line2D([0], [0], color='k', lw=2, linestyle='solid',    label=r'$K_t=10^4$'),
        Line2D([0], [0], color='k', lw=2, linestyle='dotted', label=r'$K_t=10^8$')
    ],
    loc='lower right', fontsize=12, frameon=True, fancybox=True, shadow=True
)
ax1.add_artist(legend1)
ax1.tick_params(axis='both', which='major', labelsize=14)
# Customize log grid
ax1.minorticks_on()
ax1.grid(True, which='both', ls='-', alpha=0.3)
ax1.grid(which='major', alpha=0.7, linewidth=1.2)
ax1.grid(which='minor', alpha=0.5, linewidth=1.0)
save_filename1 = os.path.join(output_dir,f"energy_balance_combined_logscale.png")
plt.savefig(save_filename1)
plt.close(fig1)