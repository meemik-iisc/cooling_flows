import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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
prefactor = (P0*mu)/kB

def density_mp_cc(df):
    return prefactor/df['T_K']

def velocity_km_s(df):
    return df['M_dot'][0]/(density_mp_cc(df)*mp*1e5)

def first_derivative(df):
    dT_dz = np.gradient(df['T_K'],df['z_pc'])
    return dT_dz

def double_derivative(df):
    dT_dz = np.gradient(df['T_K'],df['z_pc'])
    d2Tdz2 = np.gradient(dT_dz,df['z_pc'])
    return d2Tdz2

#Load data
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
df_0 = pd.read_csv(os.path.join(script_dir,"temp_profile_Kt_0.csv"))
df_4 = pd.read_csv(os.path.join(script_dir,"temp_profile_Kt_1e4.csv"))
df_5 = pd.read_csv(os.path.join(script_dir,"temp_profile_Kt_1e5.csv"))
df_6 = pd.read_csv(os.path.join(script_dir,"temp_profile_Kt_1e6.csv"))
df_7 = pd.read_csv(os.path.join(script_dir,"temp_profile_Kt_1e7.csv"))
df_8 = pd.read_csv(os.path.join(script_dir,"temp_profile_Kt_1e8.csv"))

fig1,ax1 = plt.subplots(figsize=(8,6))
ax1.plot(df_0['z_pc'],np.log10(density_mp_cc(df_0)),label=f"Kt = 0, Mdot={df_0['M_dot'][0]:.2e}")
ax1.plot(df_4['z_pc'],np.log10(density_mp_cc(df_4)),label=f"Kt = 1e4, Mdot={df_4['M_dot'][0]:.2e}")
ax1.plot(df_5['z_pc'],np.log10(density_mp_cc(df_5)),label=f"Kt = 1e5, Mdot={df_5['M_dot'][0]:.2e}")
ax1.plot(df_6['z_pc'],np.log10(density_mp_cc(df_6)),label=f"Kt = 1e6, Mdot={df_6['M_dot'][0]:.2e}")
ax1.plot(df_7['z_pc'],np.log10(density_mp_cc(df_7)),label=f"Kt = 1e7, Mdot={df_7['M_dot'][0]:.2e}")
ax1.plot(df_8['z_pc'],np.log10(density_mp_cc(df_8)),label=f"Kt = 1e8, Mdot={df_8['M_dot'][0]:.2e}")
ax1.set_xlabel('z[pc]')
ax1.set_ylabel(r'$\log_{10} \rho \, [m_p/cm^3]$')
ax1.set_title('Density Profiles for Different Kt')
ax1.grid()
ax1.legend()
save_filename = os.path.join(script_dir,"density_profile.png")
plt.savefig(save_filename)
plt.close(fig1)

fig2,ax2 = plt.subplots(figsize=(8,6))
ax2.plot(df_0['z_pc'],np.log10(np.abs(velocity_km_s(df_0))),label=f"Kt = 0, Mdot={df_0['M_dot'][0]:.2e}")
ax2.plot(df_4['z_pc'],np.log10(np.abs(velocity_km_s(df_4))),label=f"Kt = 1e4, Mdot={df_4['M_dot'][0]:.2e}")
ax2.plot(df_5['z_pc'],np.log10(np.abs(velocity_km_s(df_5))),label=f"Kt = 1e5, Mdot={df_5['M_dot'][0]:.2e}")
ax2.plot(df_6['z_pc'],np.log10(np.abs(velocity_km_s(df_6))),label=f"Kt = 1e6, Mdot={df_6['M_dot'][0]:.2e}")
ax2.plot(df_7['z_pc'],np.log10(np.abs(velocity_km_s(df_7))),label=f"Kt = 1e7, Mdot={df_7['M_dot'][0]:.2e}")
ax2.plot(df_8['z_pc'],np.log10(np.abs(velocity_km_s(df_8))),label=f"Kt = 1e8, Mdot={df_8['M_dot'][0]:.2e}")
ax2.set_xlabel('z[pc]')
ax2.set_ylabel(r'$log_{10}v \, [km/s]$')
ax2.set_title('Velocity Profiles for Different Kt')
ax2.grid()
ax2.legend()
save_filename2 = os.path.join(script_dir,"velocity_profile.png")
plt.savefig(save_filename2)
plt.close(fig2)

fig3,ax3 = plt.subplots(figsize=(8,6))
ax3.plot(np.log10(df_4['T_K']),np.log10(np.abs(double_derivative(df_4))),label=f"Kt = 1e4, Mdot={df_4['M_dot'][0]:.2e}")
ax3.plot(np.log10(df_5['T_K']),np.log10(np.abs(double_derivative(df_5))),label=f"Kt = 1e5, Mdot={df_5['M_dot'][0]:.2e}")
ax3.plot(np.log10(df_6['T_K']),np.log10(np.abs(double_derivative(df_6))),label=f"Kt = 1e6, Mdot={df_6['M_dot'][0]:.2e}")
ax3.plot(np.log10(df_7['T_K']),np.log10(np.abs(double_derivative(df_7))),label=f"Kt = 1e7, Mdot={df_7['M_dot'][0]:.2e}")
ax3.plot(np.log10(df_8['T_K']),np.log10(np.abs(double_derivative(df_8))),label=f"Kt = 1e8, Mdot={df_8['M_dot'][0]:.2e}" )
ax3.set_xlabel(r'$log_{10}T[K]$')
ax3.set_ylabel(r'$log_{10}d^2T/dz^2$')
ax3.set_title('Second Derivative Profiles for Different Kt')
ax3.grid()
ax3.legend()
save_filename3 = os.path.join(script_dir,"second_derivative_profile.png")
plt.savefig(save_filename3)
plt.close(fig3)

fig4,ax4 = plt.subplots(figsize=(8,6))
ax4.plot(np.log10(df_4['T_K']),np.log10(np.abs(first_derivative(df_4))),label=f"Kt = 1e4, Mdot={df_4['M_dot'][0]:.2e}")
ax4.plot(np.log10(df_5['T_K']),np.log10(np.abs(first_derivative(df_5))),label=f"Kt = 1e5, Mdot={df_5['M_dot'][0]:.2e}")
ax4.plot(np.log10(df_6['T_K']),np.log10(np.abs(first_derivative(df_6))),label=f"Kt = 1e6, Mdot={df_6['M_dot'][0]:.2e}")
ax4.plot(np.log10(df_7['T_K']),np.log10(np.abs(first_derivative(df_7))),label=f"Kt = 1e7, Mdot={df_7['M_dot'][0]:.2e}")
ax4.plot(np.log10(df_8['T_K']),np.log10(np.abs(first_derivative(df_8))),label=f"Kt = 1e8, Mdot={df_8['M_dot'][0]:.2e}" )
ax4.set_xlabel(r'$log_{10}T[K]$')
ax4.set_ylabel(r'$log_{10}dT/dz$')
ax4.set_title('First Derivative Profiles for Different Kt')
ax4.grid()
ax4.legend()
save_filename4 = os.path.join(script_dir,"first_derivative_profile.png")
plt.savefig(save_filename4)
plt.close(fig4)