import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import os
script_dir = os.path.dirname(os.path.abspath(__file__))
df_0 = pd.read_csv(os.path.join(script_dir,"temp_profile_Kt_0.csv"))
df_4 = pd.read_csv(os.path.join(script_dir,"temp_profile_Kt_1e4.csv"))
df_5 = pd.read_csv(os.path.join(script_dir,"temp_profile_Kt_1e5.csv"))
df_6 = pd.read_csv(os.path.join(script_dir,"temp_profile_Kt_1e6.csv"))
df_7 = pd.read_csv(os.path.join(script_dir,"temp_profile_Kt_1e7.csv"))
df_8 = pd.read_csv(os.path.join(script_dir,"temp_profile_Kt_1e8.csv"))

plt.plot(df_0['z_pc'], np.log10(df_0['T_K']), label=f"Kt = 0, Mdot={df_0['M_dot'][0]:.2e}")
plt.plot(df_4['z_pc'], np.log10(df_4['T_K']), label=f"Kt = 1e4, Mdot={df_4['M_dot'][0]:.2e}")
plt.plot(df_5['z_pc'], np.log10(df_5['T_K']), label=f"Kt = 1e5, Mdot={df_5['M_dot'][0]:.2e}")
plt.plot(df_6['z_pc'], np.log10(df_6['T_K']), label=f"Kt = 1e6, Mdot={df_6['M_dot'][0]:.2e}")
plt.plot(df_7['z_pc'], np.log10(df_7['T_K']), label=f"Kt = 1e7, Mdot={df_7['M_dot'][0]:.2e}")
plt.plot(df_8['z_pc'], np.log10(df_8['T_K']), label=f"Kt = 1e8, Mdot={df_8['M_dot'][0]:.2e}")
plt.xlabel('z[pc]')
plt.ylabel('$log_{10}T[K]$')
plt.grid()
plt.legend()
save_filename = os.path.join(script_dir,"temp_profile.png")
plt.savefig(save_filename)