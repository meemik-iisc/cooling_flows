import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import os
Tc = 1e4      # temperature at zc
Th = 1e6      # temperature at zh
# Boundary conditions
zc = -1       # example lower boundary point
zh = 1        # example upper boundary point
delz = zh - zc
script_dir = os.path.dirname(os.path.abspath(__file__))
df_0 = pd.read_csv(os.path.join(script_dir,"temp_profile_Kt_0.csv"))
df_4 = pd.read_csv(os.path.join(script_dir,"temp_profile_Kt_1e4.csv"))
df_5 = pd.read_csv(os.path.join(script_dir,"temp_profile_Kt_1e5.csv"))
df_6 = pd.read_csv(os.path.join(script_dir,"temp_profile_Kt_1e6.csv"))
df_7 = pd.read_csv(os.path.join(script_dir,"temp_profile_Kt_1e7.csv"))
df_8 = pd.read_csv(os.path.join(script_dir,"temp_profile_Kt_1e8.csv"))

fig1,ax1 = plt.subplots(figsize=(8,6))
ax1.plot(df_0['z_pc']/delz, np.log10(df_0['T_K']/Th), label=r'$K_t = 0,  '+"\t"+r'\,\dot{M} = $'+f"{df_0['M_dot'][0]:.2e}", linewidth = 2.5)
ax1.plot(df_4['z_pc']/delz, np.log10(df_4['T_K']/Th), label=r'$K_t = 10^4,'+"\t"+r'\,\dot{M} = $'+f"{df_4['M_dot'][0]:.2e}", linewidth = 2.5)
ax1.plot(df_5['z_pc']/delz, np.log10(df_5['T_K']/Th), label=r'$K_t = 10^5,'+"\t"+r'\,\dot{M} = $'+f"{df_5['M_dot'][0]:.2e}", linewidth = 2.5)
ax1.plot(df_6['z_pc']/delz, np.log10(df_6['T_K']/Th), label=r'$K_t = 10^6,'+"\t"+r'\,\dot{M} = $'+f"{df_6['M_dot'][0]:.2e}", linewidth = 2.5)
ax1.plot(df_7['z_pc']/delz, np.log10(df_7['T_K']/Th), label=r'$K_t = 10^7,'+"\t"+r'\,\dot{M} = $'+f"{df_7['M_dot'][0]:.2e}", linewidth = 2.5)
ax1.plot(df_8['z_pc']/delz, np.log10(df_8['T_K']/Th), label=r'$K_t = 10^8,'+"\t"+r'\,\dot{M} = $'+f"{df_8['M_dot'][0]:.2e}", linewidth = 2.5)
ax1.set_xlabel(r'$\tilde{r}$', fontsize = 16)
ax1.set_ylabel(r'$log_{10}\, \tilde{T}$ [K]', fontsize = 16)
ax1.set_xlim(-0.55,0.55)
ax1.set_xticks(np.arange(-0.5, 0.51, 0.1))
ax1.grid(True)
ax1.legend(fontsize = 14)
save_filename = os.path.join(script_dir,"individual_temp_profile.png")
plt.savefig(save_filename)