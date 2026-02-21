import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson
import numpy as np



# Compute tempoerature profile
# def compute_T(z):
#     num1 = (Th-Tc)*np.tanh(z/z0)
#     num2 = Th+Tc
#     return (num1+num2)/2
def compute_T(z,z0):
    num1 = (Th-Tc)*np.tanh(z/z0)
    num2 = Tc*np.tanh(zh/z0)-Th*np.tanh(zc/z0)
    den  = np.tanh(zh/z0)-np.tanh(zc/z0)
    return (num1+num2)/den


def compute_lambda(temp, filename="cooltable.dat"):
    # Load the data, assuming whitespace-delimited columns
    data = np.loadtxt(filename)
    temperatures = data[:, 0]
    lambdas = data[:, 1]
    # Find the index of the closest temperature
    idx = (np.abs(temperatures - temp)).argmin()
    # Return the lambda at that index
    return lambdas[idx]

def compute_volume_pdf(z,z0):
    T_prime = (Th-Tc)/(z0*(np.tanh(zh/z0)-np.tanh(zc/z0))*((np.cosh(z/z0))**2))
    return 1/((zh-zc)*T_prime)

def compute_mass_pdf(z,z0):
    vol_pdf = compute_volume_pdf(z, z0)
    temp = compute_T(z,z0)
    return vol_pdf/temp

def compute_emmisivity_pdf(z,z0):
    vol_pdf = compute_volume_pdf(z, z0)
    temp = compute_T(z,z0)
    lambda_T = np.vectorize(compute_lambda)(temp)
    return (vol_pdf*lambda_T)/(temp**2)

def normalize_pdf(temp, pv):
    # Sort x and p by x
    sorted_indices = np.argsort(temp)
    temp_sorted = np.array(temp)[sorted_indices]
    pv_sorted = np.array(pv)[sorted_indices]

    # Compute integral using trapezoidal rule
    integral = simpson(pv_sorted, temp_sorted)
    print(integral)

    # Normalize PDF
    pv_normalized = pv_sorted / integral

    return temp_sorted, pv_normalized

# Parameters
z0 = 0.1

# Boundary conditions
zc = -1       # example lower boundary point
zh = 1        # example upper boundary point

Tc = 1e4      # temperature at zc
Th = 1e6      # temperature at zh

# Domain for plotting
z = np.linspace(zc, zh, 300)

# Calculate temperature profile
Temp = compute_T(z, z0)
# Temp2 = compute_T_2(z)
#Calculate volume pdf
v_pdf = compute_volume_pdf(z, z0)
mass_pdf = compute_mass_pdf(z, z0)
emmisivity_pdf = compute_emmisivity_pdf(z, z0) 

# Filter arrays
mask = (Temp > 1.1 * Tc) & (Temp < 0.9 * Th)
Temp_filtered = Temp[mask]
v_pdf_filtered = v_pdf[mask]
mass_pdf_filtered = mass_pdf[mask]
emmisivity_pdf_filtered = emmisivity_pdf[mask]

# #Normalize PDF
Temp_sorted, v_pdf_normalized = normalize_pdf(Temp_filtered, v_pdf_filtered)
mass_pdf_normalized = normalize_pdf(Temp_filtered, mass_pdf_filtered)[1]
emmisivity_pdf_normalized = normalize_pdf(Temp_filtered, emmisivity_pdf_filtered)[1]

logT = np.log10(Temp_sorted)
P_logT_v = v_pdf_normalized * Temp_sorted * np.log(10.0)
P_logT_m = mass_pdf_normalized * Temp_sorted * np.log(10.0)
P_logT_e = emmisivity_pdf_normalized * Temp_sorted * np.log(10.0)

# Plot temperature profile
fig1,ax1 = plt.subplots(figsize=(8, 6))
for z0 in [0.01,0.1,1.0,10.0]:
    ax1.plot(z, np.log10(compute_T(z,z0)), label=r'$\tilde{r_0}$ ='+f'{z0}', linewidth = 2.5)
ax1.set_xlabel(r'$\tilde{r}$', fontsize = 16)
ax1.set_ylabel(r'$\log_{10}\langle T\rangle$', fontsize=16)
# ax1.set_ylim(0, 1.1)
# ax1.set_title(r'Average Temperature Profile for $<n^2 \Lambda (T) > = sech^2(r/r_0)$ in Cartesian Geometry (q=0)')
ax1.legend(fontsize = 14)
ax1.tick_params(axis='both', which='major', labelsize=14)
ax1.grid(True)




# # Plot volume pdf
fig2, ax2 = plt.subplots(figsize=(8, 6))
# ax2.plot(np.log10(Temp_pdf), v_pdf, label='Volume PDF $v(z)$')
ax2.plot(logT, P_logT_v, label='Volume PDF', linewidth = 2.5, color='orange')
ax2.plot(logT, P_logT_m, label='Mass PDF', linewidth = 2.5, color='blue')
ax2.plot(logT, P_logT_e, label='Emissivity PDF', linewidth = 2.5, color='green')
ax2.set_xlabel(r'$log_{10} \langle T \rangle$', fontsize = 16)
ax2.set_ylabel(r'$\mathcal{P}\,(\mathrm{log}_{10} \langle T \rangle)$', fontsize = 16)
ax2.set_yscale('log')
ax2.set_ylim(1e-3, 11)
# ax2.set_title('Normalized PDF for $$')
ax2.legend(fontsize = 14)
ax2.tick_params(axis='both', which='major', labelsize=14)
# Customize log grid
ax2.minorticks_on()
ax2.grid(True, which='both', ls='-', alpha=0.3)
ax2.grid(which='major', alpha=0.7, linewidth=1.2)
ax2.grid(which='minor', alpha=0.5, linewidth=1.0)


import os
script_dir = os.path.dirname(os.path.abspath(__file__))
save_path_1 = os.path.join(script_dir, "sech_outputs/temp_profile.png")
fig1.savefig(save_path_1)
save_path_2 = os.path.join(script_dir, "sech_outputs/volume_pdf.png")
fig2.savefig(save_path_2)
plt.close()