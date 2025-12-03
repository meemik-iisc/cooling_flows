import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson
import numpy as np

# Parameters
z0 = 1

# Boundary conditions
zc = -10       # example lower boundary point
zh = 10        # example upper boundary point

Tc = 1e4      # temperature at zc
Th = 1e6      # temperature at zh

# Compute tempoerature profile
# def compute_T(z):
#     num1 = (Th-Tc)*np.tanh(z/z0)
#     num2 = Th+Tc
#     return (num1+num2)/2
def compute_T(z):
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

def compute_volume_pdf(z):
    T_prime = (Th-Tc)/(z0*(np.tanh(zh/z0)-np.tanh(zc/z0))*((np.cosh(z/z0))**2))
    return 1/((zh-zc)*T_prime)

def compute_mass_pdf(z):
    vol_pdf = compute_volume_pdf(z)
    temp = compute_T(z)
    return vol_pdf/temp

def compute_emmisivity_pdf(z):
    vol_pdf = compute_volume_pdf(z)
    temp = compute_T(z)
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


# Domain for plotting
z = np.linspace(zc, zh, 300)

# Calculate temperature profile
Temp = compute_T(z)
# Temp2 = compute_T_2(z)
#Calculate volume pdf
v_pdf = compute_volume_pdf(z)
mass_pdf = compute_mass_pdf(z)
emmisivity_pdf = compute_emmisivity_pdf(z) 

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

# Plot temperature profile
fig1,ax1 = plt.subplots(figsize=(8, 6))
ax1.plot(z, Temp/Th, label='Temperature Profile $T(z)$')
ax1.scatter([zc, zh], [Tc, Th], color='red', label='Boundary Conditions')
ax1.set_xlabel('z')
ax1.set_ylabel('$<T(z)>/T_h$')
ax1.set_ylim(0, 1.1)
ax1.set_title('Temperature Profile with Boundary Conditions')
ax1.grid(True)




# # Plot volume pdf
fig2, ax2 = plt.subplots(figsize=(8, 6))
# ax2.plot(np.log10(Temp_pdf), v_pdf, label='Volume PDF $v(z)$')
ax2.plot(np.log10(Temp_sorted), v_pdf_normalized, label='Volume PDF $P_v(<T>)$')
ax2.plot(np.log10(Temp_sorted), mass_pdf_normalized, label='Mass PDF $P_m(<T>)$')
ax2.plot(np.log10(Temp_sorted), emmisivity_pdf_normalized, label='Emissivity PDF $P_E(<T>)$')
ax2.set_xlabel('$log_{10}T$')
# ax2.set_xlim(np.log10(1.1*Tc),np.log10(0.9*Th))
ax2.set_ylabel('$P_v(<T>)$')
ax2.set_yscale('log')
# ax2.set_ylim(1e-6, 1)
ax2.set_title('Volume PDF with Boundary Conditions (Excluding Boundaries)')
ax2.legend()
ax2.grid(True)


import os
script_dir = os.path.dirname(os.path.abspath(__file__))
save_path_1 = os.path.join(script_dir, "sech_outputs/temp_profile.png")
save_path_2 = os.path.join(script_dir, "sech_outputs/volume_pdf.png")
fig1.savefig(save_path_1)
fig2.savefig(save_path_2)
plt.close()