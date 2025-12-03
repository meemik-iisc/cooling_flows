import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
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
def compute_T(z):
    num1 = (Th-Tc)*erf(z/z0)
    num2 = Tc*erf(zh/z0)-Th*erf(zc/z0)
    den  = erf(zh/z0)-erf(zc/z0)
    return (num1+num2)/den

def compute_volume_pdf(z):
    A = (Th-Tc)/(erf(zh/z0)-erf(zc/z0))
    return z0*np.sqrt(np.pi)*np.exp(z**2/z0**2)/(2*A*(zh-zc))

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
#Calculate volume pdf
v_pdf = compute_volume_pdf(z)

epsilon = 1.5  # adjust as needed to exclude close to boundaries
# Create mask to exclude points near boundaries
mask = (Temp > 1.1 * Tc) & (Temp < 0.9 * Th)

# Filter arrays
# z_filtered = z[mask]
Temp_filtered = Temp[mask]
v_pdf_filtered = v_pdf[mask]

#Normalize PDF
Temp_sorted, v_pdf_normalized = normalize_pdf(Temp_filtered, v_pdf_filtered)

# Plot temperature profile
fig1,ax1 = plt.subplots(figsize=(8, 6))
ax1.plot(z, Temp/Th, label='Temperature Profile $T(z)$')
ax1.scatter([zc, zh], [Tc, Th], color='red', label='Boundary Conditions')
ax1.set_xlabel('z')
ax1.set_ylabel('$<T(z)>/T_h$')
ax1.set_ylim(0, 1.1)
ax1.set_title('Temperature Profile with Boundary Conditions')
ax1.legend()
ax1.grid(True)




# Plot volume pdf with filtered data
fig2, ax2 = plt.subplots(figsize=(8, 6))
ax2.plot(np.log10(Temp_sorted), v_pdf_normalized, label='Volume PDF $v(z)$')
# ax2.plot(np.log10(Temp_filtered), v_pdf_filtered, label='Volume PDF $v(z)$')
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
save_path_1 = os.path.join(script_dir, "outputs/temp_profile.png")
save_path_2 = os.path.join(script_dir, "outputs/volume_pdf.png")
fig1.savefig(save_path_1)
fig2.savefig(save_path_2)
plt.close()