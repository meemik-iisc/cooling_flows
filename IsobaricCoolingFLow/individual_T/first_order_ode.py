import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import quad, cumulative_trapezoid
from scipy.interpolate import interp1d, CubicSpline
import os


# ============================================================================
# CONSTANTS & PARAMETERS
# ============================================================================
kB = 1.38e-16          # erg/K
mp = 1.67e-24          # g
gamma = 5.0/3.0
mu = 1.0
pc = 3.086e18          # cm

zc = -1.0 * pc
zh = 1.0 * pc
delz = zh - zc
Tc = 1e4               # K
Th = 1e6               # K
T0 = np.sqrt(Th * Tc)
P0 = 1e3 * kB          # erg/cm^3

zc_tilde = zc / delz
zh_tilde = zh / delz
Tc_tilde = Tc / T0
Th_tilde = Th / T0

print(f"Dimensionless parameters:")
print(f"  Tc_tilde = {Tc_tilde:.6f}, Th_tilde = {Th_tilde:.6f}\n")

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

print(f"Cooling table range: {T_tab.min():.1e} to {T_tab.max():.1e} K\n")

# ============================================================================
# ANALYTICAL SOLUTION FOR FIRST-ORDER ODE
# ============================================================================

def Integrand(T_tilde):
    lambda_tilde = cool_lambda(T_tilde * T0)/cool_lambda(T0)
    return (T_tilde**2/lambda_tilde)
def build_cumulative_integral_table(n_points=1000):
    T_tilde_mesh = np.linspace(Tc_tilde, Th_tilde, n_points)
    integrand_vals = np.array([Integrand(T) for T in T_tilde_mesh])
    
    # Cumulative integral from Tc_tilde
    I_cumulative = cumulative_trapezoid(integrand_vals, T_tilde_mesh, initial=0.0)
    
    # Interpolate for fast lookup
    I_interp = interp1d(T_tilde_mesh, I_cumulative, kind='cubic', 
                       bounds_error=False, fill_value='extrapolate')
    
    return I_interp

def build_inverse_table(n_points=2000):
    """Precompute T_tilde vs integral(T_tilde) for fast inversion"""
    T_tilde_mesh = np.linspace(Tc_tilde, Th_tilde, n_points)
    I_table = build_cumulative_integral_table()
    I_mesh = np.array([I_table(T) for T in T_tilde_mesh])
    I_total = I_mesh[-1]
    
    # Normalize I to [0,1] for easier mapping
    I_norm_mesh = I_mesh / I_total
    
    # Interpolate T_tilde(I_norm)
    inverse_interp = interp1d(I_norm_mesh, T_tilde_mesh, kind='cubic',
                             bounds_error=False, fill_value='extrapolate')
    return inverse_interp, I_total
# Usage


# Total integral Tc→Th
inverse_table,I_Th = build_inverse_table()
print(f"∫[Tc_tilde→Th_tilde] T²/Λ_tilde dT_tilde = {I_Th:.6e}")

#Calculate M_dot
A=(gamma*(kB**3)*(T0**3))/((gamma-1)*mu*mp*delz*(P0**2)*cool_lambda(T0))
print("A=",A)
M_dot_sol = (zh_tilde-zc_tilde)/(A*I_Th)
print("M_dot=",M_dot_sol)


def T_tilde_of_z_fast(z_tilde):
    I_norm_target = (z_tilde - zc_tilde) / (zh_tilde - zc_tilde)
    return inverse_table(I_norm_target)

    
# Biased grid: more resolution near z_c (cold boundary)
def biased_z_mesh(n_points=1000, bias_factor=3.0):
    z_lin = np.linspace(zc_tilde, zh_tilde, n_points)
    return zc_tilde + (zh_tilde - zc_tilde) * ((z_lin - zc_tilde)/(zh_tilde - zc_tilde))**bias_factor

# Evaluate on fine grid
z_plot = biased_z_mesh(n_points=1000, bias_factor=3.0)
T_plot_tilde = T_tilde_of_z_fast(z_plot)  # Vectorized!

# Convert back to physical units
z_physical = z_plot * delz / pc
T_physical = T_plot_tilde * T0

# Create output directory
output_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(output_dir, exist_ok=True)
filename = f"temp_profile_Kt_0"
# Plot
fig, axes = plt.subplots()

# Log scale
axes.plot(z_physical, np.log10(T_physical), 'b-', linewidth=2, label='T(z)')
axes.axhline(np.log10(Tc), color='r', linestyle='--', alpha=0.7, label=f'Tc={Tc:.1e} K')
axes.axhline(np.log10(Th), color='r', linestyle='--', alpha=0.7, label=f'Th={Th:.1e} K')
axes.set_xlabel('z [pc]')
axes.set_ylabel('$log_{10}T [K]$')
axes.set_title(f'Temperature Profile K_t = {0}, M_dot={M_dot_sol:.2e}')
axes.grid(True)
axes.legend()

plt.tight_layout()
save_path = os.path.join(output_dir, f"{filename}.png")
plt.savefig(save_path, dpi=150)
print(f"\n✓ Plot saved to: {save_path}")
plt.close(fig)

# Save to CSV
df = pd.DataFrame({
    'z_pc': z_physical,
    'T_K': T_physical,
    'T_tilde': T_plot_tilde,
    'M_dot': np.full_like(z_physical, M_dot_sol),
    'dTdz_init': np.full_like(z_physical, 0)
})
save_path_csv = os.path.join(output_dir, f"{filename}.csv")
df.to_csv(save_path_csv, index=False)
print(f"✓ Data saved to: {save_path_csv}")

print("\n" + "="*70)
print(f"EIGENVALUES:")
print(f"  M_dot = {M_dot_sol:.6e}")
# print(f"  dT/dz|_zc = {dTdz_sol:.6f}")
print("="*70)