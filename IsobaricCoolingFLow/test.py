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
def solve_isothermal_analytical():
    """
    Analytical solution for first-order ODE (Kt=0) using robust numerical integration.
    
    dT/dz = A * Λ(T) / (M_dot * T²)
    
    Separable: M_dot * T² dT = A * Λ(T) dz
    
    Integrate: M_dot * I(T) = A * (z - zc)
    
    where I(T) = ∫[Tc to T] T'²/Λ(T') dT'
    
    M_dot = A * (zh - zc) / I(Th)
    """
    
    print("="*70)
    print("ANALYTICAL SOLUTION (First-Order ODE, Kt=0)")
    print("="*70)
    
    # Coefficient
    A = (cool_lambda(T0) * (delz**2) * (P0**2)) / ((kB**2) * (T0**3))
    
    print(f"\nODE: dT/dz = A * Λ(T) / (M_dot * T²)")
    print(f"Coefficient: A = {A:.6e}\n")
    
    # Integrand for I(T)
    def integrand(T):
        """T² / Λ(T) - integrand for I(T)"""
        lambda_val = cool_lambda(T)
        return T**2 / lambda_val
    
    # ====================================================================
    # Build I(T) using ADAPTIVE MESH (log-space for better resolution)
    # ====================================================================
    print("Building I(T) table using adaptive mesh...")
    
    # Use log-spaced mesh for better resolution where Λ(T) changes rapidly
    T_mesh = np.logspace(np.log10(Tc), np.log10(Th), 500)
    
    # Compute integrand on mesh
    integrand_vals = np.array([integrand(T) for T in T_mesh])
    
    # Integrate cumulatively using trapezoidal rule
    I_vals = cumulative_trapezoid(integrand_vals, T_mesh, initial=0.0)
    
    I_Th = I_vals[-1]
    
    print(f"  I(Tc) = {I_vals[0]:.6e}")
    print(f"  I(Th) = {I_Th:.6e}\n")
    
    # Compute M_dot from boundary condition
    # M_dot = A * (zh - zc) / I(Th)
    M_dot_solution = A * delz / I_Th
    
    print(f"From boundary condition: M_dot = A * (zh - zc) / I(Th)")
    print(f"  M_dot = {A:.6e} * {delz:.6e} / {I_Th:.6e}")
    print(f"  M_dot = {M_dot_solution:.6e}\n")
    
    # ====================================================================
    # RECONSTRUCT SOLUTION USING INVERSE INTEGRAL
    # ====================================================================
    print("="*70)
    print("RECONSTRUCTING TEMPERATURE PROFILE")
    print("="*70)
    
    # Create CubicSpline for smooth inversion (more robust than interp1d)
    try:
        I_to_T_spline = CubicSpline(I_vals, T_mesh, bc_type='natural', extrapolate=False)
    except:
        # Fallback to linear interpolation if CubicSpline fails
        I_to_T_spline = interp1d(I_vals, T_mesh, kind='linear', bounds_error=False)
    
    # Evaluate solution on mesh
    z_plot = np.linspace(zc_tilde, zh_tilde, 500)
    T_plot = []
    
    print(f"\nEvaluating T(z) using inverse integral...")
    for i, z_eval in enumerate(z_plot):
        # Normalized position in domain
        z_norm = (z_eval - zc_tilde) / (zh_tilde - zc_tilde)
        # Target I value at this z
        I_target = I_Th * z_norm
        # Get T from inverse
        try:
            T_eval = I_to_T_spline(I_target)
        except:
            # Fallback if out of bounds
            if I_target <= I_vals[0]:
                T_eval = Tc
            elif I_target >= I_vals[-1]:
                T_eval = Th
            else:
                # Linear interpolation fallback
                idx = np.searchsorted(I_vals, I_target)
                frac = (I_target - I_vals[idx-1]) / (I_vals[idx] - I_vals[idx-1])
                T_eval = T_mesh[idx-1] + frac * (T_mesh[idx] - T_mesh[idx-1])
        
        T_plot.append(T_eval)
    
    T_plot = np.array(T_plot)
    T_plot_tilde = T_plot / T0
    
    print(f"  Done!")
    
    return M_dot_solution, z_plot, T_plot_tilde, T_plot


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("ISOTHERMAL COOLING FLOW - ANALYTICAL SOLUTION")
    print("="*70 + "\n")
    
    M_dot_sol, z_plot, T_plot_tilde, T_plot = solve_isothermal_analytical()
    
    # Convert to physical units
    z_physical = z_plot * delz / pc
    T_physical = T_plot
    
    # Verification
    print("\n" + "="*70)
    print("VERIFICATION")
    print("="*70)
    print(f"T(zc) = {T_physical[0]:.6e} K (target: {Tc:.6e} K)")
    print(f"T(zh) = {T_physical[-1]:.6e} K (target: {Th:.6e} K)")
    print(f"T ratio: Th/Tc = {T_physical[-1]/T_physical[0]:.2f}")
    print(f"Relative error at zc: {abs(T_physical[0] - Tc) / Tc * 100:.6e}%")
    print(f"Relative error at zh: {abs(T_physical[-1] - Th) / Th * 100:.6e}%")
    
    # Create output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "individual_T")
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot
    fig, axes = plt.subplots()
    
    # Log scale
    axes.plot(z_physical, np.log10(T_physical), 'b-', linewidth=2.5, label='T(z) - Analytical')
    axes.set_xlabel('z [pc]', fontsize=11)
    axes.set_ylabel('$log_{10}T [K]$', fontsize=11)
    axes.set_title(f'Temperature Profile, Kt=0 M_dot={M_dot_sol:.2e}', fontsize=12)
    axes.grid(True)
    axes.legend(fontsize=10)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, "temp_profile_Kt_0.png")
    plt.savefig(save_path, dpi=150)
    print(f"\n✓ Plot saved to: {save_path}")
    plt.close(fig)
    
    # Save to CSV
    df = pd.DataFrame({
        'z_pc': z_physical,
        'T_K': T_physical,
        'T_tilde': T_plot_tilde,
        'M_dot': np.full_like(z_physical, M_dot_sol)
    })
    save_path_csv = os.path.join(output_dir, "temp_profile_Kt_0.csv")
    df.to_csv(save_path_csv, index=False)
    print(f"✓ Data saved to: {save_path_csv}")
    
    print("\n" + "="*70)
    print(f"EIGENVALUE (ANALYTICAL SOLUTION):")
    print(f"  M_dot = {M_dot_sol:.6e}")
    print(f"  (For isothermal case Kt=0)")
    print("="*70)
