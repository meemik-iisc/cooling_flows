import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve, root
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
Kt = 1e7               # ergcm^-1s^-1K^-1

# Dimensionless parameters
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
# ODE SYSTEM
# ============================================================================
def solve_cooling_flow_eigenvalue(Kt_val, x0):
    """
    Two-sided eigenvalue problem using shooting with fsolve.
    
    Find M_dot and dT/dz|_zc such that:
    1. T(zc) = Tc_tilde (left BC)
    2. T(zh) = Th_tilde (right BC)
    
    Parameters:
    -----------
    Kt_val : float
        Thermal conductivity parameter
    x0 : array
        Initial guess [M_dot, dT/dz|_zc]
        
    Returns:
    --------
    M_dot_solution : float
        Eigenvalue (mass flow rate parameter)
    dTdz_solution : float
        Initial slope at zc
    z_plot : ndarray
        Dimensionless z coordinates
    T_plot : ndarray
        Dimensionless temperature
    sol_final : solve_ivp result
        Full solution object
    """
    
    print("="*70)
    print("METHOD: Two-Sided Eigenvalue Shooting with fsolve")
    print("="*70)
    
    term1 = (gamma * kB * delz) / ((gamma - 1) * mu * mp * Kt_val)
    term2 = (cool_lambda(T0) * (delz**2) * (P0**2)) / ((kB**2) * (T0**3) * Kt_val)
    
    print(f"\nODE Coefficients:")
    print(f"  term1 = {term1:.6e}")
    print(f"  term2 = {term2:.6e}")
    print(f"\nInitial guess: M_dot={x0[0]:.3e}, dT/dz={x0[1]:.3f}\n")
    
    def ode_system(z, y, M_dot):
        """
        ODE system:
        y[0] = T_tilde
        y[1] = dT_tilde/dz_tilde
        """
        T_tilde = y[0]
        dT_tildedz = y[1]
        
        # Clamp to prevent domain violations
        T_tilde = np.clip(T_tilde, 0.01, 100.0)
        dT_tildedz = np.clip(dT_tildedz, -100.0, 100.0)
        
        # Physical temperature
        T_physical = np.clip(T_tilde * T0, T_tab.min() * 0.95, T_tab.max() * 1.05)
        
        # Cooling function
        lambda_physical = cool_lambda(T_physical)
        lambda_tilde = lambda_physical / cool_lambda(T0)
        lambda_tilde = np.clip(lambda_tilde, 1e-10, 1e2)
        
        # Second derivative
        d2T_tildedz2 = -term1 * M_dot * dT_tildedz + term2 * lambda_tilde / (T_tilde**2)
        d2T_tildedz2 = np.clip(d2T_tildedz2, -1e6, 1e6)
        
        return [dT_tildedz, d2T_tildedz2]
    
    call_count = [0]
    
    def residual_function(params):
        """
        Compute residual vector for root finder.
        
        Parameters:
        -----------
        params : array
            [M_dot, dT/dz|_zc]
            
        Returns:
        --------
        residuals : array
            [T(zh) - Th_tilde, dT/dz_at_zh - dT/dz_boundary_condition]
            For now: just [T(zh) - Th_tilde] as single constraint
        """
        M_dot, dTdz_init = params
        call_count[0] += 1
        
        # Physical bounds check
        if M_dot < 1e-30 or M_dot > 1e-15 or dTdz_init < 0.01 or dTdz_init > 100:
            print(f"  Call {call_count[0]:3d}: M_dot={M_dot:.3e}, dTdz={dTdz_init:.4f} | OUT OF BOUNDS")
            return [1e8, 1e8]
        
        try:
            # Integrate from zc to zh
            sol = solve_ivp(
                lambda z, y: ode_system(z, y, M_dot),
                [zc_tilde, zh_tilde],
                [Tc_tilde, dTdz_init],
                method='Radau',
                rtol=1e-8,
                atol=1e-10,
                max_step=0.005,
                first_step=1e-4,
                dense_output=True
            )
            
            # Check integration success
            if not sol.success:
                print(f"  Call {call_count[0]:3d}: M_dot={M_dot:.3e}, dTdz={dTdz_init:.4f} | FAILED: {sol.message}")
                return [1e8, 1e8]
            
            # Check for NaN/Inf
            if np.any(np.isnan(sol.y)) or np.any(np.isinf(sol.y)):
                print(f"  Call {call_count[0]:3d}: M_dot={M_dot:.3e}, dTdz={dTdz_init:.4f} | NaN/Inf detected")
                return [1e8, 1e8]
            
            # Two residuals:
            # 1. T(zh) should equal Th_tilde
            T_zh = sol.y[0, -1]
            residual_T = T_zh - Th_tilde
            
            # 2. For true eigenvalue: could add another condition
            # For now, just return T residual (system is technically underdetermined)
            # Alternative: add constraint that dT/dz at zh satisfies some condition
            # Or: prescribe both T(zh) and dT/dz(zh)
            
            print(f"  Call {call_count[0]:3d}: M_dot={M_dot:.3e}, dTdz={dTdz_init:.4f} | T(zh)={T_zh:.4f}, res={residual_T:.6e}")
            
            return [residual_T, 0.0]  # Second residual is dummy for now
            
        except Exception as e:
            print(f"  Call {call_count[0]:3d}: M_dot={M_dot:.3e}, dTdz={dTdz_init:.4f} | Exception: {str(e)}")
            return [1e8, 1e8]
    
    # Solve using fsolve (more robust than least_squares for stiff problems)
    print("Starting root finder with fsolve...\n")
    
    solution = fsolve(
        residual_function,
        x0=x0,
        full_output=True,
        xtol=1e-12,
        maxfev=500
    )
    
    params_sol, info, ier, msg = solution
    M_dot_solution, dTdz_solution = params_sol
    
    print(f"\n{msg}")
    print(f"✓ ROOT FINDING COMPLETED")
    print(f"  M_dot = {M_dot_solution:.6e}")
    print(f"  dT/dz|_zc = {dTdz_solution:.6f}")
    print(f"  Function calls: {info['nfev']}")
    print(f"  Residual norm: {np.linalg.norm(info['fvec']):.6e}")
    
    # Get final high-precision solution
    print(f"\nComputing final solution with high precision...")
    sol_final = solve_ivp(
        lambda z, y: ode_system(z, y, M_dot_solution),
        [zc_tilde, zh_tilde],
        [Tc_tilde, dTdz_solution],
        method='Radau',
        rtol=1e-11,
        atol=1e-13,
        dense_output=True,
        max_step=0.005
    )
    
    if not sol_final.success:
        print(f"⚠ WARNING: Final integration - {sol_final.message}")
    
    # Evaluate on fine grid
    z_plot = np.linspace(zc_tilde, zh_tilde, 500)
    T_plot = sol_final.sol(z_plot)[0]
    
    return M_dot_solution, dTdz_solution, z_plot, T_plot, sol_final


# ============================================================================
# MAIN: RUN SOLVER AND SAVE RESULTS
# ============================================================================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("COOLING FLOW EIGENVALUE PROBLEM SOLVER")
    print("="*70 + "\n")
    
    # Initial guess
    x0 = [1e-21, 10.0]
    
    # Solve using eigenvalue shooting
    M_dot_sol, dTdz_sol, z_plot, T_plot, sol = solve_cooling_flow_eigenvalue(Kt, x0)
    
    # Convert back to physical units
    z_physical = z_plot * delz / pc
    T_physical = T_plot * T0
    
    # Verification
    print("\n" + "="*70)
    print("VERIFICATION")
    print("="*70)
    print(f"T(zc) = {T_physical[0]:.6e} K (target: {Tc:.6e} K)")
    print(f"T(zh) = {T_physical[-1]:.6e} K (target: {Th:.6e} K)")
    print(f"T ratio: Th/Tc = {T_physical[-1]/T_physical[0]:.2f}")
    print(f"Relative error at zc: {abs(T_physical[0] - Tc) / Tc * 100:.2e}%")
    print(f"Relative error at zh: {abs(T_physical[-1] - Th) / Th * 100:.2e}%")
    
    # Create output directory
    output_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(output_dir, exist_ok=True)
    Kt_str = f"{Kt:.0e}".replace("+0", "").replace("+", "")
    filename = f"temp_profile_Kt_{Kt_str}"
    # Plot
    fig, axes = plt.subplots()
    
    # Log scale
    axes.plot(z_physical, np.log10(T_physical), 'b-', linewidth=2, label='T(z)')
    axes.axhline(np.log10(Tc), color='r', linestyle='--', alpha=0.7, label=f'Tc={Tc:.1e} K')
    axes.axhline(np.log10(Th), color='r', linestyle='--', alpha=0.7, label=f'Th={Th:.1e} K')
    axes.set_xlabel('z [pc]')
    axes.set_ylabel('$log_{10}T [K]$')
    axes.set_title(f'Temperature Profile K_t = {0.1*Kt}, M_dot={M_dot_sol:.2e}, dTcdz={dTdz_sol:.2f}')
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
        'T_tilde': T_plot,
        'M_dot': np.full_like(z_physical, M_dot_sol),
        'dTdz_init': np.full_like(z_physical, dTdz_sol)
    })
    save_path_csv = os.path.join(output_dir, f"{filename}.csv")
    df.to_csv(save_path_csv, index=False)
    print(f"✓ Data saved to: {save_path_csv}")
    
    print("\n" + "="*70)
    print(f"EIGENVALUES:")
    print(f"  M_dot = {M_dot_sol:.6e}")
    print(f"  dT/dz|_zc = {dTdz_sol:.6f}")
    print("="*70)