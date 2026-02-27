import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.integrate import solve_bvp,solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import fsolve, root_scalar

# ============================================================================
# CONSTANTS
# ============================================================================
kB = 1.38e-16
mp = 1.67e-24
gamma = 5.0/3.0
mu = 1.0
pc = 3.086e18
Msolar = 1.989e30
s_yr = 3.154e7

# ============================================================================
# Units
# ============================================================================
UNIT_PRESSURE = 1.59916e-14
# ============================================================================
# PARAMETERS (geometry & BCs)
# ============================================================================
rc = -1.0 * pc
rh =  1.0 * pc
dr = rh - rc
r0 = 0.1*pc            # choose your scale; can change
Tc = 1e4
Th = 1e6
T0 = np.sqrt(Th * Tc)
pres_0  = 14.02645*UNIT_PRESSURE
rho0 = pres_0*mu*mp/(kB*T0)
n0 = rho0/(mu*mp)
lambda0 = 7.010900e-22  # ergcm^-1s^-1K^-1
time_0 = (gamma*pres_0*mu**2*mp**2)/((gamma-1)*rho0**2*lambda0)
E0 = pres_0/time_0
# print(f"P0 = {pres_0:.2e} ergcm^-3, n0= {n0:.2e} cm^-3, rho0 = {rho0:.2e} gcm^{-3}, t0 = {time_0:.2e} s, E0 = {E0:.2e} ergcm^-3s^-1")
N = 0.3*E0
rho_cold = rho0*T0/Tc
cs_cold = np.sqrt(gamma*kB*Tc/mu/mp)
Mdot_cold = rho_cold*cs_cold
# Kt = 1.0e9           # you will vary this
# Dimensionless variables
rc_tilde = rc / r0
rh_tilde = rh / r0
Tc_tilde = Tc / T0
Th_tilde = Th / T0
print(f"Tc_tilde = {Tc_tilde:.6f}, Th_tilde = {Th_tilde:.6f}")
def sech_sqr(z):
    return 1/(np.cosh(z)**2)

def first_order_ODE(r_tilde):
    A = (Th_tilde-Tc_tilde)/(np.tanh(rh_tilde)-np.tanh(rc_tilde))
    C = (Tc_tilde*np.tanh(rh_tilde)-Th_tilde*np.tanh(rc_tilde))/(np.tanh(rh_tilde)-np.tanh(rc_tilde))
    Mdot_0 = ((gamma-1)*mu*N*mp*r0*(np.tanh(rh_tilde)-np.tanh(rc_tilde)))/(gamma*kB*T0*(Th_tilde-Tc_tilde))
    return A*np.tanh(r_tilde)+C, Mdot_0

def second_order_ode(r_tilde_eval, Kt,Mdot_guess_,gc_guess_, rtol_integrate=3e-14, atol_integrate=1e-11, rtol_root=1e-14):
    def ode_system(r_tilde, y, Mdot, Kt):
        # y[0] = T, y[1] = T'
        term1 = (gamma*kB*Mdot*r0)/((gamma-1)*mu*mp*Kt)
        term2 = (N*r0**2)/(Kt*T0)
        dT_tilde_dr_tilde = y[1]
        d2T_tilde_dr_tilde2 = -1*term1*y[1]+term2*sech_sqr(r_tilde) 
        # d2T_tilde_dr_tilde2 = term1*y[1]+term2*sech_sqr(r_tilde) 
        return np.array([dT_tilde_dr_tilde,d2T_tilde_dr_tilde2])
    
    def out_of_bounds_event(r_tilde, y):
        """Event: T_tilde exits [Tc_tilde, Th_tilde]"""
        T_tilde = y[0]
        tol = 1e-8  # Tiny buffer
        return min(T_tilde - (Tc_tilde+1e-8 + tol), (Th_tilde - tol) - T_tilde)
    def integrate_profile(Mdot, gc):
        """Integrate FROM Cold to Hot with IC: T=Tc_tilde, T'=gc"""
        y0 = np.array([Tc_tilde, gc])  # COLD side IC
        
        event = out_of_bounds_event
        event.terminal = True
        event.direction = -1
        
        sol = solve_ivp(
            lambda r, y: ode_system(r, y, Mdot, Kt),
            t_span=(rc_tilde, rh_tilde),  # cold → hot
            y0=y0,
            rtol=rtol_integrate,
            atol=atol_integrate,
            events=event,
            method='LSODA', max_step = 0.1, dense_output=True
        )
        return sol
    # def hot_slope(Mdot):
    #     """Target: hot_slope=0 at rh_tilde"""
    #     sol = integrate_profile(Mdot)
    #     return sol.y[1, -1]
    # def hot_T_tilde(Mdot):
    #     """T_tilde at HOT END (instead of slope)"""
    #     sol = integrate_profile(Mdot)
    #     T_end_tilde = sol.sol(r_tilde_eval)[0][-1]
    #     return T_end_tilde
    
    def get_residuals(vars_vec):
        """Return [T(rh) - Th_tilde, T'(rh) - 0] for fsolve."""
        Mdot, gc = vars_vec
        sol = integrate_profile(Mdot, gc)

        # If integration terminated early, penalize strongly
        if sol.t[-1] < rh_tilde - 1e-3:
            # Ended before reaching hot side
            return [1e3, 1e3]

        # Evaluate at rh_tilde via dense solution
        T_hot, dTdr_hot = sol.sol(rh_tilde)
        f1 = T_hot - Th_tilde
        f2 = dTdr_hot  # want 0
        return [f1, f2]
    def bracket_Mdot_for_gc(Mdot_start, gc_fixed, factor_Mdot=1.05, max_iter_Mdot=200, verbose=False):
        """
        For a fixed gc, try to bracket Mdot such that
        (T_hot(Mdot,gc) - Th_tilde) changes sign.

        Returns:
            (Mdot_low, Mdot_high) if successful
            None if no bracket found within max_iter_Mdot
        """
        Mdot_low = Mdot_start
        f1_low, f2_low = get_residuals([Mdot_start, gc_fixed])  # f1 = T_hot - Th_tilde
        T_tilde_low = Th_tilde + f1_low

        if verbose:
            print(f"\n  [gc = {gc_fixed:.3e}] START: "
                f"Mdot={Mdot_low:.3e}, T_hot={T_tilde_low*T0:.3e} K, "
                f"f1={f1_low:.3e}, f2={f2_low:.3e}")

        # If already extremely close, treat as trivial bracket
        if abs(f1_low) < 1e-10:
            if verbose:
                print("    f1 already ~0 at Mdot_start; trivial bracket.")
            return Mdot_low, Mdot_low

        for i in range(max_iter_Mdot):
            Mdot_high = Mdot_low * factor_Mdot
            f1_high, f2_high = get_residuals([Mdot_high, gc_fixed])
            T_high = Th_tilde + f1_high

            if verbose:
                print(f"    {i:02d}: Mdot={Mdot_high:.3e}, "
                    f"T_hot={T_high*T0:.3e} K, f1={f1_high:.3e}, gc = {gc_fixed:.3e}")

            # Check sign change of f1 = T_hot - Th_tilde
            if f1_low * f1_high < 0.0:
                if verbose:
                    print(f"    -> BRACKET in Mdot for gc={gc_fixed:.3e}: "
                        f"[{Mdot_low:.3e}, {Mdot_high:.3e}]")
                return Mdot_low, Mdot_high

            Mdot_low = Mdot_high
            f1_low = f1_high

        if verbose:
            print(f"    !! FAILED to bracket Mdot for gc={gc_fixed:.3e} "
                f"after {max_iter_Mdot} steps.")
        return None  # No bracket found for this gc

    def bracket_vars(Mdot_start,
                    gc_start,
                    factor_Mdot=1.05,
                    step_gc=0.01,
                    max_iter_gc=50,
                    max_iter_Mdot=200,
                    verbose=True):
        """
        Try to find a pair (gc, [Mdot_low, Mdot_high]) such that
        T_hot(Mdot,gc) - Th_tilde changes sign over [Mdot_low, Mdot_high].

        Algorithm:
        - Start with gc = gc_start
        - Attempt bracket_Mdot_for_gc(gc)
        - If it fails, update gc <- gc * step_gc and try again
        - Stop at first gc where Mdot bracket is found

        Returns:
            (Mdot_low, Mdot_high, gc_used)

        Raises:
            RuntimeError if no gc within max_iter_gc gives a valid Mdot bracket.
        """

        gc = gc_start

        if verbose:
            print("=== Bracketing Mdot by scanning gc ===")
            print(f"Initial guesses: Mdot_start={Mdot_start:.3e}, gc_start={gc_start:.3e}")

        for j in range(max_iter_gc):
            if verbose:
                print(f"\nGC ITER {j:02d}: trying gc={gc:.3e}")

            bracket = bracket_Mdot_for_gc(Mdot_start=Mdot_start,
                                        gc_fixed=gc,
                                        factor_Mdot=factor_Mdot,
                                        max_iter_Mdot=max_iter_Mdot,
                                        verbose=verbose)

            if bracket is not None:
                Mdot_low, Mdot_high = bracket
                if verbose:
                    print(f"\n*** SUCCESS: found Mdot bracket for gc={gc:.3e} ***")
                    print(f"    Mdot_low  = {Mdot_low:.3e}")
                    print(f"    Mdot_high = {Mdot_high:.3e}")
                return Mdot_low, Mdot_high, gc

            # If failed for this gc, nudge gc and try again
            gc += step_gc

        raise RuntimeError(
            f"FAILED to find any gc (up to {max_iter_gc} tries) "
            f"for which Mdot can be bracketed."
        )


    
    def find_eigenvals():
        Mdot_guess = Mdot_guess_
        gc_guess = gc_guess_
        # Find eigenvalue: hot_slope(Mdot) = 0
        Mdot_low, Mdot_high, gc_used = bracket_vars(Mdot_guess, gc_guess, verbose=True)

        Mdot0 = 0.5 * (Mdot_low + Mdot_high)
        x0 = np.array([Mdot0, gc_used])

        # Mdot_eigen, gc_eigen = fsolve(get_residuals, x0)
        
        # return Mdot_eigen,gc_used
        def objective(Mdot):
            # return hot_slope(Mdot)
            dT_tilde = get_residuals([Mdot,gc_used])[0]
            return dT_tilde
            # return hot_T_tilde(Mdot) - Th_tilde  # Zero when T_hot = Th_tilde
        
        result_root = root_scalar(
            objective,
            bracket=(Mdot_low, Mdot_high),
            method='bisect',
            rtol=rtol_root
        )
        if not result_root.converged:
            raise RuntimeError("Eigenvalue solver did not converge")
        Mdot_eigen = result_root.root
        return Mdot_eigen, gc_used
    
    Mdot_eigen, gc_eigen = find_eigenvals()
    # Mdot_eigen, gc_eigen = [Mdot_guess_,gc_guess_]
    # Mdot_eigen = Mdot_low_
    # Get full solution for eigen Mdot
    # sol_eigen = integrate_profile(Mdot_eigen)
    sol_eigen = integrate_profile(Mdot_eigen,gc_eigen)
    # print(sol_eigen.y[0, -1])
    # T_end_tilde = sol_eigen.sol(r_tilde_eval)[0][-1]
    # print(np.log10(T_end_tilde*T0))
    
    # Interpolate onto requested grid
    if r_tilde_eval is None:
        r_tilde_eval = np.linspace(sol_eigen.t[0], sol_eigen.t[-1], 1000)
    
    # T_interp = interp1d(sol_eigen.t, sol_eigen.y[0], kind='cubic', 
    #                    bounds_error=False, fill_value='extrapolate')
    T_tilde = sol_eigen.sol(r_tilde_eval)[0]
    
    return {
        'Mdot_eigen': Mdot_eigen,
        'gc_eigen': gc_eigen,
        'r_tilde': r_tilde_eval,
        'T_tilde': T_tilde
    }

def energy_balance(T_tilde, r_tilde, Kt, Mdot):
    dT_tildedR_tilde = np.gradient(T_tilde,r_tilde)
    advection = (gamma*kB*Mdot*T0)/((gamma-1)*mu*mp*r0)*dT_tildedR_tilde
    cooling = N*sech_sqr(r_tilde)
    if Kt==0:
        return advection, cooling 
    else:
        d2T_tildedR_tilde2 = np.gradient(dT_tildedR_tilde,r_tilde)
        heat_flux = Kt*T0*d2T_tildedR_tilde2/r0**2
        return advection, cooling, heat_flux

def plot_temp_profile(Kt_arr, Mdot_arr, gc_arr):
    r_tilde = np.linspace(rc_tilde, rh_tilde, 1000)
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    fig1,ax1=plt.subplots(figsize=(8,6))
    for i in range(len(Kt_arr)):
        Kt = Kt_arr[i]
        gc = gc_arr[i]
        if Kt ==0:
            T_tilde, Mdot = first_order_ODE(r_tilde)
            ax1.plot(r_tilde*r0/pc, np.log10(T_tilde*T0), color='k', label=f"Kt = 0, "+r'$\dot{M} = $'+f"{Mdot:.2e} g/s"+r', $T^{\prime}(r_c) = $'+f"{0:.2f}", linewidth = 3)  
        else:
            Mdot_guess = Mdot_arr[i]
            gc_guess = gc_arr[i]
            sol = second_order_ode(r_tilde, Kt=Kt, Mdot_guess_=Mdot_guess, gc_guess_=gc_guess)
            T_tilde = sol['T_tilde']
            Mdot = sol['Mdot_eigen']
            gc = sol['gc_eigen']
            r_tilde_eval = sol['r_tilde']
            expo = int(np.log10(Kt))
            ax1.plot(r_tilde_eval*r0/pc, np.log10(T_tilde*T0), label=rf"$K_t = 10^{{{expo}}}\,(cgs)$, "+r'$\dot{M} = $'+f"{Mdot:.2e} g/s"+r', $T^{\prime}(r_c) = $'+f"{gc:.2f}", linewidth = 2)
    ax1.set_xlabel('r [pc]', fontsize = 14)
    ax1.set_ylabel(r'$\log_{10} \langle T\rangle$', fontsize = 14)
    # ax1.set_title('Temperature Profiles for Different Kt')
    ax1.grid()
    ax1.tick_params(axis='both', which='major', labelsize=14)
    # Position legend outside right edge
    # ax1.legend(loc='center left', 
    #         bbox_to_anchor=(1.02, 0.5), 
    #         fontsize=12, 
    #         frameon=True)
    ax1.legend(loc='upper center', 
           bbox_to_anchor=(0.5, -0.1),  # Bottom center, slightly below
           ncol=2,                        # Horizontal: 2 rows max
           fontsize=11, 
           frameon=False,                 # Clean no-border
           columnspacing=1.0)
    # ax1.legend(fontsize = 12)
    save_filename = os.path.join(script_dir,"temperature_profile_sech_thermal.png")
    # plt.savefig(save_filename)
    plt.savefig(save_filename, bbox_inches='tight', dpi=200)
    plt.close(fig1)
    
def plot_energy_balance_vs_r(r_tilde, T_tilde, Kt, Mdot, gc):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    energy_norm = 1e-26
    fig2,ax2 = plt.subplots(figsize=(8,6))
    if Kt == 0:
        advection,cooling = energy_balance(T_tilde, r_tilde, Kt=Kt, Mdot=Mdot)
        ax2.plot(r_tilde*r0/pc, -1*advection/energy_norm, label=f"Advection", linewidth = 2, color = 'g')
        ax2.plot(r_tilde*r0/pc, cooling/energy_norm, label=f"Cooling", linewidth = 2, color = 'b')
        ax2.set_title(rf"$K_t = 0\,(cgs)$, "+r'$\dot{M} = $'+f"{Mdot:.2e} g/s"+r', $T^{\prime}(r_c) = $'+f"{gc:.2f}", fontsize = 14)
    else:
        advection, cooling, heat_flux = energy_balance(T_tilde, r_tilde, Kt=Kt, Mdot=Mdot)
        ax2.plot(r_tilde*r0/pc, -1*heat_flux/energy_norm, label=f"Heat Flux", linewidth = 2, color = 'r')
        ax2.plot(r_tilde*r0/pc, -1*advection/energy_norm, label=f"Advection", linewidth = 2, color = 'g')
        ax2.plot(r_tilde*r0/pc, cooling/energy_norm, label=f"Cooling", linewidth = 2, color = 'b')
        ax2.set_title(rf"$K_t = 10^{{{int(np.log10(Kt))}}}\,(cgs)$, "+r'$\dot{M} = $'+f"{Mdot:.2e} g/s"+r', $T^{\prime}(r_c) = $'+f"{gc:.2f}", fontsize = 14)
        
    
    ax2.set_xlabel('r [pc]', fontsize = 14)
    ax2.set_ylabel(r'($10^{-26}\,\mathrm{erg}\,\mathrm{cm}^{-3}\,\mathrm{s}^{-1}$)', fontsize = 14)
    # ax2.set_yscale('log')
    ax2.set_ylim(-2.5,2.5)
    ax2.legend(fontsize = 12)
    ax2.tick_params(axis='both', which='major', labelsize=14)
    # Customize log grid
    ax2.minorticks_on()
    ax2.grid(True, which='both', ls='-', alpha=0.3)
    ax2.grid(which='major', alpha=0.7, linewidth=1.2)
    ax2.grid(which='minor', alpha=0.5, linewidth=1.0)
    if Kt == 0:
        save_filename = os.path.join(script_dir,"energy_balance_vs_r_sech_Kt_0.png")
    else:
        save_filename = os.path.join(script_dir,"energy_balance_vs_r_sech_Kt_1e"+str(int(np.log10(Kt)))+".png")
    plt.savefig(save_filename)
    plt.close(fig2)

def plot_energy_balance_vs_logT(r_tilde, T_tilde, Kt, Mdot, gc):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    energy_norm = 1e-26
    fig3,ax3 = plt.subplots(figsize=(8,6))
    if Kt == 0:
        advection,cooling = energy_balance(T_tilde, r_tilde, Kt=Kt, Mdot=Mdot)
        ax3.plot(np.log10(T_tilde*T0), advection, label=f"Advection", linewidth = 2, color = 'g')
        ax3.plot(np.log10(T_tilde*T0), cooling, label=f"Cooling", linewidth = 2, color = 'b')
        ax3.set_title(rf"$K_t = 0\,(cgs)$, "+r'$\dot{M} = $'+f"{Mdot:.2e} g/s"+r', $T^{\prime}(r_c) = $'+f"{gc:.2f}", fontsize = 14)
        
    else:
        advection, cooling, heat_flux = energy_balance(T_tilde, r_tilde, Kt=Kt, Mdot=Mdot)
        ax3.plot(np.log10(T_tilde*T0), heat_flux, label=f"Heat Flux", linewidth = 2, color = 'r')
        ax3.plot(np.log10(T_tilde*T0), advection, label=f"Advection", linewidth = 2, color = 'g')
        ax3.plot(np.log10(T_tilde*T0), cooling, label=f"Cooling", linewidth = 2, color = 'b')
        ax3.set_title(rf"$K_t = 10^{{{int(np.log10(Kt))}}}\,(cgs)$, "+r'$\dot{M} = $'+f"{Mdot:.2e} g/s"+r', $T^{\prime}(r_c) = $'+f"{gc:.2f}", fontsize = 14)
        
    ax3.set_ylabel(r'($erg\,cm^{-3}\,s^{-1}$)', fontsize = 14)
    ax3.set_xlabel(r'$log_{10}\langle T \rangle$', fontsize = 14)
    ax3.set_yscale('log')
    ax3.set_ylim(1e-36,1e-25)
    ax3.legend(fontsize = 12)
    ax3.tick_params(axis='both', which='major', labelsize=14)
    # Customize log grid
    ax3.minorticks_on()
    ax3.grid(True, which='both', ls='-', alpha=0.3)
    ax3.grid(which='major', alpha=0.7, linewidth=1.2)
    ax3.grid(which='minor', alpha=0.5, linewidth=1.0)
    if Kt == 0:
        save_filename = os.path.join(script_dir,"energy_balance_vs_logT_sech_Kt_0.png")
    else:
        save_filename = os.path.join(script_dir,"energy_balance_vs_logT_sech_Kt_1e"+str(int(np.log10(Kt)))+".png")
    plt.savefig(save_filename)
    plt.close(fig3)
    
def plot_stuff(Kt_arr, Mdot_arr, gc_arr):
    r_tilde = np.linspace(rc_tilde, rh_tilde, 1000)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    fig1,ax1=plt.subplots(figsize=(8,6))
    for i in range(len(Kt_arr)):
        Kt = Kt_arr[i]
        Mdot_guess = Mdot_arr[i]
        gc_guess = gc_arr[i]
        if Kt == 0:
            T_tilde, Mdot = first_order_ODE(r_tilde)
            ax1.plot(r_tilde*r0/pc, np.log10(T_tilde*T0), color='k', label=rf"$K_t = 0$, "+r'$\dot{M} = $'+f"{Mdot:.2e} g/s"+r', $\tilde{T}^{\prime}(\tilde{r}_c) = $'+f"{0:.2f}", linewidth = 3)
            plot_energy_balance_vs_logT(r_tilde, T_tilde, Kt, Mdot, gc=0.0)
            plot_energy_balance_vs_r(r_tilde, T_tilde, Kt, Mdot, gc =0.0)
        else:
            sol = second_order_ode(r_tilde, Kt=Kt, Mdot_guess_=Mdot_guess, gc_guess_=gc_guess)
            T_tilde = sol['T_tilde']
            Mdot = sol['Mdot_eigen']
            gc = sol['gc_eigen']
            r_tilde_eval = sol['r_tilde']
            expo = int(np.log10(Kt))
            ax1.plot(r_tilde_eval*r0/pc, np.log10(T_tilde*T0), label=rf"$K_t = 10^{{{expo}}}\,(cgs)$, "+r'$\dot{M} = $'+f"{Mdot:.2e} g/s"+r', $\tilde{T}^{\prime}(\tilde{r}_c) = $'+f"{gc:.2f}", linewidth = 2)
            plot_energy_balance_vs_logT(r_tilde, T_tilde, Kt, Mdot, gc=gc)
            plot_energy_balance_vs_r(r_tilde, T_tilde, Kt, Mdot, gc =gc)
    ax1.set_xlabel('r [pc]', fontsize = 14)
    ax1.set_ylabel(r'$\log_{10} \langle T\rangle$', fontsize = 14)
    ax1.grid(True, which='both', ls='-', alpha=1.0)
    ax1.tick_params(axis='both', which='major', labelsize=14)
    ax1.legend(loc='upper center', 
           bbox_to_anchor=(0.5, -0.1),  # Bottom center, slightly below
           ncol=2,                        # Horizontal: 2 rows max
           fontsize=12, 
           frameon=False,                 # Clean no-border
           columnspacing=1.0)
    save_filename = os.path.join(script_dir,"temperature_profile_sech_thermal.png")
    plt.savefig(save_filename, bbox_inches='tight', dpi=200)
    plt.close(fig1)
        
if __name__ == "__main__":
    Kt_arr = [0,1e1,1e2,1e3,1e4, 1e5, 1e6, 1e7]
    Mdot_arr = [0,1e-23,1e-23,1e-23,1e-23, 1e-23, 1e-23, 1e-23]
    gc_arr = [0.0,0.0,0.0,0.0,0.0,0.4, 0.4, 0.4]
    
    # Kt_arr = [0,1e7]
    # Mdot_arr = [0,1e-23]
    # gc_arr =[0,0.4]
    # plot_temp_profile(Kt_arr, Mdot_arr, gc_arr)
    plot_stuff(Kt_arr, Mdot_arr, gc_arr)
# Kt_arr = [0,1,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8]
# plot_temp_profile(Kt_arr)
