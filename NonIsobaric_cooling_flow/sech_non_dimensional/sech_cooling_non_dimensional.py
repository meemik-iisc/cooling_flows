import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
from scipy.interpolate import interp1d
from scipy.optimize import fsolve

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
# Kt = 1.0e9           # you will vary this
# Dimensionless variables
rc_tilde = rc / r0
rh_tilde = rh / r0
Tc_tilde = Tc / T0
Th_tilde = Th / T0
print(f"Tc_tilde = {Tc_tilde:.6f}, Th_tilde = {Th_tilde:.6f}")


Mdot = ((gamma-1)*mu*N*mp*r0*(np.tanh(rh_tilde)-np.tanh(rc_tilde)))/(gamma*kB*T0*(Th_tilde-Tc_tilde))
print(f"N = {N:.2e} ergcm^-3s^-1, Mdot = {Mdot:.2e} gcm^-2s^-1")
def sech_sqr(z):
    return 1/(np.cosh(z)**2)

def first_order_ODE(r_tilde):
    A = (Th_tilde-Tc_tilde)/(np.tanh(rh_tilde)-np.tanh(rc_tilde))
    C = (Tc_tilde*np.tanh(rh_tilde)-Th_tilde*np.tanh(rc_tilde))/(np.tanh(rh_tilde)-np.tanh(rc_tilde))
    return A*np.tanh(r_tilde)+C

def second_order_ode(r_tilde, Kt):
    term1 = (gamma*kB*Mdot*r0)/((gamma-1)*mu*mp*Kt)
    term2 = (N*r0**2)/(Kt*T0)
    
    def ode_system(r, y):
        # y[0] = T, y[1] = T'
        dTdz = y[1]
        d2Tdz2 = -1*term1*y[1]+term2*sech_sqr(r) 
        return np.vstack((dTdz, d2Tdz2))

    # Boundary conditions
    def bc(ya, yb):
        return np.array([ya[0] - Tc_tilde, yb[0] - Th_tilde])

    # Initial guess for solver: linear between Tc and Th for T, zeros for derivative
    y_init = np.zeros((2, r_tilde.size))
    y_init[0] = np.linspace(Tc_tilde, Th_tilde, r_tilde.size)

    # Solve BVP
    sol = solve_bvp(ode_system, bc, r_tilde, y_init, max_nodes=10000)
    
    if sol.success:
        # Interpolate the solution T on the original mesh z
        interpolator = interp1d(sol.x, sol.y[0], kind='cubic')
        T_interpolated = interpolator(r_tilde)
        return T_interpolated
    else:
        raise RuntimeError(f"BVP solver failed to converge for Kt={Kt}")

def energy_balance(T_tilde, r_tilde, Kt):
    dT_tildedR_tilde = np.gradient(T_tilde,r_tilde)
    advection = (gamma*kB*Mdot*T0)/((gamma-1)*mu*mp*r0)*dT_tildedR_tilde
    cooling = N*sech_sqr(r_tilde)
    if Kt==0:
        return advection, cooling 
    else:
        d2T_tildedR_tilde2 = np.gradient(dT_tildedR_tilde,r_tilde)
        heat_flux = Kt*T0*d2T_tildedR_tilde2/r0**2
        return advection, cooling, heat_flux

def plot_temp_profile(Kt_arr):
    r_tilde = np.linspace(rc_tilde, rh_tilde, 1000)
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    fig1,ax1=plt.subplots(figsize=(8,6))
    for Kt in Kt_arr:
        if Kt ==0:
            T_tilde = first_order_ODE(r_tilde)
            ax1.plot(r_tilde*r0/pc, np.log10(T_tilde*T0), color='k', label=f"Kt = 0", linewidth = 3)  
        else:
            T_tilde = second_order_ode(r_tilde, Kt=Kt)
            expo = int(np.log10(Kt))
            ax1.plot(r_tilde*r0/pc, np.log10(T_tilde*T0), label=rf"$K_t = 10^{{{expo}}}\,$"+r'$\mathrm{erg}\,\mathrm{cm}^{-1}\mathrm{s}^{-1}\mathrm{K}^{-1}$', linewidth = 2)
    ax1.set_xlabel('r [pc]', fontsize = 14)
    ax1.set_ylabel(r'$\log_{10} \langle T\rangle$', fontsize = 14)
    # ax1.set_title('Temperature Profiles for Different Kt')
    ax1.grid()
    ax1.legend(fontsize = 12)
    save_filename = os.path.join(script_dir,"temperature_profile_sech_thermal.png")
    plt.savefig(save_filename)
    plt.close(fig1)
    
def plot_energy_balance_vs_r(Kt):
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    r_tilde = np.linspace(rc_tilde, rh_tilde, 1000)
    energy_norm = 1e-26
    fig2,ax2 = plt.subplots(figsize=(8,6))
    if Kt == 0:
        T_tilde = first_order_ODE(r_tilde)
        advection,cooling = energy_balance(T_tilde, r_tilde, Kt=Kt)
    else:
        T_tilde = second_order_ode(r_tilde, Kt=Kt)
        advection, cooling, heat_flux = energy_balance(T_tilde, r_tilde, Kt=Kt)
        ax2.plot(r_tilde*r0/pc, -1*heat_flux/energy_norm, label=f"Heat Flux", linewidth = 2, color = 'r')
    ax2.plot(r_tilde*r0/pc, -1*advection/energy_norm, label=f"Advection", linewidth = 2, color = 'g')
    ax2.plot(r_tilde*r0/pc, cooling/energy_norm, label=f"Cooling", linewidth = 2, color = 'b')
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

def plot_energy_balance_vs_logT(Kt):
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    r_tilde = np.linspace(rc_tilde, rh_tilde, 1000)
    fig2,ax2 = plt.subplots(figsize=(8,6))
    if Kt == 0:
        T_tilde = first_order_ODE(r_tilde)
        advection,cooling = energy_balance(T_tilde, r_tilde, Kt=Kt)
    else:
        T_tilde = second_order_ode(r_tilde, Kt=Kt)
        advection, cooling, heat_flux = energy_balance(T_tilde, r_tilde, Kt=Kt)
        ax2.plot(np.log10(T_tilde*T0), heat_flux, label=f"Heat Flux", linewidth = 2, color='r')
    ax2.plot(np.log10(T_tilde*T0), advection, label=f"Advection", linewidth = 2, color='g')
    ax2.plot(np.log10(T_tilde*T0), cooling, label=f"Cooling", linewidth = 2, color='b')
    ax2.set_xlabel('$log_{10}T$ [K]', fontsize = 14)
    ax2.set_ylabel(r'($erg\,cm^{-3}\,s^{-1}$)', fontsize = 14)
    ax2.set_yscale('log')
    ax2.set_ylim(1e-36,1e-25)
    ax2.legend(fontsize = 12)
    ax2.tick_params(axis='both', which='major', labelsize=14)
    # Customize log grid
    ax2.minorticks_on()
    ax2.grid(True, which='both', ls='-', alpha=0.3)
    ax2.grid(which='major', alpha=0.7, linewidth=1.2)
    ax2.grid(which='minor', alpha=0.5, linewidth=1.0)
    if Kt == 0:
        save_filename = os.path.join(script_dir,"energy_balance_vs_logT_sech_Kt_0.png")
    else:
        save_filename = os.path.join(script_dir,"energy_balance_vs_logT_sech_Kt_1e"+str(int(np.log10(Kt)))+".png")
    plt.savefig(save_filename)
    plt.close(fig2)

Kt_arr = [0,1,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8]
plot_temp_profile(Kt_arr)
for Kt in Kt_arr:
    plot_energy_balance_vs_r(Kt)
    plot_energy_balance_vs_logT(Kt)