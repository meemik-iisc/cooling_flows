import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import os

# Constants
gamma = 5 / 3
CONST_kB = 1.3806505e-16
CONST_mp = 1.67262171e-24

r_range = [0.01,1.5]

# Parameters
r0          = 0.5       #kpc       
init_temp   = 4.0e4     #K
bnd_temp    = 1.0e6     #K
init_num    = 1.0       #cm^(-3)
init_pres   = 5e-12     #(dyne/cm^2)

mu=6.091734e-01	
mue=1.165909e+00	
mui=1.275724e+00
delta = 0.1 * r0        # transition width, adjust as needed


def Temp(r):
    # Set initial temperature profile (a smooth step function)
    w = (r - r0) / delta
    f = 0.5 * (1.0 + np.tanh(w))
    return (1-f)*init_temp+f*bnd_temp

def rho(r):
    # Set initial density profile for constant pressure (a smooth step function)
    return (init_pres*CONST_mp*mu)/(CONST_kB*Temp(r))

def cooling_lambda(T):
    if(T<1.0e-4):
        return 0
    # Read original cooling table
    data = np.loadtxt('cooltable.dat')
    temperature = data[:, 0]
    cooling_rate = data[:, 1]
    # Find the index with the minimum absolute difference
    idx = (np.abs(temperature - T)).argmin()
    return cooling_rate[idx]
vectorized_cooling_lambda = np.vectorize(cooling_lambda)
def pres(r):
    return (rho(r)*Temp(r)*CONST_kB)/(mu*CONST_mp)

def plot_initial_profile():
    r = np.linspace(r_range[0],r_range[1])
    fig, axs = plt.subplots(2, 2, figsize=(10, 6))
    axs[0, 0].plot(r, rho(r)/(mu*CONST_mp))
    axs[0,0].set_yscale("log")
    axs[0, 0].set_title("Number Density")
    axs[0, 0].set_xlabel("r (kpc)")
    axs[0, 0].set_ylabel("Number Density")
    
    axs[0, 1].plot(r, Temp(r))
    axs[0,1].set_yscale("log")
    axs[0, 1].set_title("Temperature")
    axs[0, 1].set_xlabel("r (kpc)")
    axs[0, 1].set_ylabel("K")
    
    axs[1, 0].plot(r, pres(r))
    axs[1, 0].set_title("Pressure")
    axs[1, 0].set_xlabel("r (kpc)")
    axs[1, 0].set_ylabel("Pa")
    
    temp_prof = np.logspace(2,10)
    axs[1, 1].plot(temp_prof, vectorized_cooling_lambda(temp_prof))
    axs[1,1].set_xscale("log")
    axs[1, 1].set_title("Cooling Function")
    axs[1, 1].set_xlabel("r")
    axs[1, 1].set_ylabel(r'$erg/, cm^{-3}/, s^{-1}$')
    
    fig.suptitle("Initial Profiles")
    plt.tight_layout()
    import os
    script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),"cylindrical_cooling_flow")
    save_path = os.path.join(script_dir,"initial_profile.png")
    fig.savefig(save_path,dpi=300)
    plt.close()
    
def odes(r, y, q, v0cs0):
    v, s = y
    if abs(v) < 1e-8:
        v = -1e-8
    dens = rho(r)/rho(r0)
    pressure = pres(r)/pres(r0)
    temp0 = Temp(r0)
    temp = Temp(r)/temp0
    lam = vectorized_cooling_lambda(temp)/cooling_lambda(temp0)

    numerator = q * ((dens * lam / v) + (pressure) / (r * dens)) / (v0cs0 ** 2)
    denominator = 1 - (pressure) / (dens * (v ** 2) * (v0cs0 ** 2))
    denominator = denominator if abs(denominator) > 1e-6 else 1e-6 * np.sign(denominator) if denominator != 0 else 1e-6
    dvdr = numerator / denominator
    dvdr = max(dvdr, 1e-6)

    numerator_s = q * gamma * (dens ** 2) * lam
    denominator_s = pressure * v
    denominator_s = max(abs(denominator_s), 1e-8) * np.sign(denominator_s)
    dsdr = -numerator_s / denominator_s
    dsdr = dsdr if abs(dsdr) > 1e-8 else 1e-8

    return [dvdr, dsdr]

def solve_profile(q, v0cs0,rrange, label, axv, axrho, axT):
    v0 = -v0cs0
    s0 = 1.0
    r_start = rrange[0]
    r_end = rrange[1]
    r_inner_rev = np.linspace(r0, r_start, 300)  # decreasing from r0 to r_start
    r_outer = np.linspace(r0, r_end, 300)

    
    sol_inner = solve_ivp(odes, (r0, r_start), [v0, s0], t_eval=r_inner_rev, args=(q, v0cs0), method='LSODA')
    sol_outer = solve_ivp(odes, (r0, r_end), [v0, s0], t_eval=r_outer, args=(q, v0cs0), method='LSODA')

    r_combined = np.concatenate((sol_inner.t[::-1], sol_outer.t[1:]))
    v_combined = np.concatenate((sol_inner.y[0][::-1], sol_outer.y[0][1:]))
    # v_normalized = v_combined / c_s(r_combined)
    # Find index nearest r_0
    idx_sonic = np.argmin(np.abs(r_combined - r0))
    # # Exclude if v/c_s at r_0 equals v0cs0 (within tolerance), i.e., skip these branches
    # if np.abs(v_normalized[idx_sonic] - v0cs0) >= 0.01:  # tolerance 0.01
    #     plt.plot(r_combined, v_normalized, label=label)
    # else:
    #     print(f"Excluded: {label} (Mach = {v_normalized[idx_sonic]:.4f} at r={r_combined[idx_sonic]:.4f})")

    axv.plot(r_combined, v_combined, label=label)
    axrho.plot(r_combined, rho(r_combined), label=label)
    axT.plot(r_combined, Temp(r_combined), label=label)
    # plt.plot(sol_inner.t[::-1],sol_inner.y[0][::-1])
    # plt.plot(sol_outer.t[1:],sol_outer.y[0][1:])
    
def plot_solution():
    # Plot multiple profiles on same graph
    fig1,ax1=plt.subplots(figsize=(8,6))
    fig2,ax2=plt.subplots(figsize=(8,6))
    fig3,ax3=plt.subplots(figsize=(8,6))
    param_list = [
        (2, 1, r'$q=2, v_0/c_{s0}=1$'),
        (2, 0.1, r'$q=2, v_0/c_{s0}=0.1$'),
        (2, 2, r'$q=2, v_0/c_{s0}=2$'),
        (1, 1, r'$q=1, v_0/c_{s0}=1$'),
        (1, 0.1, r'$q=1, v_0/c_{s0}=0.1$'),
        (1, 2, r'$q=1, v_0/c_{s0}=2$')
    ]

    for q_val, v_val, label in param_list:
        solve_profile(q_val, v_val, r_range, label,ax1,ax2,ax3)
        
    ax1.axhline(-1, color='k', linestyle='--', alpha=0.3, label="sonic velocity")
    ax1.set_xlabel('Radius $r/r_0$')
    ax1.set_ylabel(r'Velocity $v / c_s$')
    ax1.set_title('Velocity Profiles for Different $q$ and $v_0/c_{s0}$ for simple density and temperature profiles')
    ax1.legend()
    ax1.grid()

    ax2.set_xlabel('Radius $r/r_0$')
    ax2.set_ylabel(r'Density $\rho / \rho_0$')
    ax2.set_title('Density Profiles for Different $q$ and $v_0/c_{s0}$')
    ax2.legend()
    ax2.grid()

    ax3.set_xlabel('Radius $r/r_0$')
    ax3.set_ylabel(r'Temperature $T / T_0$')
    ax3.set_title('Temperature Profiles for Different $q$ and $v_0/c_{s0}$')
    ax3.legend()
    ax3.grid()

    # Optional: save plot
    # plt.savefig("velocity_profiles_multi_q_v0cs0.png")
    # Save figure to same directory as script
    import os
    script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),"cylindrical_cooling_flow")
    save_path_1 = os.path.join(script_dir, "mach_number.png")
    save_path_2 = os.path.join(script_dir, "density_profile.png")
    save_path_3 = os.path.join(script_dir, "temperature_profile.png")
    save_path_4 = os.path.join(script_dir, "cooling_curve.png")

    fig1.savefig(save_path_1)
    fig2.savefig(save_path_2)
    fig3.savefig(save_path_3)
    plt.close()

if __name__ == "__main__":
    # plot_initial_profile()
    plot_solution()
    

