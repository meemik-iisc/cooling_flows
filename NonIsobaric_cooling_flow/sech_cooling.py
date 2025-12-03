import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp,cumulative_trapezoid
from scipy.interpolate import interp1d
import numpy as np

# Parameters
z0 = 0.1

# Boundary conditions
zc = -1       # example lower boundary point
zh = 1        # example upper boundary point

T_hot = 1e6
Tc = 1e4/T_hot      # temperature at zc
Th = 1e6/T_hot      # temperature at zh

B=1.0
C=(Th-Tc)/((np.tanh(zh/z0)-np.tanh(zc/z0))*z0)
def sech_sqr(z):
    return 1/(np.cosh(z)**2)
def solve_first_order_ODE(z):
    num1 = (Th-Tc)*np.tanh(z/z0)
    num2 = Tc*np.tanh(zh/z0)-Th*np.tanh(zc/z0)
    den  = np.tanh(zh/z0)-np.tanh(zc/z0)
    return (num1+num2)/den
    # T_prime = (B/C)*sech_sqr(z/z0)
    # T = cumulative_trapezoid(T_prime, z, initial=0)
    # T = T+Tc-T[0]
    # return T

def second_order_ODE(A, z, Tc, Th):
    # Define the ODE system for solve_bvp
    def ode_system(z, y):
        # y[0] = T, y[1] = T'
        dTdz = y[1]
        d2Tdz2 = -(B/A)*y[1]+(C/A)*sech_sqr(z/z0) 
        return np.vstack((dTdz, d2Tdz2))

    # Boundary conditions
    def bc(ya, yb):
        return np.array([ya[0] - Tc, yb[0] - Th])

    # Initial guess for solver: linear between Tc and Th for T, zeros for derivative
    y_init = np.zeros((2, z.size))
    y_init[0] = np.linspace(Tc, Th, z.size)

    # Solve BVP
    sol = solve_bvp(ode_system, bc, z, y_init, max_nodes=10000)
    
    if sol.success:
        # Interpolate the solution T on the original mesh z
        interpolator = interp1d(sol.x, sol.y[0], kind='cubic')
        T_interpolated = interpolator(z)
        return T_interpolated
    else:
        raise RuntimeError(f"BVP solver failed to converge for A={A}")

def energy_balance(T,z,A):
    if A == 0:
        dTdz = np.gradient(T,z)
        return -1*B*dTdz,C*sech_sqr(z/z0)
    else:
        dTdz = np.gradient(T,z)
        d2Tdz2 = np.gradient(dTdz,z)
        return -1*A*d2Tdz2,-1*B*dTdz,C*sech_sqr(z/z0) 
    

z=np.linspace(zc,zh,100)
import os
script_dir = os.path.dirname(os.path.abspath(__file__))

fig1,ax1=plt.subplots(figsize=(8,6))
for A in [10,1,0.1,0.01,0.001,0]:
    fig2,ax2=plt.subplots(figsize=(8,6))
    if A==0:
        T=solve_first_order_ODE(z)
        ax1.plot(z,np.log10(T), color='k',label=r'$K_t=0$')
        B_dT_dz, C_sech_sq_z = energy_balance(T,z,A)
        ax2.plot(z,B_dT_dz, color='g', label=r'$<\rho u_z>\frac{dB}{dz}$')
        ax2.plot(z,C_sech_sq_z, color='b', label=r'$<n^2 \Lambda(T)>$')
    else:
        T=second_order_ODE(A, z, Tc, Th)
        ax1.plot(z,np.log10(T), label=f'$K_t={A}$')
        A_d2T_dz2, B_dT_dz, C_sech_sq_z = energy_balance(T,z,A)
        ax2.plot(z,A_d2T_dz2, color='r', label=r'$-K_t \frac{d^2T}{dz^2}$')
        ax2.plot(z,B_dT_dz, color='g', label=r'$<\rho u_z>\frac{dB}{dz}$')
        ax2.plot(z,C_sech_sq_z,color='b', label=r'$<n^2 \Lambda(T)>$')
        
    ax2.set_xlabel('z')
    ax2.set_ylabel('Energy Balance')
    ax2.set_title(f'$K_t = {A}$')
    ax2.legend()
    ax2.grid()
    save_file_2 = os.path.join(script_dir,"sech_outputs",'energy_balance', f"energy_balance_Kt_{A}.png")
    plt.savefig(save_file_2, dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    fig3,ax3=plt.subplots(figsize=(8,6))    
    if A==0:
        T=solve_first_order_ODE(z)  
        B_dT_dz, C_sech_sq_z = energy_balance(T,z,A)      
        ax3.plot(np.log10(T*T_hot),B_dT_dz, color='g', label=r'$<\rho u_z>\frac{dB}{dz}$')
        ax3.plot(np.log10(T*T_hot),C_sech_sq_z, color='b', label=r'$<n^2 \Lambda(T)>$')
    else:
        T=second_order_ODE(A, z, Tc, Th)
        A_d2T_dz2, B_dT_dz, C_sech_sq_z = energy_balance(T,z,A)
        ax3.plot(np.log10(T*T_hot),A_d2T_dz2, color='r', label=r'$-K_t \frac{d^2T}{dz^2}$')
        ax3.plot(np.log10(T*T_hot),B_dT_dz, color='g', label=r'$<\rho u_z>\frac{dB}{dz}$')
        ax3.plot(np.log10(T*T_hot),C_sech_sq_z,color='b', label=r'$<n^2 \Lambda(T)>$')
    
    ax3.set_xlabel('$log_{10}T$')
    ax3.set_ylabel('Energy Balance')
    # ax3.set_xscale('log')
    # ax3.set_xlim(1e4,1e6)
    ax3.set_title(f'$K_t = {A}$')
    ax3.legend()
    ax3.grid()
    save_file_3 = os.path.join(script_dir,"sech_outputs",'energy_vs_T', f"energy_vs_T_Kt_{A}.png")
    plt.savefig(save_file_3, dpi=300, bbox_inches='tight')
    plt.close(fig3)
    
ax1.set_xlabel('z')
ax1.set_ylabel('$log_{10}(T/T_h)$')
ax1.set_title('Temperature Profile for different turbulent heating coefficient')
ax1.legend()    
ax1.grid()

save_file = os.path.join(script_dir,"sech_outputs", "temp_profile.png")
plt.savefig(save_file, dpi=300, bbox_inches='tight')
plt.close(fig1)