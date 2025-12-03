import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt

def sech_sq(z):
    return 1/(np.cosh(z)**2)

# System to integrate: convert second-order ODE to two first-order ODEs
def ode_system(z, Y, A, B, C, z0):
    T, Tp = Y
    dTdz = Tp
    dTpdz = (B * Tp - C * (1/np.cosh(z/z0))**2) / A
    return [dTdz, dTpdz]

# Shooting function to match boundary condition at z_h
def shoot(var, fixed_param, A, C, z0, z_c, z_h, T_c, T_h, find_B=True):
    if find_B:
        B = var
        pass
    else:
        B = fixed_param
        C = var
    sol = solve_ivp(
        ode_system, [z_c, z_h], [T_c, 0], args=(A, B, C, z0), t_eval=[z_h])
    T_z_h = sol.y[0][-1]
    return T_z_h - T_h

# Example workflow to find B with fixed C (or vice versa)
# 1. Set A, C, z0, z_c, z_h, T_c, T_h. Choose which parameter to shoot for.
# 2. Call root_scalar to find the value that fits the boundary condition.

A = 1.0
C = 2.0      # Try a fixed value for example
z0 = 1.0
z_c = -1.0
z_h = 1.0
T_c = 1.0e-2
T_h = 1.0

# Find B such that T(z_h)=T_h
result = root_scalar(
    shoot,
    args=(C, A, C, z0, z_c, z_h, T_c, T_h, True),  # True: finding B
    bracket=[-10, 10],
    method='bisect'
)

B_found = result.root

# Integrate with found B value for profile
sol = solve_ivp(
    ode_system, [z_c, z_h], [T_c, 0], args=(A, B_found, C, z0), dense_output=True)

z_arr = np.linspace(z_c, z_h, 100)
T_arr = sol.sol(z_arr)[0]

plt.plot(z_arr, T_arr)
plt.xlabel('z')
plt.ylabel('T(z)')
plt.title('Shooting Method: ODE Solution')
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
save_file = os.path.join(script_dir, "shooting.png")
plt.savefig(save_file, dpi=300, bbox_inches='tight')