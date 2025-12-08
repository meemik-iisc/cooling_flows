import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import os

# Simplified constants
gamma = 5/3
q = 0  # spherical geometry

# Simplified profiles: all constants set to 1 for speed/demo
def rho(r):
    return 1.0  # constant density

def T(r):
    return 1.0  # constant temperature

def Lambda(T):
    return 1.0  # constant cooling function

def n_e(r):
    return 1.0  # constant electron number density

def n_i(r):
    return 1.0  # constant ion number density

def c_s(r):
    return 1.0  # constant sound speed

# ODE for dv/dr simplified accordingly
def dvdr(r, v):
    if v == 0:
        return 0  # avoid division by zero
    
    cs = c_s(r)
    ne = n_e(r)
    ni = n_i(r)
    lam = Lambda(T(r))
    dens = rho(r)
    
    numerator = (gamma - 1)/v * (ne * ni * lam)/dens + q * cs**2 / r
    denominator = 1 - (cs**2)/(v**2)
    threshold = 1e-5
    if abs(denominator) < threshold:
        denominator = threshold if denominator > 0 else -threshold
    return numerator / denominator

# Sonic point derivative from quadratic relation (Equation 6)
# Here, cooling slope (dlnLambda/dlnT) ~0 for constant Lambda in this demo
def sonic_dvdr(q, gamma, dlnLambda_dlnT=0.0):
    A = gamma + 1
    B = q * (dlnLambda_dlnT * (gamma - 1) + 4 - gamma)
    C = q * (q * (dlnLambda_dlnT - 2) + 1)
    disc = B**2 - 4*A*C
    root1 = (-B + np.sqrt(disc)) / (2*A)
    root2 = (-B - np.sqrt(disc)) / (2*A)
    return root1 if root1 > 0 else root2

dvdr0 = sonic_dvdr(q, gamma)
r0=2.0
epsilon = 1e-4  # small offset from sonic radius to start integration
v0 = -c_s(r0)  # sonic velocity
# Integrate inward (r0 - epsilon to r_min)
r_min = 0.5
r_in = np.linspace(r0 - epsilon, r_min, 200)
v_in_init = v0 + dvdr0 * (-epsilon)
sol_in = solve_ivp(dvdr, (r0 - epsilon, r_min), [v_in_init], t_eval=r_in)
r_in_plot = sol_in.t[::-1]
v_in_plot = sol_in.y[0][::-1]

# Integrate outward (r0 + epsilon to r_max)
r_max = 10.0
r_out = np.linspace(r0 + epsilon, r_max, 200)
v_out_init = v0 + dvdr0 * epsilon
sol_out = solve_ivp(dvdr, (r0 + epsilon, r_max), [v_out_init], t_eval=r_out)

# Initial condition at r=1
r0 = 1.0
v0 = -0.5  # subsonic inflow velocity (negative)

r_max = 10.0
r_eval = np.linspace(r0, r_max, 500)

# Solve ODE
sol = solve_ivp(dvdr, (r0, r_max), [v0], t_eval=r_eval, method='RK45')

# Supersonic initial condition at r=1
v0_supersonic = -1.5  # supersonic inflow (faster than c_s = 1)

# Solve ODE for supersonic initial condition
sol_sup = solve_ivp(dvdr, (r0, r_max), [v0_supersonic], t_eval=r_eval, method='RK45')

# Plot results
plt.plot(r_in_plot, v_in_plot, 'g')
plt.plot(sol_out.t, sol_out.y[0], 'g', label='Transonic solution')
plt.plot(sol.t, sol.y[0], label='subsonic solution')

# Plot supersonic solution
plt.plot(sol_sup.t, sol_sup.y[0], label='Supersonic Solution')

plt.plot(sol.t, -np.ones_like(sol.t), 'k--', label='- Sound speed')
plt.xlabel('Radius r (normalized)')
plt.ylabel('Velocity (normalized)')
plt.title('Simplified Cooling Flow Velocity Profile (Cartesian geometry)')
plt.legend()
plt.grid(True)

# Save figure to same directory as script
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(script_dir, "cartesian_cooling_flow_velocity_simple.png")


plt.show()
plt.savefig(save_path)
