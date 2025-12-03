import numpy as np
import matplotlib.pyplot as plt

# Load the data file
filename = "subsonic_rdpv.txt"
data = np.loadtxt(filename)

# Inspect the shape of the data
print(f"Data shape: {data.shape}")

# Assuming columns are organized as follows (based on typical structure):
# col0 = radius (r), col1 = density, col2 = pressure, col3 = velocity (possibly radial)
# We will plot the first four columns

r = data[:,0]
rho = data[:,1]
prs = data[:,2]
vx = data[:,3]

# Calculate temperature assuming ideal gas law: T = P / (rho * R)
# R = kB / (mu * mH), we use mu=0.6 for mean molecular weight in ionized gas
mu = 0.6
kB = 1.380649e-16  # erg/K
mH = 1.6726219e-24 # g
R = kB / (mu * mH)

T = prs / (rho * R)

# Plotting
plt.figure(figsize=(10,7))
plt.subplot(311)
plt.plot(r, rho, label="Density")
plt.xlabel("Radius")
plt.ylabel("Density")
plt.grid()

plt.subplot(312)
plt.plot(r, prs, label="Pressure", color='orange')
plt.xlabel("Radius")
plt.ylabel("Pressure")
plt.grid()

plt.subplot(313)
plt.plot(r, T, label="Temperature", color='green')
plt.xlabel("Radius")
plt.ylabel("Temperature")
plt.grid()

plt.tight_layout()
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
plt.savefig(os.path.join(script_dir, "initial_condition.png"))
plt.show()

