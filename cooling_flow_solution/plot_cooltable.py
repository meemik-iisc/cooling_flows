import numpy as np
import matplotlib.pyplot as plt
# Read the data from 'cooltable.dat' with tab delimiter
data = np.loadtxt('cooltable_cutoff.dat')

# Assuming first column is temperature, second is cooling rate
temperature = data[:, 0]
cooling_rate = data[:, 1]

plt.plot(temperature, cooling_rate)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Temperature (K)')
plt.ylabel('Cooling Rate (erg cm^-3 s^-1)')
plt.title('Cooling Rate vs Temperature')
# Save figure to same directory as script
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(script_dir, "cooltable.png")
plt.savefig(save_path)
plt.show()
