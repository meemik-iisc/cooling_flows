import numpy as np

# Read original cooling table
data = np.loadtxt('cooltable.dat')

temperature = data[:, 0]
cooling_rate = data[:, 1]

# Set cooling rate to zero where temperature < 1e4 K
cooling_rate_modified = np.where(temperature < 1e4, 0.0, cooling_rate)

# Combine temperature and modified cooling rate into one array
modified_data = np.column_stack((temperature, cooling_rate_modified))

# Save modified data to new file
np.savetxt('cooltable_cutoff.dat', modified_data, fmt='%.6e', delimiter='\t', header='Temperature\tCooling_rate')

print("Modified cooling table saved as 'cooltable_cutoff.dat' with cooling rate=0 below 1e4 K.")
