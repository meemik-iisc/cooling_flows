import numpy as np
import matplotlib.pyplot as plt
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

z_trml = 8
delU = 31

data = np.load(os.path.join(script_dir, 'data_arrays.npz'))
emi_pdf = data['emis_vol_av']
n_row = len(emi_pdf)
z = np.linspace(-20,20,n_row)
z -= z_trml
start = 3 * n_row // 8
end = 7 * n_row // 8

plt.plot(z[start:end],emi_pdf[start:end])
plt.savefig(os.path.join(script_dir, 'fitting_outputs/emis_vol_av.png'), dpi=300, bbox_inches='tight')
plt.close()