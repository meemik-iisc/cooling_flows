import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.interpolate import griddata
from matplotlib import contour
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
data = np.load(os.path.join(script_dir,"KH_2D_Z_0_00169_C1.npz"), allow_pickle=True)
extent = data['extent']
temp_shape = data['temp'].shape
temp_data = data['temp']

# Find contour at 1e5 K (sorted by x)
level = 1e5
mask = np.isclose(temp_data, level, atol=5e3)
y_pos, x_pos = np.where(mask)
sort_idx = np.argsort(x_pos)
x_pos, y_pos = x_pos[sort_idx], y_pos[sort_idx]


# Compute tangents using np.gradient (dx, dy)
dx = np.gradient(x_pos)
dy = np.gradient(y_pos)

# Tangent vectors (unit length)
mag = np.sqrt(dx**2 + dy**2)
mag[mag == 0] = 1  # Avoid division by zero

# Tangent vectors (unit length)
tangent_x = dx / mag
tangent_y = dy / mag

# Normal vectors (rotate 90° CCW: [-dy, dx])
normal_x = -tangent_y
normal_y = tangent_x

fig, ax = plt.subplots(figsize=(12, 10))

# Plot trajectory
ax.plot(x_pos, y_pos, 'b-', linewidth=2, alpha=0.7, label='Trajectory')

# Plot tangents + normals at each point
normal_length = 10  # Adjust scale
tangent_length = 8

for i in range(len(x_pos)):
    # Tangent arrow
    ax.arrow(x_pos[i], y_pos[i], tangent_x[i]*tangent_length, tangent_y[i]*tangent_length,
             head_width=0.02, color='green', linewidth=2, alpha=0.8)
    
    # Normal arrow
    ax.arrow(x_pos[i], y_pos[i], normal_x[i]*normal_length, normal_y[i]*normal_length,
             head_width=0.02, color='red', linewidth=3, alpha=0.9)
    
    # Points
    ax.plot(x_pos[i], y_pos[i], 'mo', markersize=6, markeredgecolor='black')

# ax.scatter(x, y, c='magenta', s=40, edgecolors='black', linewidth=1.5, label='Points')
ax.set_title('Trajectory Tangents (Green) + Normals (Red) from np.gradient')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.legend()
ax.grid(alpha=0.3)
ax.axis('equal')
plt.tight_layout()
plt.savefig(os.path.join(script_dir,"T1e5_normals.png"), dpi=300, bbox_inches='tight')
plt.close(fig)


profile_lengths = np.linspace(-normal_length, normal_length, 50)  # Sample 50 points
temperatures_all = []

fig2,ax2 = plt.subplots()
# Create regular grid for interpolation
X_grid, Y_grid = np.meshgrid(np.arange(temp_shape[1]), np.arange(temp_shape[0]))
ax2.set_title('Temperature Profiles Along Normals')
ax2.set_xlabel('Distance Along Normal [pixels]')
ax2.set_ylabel('Temperature [K]')

profiles = {}  # dict of column_name -> profile array
for i in range(0, len(x_pos)):
    cx, cy = x_pos[i], y_pos[i]
    
    # Points along normal line
    profile_x = cx + normal_x[i] * profile_lengths
    profile_y = cy + normal_y[i] * profile_lengths
    
    # Interpolate temperature at these points
    temp_profile = griddata((X_grid.ravel(), Y_grid.ravel()), 
                           temp_data.ravel(), 
                           (profile_x, profile_y), 
                           method='linear', fill_value=np.nan)
    
    # Skip if too many NaNs
    if np.sum(~np.isnan(temp_profile)) > 20:
        col_name = f"Profile_{i}"
        profiles[col_name] = temp_profile
        print("here")
        
        ax2.plot(profile_lengths, np.log10(temp_profile), alpha=0.7, linewidth=1.5)
        
        # Mark contour point (T=1e5K)
        ax2.plot(0, np.log10(1e5), 'ro', markersize=4)

normal_df = pd.DataFrame({"z_pc": profile_lengths})
for name, prof in profiles.items():
    normal_df[name] = prof
normal_df.to_csv(os.path.join(script_dir, f"normal_profiles.csv"),index=False)

# # Average profile
# if temperatures_all:
#     avg_profile = np.nanmean(temperatures_all, axis=0)
#     ax2.plot(profile_lengths, avg_profile, 'k-', linewidth=3, label='Average Profile')
#     ax2.axhline(1e5, color='red', linestyle='--', alpha=0.8, label='T=1e5K')

# ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(script_dir, "normal_profiles.png"), dpi=300, bbox_inches='tight')
# plt.show()
plt.close(fig2)
data.close()