import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(script_dir, "normal_profiles.csv"))

z = df["z_pc"].values           # distance along normal
profile_cols = [c for c in df.columns if c != "z_pc"]

Tc_target = 1e4
Th_target = 1e6
Tc_tol = 1e3          # e.g. ±1000 K
Th_tol = 1e5
dTdx_tol = 1e2       # e.g. |dT/dx| < 100 K / pixel
good_cols = []

for col in profile_cols:
    T = df[col].values

    # Condition 1: T(0) ~ 1e4
    T0 = T[0]
    T_last = T[-1]
    print(T0, T_last)
    print(col, "T0=", T0, "dT0=", T0 - Tc_target,
        "Tlast=", T_last, "dTlast=", T_last - Th_target)
    cond_T0 = np.isfinite(T0) and abs(T0 - Tc_target) <= Tc_tol
    cond_Tlast = np.isfinite(T_last) and abs(T_last - Th_target) <= Th_tol
    
    # # If z is uniform, np.gradient(T) is enough; here use z explicitly for safety
    # if np.all(np.isfinite(T)):
    #     dTdz = np.gradient(T, z)
    #     dTdz0 = dTdz[0]
    #     dTdz_last = dTdz[-1]
    #     cond_grad0 = abs(dTdz0) <= dTdx_tol
    #     cond_grad_last = abs(dTdz_last) <= dTdx_tol
    # else:
    #     cond_grad0 = False
    #     cond_grad_last = False

    # All four conditions
    if cond_T0 and cond_Tlast:
        good_cols.append(col)

print("Profiles passing filters:", len(good_cols))
filtered_df = df[["z_pc"] + good_cols]
print(filtered_df.head())
filtered_df.to_csv(os.path.join(script_dir, "normal_profiles_filtered.csv"), index=False)

# Plot filtered profiles (linear and/or log)
fig, ax = plt.subplots(figsize=(8, 6))
for col in good_cols:
    T = df[col].values
    ax.plot(z, np.log10(T), alpha=0.7, linewidth=1.5, label=col)

# ax.axhline(np.log10(T_target), color="red", linestyle="--", alpha=0.7, label="T=1e4 K")
ax.set_xlabel("z_pix")
ax.set_ylabel("log10(T [K])")
ax.set_title("Filtered Temperature Profiles Along Normals")
ax.grid(alpha=0.3)
ax.legend(fontsize=8, loc="best")

plt.tight_layout()
plt.savefig(os.path.join(script_dir, "normal_profiles_filtered.png"), dpi=300, bbox_inches="tight")
plt.close(fig)