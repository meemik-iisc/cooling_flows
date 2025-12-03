import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
from save_2D_arrays_3D import ISMCoolFn

global jump
jump = 1
global n1, n2, n3, n4
n1 = 36
n4 = 125
n2 = n1 + (n4 - n1 +1)//3
n3 = n2 + (n4 - n1 +1)//3

#dir = r"../../../Downloads/Astro_zenith_data/new/snapsfidcool2D/"
dir = r"../../../Downloads/Chandra_data/snapsfiducial64cool/"
#dir = r"../../../Downloads/Aryabhatta_data/snapshalfres4xbox/"
#dir = r"../../../Downloads/Trillium_data/snapsfid3xdens2xbox/"
#dir = r"../../../Downloads/Niagara_data/snaps5xlessdens/"
#dir = r"../../../Downloads/Niagara3Dfidnpz/"
filename = dir + f"KH_PDFs_time_averaged{n1}to{n4}with{jump}.npz"

ATOMIC_MASS = 1.660539e-24  # g
LENGTH = 3.08568e+18 
TIME   = 3.15576e+13 
MASS   = 4.91417e+31  
VELOCITY = 9.77793e+4  
DENSITY  = 1.67262e-24
ENERGY   = 4.69834e+41
POWER    = 1.48881e+28
PRESSURE = 1.59916e-14
TEMPERATURE = 71.2937
MU = 0.62
N_UNIT = DENSITY/(MU*ATOMIC_MASS) 
COOLING_UNIT = PRESSURE / (TIME * (N_UNIT**2))  # Cooling rate unit
CHI = 100.0
GAMMA = 5.0/3.0

v_h = 28.18181822
T_h = 1.0e6
T_c = 1.0e4
P_0 = 14.02645
rho_h = 0.001
T_0 = 1.0e5
delU = 31.
T_0_code = T_0/TEMPERATURE
rho_0 = P_0/T_0_code
B_h = (5.0/2.0)*P_0/rho_h
TMAX = 0.9e6
TMIN = 1.1e4
NX = 16
NY = 64
NZ = 16
max_level = 0
DY = 40./NY
T_inflection = (1.1e4 + 0.9e6)/2

def P_V_fn(Temp, Th, Tc, Tmin, Tmax):
    Trange = np.linspace(Tmin, Tmax, 1000)
    P_V_range = 1./( 1 - 4*(Trange - 0.5*(Th+Tc))**2/(Th-Tc)**2 + 1.e-10 ) #avoid division by zero
    Norm_inv = np.trapz(P_V_range, Trange)
    P_V = 1./( 1 - 4*(Temp - 0.5*(Th+Tc))**2/(Th-Tc)**2 + 1.e-10 )
    P_V = np.where(Temp < Tmin, 0.0, P_V)
    P_V = np.where(Temp > Tmax, 0.0, P_V)
    P_V /= Norm_inv
    return P_V

def Lambda_CF(T): #cooling function used in the simulations
    #original data from Shure et al. paper, covers 4.12 < logt < 8.16
    Log_Lam_cool_tab = np.array([ \
    -22.5977, -21.9689, -21.5972, -21.4615, -21.4789, -21.5497, -21.6211, -21.6595, \
    -21.6426, -21.5688, -21.4771, -21.3755, -21.2693, -21.1644, -21.0658, -20.9778, \
    -20.8986, -20.8281, -20.7700, -20.7223, -20.6888, -20.6739, -20.6815, -20.7051, \
    -20.7229, -20.7208, -20.7058, -20.6896, -20.6797, -20.6749, -20.6709, -20.6748, \
    -20.7089, -20.8031, -20.9647, -21.1482, -21.2932, -21.3767, -21.4129, -21.4291, \
    -21.4538, -21.5055, -21.5740, -21.6300, -21.6615, -21.6766, -21.6886, -21.7073, \
    -21.7304, -21.7491, -21.7607, -21.7701, -21.7877, -21.8243, -21.8875, -21.9738, \
    -22.0671, -22.1537, -22.2265, -22.2821, -22.3213, -22.3462, -22.3587, -22.3622, \
    -22.3590, -22.3512, -22.3420, -22.3342, -22.3312, -22.3346, -22.3445, -22.3595, \
    -22.3780, -22.4007, -22.4289, -22.4625, -22.4995, -22.5353, -22.5659, -22.5895, \
    -22.6059, -22.6161, -22.6208, -22.6213, -22.6184, -22.6126, -22.6045, -22.5945, \
    -22.5831, -22.5707, -22.5573, -22.5434, -22.5287, -22.5140, -22.4992, -22.4844, \
    -22.4695, -22.4543, -22.4392, -22.4237, -22.4087, -22.3928])
    LogT_tab = np.linspace(4.12, 8.16, np.size(Log_Lam_cool_tab))
    #for temperatures less than 10^4 K, use Koyama & Inutsuka (2002)
    LogT = np.log10(T)
    
    #interpolate the cooling function
    Log_Lam_cool = np.interp(LogT, LogT_tab, Log_Lam_cool_tab)
    #convert to linear scale
    Lam_cool = 10**Log_Lam_cool
    
    # for temperatures less than or equal to 10^4.2 K, use Koyama & Inutsuka (2002)
    Lam_cool = np.where(
    LogT <= 4.2,
    2.0e-19 * np.exp(-1.184e5 / (T + 1.0e3)) + 2.8e-28 * np.sqrt(T) * np.exp(-92.0 / T),
    Lam_cool)
    
    # for temperatures above 10^8.15 use CGOLS fit
    Lam_cool = np.where(
    LogT > 8.15,
    10.0**(0.45 * LogT - 26.065),
    Lam_cool)
    
    Lam_cool = np.where(
    T < 1.1e4,
    0.0,
    Lam_cool)

    Lam_cool = np.where(
    T > 0.9e6,
    0.0,
    Lam_cool)
    
    return Lam_cool  

def P_E_fn(Temp, pres, Th, Tc, Tmin, Tmax):
    P_E = P_V_fn(Temp, Th, Tc, Tmin, Tmax) * Lambda_CF(Temp) / Temp**2
    P_E = np.where(Temp < Tmin, 0.0, P_E)
    P_E = np.where(Temp > Tmax, 0.0, P_E)
    Norm_inv = np.trapz(P_E, Temp)
    P_E /= Norm_inv
    return P_E

def P_M_fn(Temp, Th, Tc, Tmin, Tmax):
    P_M = P_V_fn(Temp, Th, Tc, Tmin, Tmax) / Temp
    P_M = np.where(Temp < Tmin, 0.0, P_M)
    P_M = np.where(Temp > Tmax, 0.0, P_M)
    Norm_inv = np.trapz(P_M, Temp)
    P_M /= Norm_inv
    return P_M

def cumulative_pdf(pdf, x):
    # Normalize PDF if not already normalized
    pdf = pdf / np.trapz(pdf, x)
    # Compute cumulative sum using trapezoidal rule

    cdf = np.cumsum(pdf * np.diff(x, prepend=x[0]))
    # Normalize CDF to go from 0 to 1
    print(f'normalising factor for cum pdf = {cdf[-1]}.')
    cdf = cdf / cdf[-1]
    return cdf

def load_sim_PDFs(filename):
    with np.load(filename, 'r') as f:
        hist_vol_av = f['hist_vol_av']
        hist_mass_av = f['hist_mass_av']
        hist_emis_av = f['hist_emis_av']
        hist_vol_sig = f['hist_vol_sig']
        hist_mass_sig = f['hist_mass_sig']
        hist_emis_sig = f['hist_emis_sig']
        bin_centers = f['bin_centers']   
    # cutoff the PDF below 1.05e4 K and above 0.95e6 K and normalize it. bin centers are in log10(K)
    T = 10**bin_centers

    return T, hist_vol_av, hist_mass_av, hist_emis_av, hist_vol_sig, hist_mass_sig, hist_emis_sig

with np.load(dir + f'KH_1D_arrays_time_averaged{n1}to{n4}with{jump}.npz', 'r') as f:
    temp71to250 = f['temp_vol_av']
    pressure71to250 = f['p_vol_av']

T_tanh = lambda z, z0, Th, Tc: (0.5 * (Th - Tc) * np.tanh(z / z0) + 0.5 * (Th + Tc))
T_prime_tanh = lambda z, z0, Th, Tc: 0.5 * (Th - Tc) * (1 - np.tanh(z / z0)**2) / z0
T_prime_sim = np.gradient(temp71to250, DY) 

global time_0
time_0 = 3.*P_0/(2.*rho_0*rho_0*ISMCoolFn(T_0)/COOLING_UNIT)
print(time_0)

Th = T_h
Tc = T_c
z0 = 2.3
x0 = 8
epsilon = 10000
z = np.linspace(-20, 20, NY)
# fitting a tanh functino to temp71to250
from scipy.optimize import curve_fit
def T_tanh_model(z, x0, z0, Th, Tc):
    return ((0.5 * (Th + Tc) + 0.5 * (Th - Tc) * np.tanh((z - x0) / z0)))
popt, _ = curve_fit(T_tanh_model, z, (temp71to250), p0=[x0, z0, Th, Tc])
print(f'Fitted parameters: x0={popt[0]}, z0={popt[1]}, Th={popt[2]}, Tc={popt[3]}')
Tmax = 0.9e6
Tmin = 1.1e4
z0 = popt[1]
x0 = popt[0]
z -= x0

T = T_tanh(z, z0, Th, Tc)

kB = 1.38064852e-16  # Boltzmann constant in erg/K
p = 0.001*kB*Th
mu = 0.62
mp = 1.6605e-24  # mass of proton in g

T = T_tanh(z, z0, Th, Tc)
P_V = P_V_fn(T, Th, Tc, Tmin, Tmax)  # Volume PDF

temp, hist_vol_av, hist_mass_av, hist_emis_av, hist_vol_sig, hist_mass_sig, hist_emis_sig = load_sim_PDFs(filename)

print(f'{np.trapz(P_V, T)} - from theory -P_v')
print(f'{np.trapz(hist_vol_av, np.log10(temp))} - from simulation -P_v')

P_E_const = T**0/(Tmax - Tmin)
P_E_const = np.where(T < Tmin, 0.0, P_E_const)
P_E_const = np.where(T > Tmax, 0.0, P_E_const)
P_E_const_log10T = P_E_const * T / np.log10(np.exp(1.))
P_V_log10T = P_V * T / np.log10(np.exp(1.))
P_E_log10T = P_E_fn(T, pressure71to250, Th, Tc, Tmin, Tmax) * T / np.log10(np.exp(1.))
P_M_log10T = P_M_fn(T, Th, Tc, Tmin, Tmax) * T / np.log10(np.exp(1.))

print(f'{np.trapz(P_E_log10T, np.log10(T))} - from theory -P_e')
print(f'{np.trapz(hist_emis_av, np.log10(temp))} - from simulation -P_e')
print(f'{np.trapz(P_M_log10T, np.log10(T))} - from theory -P_m')
print(f'{np.trapz(hist_mass_av, np.log10(temp))} - from simulation -P_m')

# MAKING FIGURE 3
if True:
    with np.load(dir + f'KH_1D_arrays_time_averaged{n1}to{n4}with{jump}.npz', 'r') as f:
        temp71to250__ = f['temp_vol_av']
        prs_vol_av = f['p_vol_av']
        emis_vol_av71to250__ = f['emis_vol_av']
        z__ = np.linspace(-20, 20, NY)
        P_E__T71to250 = f['P_E__T_av']
        P_E__T71to250_sig = f['P_E__T_sig']
        P_E__T71to250_log10T = f['P_E__T_log10T_av']
        P_E__T71to250_log10T_sig = f['P_E__T_log10T_sig']
        temp_range = f['temp_range']

    with np.load(dir + f'KH_fluxes_time_averaged{n1}to{n4}with{jump}.npz', 'r') as f:
        edot_cool_cum_dx2_av__ = f['edot_cool_cum_dx2']
    n2lambdaT__ = np.gradient(edot_cool_cum_dx2_av__, z__)
    T_prime__ = np.gradient(temp71to250__, z__)
    Sigma_cool__ = np.mean(edot_cool_cum_dx2_av__[-10:-1])
    P_E__T = n2lambdaT__/(T_prime__ * Sigma_cool__)
    P_E__T = np.where(temp71to250__ < Tmin, 0.0, P_E__T)
    P_E__T = np.where(temp71to250__ > Tmax, 0.0, P_E__T)
    P_E__T /= np.trapz(P_E__T, temp71to250__)
    P_E__T_log10T = P_E__T * temp71to250__ / np.log10(np.exp(1.))
    print(f'{np.trapz(P_E__T_log10T, np.log10(temp71to250__))} - from theory -P_e new log')

    plt.figure(figsize=(8, 6))
    z__ = z__[3*NY//8:7*NY//8]
    z__ -= x0
    z__ = z__ / (delU * time_0)

    plt.figure(figsize=(7, 5))
    plt.plot(np.log10(T), P_V_log10T,color = 'orange', linestyle='--')
    plt.plot(np.log10(T), P_M_log10T,color = 'blue', linestyle='--')
    plt.plot(np.log10(T), P_E_log10T, color = 'green', linestyle='--')
    plt.plot(np.log10(T), P_E_const_log10T, color = 'red', label=r'$\mathcal{P}_E = const$', linestyle='--')
    plt.plot(np.log10(temp_range), P_E__T71to250_log10T, color = 'purple', linestyle='-', label=r'$\overline{\mathcal{P}}_E$')
    plt.fill_between(np.log10(temp_range), (P_E__T71to250_log10T - P_E__T71to250_log10T_sig), (P_E__T71to250_log10T + P_E__T71to250_log10T_sig), color='purple', alpha=0.3)
    plt.plot(np.log10(temp), hist_vol_av,color = 'orange', linestyle='-', label=r'$\mathcal{P}_V$')
    plt.fill_between(np.log10(temp), hist_vol_av - hist_vol_sig, hist_vol_av + hist_vol_sig, color='orange', alpha=0.3)
    plt.plot(np.log10(temp), hist_mass_av, color = 'blue', linestyle='-', label=r'$\mathcal{P}_M$')
    plt.fill_between(np.log10(temp), hist_mass_av - hist_mass_sig, hist_mass_av + hist_mass_sig, color='blue', alpha=0.3)
    plt.plot(np.log10(temp), hist_emis_av, color = 'green', linestyle='-', label=r'$\mathcal{P}_E$')
    plt.fill_between(np.log10(temp), hist_emis_av - hist_emis_sig, hist_emis_av + hist_emis_sig, color='green', alpha=0.3)
    plt.text(0.97, 0.97, r"$\mathcal{P}\ (log_{10} T)$", fontsize=16, ha='right', va='top', transform=plt.gca().transAxes,bbox=dict(facecolor='white', edgecolor='white', boxstyle='round,pad=0.3', linewidth=1.5))
    plt.xlabel(r'$log_{10} T$', fontsize=16)
    plt.xlim([4, 6])
    plt.ylim([1e-3, 10])
    plt.yscale('log')
    plt.xscale('linear')

    plt.legend(fontsize=16, loc='lower left', ncol =2, frameon=True, handlelength=1, borderpad=0.2, labelspacing=0.2, handletextpad=0.2, bbox_to_anchor=(0.03, 0.01))
    plt.grid(which='both', axis='both', linestyle='--', linewidth=0.5, color='gray')
    plt.tick_params(labelsize=11)
    #plt.show()
    plt.savefig(dir + 'figure3.png', dpi=150, bbox_inches='tight')
    plt.clf()
    plt.close('all')
    print('PRINTED FIGURE 3')

# MAKING FITS FIGURE
if True:
    with np.load(dir + f'KH_1D_arrays_time_averaged{n1}to{n4}with{jump}.npz', 'r') as f:
        temp71to250__ = f['temp_vol_av'][3*NY//8:7*NY//8]
        prs_vol_av__ = f['p_vol_av'][3*NY//8:7*NY//8]
        emis_vol_av71to250__ = f['emis_vol_av'][3*NY//8:7*NY//8]
        z__ = np.linspace(-20, 20, NY)
        z__ = z__[3*NY//8:7*NY//8]
        vx1_vol_av__ = f['vx1_vol_av'][3*NY//8:7*NY//8]
        Be_vol_av__ = f['Be_vol_av'][3*NY//8:7*NY//8]
        den_vol_av__ = f['rho_av'][3*NY//8:7*NY//8]
    Tfits = T[3*NY//8:7*NY//8]
    with np.load(dir + f'KH_fluxes_time_averaged{n1}to{n4}with{jump}.npz', 'r') as f:
        edot_cool_cum_dx2_av__ = f['edot_cool_cum_dx2'][3*NY//8:7*NY//8]
        del_Be_del_rhov2_av__ = f['del_Be_del_rhov2_av'][3*NY//8:7*NY//8]
    n2lambdaT__ = np.gradient(edot_cool_cum_dx2_av__, z__)
    T_prime__ = np.gradient(Tfits, z__)
    Sigma_cool__ = np.mean(edot_cool_cum_dx2_av__[-10:-1])
    P_E__T_ = n2lambdaT__/(T_prime__ * Sigma_cool__ + 1e-10)
    #P_E__T_ = np.where(temp{n1}to{n4}__ < Tmin, 0.0, P_E__T_)
    #P_E__T_ = np.where(temp{n1}to{n4}__ > Tmax, 0.0, P_E__T_)
    P_E__T_ /= np.trapz(P_E__T_, Tfits)
    P_E__T_log10T_ = P_E__T_ * Tfits / np.log10(np.exp(1.))
    print(f'{np.trapz(P_E__T_log10T_, np.log10(Tfits))} - from theory -P_e new log')

    with np.load(dir + f'KH_1D_arrays_time_averaged{n1}to{n4}with{jump}.npz', 'r') as f:
        P_E__T_temprange = f['P_E__T_av']
        temp_range = f['temp_range']
        P_E__T_log10T = P_E__T_temprange * temp_range / np.log10(np.exp(1.))

    Bh = vx1_vol_av__[-1]**2/2 + 2.5 * kB*Th/(mu*mp*VELOCITY**2)
    Bc = vx1_vol_av__[0]**2/2 + 2.5 * kB*Tc/(mu*mp*VELOCITY**2)
    Bh_obs = Be_vol_av__[-1]
    Bc_obs = Be_vol_av__[0]

    P_E__T = np.interp(temp71to250__, temp_range, P_E__T_temprange)

    def P_E_function(Ts, Th, Tc, Tmin, Tmax):
        return np.interp(Ts, Tfits,P_E__T_)
    #calculate T_star
    fstar = lambda Ts: P_E_function(Ts, Th, Tc, Tmin, Tmax)*(Th-Tc) - 1.
    print("f(Tmin) =", fstar(Tmin))
    print("f(Tmax) =", fstar(Tmax))
    Ts = root_scalar(fstar, x0=4.85e5, method='secant').root

    print(f'T_star = {Ts:.2e} K')
    heat_flux_unit = MASS/(TIME**3)

    Sigma_m = Sigma_cool__*mu*mp/(2.5*kB*(Th-Tc))* MASS / TIME**3
    print(f'Bh = {Bh:.2e} using vx1_vol_av__[-1]')
    print(f'Bc = {Bc:.2e} using vx1_vol_av__[0]')
    print(f'Bh = {Be_vol_av__[-1]:.2e} using Be_vol_av__[-1]')
    print(f'Bc = {Be_vol_av__[0]:.2e} using Be_vol_av__[0]')
    print(f'Sigma_m = {Sigma_cool__/(Bh-Bc):.2e} from Bh-Bc')
    print(f'Sigma_m = {Sigma_cool__/(Be_vol_av__[-1]-Be_vol_av__[0]):.2e} from Be_vol_av__[-1] and Be_vol_av__[0]')
    print(f'Sigma_m = {Sigma_m/ (MASS / (TIME * LENGTH**2)):.2e} code units')
    print(f'hot inflow velocity = {Sigma_m / (MASS / (TIME * LENGTH**2) * rho_h):.2e} code units')
    print(f'{np.trapz(P_E__T, temp71to250__)} - from theory -P_e new')

    Q_t__ = -2.5*kB*Sigma_m*(Th-Tc)*(cumulative_pdf(P_E__T_, Tfits) - (Tfits-Tc)/(Th-Tc))/(mu*mp)
    Be_av_rhov2_av__ = -2.5 * kB * Sigma_m * (Tfits - Tc) / (mu * mp)

    z__ -= x0
    z__ = z__ / (delU * time_0)


    with np.load(dir + f'KH_fluxes_time_averaged{n1}to{n4}with{jump}.npz', 'r') as f:
        Be_av_rhov2_av = f['Be_av_rhov2_av'][3*NY//8:7*NY//8]
        Be_av_rhov2_sig = f['Be_av_rhov2_sig'][3*NY//8:7*NY//8]
        del_Be_del_rhov2_av = f['del_Be_del_rhov2_av'][3*NY//8:7*NY//8]
        del_Be_del_rhov2_sig = f['del_Be_del_rhov2_sig'][3*NY//8:7*NY//8]
        edot_cool_cum_dx2_av = f['edot_cool_cum_dx2'][3*NY//8:7*NY//8]
        edot_cool_cum_dx2_sig = f['edot_cool_cum_dx2_sig'][3*NY//8:7*NY//8]
        net_heating_av = f['net_heating'][3*NY//8:7*NY//8]
        net_heating_sig = f['net_heating_sig'][3*NY//8:7*NY//8]

    plt.figure(figsize=(8, 6))
    plt.plot(z__, del_Be_del_rhov2_av/(P_0*delU), color='red', linewidth=1.75)
    plt.fill_between(z__, (edot_cool_cum_dx2_av - edot_cool_cum_dx2_sig)/(P_0*delU), (edot_cool_cum_dx2_av + edot_cool_cum_dx2_sig)/(P_0*delU), color='blue', alpha=0.2)
    plt.plot(z__, net_heating_av/(P_0*delU), color='orange', linestyle='-', linewidth=1.75)
    plt.fill_between(z__, (Be_av_rhov2_av - Be_av_rhov2_sig)/(P_0*delU), (Be_av_rhov2_av + Be_av_rhov2_sig)/(P_0*delU), color='green', alpha=0.2)
    plt.plot(z__, Be_av_rhov2_av/(P_0*delU), color='green', linestyle='-', linewidth=1.75)
    plt.fill_between(z__, (del_Be_del_rhov2_av - del_Be_del_rhov2_sig)/(P_0*delU), (del_Be_del_rhov2_av + del_Be_del_rhov2_sig)/(P_0*delU), color='red', alpha=0.2)
    plt.plot(z__, edot_cool_cum_dx2_av/(P_0*delU), color='blue', linestyle='-', linewidth=1.75)
    plt.fill_between(z__, (net_heating_av - net_heating_sig)/(P_0*delU), (net_heating_av + net_heating_sig)/(P_0*delU), color='orange', alpha=0.2)
    plt.plot(z__, edot_cool_cum_dx2_av__/(P_0*delU), label=r'$\int n^2 \Lambda $', color='black', linewidth=1.75, linestyle='--')
    plt.plot(z__, Q_t__/(P_0*delU*PRESSURE*VELOCITY), label=r'$Q_t$-theory', color='black', linewidth=1.75, linestyle='-')
    plt.plot(z__, Be_av_rhov2_av__/(P_0*delU*PRESSURE*VELOCITY), label=r'$\langle B \rangle \langle \rho u_z \rangle$-theory', color='black', linewidth=1.75, linestyle=':')
    plt.plot(z__, Q_t__/(P_0*delU*PRESSURE*VELOCITY) + edot_cool_cum_dx2_av__/(P_0*delU) + Be_av_rhov2_av__/(P_0*delU*PRESSURE*VELOCITY), label='net-theory', color='black', linewidth=1.75, linestyle='-.')

    plt.plot(z__, np.gradient(Be_av_rhov2_av,z__*(delU*time_0))/(P_0/time_0), label=r'$\frac{d<B> <\rho u_z>}{dz}$-sim', color='cyan', linewidth=1.75, linestyle='-')
    plt.plot(z__, np.gradient(del_Be_del_rhov2_av,z__*(delU*time_0))/(P_0/time_0), label=r'$\frac{d<\delta \mathcal{B}> <\delta \rho u_z>}{dz}$-sim', color='magenta', linewidth=1.75, linestyle='-')
    plt.plot(z__, n2lambdaT__/(P_0/time_0), label=r'$\langle n^2 \Lambda (T) \rangle$', color='purple', linewidth=1.75, linestyle='--')

    plt.ylabel('normalized to '+r"$p_0/t_0 or p_0 \Delta u$")
    plt.title(f'Sigma_m = {Sigma_m*TIME*LENGTH**2/(MASS):.2e},\n T_star = {Ts:.2e}')
    plt.legend()
    diffcoeff = -2.3

    plt.grid(which='both', linestyle='--', linewidth=0.5, color='gray')
    plt.xlabel(r'$T/T_h$')
    plt.savefig(dir+'fits.png', dpi=600, bbox_inches='tight')
    plt.clf()
    plt.close()

# MAKING FIGURE 4
if True:
    print('MAKING FIGURE 4')
    with np.load(dir + f'KH_1D_arrays_time_averaged{n1}to{n4}with{jump}.npz', 'r') as f:
        temp71to2504 = f['temp_vol_av'][3*NY//8:7*NY//8]
        z4 = np.linspace(-20, 20, NY)[3*NY//8:7*NY//8]
        P_E__T71to2504 = f['P_E__T_av']
        temp_range4 = f['temp_range']
        emis_vol_av71to2504 = f['emis_vol_av'][3*NY//8:7*NY//8]

    with np.load(dir + f'KH_fluxes_time_averaged{n1}to{n4}with{jump}.npz', 'r') as f:
        Be_av_rhov2_av4 = f['Be_av_rhov2_av'][3*NY//8:7*NY//8]
        del_Be_del_rhov2_av4 = f['del_Be_del_rhov2_av'][3*NY//8:7*NY//8]
        edot_cool_cum_dx2_av4 = f['edot_cool_cum_dx2'][3*NY//8:7*NY//8]
        net_heating_av4 = f['net_heating'][3*NY//8:7*NY//8]

    from scipy.signal import savgol_filter

    z4 -= x0
    T4 = T_tanh(z4, z0, Th, Tc)
    T_prime4 = T_prime_tanh(z4, z0, Th, Tc)


    n2lambdaT4 = np.gradient(edot_cool_cum_dx2_av4, z4)
    Sigma_cool4 = np.mean(edot_cool_cum_dx2_av4[-10:-1])

    P_E__T4 = n2lambdaT4/(T_prime4*Sigma_cool4 + 1e-10)
    #P_E__T4 = np.where(T4 <= 1.0e4, 0.0, P_E__T4)
    #P_E__T4 = np.where(T4 >= 1.0e6, 0.0, P_E__T4)
    P_E__T4 /= np.trapz(P_E__T4, T4)
    P_E__T_log10T4 = P_E__T4 * T4 / np.log10(np.exp(1.))
    print(f'{np.trapz(P_E__T_log10T4, np.log10(T4))} - from theory -P_e new log')

    with np.load(dir + f'KH_1D_arrays_time_averaged{n1}to{n4}with{jump}.npz', 'r') as f:
        P_E__T_temprange = f['P_E__T_av']
        temp_range = f['temp_range']

    Sigma_m4 = Sigma_cool4*mu*mp/(2.5*kB*(Th-Tc)) * MASS / TIME**3
    Q_t_sim4 = -2.5*kB*Sigma_m4*(Th-Tc)*(cumulative_pdf(P_E__T4, T4) - (T4-Tc)/(Th-Tc))/(mu*mp)
    Be_av_rhov2_av_sim4 = -2.5 * kB * Sigma_m4 * (T4 - Tc) / (mu * mp)

    plt.figure(figsize=(8, 6))

    z4 = z4 / (delU * time_0)

    plt.plot(T4, P_E__T_log10T4, label=r'$\overline{\mathcal{P}}_E$', color='purple', linewidth=1.75, linestyle='-')
    plt.plot(temp_range4, P_E__T71to2504 * temp_range4 / np.log10(np.exp(1.)), color='purple', linestyle='--', label=r'$\overline{\mathcal{P}}_E$-sim')
    plt.plot((T), P_E_const_log10T, color='purple', linestyle=':', label=r'$\overline{\mathcal{P}}_E$-const')
    plt.yscale('log')
    plt.xscale('log')
    plt.ylim(1e-3, 1e1)
    plt.legend()
    plt.grid(which='both', axis='both', linestyle='--', linewidth=0.5, color='gray')
    plt.tick_params(labelsize=11)
    plt.savefig(dir + 'P_e_bar_comparison.png', dpi=150, bbox_inches='tight')

    # smotthening arrays
    Be_av_rhov2_av4_deriv = savgol_filter(Be_av_rhov2_av4, window_length=40, polyorder=4, deriv=1, delta=z4[1]*(delU*time_0) - z4[0]*(delU*time_0))
    del_Be_del_rhov2_av4_deriv = savgol_filter(del_Be_del_rhov2_av4, window_length=40, polyorder=4, deriv=1, delta=z4[1]*(delU*time_0) - z4[0]*(delU*time_0))

    fig,axe = plt.subplots(figsize=(7, 4))
    # plotting things direct from simulation
    line1, = axe.plot(z4,n2lambdaT4/(P_0/time_0), label=r'$\frac{\langle n^2 \Lambda (T) \rangle}{(p_0/t_0)}$', color='blue', linewidth=2.25, linestyle='-')
    line2, = axe.plot(z4,Be_av_rhov2_av4_deriv/(P_0/time_0), label=r'$\frac{1}{(p_0/t_0)} \frac{d \langle \mathcal{B} \rangle \langle \rho u_z \rangle}{dz}$', color='orange', linewidth=2.25, linestyle='-')
    line3, = axe.plot(z4,del_Be_del_rhov2_av4_deriv/(P_0/time_0), label=r'$\frac{1}{(p_0/t_0)} \frac{d \langle \delta \mathcal{B} \delta \rho u_z \rangle}{dz}$', color='green', linewidth=2.25, linestyle='-')
    axe.plot(z4,np.gradient(net_heating_av4,z4*(delU*time_0))/(P_0/time_0), color='grey', linewidth=2.25, linestyle='-')

    # calculating same quantities using tanh profile for temperature

    def P_E_function(Ts, Th, Tc, Tmin, Tmax):
        return np.interp(Ts, T4, P_E__T4)
    
    #calculate T_star
    fstar = lambda Ts: (P_E_function(Ts, Th, Tc, Tmin, Tmax)*(Th-Tc) - 1.)
    print("f(Tmin) =", fstar(Tmin))
    print("f(Tmax) =", fstar(Tmax))
    Ts = root_scalar(fstar, x0=4.85e5, method='secant').root
    print(f'T_star = {Ts:.2e} K')

    Fact = -np.max(np.gradient(Q_t_sim4, z4*(LENGTH*delU*time_0))/(P_0*PRESSURE/(time_0*TIME)))/np.max(delU*delU*time_0*time_0*np.gradient(T_prime4/Th, z4*(delU*time_0)))
    # plotting predicted terms
    axe.plot(z4, n2lambdaT4/(P_0/time_0), color='blue', linewidth=1.75, linestyle='--')
    axe.plot(z4, np.gradient(Q_t_sim4, z4*(LENGTH*delU*time_0))/(P_0*PRESSURE/(time_0*TIME)), color='green', linewidth=2.25, linestyle='--')
    axe.plot(z4, np.gradient(Be_av_rhov2_av_sim4, z4*(LENGTH*delU*time_0))/(P_0*PRESSURE/(time_0*TIME)), color='orange', linewidth=2.25, linestyle='--')
    line4, = axe.plot(z4, Fact*delU*delU*time_0*time_0*np.gradient(T_prime4/Th, z4*(delU*time_0)), color='red', linewidth=2.25, linestyle=':', label=r'$-\kappa_t\ \frac{d^2 \langle T \rangle}{dz^2}$')
    axe.plot(z4, np.gradient(Be_av_rhov2_av_sim4, z4*(LENGTH*delU*time_0))/(P_0*PRESSURE/(time_0*TIME))+
             np.gradient(Q_t_sim4, z4*(LENGTH*delU*time_0))/(P_0*PRESSURE/(time_0*TIME))+
             n2lambdaT4/(P_0/time_0), color='black', linewidth=2.25, linestyle='--')
    
    legend1 = axe.legend(handles=[line1, line2], loc='lower center', bbox_to_anchor=(0.78, 0.50),
                    ncol=1, frameon=False, fontsize=20, handlelength=1)
    # Add the first legend manually so the second doesn't overwrite it
    axe.add_artist(legend1)
    # Second legend: below the plot
    axe.legend(handles=[line3, line4], loc='upper center', bbox_to_anchor=(0.78, 0.5),
               ncol=1, frameon=False, fontsize=20, handlelength=1)
    axe.tick_params(labelsize=12)

    from scipy.interpolate import interp1d    
    f = interp1d(z4, T4/1e5, fill_value="extrapolate")   # z -> q
    f_inv = interp1d(T4/1e5, z4, fill_value="extrapolate")
    secax = axe.secondary_xaxis('top', functions=(f, f_inv))
    secax.set_xlabel(r"$\langle T  \rangle (10^5 K),(\kappa_t=$"+ f'{-Fact:.2f})', fontsize=14)
    from matplotlib.ticker import FormatStrFormatter
    secax.set_xticks(f(np.linspace(z4[int(len(z4)*1.1/3)], z4[-1], 10)))
    secax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    secax.tick_params(labelsize=12)


    plt.xlabel(r'$z/\Delta u t_0$', fontsize=14)
    plt.xlim(z4[int(len(z4)*1.1/3)],z4[-1])

    plt.grid(which='both', axis='both', linestyle='--', linewidth=0.5, color='gray')
    plt.tick_params(labelsize=12)
    #plt.show()
    plt.savefig(dir + 'figure4.png', dpi=150, bbox_inches='tight')
    plt.clf()
    plt.close('all')

    print(time_0)

