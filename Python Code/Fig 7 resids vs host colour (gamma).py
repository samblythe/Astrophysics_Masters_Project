#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import emcee

#import functions
from utilities_ import (load_data, Mb_eq, distmod_Planck18, 
Mb_err_eq, M_cor_eq_3params, intr_dis_cov,log_posterior_step_two,step_model_two)

#load the data into a pandas dataframe
DataFrame = load_data(r"C:\Users\sambl\Documents\MPhys Project\ztfdr2_salt2_params.csv")
df_classifications = load_data(r"C:\Users\sambl\Documents\MPhys Project\ztfdr2_classifications.csv")
df_host_prop = load_data(r"C:\Users\sambl\Documents\MPhys Project\ztfdr2_local2kpc_prop.csv")
spec_df = load_data(r"C:\Users\sambl\Documents\MPhys Project\ZTF_DR2_Spec_Div_Burgaz_2024.csv")

#merge dataframes
merged_df = pd.merge(DataFrame, df_classifications, on='ztfname')
merged_df_2 = pd.merge(merged_df,df_host_prop, on='ztfname')

#remove all data that doesnt meet criteria for the basic data cuts
cut = merged_df_2[merged_df_2["fitquality_flag"]==1][merged_df_2["lccoverage_flag"]==1][merged_df_2["sn_type"] != 'snia-pec' ]
cut = cut.dropna()

#make cut in redshift to create Volume Limited Sample
cut = cut[cut["z"]<= 0.06]

#---define variables---

#redshift values 
z = cut["z"]
z_err = cut["z_err"]

#x1 values
x1 =cut["x1"]
x1_err=cut["x1_err"]

#c values
c = cut["c"]
c_err=cut["c_err"]

#x0 values
x0 = cut["x0"]
x0_err= cut["x0_err"]

#Getting Mb values from the function created in Utilities,py
Mb = Mb_eq(cut["x0"])
#error in Mb
Mb_err = Mb_err_eq(x0, x0_err)

#distance modulus (mu) using Planck 18 cosmology
mu_Planck18 = distmod_Planck18(cut)

host_c = cut["restframe_gz"] 
host_c_err = cut["restframe_gz_err"]
mass = cut["mass"]
mass_err = cut["mass_err"]

#alpha and beta values from independent fits (figs 4 and 5)

#Alpha values and uncertainties
a_value = 0.119
a_err = 0.007

#Beta values and uncertainties
B_value = 2.942
B_err =0.043

#gamma values and uncertainties:
g_value =0
g_err = 0

#find corrected M values using function in Utilities
M_cor = M_cor_eq_3params(Mb, mu_Planck18, a_value,x1,B_value,c,g_value,host_c)

intr_dis_values = np.arange(0.12,0.21,0.0001 )

#set to zero for simplicity
sigma_z = 0.0

#define covariances
cov_mb_x1 = cut["cov_x0_x1"]
cov_mb_c = cut["cov_x0_c"]
cov_x1_c = cut["cov_x1_c"]

#intrinsic dispersion estimate using function from Utilities
best_intr_dis, chisqr_red_min, best_inter = intr_dis_cov(
    intr_dis_values, sigma_z,M_cor, z,a_value, B_value, Mb_err, x1_err, c_err,cov_mb_x1,cov_mb_c,cov_x1_c)

#Error on corrected M using intrinsic dispersion estimate
M_cor_err = np.sqrt((best_intr_dis)**2 + (sigma_z)**2 + (a_value*x1_err)**2 + (B_value*c_err)**2 + 2*a_value*cov_mb_x1 - 2*B_value*cov_mb_c - 2*a_value*B_value*cov_x1_c)

#----Plotting----

#define axis (Hubble residuals)
y2 = M_cor + 19.05798

#apply Seaborn style
sns.set_context("notebook")
sns.set_style("whitegrid", {
    "axes.edgecolor": "0.3",
    "grid.color": "0.85",
    "axes.spines.right": True,
    "axes.spines.top": True
})

#set Seaborn color palette
palette = sns.color_palette("dark")
color_uncorrected = palette[0]  
color_corrected = palette[2]    

#define figure
fig, (ax1) = plt.subplots(1, 1, figsize=(7, 5), sharex=False)

# ---plot Hubble residuals against host colour ---
ax1.errorbar(host_c, y2, yerr=M_cor_err,xerr=host_c_err ,fmt='o', markersize=4,
             color='grey', alpha=0.6, ecolor='lightgrey', capsize=0.9,label='SNe Ia events corrected \n for $x_1$ and $c$ ')

#---binned averages---

host_c_sorted = np.sort(host_c)

#bin parameters
num_bins = 70
bin_edges = np.linspace(host_c_sorted.min(), host_c_sorted.max(), num_bins + 1) 
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])  # midpoints for plotting

#compute mean y and standard error in each bin
binned_means = []
binned_errors = []
for i in range(num_bins):
    in_bin = (host_c >= bin_edges[i]) & (host_c < bin_edges[i+1])
    if np.sum(in_bin) == 0:
        #no data in this bin
        binned_means.append(np.nan)
        binned_errors.append(np.nan)
        continue
    
    #weighted mean
    y_in_bin = y2[in_bin]
    w = 1.0 / (M_cor_err[in_bin]**2)
    mean_val = np.sum(w * y_in_bin) / np.sum(w)
    
    #standard error of the mean:
    sem = np.sqrt(np.sum(M_cor_err[in_bin]**2)) / np.sum(in_bin)
    
    #store results
    binned_means.append(mean_val)
    binned_errors.append(sem)

binned_means = np.array(binned_means)
binned_errors = np.array(binned_errors)

#overlay the binned averages with bigger markers
ax1.errorbar(bin_centers, binned_means, yerr=binned_errors, fmt='o',
             color=color_corrected, ecolor=color_corrected, markersize=10, capsize=1,
             label='Binned Averages',zorder=11,alpha=0.9)


host_c_range = np.arange(-1, 4, 0.01)

#----plot formatting ----
ax1.axhline(0, color='black', linestyle=':', linewidth=1, zorder =12)
plt.subplots_adjust(left=0.09, right=0.97, top=0.95, bottom=0.12)
ax1.set_ylabel(fr'$\mu - \mu_{{cosmo}}-\beta c +\alpha x_{1}$', fontsize = 18)
ax1.set_xlabel(r'$(g-z)_{\mathrm{local}}$', fontsize = 18)

#--- fittig a step function using MCMC -----

#define simpler variables for MCMC
x = host_c
y = y2             
yerr = M_cor_err  

ndim = 2     #Number of parameters: break_point, step
nwalkers = 32   #Number of MCMC walkers

#set initial starting points for the walkers to begin on
initial_break = np.median(x)
initial_gamma = 0.0
initial = np.array([initial_break, initial_gamma])

#allow walkers to search the parameter space by deviating from the starting point by a small random amount
pos = initial + 1e-4 * np.random.randn(nwalkers, ndim)

#MCMC sampler
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior_step_two, args=(x, y, yerr))
nsteps = 5000

#run sampler
sampler.run_mcmc(pos, nsteps, progress=True)

#discard burn-in samples and flatten the chain
burnin = 1000


#extract the MAP (best fit/most likely) estimates from the sampler chain
flat_samples = sampler.get_chain(discard=burnin, flat=True)
flat_log_probs = sampler.get_log_prob(discard=burnin, flat=True)

max_index = np.argmax(flat_log_probs)
map_params = flat_samples[max_index]  # [break_point, step_low, step_high]

break_map, gamma_map = map_params

#use standard deviation as an estimate for uncertainty
break_std = np.std(flat_samples[:, 0])
gamma_std = np.std(flat_samples[:, 1])

#print results of MCMC

print("MCMC MAP results (reparameterized two-parameter step model):")
print(f"Break point: {break_map:.3f} ± {break_std:.3f}")
print(f"Gamma: {gamma_map:.3f} ± {gamma_std:.3f}")

#use MCMC results to deinfe a best fitting step function model using the step function defined in Utilities
x_range = np.linspace(np.min(x), np.max(x), 1000)
best_model_map = step_model_two(x_range, break_map, gamma_map)

#plot step function on axis
plt.plot(x_range, best_model_map, color=color_corrected, linewidth=2,
         label=fr'Fitted Step Function:' f'\n'fr'break={break_map:.3f}, $\gamma$={gamma_map:.3f}±{gamma_std:.3f}', zorder = 30)

#plot formatting
ax1.legend(title='', loc='upper right', fontsize=12, title_fontsize=14, labelspacing=0.4)
ax1.set_ylim(-0.4, 0.4)
ax1.set_xlim(0, 2)
for ax in [ax1]:
    ax.tick_params(axis='both', which='major', labelsize=16.5)
    ax.grid(True, linestyle='--', alpha=0.1)
    ax.tick_params(axis='both', which='both', direction='out', length=5, width=1, colors='black')
    ax.tick_params(bottom=True, top=False, left=True, right=False)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.subplots_adjust(wspace=0.1)

plt.show()

