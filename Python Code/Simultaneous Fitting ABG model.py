#import libraries
import numpy as np
import pandas as pd
import emcee
import corner
import matplotlib.pyplot as plt

#import functions
from utilities_ import (load_data, Mb_eq, distmod_Planck18, Mb_err_eq,
                       log_posterior_abg)

#load the data into a pandas dataframe
DataFrame = load_data(r"C:\Users\sambl\Documents\MPhys Project\ztfdr2_salt2_params.csv")
df_classifications = load_data(r"C:\Users\sambl\Documents\MPhys Project\ztfdr2_classifications.csv")
df_host_prop = load_data(r"C:\Users\sambl\Documents\MPhys Project\ztfdr2_globalhost_prop.csv")
spec_df = load_data(r"C:\Users\sambl\Documents\MPhys Project\ZTF_DR2_Spec_Div_Burgaz_2024.csv")

#merge dataframes
merged_df = pd.merge(DataFrame, df_classifications, on='ztfname')
merged_df_2 = pd.merge(merged_df,df_host_prop, on='ztfname')
pEW_sil_df = spec_df[["ztfname","pEW_Sil_6355","pEW_6355_unc_low","pEW_6355_unc_high","Branch_Type"]]
merged_df_3 = pd.merge(merged_df_2,pEW_sil_df, on='ztfname')

#perform data quality cuts
cut = merged_df_3[merged_df_3["fitquality_flag"]==1][merged_df_3["lccoverage_flag"]==1][merged_df_3["sn_type"] != 'snia-pec' ]
cut = cut.dropna()

#---define variables---

host_c = cut["restframe_gz"] 

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

#define covariances
cov_mb_x1 = cut["cov_x0_x1"]
cov_mb_c = cut["cov_x0_c"]
cov_x1_c = cut["cov_x1_c"]

#fix break point in the step function (gamma) for simplicity
bp_fixed = 1.0

#define y axis for plot (Hubble residuals)
M_uncor = Mb - mu_Planck18  
yerr = Mb_err


#--- alpha beta gamma model ---
  
y_target = np.zeros_like(Mb)
w = 1.0 / (yerr**2)

#sampler parameters
ndim = 4  # alpha, beta, gamma, sigma_int
nwalkers = 32

# Initial guesses
initial = np.array([0.14, 2.8, 0.1, 0.1])

#small random pertubations from the inital guess to allow walkers to explore the parameter space
pos = initial + 1e-5 * np.random.randn(nwalkers, ndim)

sigma_z = 0 #set to zero for simplicity

#set up MCMC sampler
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior_abg,
    args=(Mb, mu_Planck18, x1, c, host_c, bp_fixed,
          Mb_err, sigma_z, x1_err, c_err,
          cov_mb_x1, cov_mb_c, cov_x1_c,y_target)
)

#3000 steps is enough to get good sampling (checked the trace plots and all walkers were well mixed)
nsteps = 3000

#run the sampler
sampler.run_mcmc(pos, nsteps, progress=True)

#Discard burn-in and flatten the chain:
burnin = 1000
flat_samples = sampler.get_chain(discard=burnin, flat=True)
flat_log_probs = sampler.get_log_prob(discard=burnin, flat=True)

#Get highest log probability (best fit) values
max_index = np.argmax(flat_log_probs)
map_params = flat_samples[max_index]
alpha_map, beta_map, gamma_map, intr_map = map_params

#uncertainties using standard deviation
alpha_std = np.std(flat_samples[:, 0])
beta_std  = np.std(flat_samples[:, 1])
gamma_std = np.std(flat_samples[:, 2])
intr_std = np.std(flat_samples[:,3])

#print results
print("MCMC MAP Results (Joint Fit):")
print(f"Alpha: {alpha_map:.3f} ± {alpha_std:.3f}")
print(f"Beta:  {beta_map:.3f} ± {beta_std:.3f}")
print(f"Gamma: {gamma_map:.3f} ± {gamma_std:.3f}")
print(f"Intrinsic Dispersion: {intr_map} ±{intr_std}")

#corner plot to show relationships between distributions
flat_samples = sampler.get_chain(discard=burnin, flat=True)
labels = ["alpha", "beta", "gamma","Intr Dis"]
fig_corner = corner.corner(flat_samples, labels=labels, show_titles=True,
                           title_fmt=".3f")

plt.suptitle("Corner Plot of Posterior Distributions", y=1.02)
plt.show()