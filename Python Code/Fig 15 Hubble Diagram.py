#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM, wCDM
import numpy as np
import seaborn as sns

#import functions
from utilities_ import (load_data, Mb_eq, distmod_Planck18, 
Mb_err_eq, M_cor_eq_4params, intr_dis_cov_errs_delta)

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
pEW_sil = cut["pEW_Sil_6355"]

#estimate error in pEW by adding uncertainties in quadrature
pEW_sil_err = np.sqrt((cut["pEW_6355_unc_low"])**2 +(cut["pEW_6355_unc_high"])**2)

#best fit alpha beta gamma delta values taken from MCMC

#Alpha values and uncertainties
a_value = 0.142
a_err = 0.012
#Beta values and uncertainties
B_value = 3.03
B_err =0.06
#gamma values and uncertainties:
g_value =0.147
g_err = 0.02
#delta values and uncertainties
d_value = 0.0018
d_err = 0.00036

#corrected M using 4 parameter equation (alpha beta gamma delta)
M_cor = M_cor_eq_4params(Mb, mu_Planck18, a_value,x1,B_value,c,g_value,d_value,pEW_sil,host_c)
#intrinsic dis value from the MCMC 
intr_dis = 0.176
intr_dis_values = np.arange(0.12,0.6,0.0001 )

#set to zero for simplicity
sigma_z = 0.0

#define covariances
cov_mb_x1 = cut["cov_x0_x1"]
cov_mb_c = cut["cov_x0_c"]
cov_x1_c = cut["cov_x1_c"]

#intrinsic dispersion function that takes d into account!!!! need this to get an estimate for y intercep to create residuals
best_intr_dis1, intr_dis_err1 ,chisqr_red_min, best_inter1, df = intr_dis_cov_errs_delta(
    intr_dis_values, sigma_z,M_cor, z,a_value, B_value, Mb_err, x1_err, c_err,cov_mb_x1,cov_mb_c,cov_x1_c,d_value,pEW_sil_err,pEW_sil,d_err)

#Error on corrected M using intr dis value from MCMC
M_cor_err = np.sqrt((intr_dis)**2 + (sigma_z)**2 + (a_value*x1_err)**2 + (B_value*c_err)**2 +(d_value*pEW_sil_err)**2 +2*a_value*cov_mb_x1 - 2*B_value*cov_mb_c - 2*a_value*B_value*cov_x1_c)

#set alpha beta gamma delta = 0 for comparison with undstandardised sample

#Alpha values and uncertainties
a_value = 0
a_err = 0
#Beta values and uncertainties
B_value = 0
B_err =0
#gamma values and uncertainties:
g_value =0
g_err = 0
#delta values and uncertainties
d_value = 0
d_err = 0


M_uncor = M_cor_eq_4params(Mb, mu_Planck18, a_value,x1,B_value,c,g_value,d_value,pEW_sil,host_c)
#intr dis from MCMC when no alpha beta gamma or delta corrections used
intr_dis = 0.578
intr_dis_values = np.arange(0.12,0.6,0.0001 )

#set to zero for simplicity
sigma_z = 0.0


#intrinsic dispersion function that takes d into account!!!!
best_intr_dis1, intr_dis_err1 ,chisqr_red_min, best_inter2, df = intr_dis_cov_errs_delta(
    intr_dis_values, sigma_z,M_cor, z,a_value, B_value, Mb_err, x1_err, c_err,cov_mb_x1,cov_mb_c,cov_x1_c,d_value,pEW_sil_err,pEW_sil,d_err)

M_uncor_err = np.sqrt((intr_dis)**2 + (sigma_z)**2 + (a_value*x1_err)**2 + (B_value*c_err)**2 +(d_value*pEW_sil_err)**2 +2*a_value*cov_mb_x1 - 2*B_value*cov_mb_c - 2*a_value*B_value*cov_x1_c)

#define y axes for plot (Hubble residuals)
resids = M_cor - best_inter1
mu = resids + mu_Planck18

resids2 = M_uncor - best_inter2
mu_uncor = resids2 + mu_Planck18

#define cosmology model used as example
planck = FlatLambdaCDM(H0=67.66, Om0=0.311)

#show the impact H0 has on shape of model
high_H0 = FlatLambdaCDM(H0=75, Om0=0.30966)
low_H0 = FlatLambdaCDM(H0=60, Om0=0.30966)

# wCDM (dark energy equation of state w != -1)
w_model = wCDM(H0=67.66, Om0=0.3, Ode0=0.7, w0=-3)

z_vals = np.linspace(0.01, 0.08, 500)
mu_high_H0 = high_H0.distmod(z_vals)
mu_low_H0 = low_H0.distmod(z_vals)

#---plotting---

#set seaborn style
sns.set_style("whitegrid", {
    "axes.edgecolor": "0.3",
    "grid.color": "0.85",
    "axes.spines.right": True,
    "axes.spines.top": True
})

#set Seaborn color palette
palette = sns.color_palette("deep")
color_uncorrected = palette[3]  # Blue
color_corrected = palette[0]    # Orange/red
color3 = palette[2]

#define figure
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

# ---plot uncorrected residuals---
ax1.errorbar(z, mu_uncor, yerr=M_uncor_err,xerr=z_err ,fmt='o', markersize=8,
             color=color_uncorrected, alpha=0.4, ecolor='lightgrey', capsize=0.9,label='Unstandardised SNe Ia events')


#overlay example cosmology models to compare with data
z_np = np.array(z)
mu_planck18_np = np.array(mu_Planck18)
sorted_indices = np.argsort(z_np)
z_sorted = z_np[sorted_indices]
mu_Planck18_sorted = mu_planck18_np[sorted_indices]
ax1.plot(z_sorted, mu_Planck18_sorted, color='green', linestyle='-', linewidth=3,
         label='Planck18 $\Lambda$CDM',zorder = 30)
ax1.plot(z_vals, mu_high_H0, color='black', linestyle='--')
ax1.plot(z_vals, mu_low_H0, color='black', linestyle='--', label='H0 Constraints')

#plot formatting
ax1.text(0.015, 33.4, r'$H_0 = 75\mathrm{km/s/Mpc}$', color='black', fontsize=12,rotation=20)
ax1.text(0.013, 34.3, r'$H_0 = 60\mathrm{km/s/Mpc}$', color='black', fontsize=12,rotation=26)
ax1.set_ylabel(r'$\mu_\mathrm{uncor}$', fontsize = 18)
ax1.legend(title=fr'$\sigma_\mathrm{{intr}}$ = {0.578}±{0.02}', loc='lower right', fontsize = 13, title_fontsize = 13)
ax1.set_ylim(33, 39)
ax1.set_xlim(0.013, 0.06)

# ---plot corrected residuals ---
ax2.errorbar(z, mu, yerr=M_cor_err, xerr=z_err,fmt='o', markersize=8,
             color=color_corrected, alpha=0.4, ecolor='lightgrey',capsize=0.9 ,label='Standardised SNe Ia events')
#include cosmology models
ax2.plot(z_sorted, mu_Planck18_sorted, color='green', linestyle='-', linewidth=3,
         label='Planck18 $\Lambda$CDM',zorder = 30)
ax2.plot(z_vals, mu_high_H0, color='black', linestyle='--', label ='H0 Constraints' )
ax2.plot(z_vals, mu_low_H0, color='black', linestyle='--')
#formatting:
ax2.text(0.015, 33.4, r'$H_0 = 75\mathrm{km/s/Mpc}$', color='black', fontsize=12,rotation=20)
ax2.text(0.013, 34.3, r'$H_0 = 60\mathrm{km/s/Mpc}$', color='black', fontsize=12,rotation=26)
ax2.set_xlabel('Redshift', fontsize = 18)
ax2.set_ylabel(r'$\mu_\mathrm{data}$', fontsize = 18)
ax2.legend(title=fr'$\sigma_\mathrm{{intr}}$ = {0.176}±{0.009}', loc='lower right', fontsize = 13, title_fontsize = 13)
ax2.set_ylim(33, 39)
ax2.set_xlim(0.013, 0.06)

#general formatting
for ax in [ax1, ax2]:
    ax.tick_params(axis='x', which='major', labelsize=16.5)
    ax.tick_params(axis='y', which='major', labelsize=16.5)
    ax.grid(True, linestyle='--', alpha=0.1)
    ax.tick_params(axis='both', which='both', direction='out', length=5, width=1, colors='black')
    ax.tick_params(bottom=True, top=False, left=True, right=False)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.subplots_adjust(wspace=0.3, hspace = 0.1)

plt.show()