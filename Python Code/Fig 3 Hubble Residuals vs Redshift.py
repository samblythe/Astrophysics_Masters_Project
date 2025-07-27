#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#import functions
from utilities_ import (load_data, Mb_eq, distmod_Planck18, 
Mb_err_eq,M_cor_eq_3params,intr_dis_cov_errs)

#load the data into a pandas dataframe
DataFrame = load_data(r"C:\Users\sambl\Documents\MPhys Project\ztfdr2_salt2_params.csv")
df_classifications = load_data(r"C:\Users\sambl\Documents\MPhys Project\ztfdr2_classifications.csv")
df_host_prop = load_data(r"C:\Users\sambl\Documents\MPhys Project\ztfdr2_local2kpc_prop.csv")
spec_df = load_data(r"C:\Users\sambl\Documents\MPhys Project\ZTF_DR2_Spec_Div_Burgaz_2024.csv")

#merge data frames
merged_df = pd.merge(DataFrame, df_classifications, on='ztfname')
merged_df_2 = pd.merge(merged_df,df_host_prop, on='ztfname')

#remove all data that doesnt meet criteria for the basic data cuts
cut = merged_df_2[merged_df_2["fitquality_flag"]==1][merged_df_2["lccoverage_flag"]==1][merged_df_2["sn_type"] != 'snia-pec' ]

#remove empty cells
cut = cut.dropna()

#----define variables-------

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

#Getting Mb values from the function created in Utilities.py
Mb = Mb_eq(cut["x0"])
#error in Mb
Mb_err = Mb_err_eq(x0, x0_err)

#distance modulus (mu) using Planck 18 cosmology
mu_Planck18 = distmod_Planck18(cut)

host_c = cut["restframe_gz"] 

#alpha beta gamma values taken from Ginolin 2024 paper

#Alpha values and uncertainties
a_value = 0.161
a_err = 0.01

#Beta values and uncertainties
B_value = 3.05
B_err =0.06

#gamma values and uncertainties:
g_value =0.143
g_err = 0.025

#compute corrected M values:

M_cor = M_cor_eq_3params(Mb, mu_Planck18, a_value,x1,B_value,c,g_value,host_c)

intr_dis_values = np.arange(0.12,0.5,0.0001 )

sigma_z = 0.0

#define covariances
cov_mb_x1 = cut["cov_x0_x1"]
cov_mb_c = cut["cov_x0_c"]
cov_x1_c = cut["cov_x1_c"]

#use intrisic dispersion estimation function from utilities

best_intr_dis, intr_dis_err ,chisqr_red_min, best_inter, df = intr_dis_cov_errs(
    intr_dis_values, sigma_z,M_cor, z,a_value, B_value, Mb_err, x1_err, c_err,cov_mb_x1,cov_mb_c,cov_x1_c)


#compute error on corrected M using intr dis estimate and other errors added in quadrature
#include covariances!! 
M_cor_err = np.sqrt((best_intr_dis)**2 + (sigma_z)**2 + (a_value*x1_err)**2 + (B_value*c_err)**2 + 2*a_value*cov_mb_x1 - 2*B_value*cov_mb_c - 2*a_value*B_value*cov_x1_c)


#repeat using alpha beta gamma = 0 

#Alpha values and uncertainties
a_value = 0
a_err = 0

#Beta values and uncertainties
B_value = 0
B_err =0

#gamma values and uncertainties:
g_value =0
g_err = 0

#compoute uncorrected M
M_uncor = M_cor_eq_3params(Mb, mu_Planck18, a_value,x1,B_value,c,g_value,host_c)


#use intrisic dispersion estimation function from utilities

best_intr_dis1, intr_dis_err1 ,chisqr_red_min1, best_inter1, df = intr_dis_cov_errs(
    intr_dis_values, sigma_z,M_uncor, z,a_value, B_value, Mb_err, x1_err, c_err,cov_mb_x1,cov_mb_c,cov_x1_c)


#compute error on corrected M using intr dis estimate and other errors added in quadrature
#include covariances!! 
M_uncor_err = np.sqrt((best_intr_dis1)**2 + (sigma_z)**2 + (a_value*x1_err)**2 + (B_value*c_err)**2 + 2*a_value*cov_mb_x1 - 2*B_value*cov_mb_c - 2*a_value*B_value*cov_x1_c)


#define axes for plot (Hubble residuals) using corrected and uncorrected M
y1 = M_uncor - best_inter1
y2 = M_cor - best_inter

#set Seaborn style
sns.set_context("notebook")
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

#define figure
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5), sharex=True)

# ---plot uncorrected Hubble residuals---
ax1.errorbar(z, y1, yerr=M_uncor_err,xerr=z_err ,fmt='o', markersize=3,
             color=color_uncorrected, alpha=0.4, ecolor='lightgrey', capsize=0.9,label='Unstandardised \n SNe Ia events')

ax1.axhline(0, color='black', linestyle=':', linewidth=1, zorder =12)
ax1.set_ylabel(r'$\mu_\mathrm{uncor} - \mu_\mathrm{CDM}$', fontsize = 19)
ax1.legend(title=fr'$\sigma_\mathrm{{intr}}$ = {best_intr_dis1:.3f}±{intr_dis_err1}', loc='upper right', fontsize = 15, title_fontsize = 15)
ax1.set_ylim(-1.2, 2)
ax1.set_xlim(0.01, 0.175)

# ---plot corrected Hubble residuals ---
ax2.errorbar(z, y2, yerr=M_cor_err, xerr=z_err,fmt='o', markersize=3,
             color=color_corrected, alpha=0.4, ecolor='lightgrey',capsize=0.9 ,label='Standardised \n SNe Ia events')

ax2.axhline(0, color='black', linestyle=':', linewidth=1, zorder =12)
ax2.set_xlabel('Redshift', fontsize = 19)
ax2.set_ylabel(r'$\mu_\mathrm{data} - \mu_\mathrm{CDM}$', fontsize = 19)
ax2.legend(title=fr'$\sigma_\mathrm{{intr}}$ = {best_intr_dis:.3f}±{intr_dis_err}', loc='upper right', fontsize = 15, title_fontsize = 15)
ax2.set_ylim(-1.2, 2)
ax2.set_xlim(0.01, 0.175)

#improve tick size and spacing
for ax in [ax1, ax2]:
    ax.tick_params(axis='both', which='major', labelsize=12.5)
    ax.grid(True, linestyle='--', alpha=0.1)
    ax.tick_params(axis='both', which='both', direction='out', length=5, width=1, colors='black')
    ax.tick_params(bottom=True, top=False, left=True, right=False)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.subplots_adjust(wspace=0.1,hspace=0.05)

plt.show()


