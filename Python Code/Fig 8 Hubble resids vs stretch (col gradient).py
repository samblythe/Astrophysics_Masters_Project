#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from scipy.optimize import curve_fit

#import functions
from utilities_ import (load_data, Mb_eq, distmod_Planck18, linearFunc, 
Mb_err_eq, M_cor_eq_3params, intr_dis_cov)

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

#set alpha beta and gamma to zero for this (want to see results before standardisation is applied)

#Alpha values and uncertainties
a_value = 0
a_err = 0

#Beta values and uncertainties
B_value = 0 
B_err = 0

#gamma values and uncertainties:
g_value =0
g_err = 0

#corrected M values using function in Utilities
M_cor = M_cor_eq_3params(Mb, mu_Planck18, a_value,x1,B_value,c,g_value,host_c)

intr_dis_values = np.arange(0.12,0.21,0.0001 )

#set to zero for simplicity
sigma_z = 0.0

#define covariances
cov_mb_x1 = cut["cov_x0_x1"]
cov_mb_c = cut["cov_x0_c"]
cov_x1_c = cut["cov_x1_c"]

#intrinsic dispersion estimate using function from utilities
best_intr_dis, chisqr_red_min, best_inter = intr_dis_cov(
    intr_dis_values, sigma_z,M_cor, z,a_value, B_value, Mb_err, x1_err, c_err,cov_mb_x1,cov_mb_c,cov_x1_c)

#error on corrected M using intrinsic dispersion estimate and covariances
M_cor_err = np.sqrt((best_intr_dis)**2 + (sigma_z)**2 + (a_value*x1_err)**2 + (B_value*c_err)**2 + 2*a_value*cov_mb_x1 - 2*B_value*cov_mb_c - 2*a_value*B_value*cov_x1_c)


#---plotting---

#define axis for plot (Hubble residuals)
y2 = M_cor - best_inter

#apply Seaborn style
sns.set_context("notebook")
sns.set_style("whitegrid", {
    "axes.edgecolor": "0.3",
    "grid.color": "0.85",
    "axes.spines.right": True,
    "axes.spines.top": True
})

#set Seaborn color palette
palette = sns.color_palette("deep")
color_uncorrected = palette[3]  
color_corrected = palette[3]    

#define figure
fig, (ax1) = plt.subplots(1, 1, figsize=(9, 5), sharex=False)

#plot uncorrected Hubble residuals against stretch (x1)
ax1.errorbar(x1, y2, yerr=M_cor_err,xerr=x1_err ,fmt='o', markersize=4,
             color='grey', alpha=0.6, ecolor='lightgrey', capsize=0.9,label='Unstandardised \n SNe Ia events')

#add a colour gradient showing the distribution of a 3rd variable: C
ax1.scatter(x1, y2,  c=c, cmap='seismic', s=20, alpha=0.8, zorder = 30)
norm = plt.Normalize(c.min(), c.max())
cax = fig.add_axes([0.90, 0.15, 0.02, 0.7]) 
cax.set_title('Light \n Curve \n Colour')
#inlcude colourbar on the axes
cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='seismic'), cax=cax, orientation='vertical')


#add a linear fit to the plot

x1_range = np.arange(-3.3, 3.3, 0.01)
    
a_fit,cov=curve_fit(linearFunc,x1,y2,sigma=M_cor_err,absolute_sigma=True)
inter = a_fit[0]
slope1 = a_fit[1]
d_inter = np.sqrt(cov[0][0])
d_slope1 = np.sqrt(cov[1][1])

yfit1 = inter + slope1*x1_range

ax1.plot(x1_range, yfit1, 
         label=fr'Linear Fit: $\alpha$ = {slope1:.3f}$\pm${d_slope1:.3f} ' ,
               
         color='black',linewidth=2.0 ,zorder=30, linestyle = ':')

#---plot formatting---
plt.subplots_adjust(left=0.08, right=0.88, top=0.95, bottom=0.12)
ax1.set_xlabel('Light-Curve Stretch', fontsize = 15)
ax1.set_ylabel(r'$\mu_\mathrm{{uncor}} - \mu_\mathrm{{CDM}}$', fontsize = 15)
ax1.set_title('', fontsize = 15)
ax1.legend(title='', loc='upper right', fontsize=12, title_fontsize=12, labelspacing=0.4)
ax1.set_ylim(-1.2, 3)
ax1.set_xlim(-3, 2)

for ax in [ax1]:
    ax.tick_params(axis='both', which='major', labelsize=12.5)
    ax.grid(True, linestyle='--', alpha=0.1)
    ax.tick_params(axis='both', which='both', direction='out', length=5, width=1, colors='black')
    ax.tick_params(bottom=True, top=False, left=True, right=False)
plt.subplots_adjust(wspace=0.1)

plt.show()