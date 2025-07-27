#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

#make cut in redshift to create a the Volume Limited Sample
cut = cut[cut["z"]<= 0.06]

#---define variables----

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

#set alpha beta and gamma to zero for the purpose of this (trying to estimate beta before standardisation)

#Alpha values and uncertainties
a_value = 0
a_err = 0

#Beta values and uncertainties
B_value = 0
B_err =0

#gamma values and uncertainties:
g_value =0
g_err = 0

#corrected M values
M_cor = M_cor_eq_3params(Mb, mu_Planck18, a_value,x1,B_value,c,g_value,host_c)

intr_dis_values = np.arange(0.12,0.21,0.0001 )

#set to zero for simplicity 
sigma_z = 0.0

#define covarainces
cov_mb_x1 = cut["cov_x0_x1"]
cov_mb_c = cut["cov_x0_c"]
cov_x1_c = cut["cov_x1_c"]

#use function from utilities to estimate intrinsic dispersion
best_intr_dis, chisqr_red_min, best_inter = intr_dis_cov(
    intr_dis_values, sigma_z,M_cor, z,a_value, B_value, Mb_err, x1_err, c_err,cov_mb_x1,cov_mb_c,cov_x1_c)

#use intrinsic dispersion estimate to get error on corrected M
M_cor_err = np.sqrt((best_intr_dis)**2 + (sigma_z)**2 + (a_value*x1_err)**2 + (B_value*c_err)**2 + 2*a_value*cov_mb_x1 - 2*B_value*cov_mb_c - 2*a_value*B_value*cov_x1_c)

#define axis for plot (Hubble residuals)
y2 = M_cor - best_inter

#----Plotting-----

#apply Seaborn style
sns.set_context("notebook")
sns.set_style("whitegrid", {
    "axes.edgecolor": "0.3",
    "grid.color": "0.85",
    "axes.spines.right": True,
    "axes.spines.top": True
})

#set Seaborn color palette
palette = sns.color_palette("bright")
color_uncorrected = palette[0]  
color_corrected = palette[0]    

#define figure
fig, (ax1) = plt.subplots(1, 1, figsize=(9, 5), sharex=False)

#Plot residuals against C
ax1.errorbar(c, y2, yerr=M_cor_err,xerr=c_err ,fmt='o', markersize=4,
             color='grey', alpha=0.6, ecolor='lightgrey', capsize=0.9,label='Unstandardised \n SNe Ia events')


#----binned averages----

#sort C
c_sorted = np.sort(c)

#bin parameters
num_bins = 20
bin_edges = np.linspace(c_sorted.min(), c_sorted.max(), num_bins + 1) 
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])  # midpoints for plotting

#Compute mean y and standard error in each bin
binned_means = []
binned_errors = []
for i in range(num_bins):
    in_bin = (c >= bin_edges[i]) & (c < bin_edges[i+1])
    if np.sum(in_bin) == 0:
        #no data in this bin
        binned_means.append(np.nan)
        binned_errors.append(np.nan)
        continue
    
    #Weighted mean
    y_in_bin = y2[in_bin]
    w = 1.0 / (M_cor_err[in_bin]**2)
    mean_val = np.sum(w * y_in_bin) / np.sum(w)
    
    #standard error of the mean
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


#---fitting a linear function to the data-----

c_range = np.arange(-0.3, 1, 0.01)

#define the linear function using the function in Utilities
a_fit,cov=curve_fit(linearFunc,c,y2,sigma=M_cor_err,absolute_sigma=True)
inter = a_fit[0]
slope1 = a_fit[1]
d_inter = np.sqrt(cov[0][0])
d_slope1 = np.sqrt(cov[1][1])

#define line of best fit
yfit1 = inter + slope1*c_range

#plot line of best fit/ linear fit
ax1.plot(c_range, yfit1, 
         label=fr'Linear Fit: $\beta$ = {slope1:.3f}$\pm${d_slope1:.3f} ' ,
               
         color=color_corrected,linewidth=2.0 ,zorder=40)

#---plot formatting----
plt.subplots_adjust(left=0.09, right=0.97, top=0.95, bottom=0.12)
plt.subplots_adjust(left=0.2, right=0.88, top=0.95, bottom=0.12)
ax1.set_xlabel('Light-Curve Colour ', fontsize = 18)
ax1.set_ylabel(r'$\mu_\mathrm{uncor} - \mu_\mathrm{CDM}$', fontsize = 18)
ax1.set_title('', fontsize = 15)
ax1.legend(title='', loc='lower right', fontsize=14, title_fontsize=14, labelspacing=0.6)
ax1.set_ylim(-1.2, 2)
ax1.set_xlim(-0.2, 0.8)

for ax in [ax1]:
    ax.tick_params(axis='both', which='major', labelsize=16.5)
    ax.grid(True, linestyle='--', alpha=0.1)
    ax.tick_params(axis='both', which='both', direction='out', length=5, width=1, colors='black')
    ax.tick_params(bottom=True, top=False, left=True, right=False)

plt.subplots_adjust(wspace=0.01)

plt.show()