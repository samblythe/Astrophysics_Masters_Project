#import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp
#import functions
from utilities_ import load_data, Mb_eq


#load CSV data into pandas dataframes
DataFrame = load_data(r"C:\Users\sambl\Documents\MPhys Project\ztfdr2_salt2_params.csv")
df_classifications = load_data(r"C:\Users\sambl\Documents\MPhys Project\ztfdr2_classifications.csv")
spec_df = load_data(r"C:\Users\sambl\Documents\MPhys Project\ZTF_DR2_Spec_Div_Burgaz_2024.csv")
df_host_prop = load_data(r"C:\Users\sambl\Documents\MPhys Project\ztfdr2_globalhost_prop.csv")

#merge dataframes
merged_df = pd.merge(DataFrame, df_classifications, on='ztfname')
merged_df_2 = pd.merge(merged_df,df_host_prop, on='ztfname')

#perform data quality cuts
merged_df_2 = merged_df_2[merged_df["fitquality_flag"]==1][merged_df_2["lccoverage_flag"]==1][merged_df_2["sn_type"] != 'snia-pec' ]

#define variables for full data set
x1 =merged_df_2["x1"]
c=merged_df_2["c"]
host =merged_df_2["restframe_gz"]
mass = merged_df_2["mass"]
Mb = Mb_eq(merged_df_2["x0"])

#perfrom data cut to create Volume Limited Sample
low_Z_cut = merged_df_2[merged_df_2["z"]<=0.06]

#define variables for Volume Limited Sample
x1_low_z =low_Z_cut["x1"]
c_low_z=low_Z_cut["c"]
host_low_z = low_Z_cut["restframe_gz"]
mass_low_z = low_Z_cut["mass"]
Mb_low_z = Mb_eq(low_Z_cut["x0"])

#perform data cut (by merging) to create spectroscopic sample
spec_df_merged = pd.merge(merged_df_2,spec_df, on='ztfname')

#define variables for spectroscopic sample
x1_spec = spec_df_merged["x1"]
c_spec = spec_df_merged["c"]
host_spec = spec_df_merged["restframe_gz"]
mass_spec = spec_df_merged["mass"]
Mb_spec = Mb_eq(spec_df_merged["x0"])



#--------Histogram Plotting------



#set for a clean whitye background withouth gridlines
plt.style.use("default")  
sns.set_style("white")    

#define figure axes and dimensions
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 7),
                                             sharex=False, sharey=False,
                                             gridspec_kw={'hspace': 0.2, 'wspace': 0.01})

#universal histogram params
hist_kw = dict(bins=25, kde=False, alpha=0.65, stat="density",element="step", fill = True)

#panel 1: x1 distributions
sns.histplot(x1_low_z, color="blue", label="VLS", ax=ax1, **hist_kw)
sns.histplot(x1_spec, color="orange", label="SDA Sample", ax=ax1, **hist_kw)
sns.histplot(x1, color="black", label="Full DR2cut Sample", ax=ax1, bins=25, kde=False, alpha=0.8, fill = False,linewidth = 1.7, stat="density", element="step")
ax1.set_xlabel(r'Light Curve Stretch $x_1$', fontsize=15)
ax1.set_ylabel("")
ax1.tick_params(left=False, labelleft=False)


#panel 2: c distributions
sns.histplot(c_low_z, color="blue", label="VLS", ax=ax2, **hist_kw)
sns.histplot(c_spec, color="orange", label="SDA Sample", ax=ax2, **hist_kw)
sns.histplot(c, color="black", label="Full DR2cut Sample", ax=ax2, bins=25, kde=False, alpha=0.8, fill = False,linewidth = 1.7, stat="density", element="step")
ax2.set_xlabel(r'Light Curve Colour $c$', fontsize=15)
ax2.set_ylabel("")
ax2.tick_params(left=False, labelleft=False)
ax2.legend(loc="upper right",fontsize=14)

#panel 3: host galaxy colour distributions
sns.histplot(host_low_z, color="blue", label="VLS", ax=ax3, **hist_kw)
sns.histplot(host_spec, color="orange", label="SDA Sample", ax=ax3, **hist_kw)
sns.histplot(host, color="black", label="Full DR2cut Sample", ax=ax3, bins=25, kde=False, alpha=0.8, fill = False,linewidth = 1.7, stat="density", element="step")
ax3.set_xlabel(r'$(g-z)_{\mathrm{global}}$', fontsize=15)
ax3.set_ylabel("")
ax3.tick_params(left=False, labelleft=False)


#panel 4: Mb distributions
sns.histplot(Mb_low_z, color="blue", label="VLS", ax=ax4, **hist_kw)
sns.histplot(Mb_spec, color="orange", label="SDA Sample", ax=ax4, **hist_kw)
sns.histplot(Mb, color="black", label="Full DR2cut Sample", ax=ax4, bins=25, kde=False, alpha=0.8, fill = False,linewidth = 1.7, stat="density", element="step")
ax4.set_xlabel(r'$m_{b}$', fontsize=15)
ax4.set_ylabel("")
ax4.tick_params(left=False, labelleft=False)


#set all spines black and visible
for ax in [ax1, ax2, ax3, ax4]:
    ax.tick_params(axis='x', which='major', labelsize=13.5)
    ax.grid(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['bottom'].set_color('black')
    ax.spines['bottom'].set_linewidth(1.3)

    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='both', direction='out', length=5, width=1, colors='black')
    ax.tick_params(bottom=True, top=False, left=False, right=False)


#Adjust final layout
plt.tight_layout(pad=0.1)
fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.1, wspace=0.01, hspace=0.15)


#---------K-S Testing on distributions--------


stat_x1, pval_x1 = ks_2samp(x1_low_z, x1)
stat_c, pval_c = ks_2samp(c_low_z, c)
stat_host, pval_host = ks_2samp(host_low_z, host)
stat_mb, pval_mb = ks_2samp(Mb_low_z, Mb)

print("K-S test Results:")
print("H0 = The parameter samples are drawn from the same distribution")
print(f"x1 p-value: {pval_x1}")
print(f"c p-value: {pval_c}")
print(f"host c p-value: {pval_host}")
print(f"mb p-value: {pval_mb}")
print("If P value < 0.05 then reject H0")

plt.show()