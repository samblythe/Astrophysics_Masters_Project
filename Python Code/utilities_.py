#import librarires
import numpy as np
import pandas as pd
import astropy as ap
import astropy.cosmology
import astropy.coordinates
import matplotlib.pyplot as plt
import csv
import math
from scipy.stats import linregress
from scipy.optimize import curve_fit
from scipy.stats import chi2

def load_data(filename):
    """
    Loads the csv data and reads it into a pandas dataframe.
    Takes the name of the file being read as the argument
    """

    DF = pd.read_csv(filename)
    return DF

def Mb_eq(x0_col):
    """
    Finds the observed brightness using the x0 variable in the data set
    10.652 is a constant 

    """
    Mb = -2.5*(np.log10(x0_col)) +10.652
    return Mb

def Mb_err_eq(x0, x0_err):
    """
    from error propagation laws carried out on Mb_eq above
    """
    Mb_err = (1.08574 / x0 )*x0_err
    return Mb_err

def distmod_Planck18(z_value):
    """
    Finds the distance modulus (mu) from the redshfit
    Uses the cosmology Planck 18 to compute distances
    """
    from astropy.cosmology import Planck18
    from astropy import units as u
    mu_theory_value = Planck18.distmod(z=z_value["z"].to_list()) /u.mag
    return mu_theory_value

def linearFunc(x,intercept,slope):
    """
    defines a linear function to use for plottig linear regressions
    """
    y = intercept + slope * x
    return y

def M_cor_eq_3params(Mb,mu,alpha,x1,beta,c,gamma,host_c):
    """
    standardised M equation by including alpha beta gamma corrections
    """
    #p is a step function:
    p = np.where(host_c < 1, 0, 1)  #bp = 0.924 for vol lim sample, 0.939 for full
    return Mb - mu + alpha*x1 - beta*c + gamma*p

def M_cor_eq_4params(Mb,mu,alpha,x1,beta,c,gamma,delta,E,host_c):
    """
    standardised M equation by including alpha beta gamma corrections
    """
    #p is a step function:

    p = np.where(host_c < 1, 0, 1)  #bp = 0.924 for vol lim sample, 0.939 for full
    return Mb - mu + alpha*x1 - beta*c + gamma*p - delta*E

def intr_dis_cov(intr_dis_values, sigma_z,M_cor, z,alpha, beta, Mb_err, x1_err, c_err,cov_mb_x1,cov_mb_c,cov_x1_c):
    #Precompute fixed errors
    base_err = np.sqrt(Mb_err**2 + sigma_z**2+ (alpha * x1_err)**2 + (beta * c_err)**2 + 2*alpha*cov_mb_x1 - 2*beta*cov_mb_c - 2*alpha*beta*cov_x1_c)

    #Store results directly in a list
    results = []

    for intr_dis in intr_dis_values:
        #Calculate M_cor_err
        M_cor_err = np.sqrt(base_err**2 + intr_dis**2)

        #Perform curve fit
        a_fit, cov = curve_fit(lambda z, inter: inter, z, M_cor, sigma=M_cor_err, absolute_sigma=True)
        inter = a_fit[0]
        d_inter = np.sqrt(cov[0][0]) if cov.size > 0 else np.nan
         
        #Compute reduced chi-squared
        chisqr = np.sum((M_cor - inter) ** 2 / M_cor_err ** 2)
        dof = len(M_cor) - 1
        chisqr_red = chisqr / dof if dof > 0 else np.nan

        #Compute the difference from target chi-squared
        chi_diff = np.abs(chisqr_red - 1, dtype=float)

        #Store results
        results.append([intr_dis, float(chisqr_red), float(chi_diff), float(inter)])
        #Create DataFrame from results
    columns = ['intr_dis', 'chisqr_red', 'abs chi^2 -1', 'inter']
    Table_df = pd.DataFrame(results, columns=columns)
    
    #Ensure numeric data and handle NaNs
    Table_df["abs chi^2 -1"] = pd.to_numeric(Table_df["abs chi^2 -1"], errors="coerce")

    #Find the best fit result while skipping NaNs
    best_idx = Table_df["abs chi^2 -1"].idxmin(skipna=True)
    
    #retun results
    return (Table_df.loc[best_idx, "intr_dis"],
            Table_df.loc[best_idx, "chisqr_red"],
            Table_df.loc[best_idx, "inter"])

def intr_dis_cov_errs(intr_dis_values, sigma_z, M_cor, z, alpha, beta,
                 Mb_err, x1_err, c_err, cov_mb_x1, cov_mb_c, cov_x1_c):
    
    #Precompute fixed errors
    base_err = np.sqrt(Mb_err**2 + sigma_z**2 + (alpha * x1_err)**2 +
                       (beta * c_err)**2 + 2*alpha*cov_mb_x1 - 2*beta*cov_mb_c - 2*alpha*beta*cov_x1_c)

    #store results
    results = []

    for intr_dis in intr_dis_values:
        M_cor_err = np.sqrt(base_err**2 + intr_dis**2)
        
        try:
            # Fit to constant model: mu_model = intercept
            a_fit, cov = curve_fit(lambda z, inter: inter, z, M_cor,
                                   sigma=M_cor_err, absolute_sigma=True)
            inter = a_fit[0]
        except Exception:
            inter = np.nan

        chisqr = np.sum((M_cor - inter) ** 2 / M_cor_err ** 2)
        dof = len(M_cor) - 1
        chisqr_red = chisqr / dof if dof > 0 else np.nan
        chi_diff = np.abs(chisqr_red - 1,dtype=float)
        results.append([intr_dis, float(chisqr_red), float(chi_diff), float(inter)])
        

    # Create DataFrame and force all values to numeric types
    columns = ['intr_dis', 'chisqr_red', 'abs chi^2 -1', 'inter']
    df = pd.DataFrame(results, columns=columns)
    
    df["abs chi^2 -1"] = pd.to_numeric(df["abs chi^2 -1"],errors='coerce')

    # Drop rows with NaNs in the relevant columns
    #df_clean = df.dropna(subset=["intr_dis", "chisqr_red", "abs chi^2 -1"])

    # Find the best-fit intrinsic dispersion (min abs(chi²_red - 1))
    best_idx = df["abs chi^2 -1"].idxmin(skipna=True)
    
    best_intr_dis = df.loc[best_idx, "intr_dis"]
    best_chi_red = df.loc[best_idx, "chisqr_red"]
    best_inter = df.loc[best_idx, "inter"]

    # --- Confidence interval based on chi² distribution ---
    dof = len(M_cor) - 1
    chi2_lower = chi2.ppf(0.1587, dof) / dof  # ~1 - 1σ
    chi2_upper = chi2.ppf(0.8413, dof) / dof  # ~1 + 1σ

    # Filter all points within 1σ range
    within_conf = df[
        (df["chisqr_red"] >= chi2_lower) &
        (df["chisqr_red"] <= chi2_upper)
    ]

    if within_conf.empty:
        intr_dis_uncertainty = np.nan
    else:
        intr_dis_uncertainty = 0.5 * (within_conf["intr_dis"].max() - within_conf["intr_dis"].min())

    # Round to 3 significant figures
    best_rounded = float(f"{best_intr_dis:.3g}")
    err_rounded = float(f"{intr_dis_uncertainty:.1g}") if not np.isnan(intr_dis_uncertainty) else np.nan

    return best_rounded, err_rounded, best_chi_red, best_inter, df

def intr_dis_cov_errs_delta(intr_dis_values, sigma_z, M_cor, z, alpha, beta,
                 Mb_err, x1_err, c_err, cov_mb_x1, cov_mb_c, cov_x1_c,delta,pEW_err,pEW,d_err):
    
    # Precompute fixed errors
    base_err = np.sqrt(Mb_err**2 + sigma_z**2 + (alpha * x1_err)**2 +
                       (beta * c_err)**2 +(delta*pEW_err)**2+ (pEW*d_err)**2+2*alpha*cov_mb_x1 - 2*beta*cov_mb_c - 2*alpha*beta*cov_x1_c)

    results = []

    for intr_dis in intr_dis_values:
        M_cor_err = np.sqrt(base_err**2 + intr_dis**2)
        
        try:
            # Fit to constant model: mu_model = intercept
            a_fit, cov = curve_fit(lambda z, inter: inter, z, M_cor,
                                   sigma=M_cor_err, absolute_sigma=True)
            inter = a_fit[0]
        except Exception:
            inter = np.nan

        chisqr = np.sum((M_cor - inter) ** 2 / M_cor_err ** 2)
        dof = len(M_cor) - 1
        chisqr_red = chisqr / dof if dof > 0 else np.nan
        chi_diff = np.abs(chisqr_red - 1,dtype=float)
        results.append([intr_dis, float(chisqr_red), float(chi_diff), float(inter)])
        #results.append([intr_dis, chisqr_red, chi_diff, inter])

    # Create DataFrame and force all values to numeric types
    columns = ['intr_dis', 'chisqr_red', 'abs chi^2 -1', 'inter']
    df = pd.DataFrame(results, columns=columns)
    
    df["abs chi^2 -1"] = pd.to_numeric(df["abs chi^2 -1"],errors='coerce')

    # Drop rows with NaNs in the relevant columns
    #df_clean = df.dropna(subset=["intr_dis", "chisqr_red", "abs chi^2 -1"])

    # Find the best-fit intrinsic dispersion (min abs(chi²_red - 1))
    best_idx = df["abs chi^2 -1"].idxmin(skipna=True)
    
    best_intr_dis = df.loc[best_idx, "intr_dis"]
    best_chi_red = df.loc[best_idx, "chisqr_red"]
    best_inter = df.loc[best_idx, "inter"]

    # --- Confidence interval based on chi² distribution ---
    dof = len(M_cor) - 1
    chi2_lower = chi2.ppf(0.1587, dof) / dof  # ~1 - 1σ
    chi2_upper = chi2.ppf(0.8413, dof) / dof  # ~1 + 1σ

    # Filter all points within 1σ range
    within_conf = df[
        (df["chisqr_red"] >= chi2_lower) &
        (df["chisqr_red"] <= chi2_upper)
    ]

    if within_conf.empty:
        intr_dis_uncertainty = np.nan
    else:
        intr_dis_uncertainty = 0.5 * (within_conf["intr_dis"].max() - within_conf["intr_dis"].min())

    # Round to 3 significant figures
    best_rounded = float(f"{best_intr_dis:.3g}")
    err_rounded = float(f"{intr_dis_uncertainty:.1g}") if not np.isnan(intr_dis_uncertainty) else np.nan

    return best_rounded, err_rounded, best_chi_red, best_inter, df

def step_model_two(x, break_point, gamma):
    """
    Reparameterized step function with two free parameters:
      - break_point: the x-value at which the step occurs
      - gamma: the jump (difference between high and low levels)
    The model is defined to be 0 for x < break_point and gamma for x >= break_point.
    """
    return np.where(x < break_point, 0.0, gamma)

# ----------------------------
# Log likelihood, assuming Gaussian errors.
def log_likelihood_step_two(theta, x, y, yerr):
    break_point, gamma = theta
    model = step_model_two(x, break_point, gamma)
    chi2 = np.sum(((y - model) / yerr)**2)
    return -0.5 * chi2

# ----------------------------
# Log prior: restrict break_point to be within the range of x, and gamma to a reasonable range.
def log_prior_step_two(theta, x):
    break_point, gamma = theta
    if break_point < np.min(x) or break_point > np.max(x):
        return -np.inf
    if not (-10 < gamma < 10):
        return -np.inf
    return 0.0  # flat priors

# ----------------------------
# Log posterior function
def log_posterior_step_two(theta, x, y, yerr):
    lp = log_prior_step_two(theta, x)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_step_two(theta, x, y, yerr)


#----------------

def step_function_combined(host, bp_fixed):
    """
    Step function that returns 0 for host colour < bp_fixed and 1 for host colour >= bp_fixed.
    """
    return np.where(host < bp_fixed, 0.0, 1.0)

def combined_model(theta, mb, mu, x1, c, host, bp_fixed, w):
    """
    Combined model for SN corrections.
    
    Parameters:
      theta = [alpha, beta, gamma]
      
    The model is defined as:
      M_model = (mb - mu) + alpha*x1 - beta*c - offset + p(gamma)
      where offset is the weighted mean of (mb - mu + alpha*x1 - beta*c) for objects with host < bp_fixed,
      and p(host) = 0 if host < bp_fixed, 1 if host >= bp_fixed.
      
    The fit is done on residuals (target = 0).
    """
    alpha, beta, gamma,_= theta
    M_uncor = mb - mu
    #calculate prelim m_cor values in ordere to calculate gamma:
    M_corr_prelim = M_uncor + alpha * x1 - beta * c
    #Compute offset from objects with host < bp_fixed:
    #find which objects have host colour < break point
    mask = (host < bp_fixed)
    # Avoid division by zero if no object in the low region:
    if np.sum(mask) == 0:
        offset = 0.0
    else:
        #weighted mean of all objects with host colour below break point
        offset = np.sum(w[mask] * M_corr_prelim[mask]) / np.sum(w[mask])
    #Step function:
    p = step_function_combined(host, bp_fixed)
    #Full model: subtract offset, then add step correction (gamma applied only above break).
    model = M_corr_prelim - offset + p * gamma
    return model

def log_likelihood_abg(theta, mb, mu, x1, c, host, bp_fixed, Mb_err, sigma_z, x1_err, c_err, cov_mb_x1, cov_mb_c, cov_x1_c,y_target):
    alpha, beta, gamma, sigma_int = theta

    # Compute total error
    total_var = (
        Mb_err**2 + sigma_z**2 +
        (alpha * x1_err)**2 + (beta * c_err)**2 +
        2 * alpha * cov_mb_x1 - 2 * beta * cov_mb_c - 2 * alpha * beta * cov_x1_c +
        sigma_int**2
    )
    total_err = np.sqrt(total_var)

    #model = combined_model_delta(theta, mb, mu, x1, c, host, bp_fixed, 1 / total_var, pEW)
    model = combined_model(theta, mb, mu, x1, c, host, bp_fixed, 1/ total_var)
    #residuals = (mb - mu) - model
    chi2 = np.sum((y_target - model / total_err) ** 2)
    logL = -0.5 * (len(mb) * np.log(2 * np.pi) + 2*(np.sum(np.log(total_err))) + chi2)
    return logL

def log_prior_abg(theta):
    alpha, beta, gamma, sigma_int = theta
    
    if not (-1 < alpha < 1):
        return -np.inf
    if not (2 < beta < 4):
        return -np.inf
    if not (-1 < gamma < 1):
        return -np.inf
    if not (0 < sigma_int < 0.5):  # typical values 0.1–0.2
        return -np.inf
    
    return 0.0  # uniform priors

def log_posterior_abg(theta, mb, mu, x1, c, host, bp_fixed,
                           Mb_err, sigma_z, x1_err, c_err,
                           cov_mb_x1, cov_mb_c, cov_x1_c,y_target):
    lp = log_prior_abg(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_abg(theta, mb, mu, x1, c, host, bp_fixed,
                                        Mb_err, sigma_z, x1_err, c_err,
                                        cov_mb_x1, cov_mb_c, cov_x1_c,y_target)