# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 11:31:42 2026

@author: 32489
"""

# %% 

import numpy as np 
import Bargaining_numba as brg  
import UserFunctions_numba as usr 
import pandas as pd
import getpass
import statsmodels.api as sm 
import matplotlib.pyplot as plt


#Initialize seed 
np.random.seed(10) 
 
#Root
# root='C:/Users/32489/Dropbox/Family Risk Sharing'

user = getpass.getuser()
if user == "sara":
      root = '/Users/sara/Dropbox/Family Risk Sharing'
elif user == "32489":
      root = '/Users/32489/Dropbox/Family Risk Sharing'
else:
      raise RuntimeError(f"Unknown user: {user}")



#estimating the model (True) or compute tables given paramters in xc below (BPP, paramters, fitt) (False)
ESTIMATE=False


#Create sample with replacement 
N=10_000#sample size 


#Import information for the sample, then store the relevant variables
baseline_sample=np.array(pd.read_excel(root+'/Output files/data_sample.csv'))

#Without those two lines below there are nan which screw up things
baseline_sample=baseline_sample[~np.isnan(baseline_sample).any(axis=1)]
baseline_sample[:, 0] = np.arange(len(baseline_sample))

pr=np.ones(baseline_sample.shape[0])/baseline_sample.shape[0]
indexes=np.array(np.random.choice(baseline_sample[:,0], size=N, p=pr, replace=True),dtype=np.int64)-1
final_sample= baseline_sample[:,1:][indexes] 

age_initial=final_sample[:,0]
age_final=final_sample[:,1]
cw_cons_share=final_sample[:,2]
h_income=final_sample[:,3]
w_income=final_sample[:,4]
age_marriage=final_sample[:,5]
year=final_sample[:,6]
assets=final_sample[:,7]*np.mean(np.exp(h_income))



# Guess of internal parameters: [ω,σL,α,ρ,wedge,β]

#xc=np.array([0.55, 0.1       , 0.85, 1.2, 0.929     ,1.        ])

# Higher disutility of working
#xc=np.array([0.7, 0.1       , 0.85, 1.2, 0.929     ,1.        ])

# Lower disutility of working
xc=np.array([0.1, 0.1       , 0.85, 1.2, 0.929     ,1.        ])


# No non-homotheticity in home production
#xc=np.array([0.55, 0.1       , 0.85, 1.5, 0.929     ,1.        ])



# Lower and higher bounds of parameters
xl=np.array([0.00001,0.000082,0.1,0.5,0.01,0.9]) 
xu=np.array([0.8,0.4,0.999,2.5,1.0,1.1]) 

#Parametrize the model 
par = {'simN':N,'ω': xc[0],'σL':xc[1],'α':xc[2],'ρ':xc[3],'wedge':xc[4],'β':xc[5],'sample_init':np.array(age_marriage-20,dtype=np.int_)}
model=brg.HouseholdModelClass(par=par)


#####################################################################
# Set the initial conditions for the couples based on baseline sample
#####################################################################


#Given the parameters, set the initial pareto weight for couples
param=(cw_cons_share/(1.0-cw_cons_share))**model.par.ρ
model.sim.init_power=param/(1.0+param)

#Set the initial income gridpoints for income, the closest to our value

gridzw=model.par.grid_zw[:,:,np.linspace(0,model.par.num_z-1,model.par.num_zm,dtype=np.int_)]
gridzm=model.par.grid_zm[:,:,:model.par.num_zw]

izm=np.array([np.argmin(np.abs(np.log(gridzm)[int(model.par.sample_init[i]),0,:,0]-h_income[i])) for i in range(model.par.simN)],dtype=np.int64)
izm[np.isnan(h_income)]=(model.par.num_pm*model.par.num_ϵm)//2
izw=np.array([np.argmin(np.abs(np.log(gridzw)[int(model.par.sample_init[i]),0,:,0]-w_income[i])) for i in range(model.par.simN)],dtype=np.int64)
izw[np.isnan(w_income)]=(model.par.num_pw*model.par.num_ϵw)//2     
model.sim.init_z=izm*model.par.num_zm+izw
model.sim.init_A=assets


#Create variable for policy change
age=(np.cumsum(np.ones((model.par.simN,model.par.T)),axis=1)-1)+20#age of hh  
calendar_year=age-age_initial[:,None]+year[:,None]

policy=np.maximum(calendar_year[:,0],2007)
age_policy=np.array(np.where(policy[:,None]==calendar_year)[1],dtype=np.int32)

###################
#Pre reform model
####################
#tic=time.time()


#Solve the model at baseline
M = model.copy(name='numba_new_copy')    
M.solve() 
M.simulate() 



#Sample 
sample= (age>age_initial[:,None]) & (age<=age_final[:,None]) & (M.sim.couple_lag==1) #& (M.sim.couple==1) 

sample1  = np.roll(sample,1,axis=1)
sample2  = np.roll(sample,2,axis=1)
sample_1 = np.roll(sample,-1,axis=1)

# Select different samples depending of WLP and being in a couple
sm   = (M.sim.couple[sample1]==1) & (M.sim.couple[sample]==1)

# Changes in income one period ahed (t+1) (husband m, wife w and total)
ΔYm    = np.log(M.sim.incmg[sample1]/M.sim.incmg[sample]) # husband gross income
ΔYw    = np.log(M.sim.incwg[sample1]/M.sim.incwg[sample]) # wife gross income


# Change in WLP
ΔWLP=np.array([(M.par.grid_wlp[M.sim.WLP][sample1]>0)],dtype=np.float64)[0]-np.array([(M.par.grid_wlp[M.sim.WLP][sample]>0)],dtype=np.float64)[0]

##########################################
# Compact OLS regression
##########################################
def ols(indep,dep,cond,take=1,cov=None):
    """
    Perform Ordinary Least Squares (OLS) regression on a filtered subsample of 1D arrays.
    
    Parameters:
    -----------
    indep : ndarray
        Main independent variable (1D array).
    dep : ndarray
        Dependent variable (1D array).
    cond : ndarray (boolean)
        Boolean array used to select observations (e.g. based on a condition).
    take : int, optional (default=1)
        Index of the coefficient to return. 
        - 0 corresponds to the intercept,
        - 1 corresponds to the main independent variable,
        - 2+ are for additional covariates (if any).
    cov : list of ndarray, optional
        List of additional 1D covariate arrays to include in the regression, stored in a tuple
    
    Returns:
    --------
    β : float
        The `take`-th OLS coefficient from the regression. 

    """
    
    intercept=np.ones(sample.shape)[sample][cond]
    
    X=np.hstack((intercept[:,None],indep[cond][:,None]))#explicative variables 
    
    if (cov!=None):
     for j in range(len(cov)):
         
         X=np.hstack((X,cov[j][cond][:,None]))
 
    try:  
        #OLS on untreated observations 
        β = (np.linalg.inv(X.T @ X) @ (X.T @ dep[cond].flatten()))[take]
        
    except:
        
        β=100000.0
     
    return β

# Regression:

AWE=ols(ΔYm,ΔWLP,sm)

print("Change in WLP (p.p.) given a 1% increase in men's earnings: {}".format(AWE))

##########################################################################
# Asymmetry test — two-slope spline of ΔWLP on ΔYm at ΔYm = 0
# Pooled sm sample. Wald test of H0: β_neg = β_pos.
##########################################################################
from scipy.stats import chi2 as _chi2

ΔYm_neg = ΔYm * (ΔYm <  0.0)
ΔYm_pos = ΔYm * (ΔYm >  0.0)

X_a = np.column_stack([np.ones(int(sm.sum())),
                       ΔYm_neg[sm],
                       ΔYm_pos[sm]])
y_a = ΔWLP[sm]

b_a, *_ = np.linalg.lstsq(X_a, y_a, rcond=None)
β_neg, β_pos = b_a[1], b_a[2]

resid_a = y_a - X_a @ b_a
n_a, k_a = X_a.shape
σ2_a = (resid_a @ resid_a) / (n_a - k_a)
V_a  = σ2_a * np.linalg.inv(X_a.T @ X_a)
se_neg, se_pos = float(np.sqrt(V_a[1, 1])), float(np.sqrt(V_a[2, 2]))

# Wald test:  c'b = β_neg − β_pos = 0
c_vec   = np.array([0.0, 1.0, -1.0])
diff    = c_vec @ b_a
var_d   = c_vec @ V_a @ c_vec
wald    = float(diff**2 / var_d)
p_value = float(1.0 - _chi2.cdf(wald, df=1))

print("\n--- Asymmetry test: two-slope spline of ΔWLP on ΔYm at ΔYm = 0 ---")
print("  Pooled stay-married sample (n={})".format(int(sm.sum())))
print("  β_neg  (slope when Δlog y_m < 0)  : {:+.4f}   (s.e. {:.4f})".format(β_neg, se_neg))
print("  β_pos  (slope when Δlog y_m >0)  : {:+.4f}   (s.e. {:.4f})".format(β_pos, se_pos))
print("  Wald test  H0: β_neg = β_pos      : χ²(1) = {:.3f},   p-value = {:.4f}"
      .format(wald, p_value))

##########################################################################
# Diagnostic plots: distribution of ΔYm, ΔWLP, and the joint scatter
##########################################################################

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# (a) Histogram of ΔYm on the sm sample
axes[0].hist(ΔYm[sm], bins=60, edgecolor='black', alpha=0.7)
axes[0].axvline(0.0, color='red', linestyle='--', linewidth=1)
axes[0].set_xlabel(r"$\Delta \log y_m$ (husband gross earnings)")
axes[0].set_ylabel("Frequency")
axes[0].set_title(r"Distribution of $\Delta \log y_m$ (n={})".format(int(sm.sum())))

# (b) Histogram of ΔWLP on the sm sample (values in {-1, 0, 1})
axes[1].hist(ΔWLP[sm], bins=[-1.5, -0.5, 0.5, 1.5], edgecolor='black',
             alpha=0.7, rwidth=0.8)
axes[1].set_xticks([-1, 0, 1])
axes[1].set_xticklabels(["exit (-1)", "no change (0)", "entry (+1)"])
axes[1].set_xlabel(r"$\Delta\,\mathbf{1}\{\mathrm{WLP}>0\}$")
axes[1].set_ylabel("Frequency")
axes[1].set_title(r"Distribution of $\Delta$WLP")

# (c) Scatter ΔWLP vs ΔYm with OLS line, on the sm sample
x = ΔYm[sm];  y = ΔWLP[sm]
axes[2].scatter(x, y, s=4, alpha=0.15)
xs = np.linspace(x.min(), x.max(), 100)
# Use the AWE coefficient (slope) and reconstruct the intercept from the means
intercept = y.mean() - AWE * x.mean()
axes[2].plot(xs, intercept + AWE * xs, color='red', linewidth=1.5,
             label=r"OLS fit ($\hat\beta$={:.3f})".format(AWE))
axes[2].axhline(0.0, color='gray', linewidth=0.5)
axes[2].axvline(0.0, color='gray', linewidth=0.5)
axes[2].set_xlabel(r"$\Delta \log y_m$")
axes[2].set_ylabel(r"$\Delta$WLP")
axes[2].set_title(r"$\Delta$WLP vs $\Delta \log y_m$ (sm sample)")
axes[2].legend()

plt.tight_layout()
plt.show()

##########################################################################
# Binscatter — mean ΔWLP per decile of ΔYm (non-parametric look at asymmetry)
##########################################################################
n_decile = 20
edges    = np.quantile(ΔYm[sm], np.linspace(0.0, 1.0, n_decile + 1))
edges    = np.unique(edges)            # avoid duplicate edges if shocks tie
n_bins   = len(edges) - 1

x_bin = np.full(n_bins, np.nan)
y_bin = np.full(n_bins, np.nan)
for k in range(n_bins):
    upper = (ΔYm <= edges[k+1]) if k == n_bins - 1 else (ΔYm < edges[k+1])
    mask_k = sm & (ΔYm >= edges[k]) & upper
    if mask_k.sum() > 0:
        x_bin[k] = ΔYm[mask_k].mean()
        y_bin[k] = ΔWLP[mask_k].mean()

fig_bin, ax_bin = plt.subplots(figsize=(7, 5))
ax_bin.plot(x_bin, y_bin, marker='o', linewidth=2, color='steelblue',
            label="Mean ΔWLP per decile")
ax_bin.axhline(0.0, color='gray', linewidth=0.5)
ax_bin.axvline(0.0, color='gray', linewidth=0.5)
ax_bin.set_xlabel(r"$\Delta \log y_m$ (decile mean)")
ax_bin.set_ylabel(r"Mean $\Delta$WLP")
ax_bin.set_title(r"Binscatter: $\Delta$WLP vs $\Delta \log y_m$ (sm sample)")
ax_bin.legend()
plt.tight_layout()
plt.show()

##########################################################################
# Added Worker Effect (AWE) vs Subtracted Worker Effect (SWE)
# -----------------------------------------------------------------------
# Split the sample by the wife's labor force status in the *baseline* period t:
#   - AWE  : wives who were OUT of the labor force at t  (WLP_t == 0)
#            ΔWLP can only be 0 or +1   ->  measures probability of ENTRY
#   - SWE  : wives who were WORKING at t                  (WLP_t  > 0)
#            ΔWLP can only be 0 or -1   ->  measures probability of EXIT
##########################################################################

# Wife's participation status at baseline t (using the sample mask)
working_t   = (M.par.grid_wlp[M.sim.WLP][sample] > 0)

# Conditioning sets: stay married AND wife not-working / working at t
sm_outoflf  = sm & (~working_t)   # AWE sample
sm_working  = sm & ( working_t)   # SWE sample

# OLS coefficients (pp change in P(participate) per 1% change in husband earnings)
AWE_entry = ols(ΔYm, ΔWLP, sm_outoflf)   # expected sign: negative
                                          # husband earnings up -> less likely to enter
SWE_exit  = ols(ΔYm, ΔWLP, sm_working)    # expected sign: negative
                                          # husband earnings up -> more likely to exit

print("\n--- Added vs. Subtracted Worker Effect ---")
print("AWE (women OLF at t, n={}):  ΔP(enter) per 1% Δy_m = {:.4f} pp"
      .format(int(sm_outoflf.sum()), AWE_entry))
print("SWE (women working at t, n={}): ΔP(stay)  per 1% Δy_m = {:.4f} pp"
      .format(int(sm_working.sum()), SWE_exit))
print("Interpretation (timing: status measured in t-1, change from t-1 to t):")
print("  AWE: a 1% rise in husband's earnings changes the probability that a")
print("       wife who was OLF at t-1 ENTERS the labor force at t by {:.4f} pp.".format(AWE_entry))
print("  SWE: a 1% rise in husband's earnings changes the probability that a")
print("       wife who was WORKING at t-1 EXITS the labor force at t by {:.4f} pp"
      .format(-SWE_exit))
print("       (note: -SWE_exit because ΔWLP=-1 on exit; positive value = more exits).")


##########################################################################
# Asymmetry test by t-1 LF status:
# Two-slope spline of ΔWLP on ΔYm at ΔYm = 0,
# separately on the AWE sample (OLF at t-1) and the SWE sample (working at t-1).
##########################################################################
def spline_asymmetry(mask, label):
    X = np.column_stack([np.ones(int(mask.sum())),
                         ΔYm_neg[mask],
                         ΔYm_pos[mask]])
    y = ΔWLP[mask]
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    βn, βp = b[1], b[2]
    res = y - X @ b
    n_, k_ = X.shape
    σ2  = (res @ res) / (n_ - k_)
    V   = σ2 * np.linalg.inv(X.T @ X)
    se_n, se_p = float(np.sqrt(V[1, 1])), float(np.sqrt(V[2, 2]))
    c   = np.array([0.0, 1.0, -1.0])
    diff = c @ b
    var_d = c @ V @ c
    w   = float(diff**2 / var_d)
    pv  = float(1.0 - _chi2.cdf(w, df=1))
    print("  [{}]  n={}".format(label, n_))
    print("    β_neg (Δlog y_m < 0) : {:+.4f}   (s.e. {:.4f})".format(βn, se_n))
    print("    β_pos (Δlog y_m ≥ 0) : {:+.4f}   (s.e. {:.4f})".format(βp, se_p))
    print("    Wald  H0: β_neg=β_pos: χ²(1) = {:.3f},  p = {:.4f}".format(w, pv))
    return βn, βp

print("\n--- Asymmetry of AWE / SWE: two-slope spline by t-1 LF status ---")
spline_asymmetry(sm_outoflf, "AWE  — OLF at t-1, only entries possible (ΔWLP ∈ {0,+1})")
spline_asymmetry(sm_working, "SWE  — working at t-1, only exits possible (ΔWLP ∈ {-1,0})")


##########################################################################
# Structural decomposition of WLP response into persistent vs transitory shocks
##########################################################################

# 1. Recover gender-specific composite shock indices
#    Couple index built as iz = iz_w * num_zm + iz_m  (cf. UserFunctions_numba.labor_income)
iz_full  = M.sim.iz
iz_w_idx = iz_full // M.par.num_zm   # wife composite shock index   ∈ [0, num_zw)
iz_m_idx = iz_full %  M.par.num_zm   # husband composite shock index ∈ [0, num_zm)

# 2. par.grid_pw / par.grid_ϵw store XPw / XTw with shape (T, num_perdiv, num_zw, num_h);
#    values do not vary across the (iD, ih) axes, so we pick any slice.
logP_w_grid = np.asarray(M.par.grid_pw)[:, 0, :, 0]   # (T, num_zw)  — log persistent, wife
logϵ_w_grid = np.asarray(M.par.grid_ϵw)[:, 0, :, 0]   # (T, num_zw)  — log transitory, wife
logP_m_grid = np.asarray(M.par.grid_pm)[:, 0, :, 0]   # (T, num_zm)  — log persistent, husband
logϵ_m_grid = np.asarray(M.par.grid_ϵm)[:, 0, :, 0]   # (T, num_zm)  — log transitory, husband

T_idx = np.tile(np.arange(M.par.T), (M.par.simN, 1))   # (N, T)

logPw = logP_w_grid[T_idx, iz_w_idx]   # (N, T)
logϵw = logϵ_w_grid[T_idx, iz_w_idx]
logPm = logP_m_grid[T_idx, iz_m_idx]
logϵm = logϵ_m_grid[T_idx, iz_m_idx]

# 3. Period-over-period changes (t -> t+1) in each component (1-D, aligned with ΔYm/ΔYw/ΔWLP)
ΔlogPw = logPw[sample1] - logPw[sample]
Δlogϵw = logϵw[sample1] - logϵw[sample]
ΔlogPm = logPm[sample1] - logPm[sample]
Δlogϵm = logϵm[sample1] - logϵm[sample]

# (1) WLP response to HUSBAND'S persistent vs transitory shocks
β_perm_h_WLP = ols(ΔlogPm, ΔWLP, sm)
β_tran_h_WLP = ols(Δlogϵm, ΔWLP, sm)

# (2) WLP response to WIFE'S OWN persistent vs transitory shocks
#     (no need to condition on working: structural shocks are observed regardless of WLP)
β_perm_w_WLP = ols(ΔlogPw, ΔWLP, sm)
β_tran_w_WLP = ols(Δlogϵw, ΔWLP, sm)

print("\n--- Structural decomposition of ΔWLP ---")
print("Husband shock response (n={}):".format(int(sm.sum())))
print("  persistent  P_m : {:+.4f} pp per 1 log-point of Δlog P_m".format(β_perm_h_WLP))
print("  transitory  ε_m : {:+.4f} pp per 1 log-point of Δlog ε_m".format(β_tran_h_WLP))
print("Own shock response (n={}):".format(int(sm.sum())))
print("  persistent  P_w : {:+.4f} pp per 1 log-point of Δlog P_w".format(β_perm_w_WLP))
print("  transitory  ε_w : {:+.4f} pp per 1 log-point of Δlog ε_w".format(β_tran_w_WLP))


##########################################################################
# Home production response to husband's earnings
# ------------------------------------------------------------------------
# (i)   d   = home inputs (money) -> M.sim.dw  (couples: produces with d/px)
# (ii)  Q   = home good (output)  -> Q = home_time^(1-ν) * (d/px)^ν       , couples
#            with home_time = 2*ϕ + ishom*(1-ϕ),  ishom = 1 - wlp (or 2 if retired)
# (iii) U_Q = utility from Q       -> α * Q^(1-χ) / (1-χ)
##########################################################################

ϕ_p, ν_p, α_p, χ_p, px_p = M.par.ϕ, M.par.ν, M.par.α, M.par.χ, M.par.px

wlp_val = M.par.grid_wlp[M.sim.WLP]                        # (N, T) — fraction worked
ret_t   = (np.arange(M.par.T) >= M.par.Tr)                  # (T,)   — retirement flag

# ishom passed to util: 1 - wlp before retirement, 2 after (matches UserFunctions_numba.py:135)
ishom         = np.where(ret_t[None, :], 2.0, 1.0 - wlp_val)
home_time_eff = 2.0*ϕ_p + ishom*(1.0 - ϕ_p)                 # (N, T)

d   = M.sim.dw                                              # (N, T) — home-input expenditure
Q   = home_time_eff**(1.0 - ν_p) * (d / px_p)**ν_p          # (N, T) — home good
U_Q = α_p * Q**(1.0 - χ_p) / (1.0 - χ_p)                    # (N, T) — utility from Q

# Period-over-period changes on the sm subsample (aligned with ΔYm)
Δlogd     = np.log(d[sample1] / d[sample])
ΔlogQ     = np.log(Q[sample1] / Q[sample])
Δloghome_t= np.log(home_time_eff[sample1] / home_time_eff[sample])
ΔU_Q      = U_Q[sample1] - U_Q[sample]                      # level (logging would fail since U_Q<0 when χ>1)

# Elasticities / response slopes w.r.t. husband's log earnings change
β_d_ym    = ols(ΔYm, Δlogd,     sm)
β_t_ym    = ols(ΔYm, Δloghome_t, sm)
β_Q_ym    = ols(ΔYm, ΔlogQ,     sm)
β_U_ym    = ols(ΔYm, ΔU_Q,      sm)

print("\n--- Home production response to husband earnings ---")
print("Sample: stay-married (n={})".format(int(sm.sum())))
print("  (i)   Elasticity of home INPUTS  d         w.r.t. Δlog y_m :  {:+.4f}".format(β_d_ym))
print("  (ii)  Elasticity of home TIME    home_time w.r.t. Δlog y_m :  {:+.4f}".format(β_t_ym))
print("  (iii) Elasticity of home GOOD    Q         w.r.t. Δlog y_m :  {:+.4f}".format(β_Q_ym))
print("  (iv)  Δ(α·Q^(1-χ)/(1-χ)) per log-pt of Δlog y_m             :  {:+.4f} util / log-pt"
      .format(β_U_ym))


##########################################################################
# Same four home-production elasticities, split by wife's WLP at t-1
#   - sm_outoflf : wife OLF at t-1 (would-be entrants if she switches)
#   - sm_working : wife working at t-1 (would-be exiters if she switches)
# (sm_outoflf and working_t are already defined in the AWE/SWE block above)
##########################################################################

# OLF at t-1
β_d_ym_olf = ols(ΔYm, Δlogd,     sm_outoflf)
β_t_ym_olf = ols(ΔYm, Δloghome_t, sm_outoflf)
β_Q_ym_olf = ols(ΔYm, ΔlogQ,     sm_outoflf)
β_U_ym_olf = ols(ΔYm, ΔU_Q,      sm_outoflf)

# Working at t-1
β_d_ym_w   = ols(ΔYm, Δlogd,     sm_working)
β_t_ym_w   = ols(ΔYm, Δloghome_t, sm_working)
β_Q_ym_w   = ols(ΔYm, ΔlogQ,     sm_working)
β_U_ym_w   = ols(ΔYm, ΔU_Q,      sm_working)

print("\n--- Home production response, by wife's labor status at t-1 ---")
print("                                          OLF at t-1   |  Working at t-1")
print("                                          (n={:>5})    |  (n={:>5})"
      .format(int(sm_outoflf.sum()), int(sm_working.sum())))
print("  (i)   Elasticity of home INPUTS  d   :  {:+.4f}      |  {:+.4f}"
      .format(β_d_ym_olf, β_d_ym_w))
print("  (ii)  Elasticity of home TIME    h   :  {:+.4f}      |  {:+.4f}"
      .format(β_t_ym_olf, β_t_ym_w))
print("  (iii) Elasticity of home GOOD    Q   :  {:+.4f}      |  {:+.4f}"
      .format(β_Q_ym_olf, β_Q_ym_w))
print("  (iv)  Δ(α·Q^(1-χ)/(1-χ)) / Δlog y_m  :  {:+.4f}      |  {:+.4f}"
      .format(β_U_ym_olf, β_U_ym_w))


# %%
