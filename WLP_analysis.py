# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 11:31:42 2026

@author: 32489
"""


import numpy as np 
import Bargaining_numba as brg  
import UserFunctions_numba as usr 
import pandas as pd
import getpass
import statsmodels.api as sm 


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
indexes=np.array(np.random.choice(baseline_sample[:,0], size=N, p=pr, replace=True),dtype=np.int32)-1
final_sample= baseline_sample[:,1:][indexes] 

age_initial=final_sample[:,0]
age_final=final_sample[:,1]
cw_cons_share=final_sample[:,2]
h_income=final_sample[:,3]
w_income=final_sample[:,4]
age_marriage=final_sample[:,5]
year=final_sample[:,6]
assets=final_sample[:,7]*np.mean(np.exp(h_income))



# Guess of internal parameters: [ν,σL,α,ρ,wedge,β]

xc=np.array([0.53315238, 0.1      , 0.80884379, 1.15375904, 0.915     ,1.0    ])


# Lower and higher bounds of parameters
xl=np.array([0.00001,0.000082,0.1,0.5,0.01,0.9]) 
xu=np.array([0.8,0.4,0.999,2.5,1.0,1.1]) 

#Parametrize the model 
par = {'simN':N,'ν': xc[0],'σL':xc[1],'α':xc[2],'ρ':xc[3],'wedge':xc[4],'sample_init':np.array(age_marriage-20,dtype=np.int_)}
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

izm=np.array([np.argmin(np.abs(np.log(gridzm)[int(model.par.sample_init[i]),0,:,0]-h_income[i])) for i in range(model.par.simN)],dtype=np.int32)
izm[np.isnan(h_income)]=(model.par.num_pm*model.par.num_ϵm)//2
izw=np.array([np.argmin(np.abs(np.log(gridzw)[int(model.par.sample_init[i]),0,:,0]-w_income[i])) for i in range(model.par.simN)],dtype=np.int32)
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

print("Change in WLP given a 1% increase in men's earnings: {}".format(AWE))
    
 