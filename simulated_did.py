# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 14:16:12 2026

@author: 32489

Solution, simulation and DiD
"""


import numpy as np 
import Bargaining_numba as brg  
import UserFunctions_numba as usr 
import pandas as pd
import getpass


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


# Guess of internal parameters: [ω,σL,α,ρ,wedge,β]
pt=np.array([0.55, 0.1       , 0.85, 1.2, 0.929     ,1.        ])

#Parametrize the model
par = {'simN':N,'ω': pt[0],'σL':pt[1],'α':pt[2],'ρ':pt[3],'wedge':pt[4],'sample_init':np.array(np.maximum(age_marriage-20,0),dtype=np.int_)}
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


########################################################################################################################################
#Solve and simulate the model for given parametrization, then compute moments and their distance distance from the data
#######################################################################################################################################

###################
#Pre reform model
####################

# Set up the model with the input parameters pt
M_bef = model.copy(name='numba_new_copy')
   
M_bef.par.ω=pt[0]
M_bef.par.grid_lovew,M_bef.par.Πlw,M_bef.par.Πlw0= usr.rouw_nonst(M_bef.par.T,pt[1],pt[1]*0+0.03,M_bef.par.num_lovew) 
M_bef.par.grid_lovem,M_bef.par.Πlm,M_bef.par.Πlm0= usr.rouw_nonst(M_bef.par.T,pt[1],pt[1]*0+0.03,M_bef.par.num_lovem) 


M_bef.par.Πl=[np.kron(M_bef.par.Πlw[t],M_bef.par.Πlm[t]) for t in range(M_bef.par.T-1)] # couples trans matrix
M_bef.par.Πl0=[np.kron(M_bef.par.Πlw0[t],M_bef.par.Πlm0[t]) for t in range(M_bef.par.T-1)] # couples trans matrix 
M_bef.par.α=pt[2] 
M_bef.par.ρ=pt[3]
M_bef.par.wedge=pt[4]        

# Solve and simulate the model
M_bef.solve() 
M_bef.simulate() 



###################
#After reform model
####################

M = M_bef.copy(name='numba_new_copy')  
M.par.pens_reform=True #set up pension reform
M.par.policy_init=age_policy
M.par.ω=pt[0]
M.par.grid_lovew,M.par.Πlw,M.par.Πlw0= usr.rouw_nonst(M.par.T,pt[1],pt[1]*0+0.03,M.par.num_lovew) 
M.par.grid_lovem,M.par.Πlm,M.par.Πlw0= usr.rouw_nonst(M.par.T,pt[1],pt[1]*0+0.03,M.par.num_lovem) 


M.par.Πl=[np.kron(M.par.Πlw[t],M.par.Πlm[t]) for t in range(M.par.T-1)] # couples trans matrix 
M.par.Πl0=[np.kron(M.par.Πlw0[t],M.par.Πlm0[t]) for t in range(M.par.T-1)] # couples trans matrix 

# Income shocks grids: couples
M.par.grid_zw,M.par.grid_ϵw,M.par.grid_pw,M.par.Π_zw0, \
    M.par.grid_zm,M.par.grid_ϵm,M.par.grid_pm,M.par.Π_zm0, \
                                M.par.Π=usr.labor_income(M.par,pens_reform=M.par.pens_reform)     

M.par.grid_zws,M.par.grid_ϵw,M.par.grid_pw,M.par.Π_zw0, \
    M.par.grid_zms,M.par.grid_ϵm,M.par.grid_pm,M.par.Π_zm0, \
                                        M.par.Πs=usr.labor_income(M.par,single=True,pens_reform=M.par.pens_reform) 
                                        
M.par.α=pt[2] 
M.par.ρ=pt[3]
M.par.wedge=pt[4]        

# Solve and simulate the model
M.solve() 
M.simulate() 


#############################################
#DiD below
############################################

"""

Choose MM=M if want simulated data where reform was introduced,
otherwise choose MM=M_bef if you want simulated data where
the reform was never introduced

"""

#Choose your model
MM = M.copy()

#Time to policy event
event_time= (np.cumsum(np.ones((MM.par.simN,MM.par.T)),axis=1)-1)-MM.par.policy_init[:,None]

#Individual fixed effects
idd=np.repeat(np.cumsum(np.ones(MM.par.simN))[:,None],MM.par.T,axis=1)

#Age at marriage
agei=np.repeat((age_marriage)[:,None],MM.par.T,axis=1) 

#Age at which policy started
agepolicy= np.repeat((MM.par.policy_init)[:,None],MM.par.T,axis=1) 

#Assets, productivity index and bargaining power at marriage
assetsi=np.repeat((MM.sim.init_A)[:,None],MM.par.T,axis=1) 
izi=np.repeat((MM.sim.init_z)[:,None],MM.par.T,axis=1) 
poweri=np.repeat((param)[:,None],MM.par.T,axis=1) 

#Bargaining power
power=MM.sim.power

#Wife consumption share and ratio
wife_share=MM.sim.Cw/(MM.sim.Cw+MM.sim.Cm)
wife_ratio=MM.sim.Cw/(MM.sim.Cm)

#Actual income of wife and husband
incw_a=MM.sim.incw
incm_a=MM.sim.incm 

#Potential income of wife and husband
incm_p=MM.par.grid_zm[np.arange(MM.par.T),MM.sim.ID,MM.sim.iz,MM.sim.ih]
incw_p=MM.par.grid_zm[np.arange(MM.par.T),MM.sim.ID,MM.sim.iz,MM.sim.ih]

#Marriage surplus of wife and husband
Sw=MM.sim.Vcw-MM.sim.Vsw
Sm=MM.sim.Vcm-MM.sim.Vsm

#Time elaspsed since marriage
time=age-agei

# Combine into a DataFrame 
df = pd.DataFrame({ 
    "assetsi":assetsi.flatten(),
    "agei":agei.flatten(),
    "poweri":poweri.flatten(),
    "power":power.flatten(),
    "time":time.flatten(),
    "age_policy":agepolicy.flatten(),
    "wife_share":wife_share.flatten(), 
    "wife_ratio":wife_ratio.flatten(), 
    "event_time":event_time.flatten(), 
    "income_wife":incw_a.flatten(),
    "income_husband":incm_a.flatten(),
    "income_wife_potential":incw_p.flatten(),
    "income_husband_potential":incm_p.flatten(),
    "surplus_wife":Sw.flatten(),
    "surplus_husband":Sm.flatten(),
    "izi":izi.flatten(),
    "idd":idd.flatten(), 
    "age":age.flatten()
}) 
 
#Save to stata
df.to_stata('simulated_did_mod.dta')