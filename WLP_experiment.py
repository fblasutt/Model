# -*- coding: utf-8 -*- 
""" 
Created on Fri Feb 23 15:31:56 2024 
 
@author: 32489 
""" 
 
import numpy as np 
import Bargaining_numba as brg  
import UserFunctions_numba as usr 
import pandas as pd
from reg_cons_insurance import insurance
import matplotlib.pyplot as plt
 
#Initialize seed 
np.random.seed(10) 
 
#Root
root='C:/Users/32489/Dropbox/Family Risk Sharing'


#Create sample with replacement 
N=10_000#sample size 


#Import information for the sample, then store the relevant variables
baseline_sample=np.array(pd.read_excel(root+'/Output files/data_sample.csv'))

pr=np.ones(baseline_sample.shape[0])/baseline_sample.shape[0]
indexes=np.array(np.random.choice(baseline_sample[:,0], size=N, p=pr, replace=True),dtype=np.int32)-1
final_sample= baseline_sample[:,1:][indexes] 

age_initial=final_sample[:,0]
age_final=final_sample[:,1]
cw_cons_share=final_sample[:,2]
h_income=final_sample[:,3]
w_income=final_sample[:,4]
age_marriage=final_sample[:,5]


# Parametrization: [ν,σL,α,χ,wedge]
xc=np.array([0.35870364, 0.00596541, 0.89223739, 0.96875457, 0.71759143])
#Parametrize the model 
par = {'simN':N,'ν': xc[0],'σL':xc[1],'α':xc[2],'χ':xc[3],'wedge':xc[4]} 
model = brg.HouseholdModelClass(par=par)  



#####################################################################
# Set the initial conditions for the couples based on baseline sample
#####################################################################

#We start simulating the agent at age_initial
model.par.sample_init=age_initial-20

#Given the parameters, set the initial pareto weight for couples
param=(cw_cons_share/(1.0-cw_cons_share))**model.par.ρ
model.sim.init_power=param/(1.0+param)

#Set the initial income gridpoints for income, the closest to our value
izm=np.array([np.argmin(np.abs(np.log(model.par.grid_zm)[int(model.par.sample_init[i]),:,0]-h_income[i])) for i in range(model.par.simN)],dtype=np.int32)
izm[np.isnan(h_income)]=(model.par.num_pm*model.par.num_ϵm)//2
izw=np.array([np.argmin(np.abs(np.log(model.par.grid_zw)[int(model.par.sample_init[i]),:,0]-w_income[i])) for i in range(model.par.simN)],dtype=np.int32)
izw[np.isnan(w_income)]=(model.par.num_pw*model.par.num_ϵw)//2     
model.sim.init_z=izm*model.par.num_zm+izw


############################################################################
# Solve the model for different gender wage gap, with and without commitment
############################################################################

#Values of the gender wage gap we consider
gridτ=np.linspace(model.par.ι0w,model.par.ι0m,3)#gender wage gap grid


#These lists will store all them models we solve and simulate
Bmodel=list()#limited commitment
Bfmodel=list()#full commitment


#Loop over gender wage gap grid and solve the model - limited commitment
for i in range(len(gridτ)):

    # Set up the model - limited commitment
    M = model.copy(name='numba_new_copy')    
    M.par.ι0w=gridτ[i]    
    
    # income shocks grids: singles and couples
    M.par.grid_zw,M.par.grid_ϵw,M.par.grid_pw,M.par.Π_zw0, \
        M.par.grid_zm,M.par.grid_ϵm,M.par.grid_pm,M.par.Π_zm0, \
                                    M.par.Π=usr.labor_income(M.par) 
                                    
                                    
    # income shocks grids: singles and couples
    M.par.grid_zw,M.par.grid_ϵw,M.par.grid_pw,M.par.Π_zw0, \
        M.par.grid_zm,M.par.grid_ϵm,M.par.grid_pm,M.par.Π_zm0, \
                                            M.par.Πs=usr.labor_income(M.par,single=True) 
    M.solve() 
    M.simulate()  
    Bmodel.append(M)
 
#Loop over gender wage gap grid and solve the model - full commitment
for i in range(len(gridτ)):
    
    # Set up the model - full
    Mf = model.copy(name='numba_new_copy')    
    Mf.par.ι0w=gridτ[i]    
    Mf.par.full=True
    
        
    # income shocks grids: singles and couples
    Mf.par.grid_zw,Mf.par.grid_ϵw,Mf.par.grid_pw,Mf.par.Π_zw0, \
        Mf.par.grid_zm,Mf.par.grid_ϵm,Mf.par.grid_pm,Mf.par.Π_zm0, \
                                    Mf.par.Π=usr.labor_income(Mf.par) 
                                    
                                    
    # income shocks grids: singles and couples
    Mf.par.grid_zw,Mf.par.grid_ϵw,Mf.par.grid_pw,Mf.par.Π_zw0, \
        Mf.par.grid_zm,Mf.par.grid_ϵm,Mf.par.grid_pm,Mf.par.Π_zm0, \
                                            Mf.par.Πs=usr.labor_income(Mf.par,single=True) 

    Mf.solve() 
    Mf.simulate()    
    Bfmodel.append(Mf)


#########################################
#Sample selection for Insurance analysis
########################################

#We take individuals that stays married across spefifications
age=(np.cumsum(np.ones((M.par.simN,M.par.T)),axis=1)-1)+20#age of hh   

alwayscouple=np.array([(Bmodel[i].sim.couple_lag==1) & (Bfmodel[i].sim.couple_lag==1)  for i in range(len(gridτ))])
alwayscouplep=np.array([(Bmodel[i].sim.couple==1) & (Bfmodel[i].sim.couple==1)  for i in range(len(gridτ))])

sample =  (age>age_initial[:,None]) & (age<=age_final[:,None]) & (alwayscouple.min(axis=0)) & (alwayscouplep.min(axis=0))
sample1=np.roll(sample,1,axis=1)
    


#########################################
#Insurance analysis
########################################

#Lists with results
Bgrid=list()#limited commitment
Bfgrid=list()#full commitment

#Names of the file and of the table line associated with a model version (if gridτ has len()>3, names should be adapted)
Names=['Baseline', 'LowGG1', 'LowGG2']
Names_line=['Baseline', 'Low GWG', 'No GWG']

#Obtain pass-throughs and do the decomposition calling function insurance
for i in range(len(gridτ)):


    B=insurance(Bmodel[i],sample,
                shock_type='permanent',
                shock_gender='Male',
                consumption_gender='Male',
                name_file=Names[i],
                name_line=Names_line[i])
                
    
    B['par']=gridτ[i]
    Bgrid.append(B)
    
    Bf=insurance(Bfmodel[i],sample,                
                 shock_gender='Male',
                 shock_type='permanent',
                 consumption_gender='Male',
                 name_file=Names[i]+'full',
                 name_line=Names_line[i])
                 
    
    Bf['par']=gridτ[i]
    Bfgrid.append(Bf)
       
    
# sh_perm=np.array([Bgrid[i]['w_sh']['per_m'] for i in range(len(gridτ))])

# perm_m_m=np.array([Bgrid[i]['indc']['per_m_m'] for i in range(len(gridτ))])
# perm_m_w=np.array([Bgrid[i]['indc']['per_m_w'] for i in range(len(gridτ))])
# perm_m_Q=np.array([Bgrid[i]['Qins']['per_m'] for i in range(len(gridτ))])

# rebargainings=np.array([((Bmodel[i].sim.power!=Bmodel[i].sim.power_lag) & (Bmodel[i].sim.power>=0.0))[sample].mean() for i in range(len(gridτ))])
# reb_size=np.array([(abs(Bmodel[i].sim.power[sample]-Bmodel[i].sim.power_lag[sample])[((Bmodel[i].sim.power[sample]!=Bmodel[i].sim.power_lag[sample]) & (Bmodel[i].sim.power[sample]>=0.0))]).mean() for i in range(len(gridτ))])

# sh_perm_f=np.array([Bfgrid[i]['w_sh']['per_m'] for i in range(len(gridτ))])

# perm_m_m_f=np.array([Bfgrid[i]['indc']['per_m_m'] for i in range(len(gridτ))])
# perm_m_w_f=np.array([Bfgrid[i]['indc']['per_m_w'] for i in range(len(gridτ))])
# perm_m_Q_f=np.array([Bfgrid[i]['Qins']['per_m'] for i in range(len(gridτ))])



# #Plot consumption insurance graph 
# plt.plot(gridτ,perm_m_m-perm_m_m[0],label="Husband, Lim.",color='blue')
# plt.plot(gridτ,perm_m_w-perm_m_w[0],label="Wife, Lim.",color='red')
# plt.plot(gridτ,perm_m_Q-perm_m_Q[0],label="Common, Lim.",color='black')


# plt.plot(gridτ,perm_m_m_f-perm_m_m_f[0],label="Husband, Full",color='blue',linestyle='--')
# plt.plot(gridτ,perm_m_w_f-perm_m_w_f[0],label="Wife, Full",color='red',linestyle='--')
# plt.plot(gridτ,perm_m_Q_f-perm_m_Q_f[0],label="Common, Full",color='black',linestyle='--')

# plt.xlabel("Deduction")  
# plt.ylabel("Δ Insurance")    
# #plt.ylim(0, 20)    
# plt.legend()                              
# #plt.savefig(root+'/Model/results/lifecycle_singlew.eps', format='eps', bbox_inches="tight")  
# plt.show()

