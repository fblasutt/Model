# -*- coding: utf-8 -*- 
""" 
Created on Fri Feb 23 15:31:56 2024 
 
@author: 32489 
""" 
 
import numpy as np 
import Bargaining_numba as brg  
import pandas as pd
from reg_cons_insurance import insurance
import matplotlib.pyplot as plt
from scipy import optimize

#Initialize seed 
np.random.seed(10) 
 
#Root
root='C:/Users/32489/Dropbox/Family Risk Sharing'




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



#target 0.065 + real pension function+love sd=0.15
xc=np.array([0.55, 0.1       , 0.85, 1.2, 0.929     ,1.        ])


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



#############################################################################
#     INITIAL INFORMATION
############################################################################

#Tax progressivity parameters to consider
gridτ=np.linspace(model.par.τ,0.3,3)

#Lists that contains model information
Bmodel=list()
Bfmodel=list()

############################################################################
# BALANCE THE GOVERNMENT BUDGET
########################################################################

#All limited commitment models should imply the same government surplus:
#First, we want to adjust tax level Λ accordingly

#age of hh
age=(np.cumsum(np.ones((model.par.simN,model.par.T)),axis=1)-1)+20#age of hh  

#discounting to check present values
discounting=(1/model.par.R)**(np.cumsum(age>=age_initial[:,None],axis=1)-1)

#Solve the model at baseline
M = model.copy(name='numba_new_copy')    
M.solve() 
M.simulate() 


#Obtain taxes
taxes_baseline=((discounting*M.sim.tax)[(age>=age_initial[:,None])]).sum()
    
#create function to find deviations from baseline budget
def budget(x,i):
    
    m=model.copy(name='numba_new_copy')    
    m.par.τ=gridτ[i]
    m.par.Λ=x
    m.solve() 
    m.simulate()
    
    print(((discounting*m.sim.tax)[(age>=age_initial[:,None])]).sum()-taxes_baseline,x)
    return ((discounting*m.sim.tax)[(age>=age_initial[:,None])]).sum()-taxes_baseline

#Find the level of Λ so that goivernment surplus is the same than at baseline
gridΛ=np.array([optimize.bisect(budget,0.87,1.03,args=(i,),xtol=0.001) for i in range(1,len(gridτ))])
gridΛ=np.append(M.par.Λ,gridΛ)
#array([0.92    , 0.946875, 0.964375])
############################################################################
#                        BALANCE THE BUDGET
########################################################################


#Loop over gender wage gap grid and solve the model - limited commitment
for i in range(len(gridτ)):

    # Set up the model - limited commitment
    M = model.copy(name='numba_new_copy')    
    M.par.τ=gridτ[i]    
    M.par.Λ=gridΛ[i]
  
    M.solve() 
    M.simulate() 
    Bmodel.append(M)
 
#Loop over gender wage gap grid and solve the model - full commitment
for i in range(len(gridτ)):
    
    # Set up the model - full commitment
    Mf = model.copy(name='numba_new_copy')    
    Mf.par.τ=gridτ[i]    
    Mf.par.Λ=gridΛ[i]
   
    Mf.par.full=True

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
Names=['Baselineprog', 'Progr1', 'Progr2']
Names_line=['Baseline', 'Higher tax progr', 'Even higher tax progr']

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
                 shock_type='permanent',
                 shock_gender='Male',
                 consumption_gender='Male',
                 name_file=Names[i]+'full',
                 name_line=Names_line[i])
                 
    
    Bf['par']=gridτ[i]
    Bfgrid.append(Bf)
    


    
# sh_perm=np.array([Bgrid[i]['w_sh']['per_m'] for i in range(len(gridτ))])

# perm_m_m=np.array([Bgrid[i]['indc']['per_m_m'] for i in range(len(gridτ))])
# perm_m_w=np.array([Bgrid[i]['indc']['per_m_w'] for i in range(len(gridτ))])
# perm_m_Q=np.array([Bgrid[i]['Qins']['per_m'] for i in range(len(gridτ))])

# rebargainings=np.array([((Bmodel[i].sim.power!=Bmodel[i].sim.power_lag) & (Bfmodel[i].sim.power>=0.0))[sample].mean() for i in range(len(gridτ))])
# reb_size=np.array([(abs(Bmodel[i].sim.power[sample]-Bmodel[i].sim.power_lag[sample])[((Bmodel[i].sim.power[sample]!=Bmodel[i].sim.power_lag[sample]) & (Bfmodel[i].sim.power[sample]>=0.0))]).mean() for i in range(len(gridτ))])


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
# #plt.ylim(0, 20)    
# plt.legend()                              
# #plt.savefig(root+'/Output files/model/lifecycle_singlew.eps', format='eps', bbox_inches="tight")  
# plt.show()

