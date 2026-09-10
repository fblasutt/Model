# -*- coding: utf-8 -*- 
""" 
Created on Fri Feb 23 15:31:56 2024 
 
@author: 32489 
""" 
 
import numpy as np 
import Bargaining_numba as brg  
import init_conditions as ic
import pandas as pd
from reg_cons_insurance import insurance, vol_stack_figure, share_var_decomposition
import matplotlib.pyplot as plt
from scipy import optimize
import UserFunctions_numba as usr 

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

age_initial=final_sample[:,0]*0+25 # forced to 25, as in calibration.py; also prevents
                                   # negative sample_init (data has age_initial=24 for ~5%)
age_final=final_sample[:,1]
cw_cons_share=final_sample[:,2]
h_income=final_sample[:,3]
w_income=final_sample[:,4]
age_marriage=final_sample[:,5]*0+25 # forced to 25, as in calibration.py
assets=final_sample[:,7]*np.mean(np.exp(h_income))

# Pre-drawn uniforms for the posterior-draw initial income split (drawn AFTER
# the sample so sample selection is unchanged; FIXED across evaluations so
# SMM objectives stay deterministic)
u_init_w=np.random.rand(N);u_init_m=np.random.rand(N)
σME2_init=0.0  # measurement-error variance in observed entry income (0 = off)


#Current estimates [η,σL,α,ρ,Ω,β] from the shared module (sync with calibration.py)
from estimated_params import xc, par_dict, apply_fc_params

#Parametrize the model (NB: position 4 is Ω, the match-quality disagreement shock)
par = par_dict(N, np.array(age_initial-25,dtype=np.int_))
model = brg.HouseholdModelClass(par=par)



#####################################################################
# Set the initial conditions for the couples based on baseline sample
#####################################################################

#Given the parameters, set the initial pareto weight for couples
param=(cw_cons_share/(1.0-cw_cons_share))**model.par.ρ
model.sim.init_power=param/(1.0+param)

#Set the initial income gridpoints for income, the closest to our value
gridzw=model.par.grid_zw[:,:,np.linspace(0,model.par.num_z-1,model.par.num_zm,dtype=np.int_)]
gridzm=model.par.grid_zm[:,:,:model.par.num_zw]

izm=ic.draw_init_iz(h_income,model.par.sample_init,gridzm,model.par.grid_pm,model.par.grid_ϵm,u_init_m,σME2=σME2_init)
izm[np.isnan(h_income)]=(model.par.num_pm*model.par.num_ϵm)//2
izw=ic.draw_init_iz(w_income,model.par.sample_init,gridzw,model.par.grid_pw,model.par.grid_ϵw,u_init_w,σME2=σME2_init)
izw[np.isnan(w_income)]=(model.par.num_pw*model.par.num_ϵw)//2
model.sim.init_z=izw*model.par.num_zm+izm   # FIXED gender swap: wife is the SLOW joint-index component
model.sim.init_A=assets





############################################################################
# Solve the model for different gender wage gap, with and without commitment
############################################################################

#Values of the gender wage gap we consider
gridτ=np.linspace(model.par.ι0w,model.par.ι0m,3)#gender wage gap grid


#This list will store all the models we solve and simulate (LC only)
Bmodel=list()#limited commitment


#Loop over gender wage gap grid and solve the model - limited commitment
for i in range(len(gridτ)):

    # Set up the model - limited commitment
    M = model.copy(name='numba_new_copy')    
    M.par.ι0w=gridτ[i]    
    
    # income shocks grids: singles and couples
    M.par.grid_zw,M.par.grid_ϵw,M.par.grid_pw,M.par.Π_zw0, \
        M.par.grid_zm,M.par.grid_ϵm,M.par.grid_pm,M.par.Π_zm0, \
                                    M.par.Π=usr.labor_income(M.par) 
                                    
                                    
    # income shocks grids: SINGLES (grid_zws/grid_zms; do not overwrite couples' grids)
    M.par.grid_zws,M.par.grid_ϵw,M.par.grid_pw,M.par.Π_zw0, \
        M.par.grid_zms,M.par.grid_ϵm,M.par.grid_pm,M.par.Π_zm0, \
                                            M.par.Πs=usr.labor_income(M.par,single=True) 
    M.solve()

    # The initial love distribution is drawn ONCE, in the BASELINE LC model
    # (i=0, gridτ[0]=baseline ι0w), then held FIXED across every other version
    # (other gap levels and all FC models): comparisons are not contaminated
    # by initial-composition differences.
    if i==0:
        M.simulate()
        init_love_base = np.array(
            [M.sim.love[k, int(M.par.sample_init[k])] for k in range(M.par.simN)],
            dtype=np.int_,
        )
    else:
        M.sim.force_init_love[:] = init_love_base
        M._init_love_rationalized=True
        M.simulate()

    Bmodel.append(M)
 
#########################################
#Sample selection for Insurance analysis
########################################

#Common age window; the married-couple requirement is MODEL-SPECIFIC (each
#policy variant's statistics use its own surviving couples, built in-loop)
age=(np.cumsum(np.ones((M.par.simN,M.par.T)),axis=1)-1)+25#age of hh
    


#########################################
#Insurance analysis
########################################

#Lists with results (LC only; male-shock and female-shock decompositions)
Bgrid=list()   # permanent MALE shock   -> HIS consumption
Bwgrid=list()  # permanent FEMALE shock -> HER consumption
samples=list() # per-variant samples (for the volatility figure)

#Names of the file and of the table line associated with a model version (if gridτ has len()>3, names should be adapted)
Names=['Baseline', 'LowGG1', 'LowGG2']
Names_line=['Baseline', 'Lower gender gap', 'No gender gap']

#Obtain pass-throughs and do the decomposition calling function insurance
for i in range(len(gridτ)):

    sample = (age>age_initial[:,None]) & (age<=age_final[:,None]) & (Bmodel[i].sim.couple_lag==1) &  (Bmodel[i].sim.couple==1) 
    samples.append(sample)

    B=insurance(Bmodel[i],sample,
                shock_type='permanent',
                shock_gender='Male',
                consumption_gender='Male',
                name_file=Names[i],
                name_line=Names_line[i])

    B['par']=gridτ[i]
    Bgrid.append(B)

    Bw=insurance(Bmodel[i],sample,
                 shock_type='permanent',
                 shock_gender='Female',
                 consumption_gender='Female',
                 name_file=Names[i]+'w',
                 name_line=Names_line[i])

    Bw['par']=gridτ[i]
    Bwgrid.append(Bw)
       
    
#########################################
# Private-consumption volatility figure: one stacked bar per variant x
# spouse (household part + share component + 2Cov; raw variances)
########################################
vol_stack_figure(Bmodel, samples, Names_line, 'volbars_gwg')


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
# #plt.savefig(root+'/Output files/model/lifecycle_singlew.eps', format='eps', bbox_inches="tight")  
# plt.show()




#########################################
# Diagnostic: level effect vs transition lumpiness behind the share
# 'scissors'. Per variant: the share LEVEL (denominator of the mechanical
# effect), the dispersion of share changes in LEVELS (bargaining risk with
# the level effect stripped out), the DRIFT of log share growth (fingerprint
# of the catch-up transition), and renegotiation frequency/direction.
########################################
print()
print('=== Share-volatility diagnostic (variants) ===')
print(f"{'variant':18s} {'mean s^w':>9s} {'Var(dS)x100':>12s} {'mean dlog s^w':>14s} "
      f"{'reneg freq':>11s} {'tow. wife':>10s} {'tow. husb':>10s}")
for i,m in enumerate(Bmodel):
    s  = samples[i]; s1 = np.roll(s,1,axis=1)
    sm = np.roll(s,-1,axis=1)[s]              # sample at t and t+1
    swl = m.sim.Cw/(m.sim.Cw+m.sim.Cm)
    dS   = (swl[s1]-swl[s])[sm]               # share growth in LEVELS
    dls  = np.log(swl[s1]/swl[s])[sm]         # share growth in LOGS
    pw_now,pw_lag = m.sim.power[s1][sm], m.sim.power_lag[s1][sm]
    ren = pw_now!=pw_lag
    up  = pw_now>pw_lag
    print(f"{Names_line[i]:18s} {swl[s].mean():9.3f} {100*dS.var(ddof=1):12.3f} "
          f"{dls.mean():14.4f} {ren.mean():11.4f} {(ren&up).mean():10.4f} {(ren&~up).mean():10.4f}")


# Size-vs-level decomposition of share-growth variance (both spouses)
share_var_decomposition(Bmodel, samples, Names_line)
