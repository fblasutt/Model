# -*- coding: utf-8 -*- 
""" 
Created on Fri Feb 23 15:31:56 2024 
 
@author: 32489 
""" 
 
import numpy as np 
import Bargaining_numba as brg  
import init_conditions as ic
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

#Without those two lines below there are nan which screw up things
baseline_sample=baseline_sample[~np.isnan(baseline_sample).any(axis=1)]
baseline_sample[:, 0] = np.arange(len(baseline_sample))

pr=np.ones(baseline_sample.shape[0])/baseline_sample.shape[0]
indexes=np.array(np.random.choice(baseline_sample[:,0], size=N, p=pr, replace=True),dtype=np.int32)-1
final_sample= baseline_sample[:,1:][indexes] 

age_initial=final_sample[:,0]*0+25 # forced to 25, as in calibration.py (t=0 = age 25)
age_final=final_sample[:,1]
cw_cons_share=final_sample[:,2]
h_income=final_sample[:,3]
w_income=final_sample[:,4]
age_marriage=final_sample[:,5]*0+25 # forced to 25, as in calibration.py
year=final_sample[:,6]
assets=final_sample[:,7]*np.mean(np.exp(h_income))

# Pre-drawn uniforms for the posterior-draw initial income split (drawn AFTER
# the sample so sample selection is unchanged; FIXED across evaluations so
# SMM objectives stay deterministic)
u_init_w=np.random.rand(N);u_init_m=np.random.rand(N)
σME2_init=0.0  # measurement-error variance in observed entry income (0 = off)



#Current estimates [η,σL,α,ρ,Ω,β] from the shared module (sync with calibration.py)
from estimated_params import xc, par_dict, apply_fc_params

#Parametrize the model (NB: position 4 is Ω, the match-quality disagreement shock)
par = par_dict(N, np.array(age_marriage-25,dtype=np.int_))
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

izm=ic.draw_init_iz(h_income,model.par.sample_init,gridzm,model.par.grid_pm,model.par.grid_ϵm,u_init_m,σME2=σME2_init)
izm[np.isnan(h_income)]=(model.par.num_pm*model.par.num_ϵm)//2
izw=ic.draw_init_iz(w_income,model.par.sample_init,gridzw,model.par.grid_pw,model.par.grid_ϵw,u_init_w,σME2=σME2_init)
izw[np.isnan(w_income)]=(model.par.num_pw*model.par.num_ϵw)//2     
model.sim.init_z=izw*model.par.num_zm+izm   # FIXED gender swap: wife is the SLOW joint-index component
model.sim.init_A=assets


#Create variable for policy change
age=(np.cumsum(np.ones((model.par.simN,model.par.T)),axis=1)-1)+25#age of hh  
calendar_year=age-age_initial[:,None]+year[:,None]

policy=np.maximum(calendar_year[:,0],2007)
age_policy=np.array(np.where(policy[:,None]==calendar_year)[1],dtype=np.int32)


############################################################################
# Solve the model for different alimony levels
############################################################################

#Grid of alimony values we will consider
gridτ=np.linspace(0.0,0.1,3)

#These lists will store all the models we solve and simulate
Bmodel=list() #limited commitment
Bfmodel=list()#full commitment


#Solve and simulate the model - limited commitment.
#The initial love distribution is drawn ONCE, in the BASELINE LC model
#(conditional draw under the baseline solution), and then held FIXED across
#every other version (LC at other alimony levels AND all FC models below):
#policy comparisons are not contaminated by initial-composition differences.
for i in range(len(gridτ)):

    # Set up the model - limited commitment
    M = model.copy(name='numba_new_copy')
    M.par.alimony=gridτ[i]

    M.solve()

    if i==0:
        # baseline: conditional draw runs inside simulate(); then extract the
        # realized entry love of every agent as THE initial love distribution
        M.simulate()
        init_love_base = np.array(
            [M.sim.love[k, int(M.par.sample_init[k])] for k in range(M.par.simN)],
            dtype=np.int_,
        )
    else:
        # other alimony levels: transplant the baseline draw, skip the re-draw
        M.sim.force_init_love[:] = init_love_base
        M._init_love_rationalized=True
        M.simulate()

    Bmodel.append(M)


#Solve and simulate the model - full commitment
for i in range(len(gridτ)):

    # Set up the model - full commitment
    Mf = model.copy(name='numba_new_copy')
    Mf.par.alimony=gridτ[i]

    Mf.par.full=True
    apply_fc_params(Mf)   # FC-specific [η, σL, β] (estimated_params.xc_full)

    Mf.solve()

    # Same BASELINE-LC initial love as every other version (see above)
    Mf.sim.force_init_love[:] = init_love_base

    Mf.simulate()
    Bfmodel.append(Mf)



#########################################
#Sample selection for Insurance analysis
########################################


#We take individuals that stays married across spefifications
age=(np.cumsum(np.ones((M.par.simN,M.par.T)),axis=1)-1)+25#age of hh
alwayscouple=np.array([(Bmodel[i].sim.couple_lag==1) & (Bfmodel[i].sim.couple_lag==1)  for i in range(len(gridτ))])
alwayscouplep=np.array([(Bmodel[i].sim.couple==1) & (Bfmodel[i].sim.couple==1)  for i in range(len(gridτ))])

sample =  (age>age_initial[:,None]) & (age<=age_final[:,None]) & (alwayscouple.min(axis=0)) & (alwayscouplep.min(axis=0))
sample1=np.roll(sample,1,axis=1)
    


#########################################
#Insurance analysis
########################################

#Lists with results
Bgrid=list() #limited commitment
Bfgrid=list()#full commitment


#Names of the file and of the table line associated with a model version (if gridτ has len()>3, names should be adapted)
Names=['Baselinealimony', 'Alimony1', 'Alimony2']
Names_line=['Baseline', 'Alimony, low', 'Alimony, high']

#Obtain pass-throughs and do the decomposition calling function insurance
for i in range(len(gridτ)):

    sample   = (age>age_initial[:,None]) & (age<=age_final[:,None]) & (Bmodel[i].sim.couple_lag==1)
    sample_f = (age>age_initial[:,None]) & (age<=age_final[:,None]) & (Bfmodel[i].sim.couple_lag==1)

    B=insurance(Bmodel[i],sample,
                shock_type='permanent',
                shock_gender='Male',
                consumption_gender='Male',
                name_file=Names[i],
                name_line=Names_line[i])

    B['par']=gridτ[i]
    Bgrid.append(B)

    Bf=insurance(Bfmodel[i],sample_f,
                 shock_type='permanent',
                 shock_gender='Male',
                 consumption_gender='Male',
                 name_file=Names[i]+'full',
                 name_line=Names_line[i])

    Bf['par']=gridτ[i]
    Bfgrid.append(Bf)


#########################################################
# Volatility ladder (waterfall) tables: consumption
# volatility by insurance channel. All rungs are exact
# sample variances on identical cells; the '=' rows are
# risk levels remaining at each stage of the budget flow,
# the '-' rows the variance absorbed by the channel in
# between (telescoping is exact). Six columns:
# (LC, FC) x alimony levels; one table per spouse.
#########################################################

def waterfall_table(gender):
    Ls=[Bgrid[i]['vol_ladder'] for i in range(len(gridτ))] \
      +[Bfgrid[i]['vol_ladder'] for i in range(len(gridτ))]
    alloc = 'alloc_m' if gender=='m' else 'alloc_w'
    Vlast = 'V_cm'    if gender=='m' else 'V_cw'
    who   = 'Husband' if gender=='m' else 'Wife'
    v=lambda x:'%.2f'%(100.0*x)
    rows=[]
    def level(label,key,bold=False):
        cells=[v(L[key]) for L in Ls]
        if bold: cells=[r'\textbf{'+c+'}' for c in cells]
        rows.append(('$=$ ' if rows else '')+label+' & '+' & '.join(cells))
    def absorbed(label,key):
        rows.append(r'\quad $-$ '+label+' & '+' & '.join(v(L[key]) for L in Ls))
    level('Potential household income risk','V_pot')
    absorbed('non-participation (risk concentration)','compos')
    level('Earnings risk, fixed participation','V_fp')
    absorbed('participation changes','partchg')
    level('Earnings risk','V_Y')
    absorbed('taxes','taxes');                              level('Net income risk','V_Ynet')
    absorbed('savings','savings');                          level('Household consumption risk','V_C')
    absorbed('private/public expenditure shift','pubpriv'); level('Private consumption risk','V_cp')
    absorbed('intra-household allocation',alloc);           level(who+' consumption risk',Vlast,bold=True)
    return (' \\\\\n'.join(rows))

with open(root+'/Output files/model/volladder_alimony_m.tex','w') as f: f.write(waterfall_table('m'))
with open(root+'/Output files/model/volladder_alimony_w.tex','w') as f: f.write(waterfall_table('w'))

# Console echo with the exact telescoping check
for tag,grid in (('LC',Bgrid),('FC',Bfgrid)):
    for i in range(len(gridτ)):
        L=grid[i]['vol_ladder']
        print(f"{tag} {Names_line[i]:14s}: V_pot={100*L['V_pot']:.2f} compos={100*L['compos']:.2f} "
              f"partchg={100*L['partchg']:.2f} V_Y={100*L['V_Y']:.2f} tax={100*L['taxes']:.2f} "
              f"sav={100*L['savings']:.2f} pub/priv={100*L['pubpriv']:.2f} "
              f"alloc_m={100*L['alloc_m']:.2f} -> V_cm={100*L['V_cm']:.2f}  "
              f"alloc_w={100*L['alloc_w']:.2f} -> V_cw={100*L['V_cw']:.2f}  "
              f"[resid {L['resid_m']:.1e}/{L['resid_w']:.1e}, cells {L['n_cells']}]")



    
# sh_perm=np.array([Bgrid[i]['w_sh']['per_m'] for i in range(len(gridτ))])

# perm_m_m=np.array([Bgrid[i]['indc']['per_m_m'] for i in range(len(gridτ))])
# perm_m_w=np.array([Bgrid[i]['indc']['per_m_w'] for i in range(len(gridτ))])
# perm_m_Q=np.array([Bgrid[i]['Qins']['per_m'] for i in range(len(gridτ))])


# share_w=(1+((1-M.par.grid_power)/M.par.grid_power)**(1/M.par.ρ))**-1

# rebargainings=np.array([((Bmodel[i].sim.power!=Bmodel[i].sim.power_lag) & (Bmodel[i].sim.power>=0.0))[sample].mean() for i in range(len(gridτ))])
# reb_size=np.array([(abs(Bmodel[i].sim.power[sample]-Bmodel[i].sim.power_lag[sample])[((Bmodel[i].sim.power[sample]!=Bmodel[i].sim.power_lag[sample]) & (Bmodel[i].sim.power[sample]>=0.0))]).mean() for i in range(len(gridτ))])

# power_at_rebarg=np.array([(Bmodel[i].sim.power_lag[sample][((Bmodel[i].sim.power[sample]!=Bmodel[i].sim.power_lag[sample]) & (Bmodel[i].sim.power[sample]>=0.0))]).mean() for i in range(len(gridτ))])



# #Plot consumption insurance graph 
# plt.plot(gridτ,perm_m_m-perm_m_m[0],label="Husband, Lim.",color='blue')
# plt.plot(gridτ,perm_m_w-perm_m_w[0],label="Wife, Lim.",color='red')
# plt.plot(gridτ,perm_m_Q-perm_m_Q[0],label="Common, Lim.",color='black')


# plt.xlabel("Alimony")   
# plt.ylabel("Δ Insurance")   
# #plt.ylim(0, 20)    
# plt.legend()                              
# #plt.savefig(root+'/Output files/model/lifecycle_singlew.eps', format='eps', bbox_inches="tight")  
# plt.show()

