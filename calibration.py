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
import dfols 
import getpass
import time
import quantecon.markov.approximation as quant
import scipy
import pandas as pd 
import statsmodels.formula.api as smf 
import statsmodels.api as sm 
from pyhdfe import create
import matplotlib.pyplot as plt
import TikTak 

import pybobyqa

#Initialize seed 
np.random.seed(10) 
 
#Root
# root='C:/Users/32489/Dropbox/Family Risk Sharing'

user = getpass.getuser()
if user == "sara":
      root = '/Users/sara/Dropbox/Family Risk Sharing'
elif user == "32489":
      root = '/Users/32489/Dropbox/Family Risk Sharing'
elif user=='blasutto':
      root = '/home/users/b/l/blasutto/Family-Risk-Sharing'
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

age_initial=final_sample[:,0]*0+25
age_final=final_sample[:,1]
cw_cons_share=final_sample[:,2]
h_income=final_sample[:,3]
w_income=final_sample[:,4]
age_marriage=final_sample[:,5]*0+25
year=final_sample[:,6]
assets=final_sample[:,7]*np.mean(np.exp(h_income))

# Pre-drawn uniforms for the posterior-draw initial income split (drawn AFTER
# the sample so sample selection is unchanged; FIXED across evaluations so
# SMM objectives stay deterministic)
u_init_w=np.random.rand(N);u_init_m=np.random.rand(N)
σME2_init=0.0  # measurement-error variance in observed entry income (0 = off)
marr_durr_pol=final_sample[:,8]


# Guess of internal parameters: [η,σL,α,ρ,wedge_w,wedge_m,β,ι0w]


#Target 0.97 (large diff of pass thorughs + ridiculous real effect of reform)
xc=np.array([4.46931249,  0.04134383,  0.92836788,  1.68926261,  2.40723492, -0.09099960,.98391886, -0.64534907])

#Target 1.03.. (resonable pass throughs, , ridiculous effect of the reorm)
xc=np.array([5.61375811,  0.03893,     0.95715273,  1.97497468,  3.05150031, -0.10250626, 0.98321172, -0.64221611])

# 8-parameter layout: [η, σL, α, ρ, wedge_w, wedge_m, β, ι0w] — ι0w (female
# income-trend level) now internally estimated, targeting the mean wife-to-
# husband earnings ratio among working wives (0.46 in the data).
xl=np.array([0.2,0.001,0.78 ,1.0,-0.2,-0.2,0.94,-1.5])
xu=np.array([12.2,0.25 ,0.98 ,2.5 ,3.5 ,3.5 ,1.005,-0.05])

#Parametrize the model 
par = {'simN':N,'η': xc[0],'σL':xc[1],'α':xc[2],'ρ':xc[3],'wedge_w':xc[4],'wedge_m':xc[5],'β':xc[6],'ι0w':xc[7],'sample_init':np.array(age_marriage-25,dtype=np.int_)}
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
model.sim.init_z=izm*model.par.num_zm+izw
model.sim.init_A=assets


#Create variable for policy change
age=(np.cumsum(np.ones((model.par.simN,model.par.T)),axis=1)-1)+25#age of hh  
calendar_year=age-age_initial[:,None]+year[:,None]

policy=np.maximum(calendar_year[:,0],2007)
age_policy=np.array(marr_durr_pol,dtype=np.int32)#np.array(np.where(policy[:,None]==calendar_year)[1],dtype=np.int32)


#Function to minimize 
def q(pt,table=False): 
 

    try:
    
        #########################################################################################################################################
        #Solve and simulate the model for given parametrization, then compute moments and their distance distance from the data
        #######################################################################################################################################
       
              
        ###################
        #Pre reform model
        ####################
        #tic=time.time()
        
        # Set up the model with the input parameters pt
        M_bef = model.copy(name='numba_new_copy')
       
        M_bef.par.η=pt[0]
        M_bef.par.α=pt[2]
        M_bef.par.ρ=pt[3]
        M_bef.par.wedge_w=pt[4]
        M_bef.par.wedge_m=pt[5]
        M_bef.par.β=pt[6]
        M_bef.par.ι0w=pt[7]

        # ι0w is estimated: rebuild the income grids (couples AND singles) with pt[7]
        M_bef.par.grid_zw,M_bef.par.grid_ϵw,M_bef.par.grid_pw,M_bef.par.Π_zw0, \
            M_bef.par.grid_zm,M_bef.par.grid_ϵm,M_bef.par.grid_pm,M_bef.par.Π_zm0, \
                                            M_bef.par.Π=usr.labor_income(M_bef.par,pens_reform=M_bef.par.pens_reform)
        M_bef.par.grid_zws,M_bef.par.grid_ϵw,M_bef.par.grid_pw,M_bef.par.Π_zw0, \
            M_bef.par.grid_zms,M_bef.par.grid_ϵm,M_bef.par.grid_pm,M_bef.par.Π_zm0, \
                                                    M_bef.par.Πs=usr.labor_income(M_bef.par,single=True,pens_reform=M_bef.par.pens_reform)

        # The female grid shifted with pt[7]: redo the wife's initial-income
        # nearest-gridpoint mapping so the in-loop economy matches a fresh run
        # parametrized at pt (same logic as the init_power/ρ fix below).
        gridzw_pt=M_bef.par.grid_zw[:,:,np.linspace(0,M_bef.par.num_z-1,M_bef.par.num_zm,dtype=np.int_)]
        izw_pt=ic.draw_init_iz(w_income,M_bef.par.sample_init,gridzw_pt,M_bef.par.grid_pw,M_bef.par.grid_ϵw,u_init_w,σME2=σME2_init)
        izw_pt[np.isnan(w_income)]=(M_bef.par.num_pw*M_bef.par.num_ϵw)//2
        M_bef.sim.init_z=izm*M_bef.par.num_zm+izw_pt

        # Initial Pareto weights depend on ρ: recompute them with pt[3] so the
        # in-loop economy matches a fresh run parametrized at pt. (The module-top
        # init_power was built with the construction-time ρ and is otherwise just
        # copied along, making the reported fit at pt differ from a restart at pt.)
        param_pt=(cw_cons_share/(1.0-cw_cons_share))**pt[3]
        M_bef.sim.init_power=param_pt/(1.0+param_pt)
    
        # Two individual-specific random-walk love shocks with innovation sd pt[1],
        # initial sd σL0 = 0.1 fixed (set in setup_grids: t=0 grid width for the
        # init-love rationalization; median point stays exactly 0). Joint index iL = iψw*num_lovem + iψm.
        M_bef.par.grid_lovew_,M_bef.par.Πlw_,M_bef.par.Πlw0_= usr.rouw_nonst(M_bef.par.T,pt[1],M_bef.par.σL0,M_bef.par.num_lovew)
        M_bef.par.grid_lovem_,M_bef.par.Πlm_,M_bef.par.Πlm0_= usr.rouw_nonst(M_bef.par.T,pt[1],M_bef.par.σL0,M_bef.par.num_lovem)
        M_bef.par.grid_lovew=[np.repeat(M_bef.par.grid_lovew_[t],M_bef.par.num_lovem) for t in range(M_bef.par.T)]
        M_bef.par.grid_lovem=[np.tile(M_bef.par.grid_lovem_[t],M_bef.par.num_lovew) for t in range(M_bef.par.T)]
        M_bef.par.Πl=[np.kron(M_bef.par.Πlw_[t],M_bef.par.Πlm_[t]) for t in range(M_bef.par.T-1)]
        M_bef.par.Πl0=[np.kron(M_bef.par.Πlw0_[t],M_bef.par.Πlm0_[t]) for t in range(M_bef.par.T-1)]
               
                
    
        # Solve the pre-reform model (SIMULATION deferred: the initial-love
        # rationalization needs both regimes' solutions first, see below)
        M_bef.solve()



        ###################
        #After reform model
        ####################

        M = M_bef.copy(name='numba_new_copy')
        M.par.pens_reform=True #set up pension reform
        M.par.policy_init=age_policy
        M.par.η=pt[0]
        M.par.α=pt[2]
        M.par.ρ=pt[3]
        M.par.wedge_w=pt[4]
        M.par.wedge_m=pt[5]
        M.par.β=pt[6]
        M.par.ι0w=pt[7]
    
    
        # Two individual-specific random-walk love shocks with innovation sd pt[1]
        M.par.grid_lovew_,M.par.Πlw_,M.par.Πlw0_= usr.rouw_nonst(M.par.T,pt[1],M.par.σL0,M.par.num_lovew)
        M.par.grid_lovem_,M.par.Πlm_,M.par.Πlm0_= usr.rouw_nonst(M.par.T,pt[1],M.par.σL0,M.par.num_lovem)
        M.par.grid_lovew=[np.repeat(M.par.grid_lovew_[t],M.par.num_lovem) for t in range(M.par.T)]
        M.par.grid_lovem=[np.tile(M.par.grid_lovem_[t],M.par.num_lovew) for t in range(M.par.T)]
        M.par.Πl=[np.kron(M.par.Πlw_[t],M.par.Πlm_[t]) for t in range(M.par.T-1)]
        M.par.Πl0=[np.kron(M.par.Πlw0_[t],M.par.Πlm0_[t]) for t in range(M.par.T-1)]
    
        
        # Income shocks grids: couples
        M.par.grid_zw,M.par.grid_ϵw,M.par.grid_pw,M.par.Π_zw0, \
            M.par.grid_zm,M.par.grid_ϵm,M.par.grid_pm,M.par.Π_zm0, \
                                        M.par.Π=usr.labor_income(M.par,pens_reform=M.par.pens_reform)     
        
        M.par.grid_zws,M.par.grid_ϵw,M.par.grid_pw,M.par.Π_zw0, \
            M.par.grid_zms,M.par.grid_ϵm,M.par.grid_pm,M.par.Π_zm0, \
                                                M.par.Πs=usr.labor_income(M.par,single=True,pens_reform=M.par.pens_reform) 
                                                
    
       
    
        M.solve()

        # Initial-love rationalization by ENTRY REGIME: couples whose policy is
        # already in place at entry (policy_init <= sample_init) face the
        # REFORM environment at t=0, so their revealed-preference love is
        # computed under M's solution; the rest under M_bef's. Both models then
        # share the SAME merged initial love, so pre/post-reform economies
        # differ only in the reform itself (clean policy moment).
        # --- TEST CONFIGURATION (reproduces the pre-two-regime behavior): ----
        # everyone rationalized under the PRE-REFORM solution only. Should
        # replicate the old near-perfect fit exactly; if it does, the residual
        # difference of the two-regime version below is purely the
        # entry-regime refinement. Swap the comment blocks to switch back.
        # M_bef.rationalize_init_love()          # everyone under the pre-reform solution
        # M.sim.force_init_love[:]=M_bef.sim.force_init_love
        # M_bef._init_love_rationalized=True;M._init_love_rationalized=True

        # --- TWO-REGIME VERSION (the intended final configuration): ----------
        treated0=(np.array(age_policy)<=np.array(M.par.sample_init))
        M_bef.rationalize_init_love(agents=~treated0)
        M.rationalize_init_love(agents=treated0)
        merged=np.where(treated0,M.sim.force_init_love,M_bef.sim.force_init_love)
        M_bef.sim.force_init_love[:]=merged;M.sim.force_init_love[:]=merged
        M_bef._init_love_rationalized=True;M._init_love_rationalized=True

        # Simulate the pre-reform model first
        M_bef.simulate()

        # PREFILL the reform model's sim arrays with the pre-reform simulated
        # history: for agents whose policy arrives AFTER entry the reform
        # simulation runs with copy=True from the first period and reads
        # lagged states from the sim arrays themselves — under the original
        # code order M was copied from an already-simulated M_bef, so this
        # replicates that behavior (without it those agents' paths are NaN).
        for _k,_v in M_bef.sim.__dict__.items():
            if isinstance(_v,np.ndarray): getattr(M.sim,_k)[...]=_v

        M.simulate()

        #toc=time.time()
        #print('Time elapsed for model solution is {}'.format(toc-tic))
        
        aaa=M.sim.power[age==age_marriage[:,None]]>0
        print(np.mean(aaa))
        
        
        ok_M = M.sim.power[age == age_marriage[:,None]] > 0
        print('treated0:', ok_M[treated0].mean(), '| untreated:', ok_M[~treated0].mean())
        
        #############################################
        #Event Study: policy and consumption ratio
        ############################################
      
        # #Covariates
        time_to_policy= (np.cumsum(np.ones((M.par.simN,M.par.T)),axis=1)-1)-M.par.policy_init[:,None]
        event_time=time_to_policy.copy()
        idd=np.repeat(np.cumsum(np.ones(M.par.simN))[:,None],M.par.T,axis=1)   
        policy_init= np.repeat((M.par.policy_init)[:,None],M.par.T,axis=1) 
        agei=np.repeat((age_marriage)[:,None],M.par.T,axis=1) 
        time=age-agei
        
        #Event-study specific variables
        treat_group=np.repeat((M.par.policy_init-(age_marriage-25)>=15)[:,None],M.par.T,axis=1) 
       
        #treat_group=np.repeat((M.par.policy_init>=20)[:,None],M.par.T,axis=1) 
            
       
        event_time[event_time<=-5]=-5
        event_time[event_time>=5]=5
        event_time_PER_treat=event_time*treat_group 
        wife_ratio=M.sim.Cw/(M.sim.Cm+M.sim.Cw)
    
        #Sample
        subset= (event_time>=-5) & (age>=np.maximum(age_marriage,age_initial)[:,None])  & (age<=age_final[:,None])  & (M.sim.power>=0)  &(age_policy>np.maximum(age_marriage,age_initial)-25)[:,None] 
    
        # Combine into a DataFrame 
        df = pd.DataFrame({ 
            "time":time[subset],
            "policy_init":policy_init[subset],
            "post":(event_time>=0)[subset],
            "inter":(treat_group*(event_time>=0))[subset],
            "wife_ratio":wife_ratio[subset], 
            "event_time":event_time[subset], 
            "idd":idd[subset], 
            "age":age[subset],
            "agesq":age[subset]**2,
            "ageth":age[subset]**3,
            "duration":(age-age_marriage[:,None])[subset],
            "treat_group":treat_group[subset],
            "event_time_PER_treat":event_time_PER_treat[subset] 
        }) 
         
       
        reference_value=-1
        event_cats = sorted(df['event_time'].unique()) 
        if reference_value in event_cats: 
            event_cats.remove(reference_value) 
            event_cats = [reference_value] + event_cats 
             
     
       # Step 1: Create the fixed effects structure (age removed)
        fe_df = df[['post','idd']].astype('category')
         
        # Step 2: Create the HDFE projector 
        hdfe = create(fe_df) 
         
        # Create categorical with this ordering 
        df['event_cat'] = pd.Categorical(df['event_time_PER_treat'], categories=event_cats) 
     
        # Create dummies, drop_first will now drop your reference group 
        event_dummies = pd.get_dummies(df['event_cat'], prefix='event', drop_first=True) 
         
        # Continuous age controls (replace age dummies)
        age_controls = df[['age','agesq','ageth']].astype(float).values
    
        # Residualize y and X (event dummies + age, age2) on the FEs
        y_resid = hdfe.residualize(df[['wife_ratio']].values) 
        #X_event_resid = hdfe.residualize(event_dummies.values.astype(float)) 
        X_event_resid = hdfe.residualize(df[['inter']].values) 
        X_age_resid   = hdfe.residualize(age_controls)
    
        # Stack event dummies and age controls into one design matrix
        X_resid = np.hstack([X_event_resid, X_age_resid])
         
        # OLS on residuals 
        model_ = sm.OLS(y_resid, X_resid) 
        results = model_.fit() 
        
        policy_effect_wife_ratio = results.params[0]
        
        print(policy_effect_wife_ratio)
    
        #plt.plot(np.insert(results.params[1:-4], 3, 0))
         
        ######################################
        #Other moments here
        ###################################### 
        # from consav import linear_interp,upperenvelope
        # Vcw=np.zeros(M.par.simN)
        # for i in range(M.par.simN):
            
            
            
        #     idx = (44,M.sim.ih[i,44],M.sim.iz[i,44],slice(None),M.sim.love[i,44])
            
        #     Vcw[i]=linear_interp.interp_2d(M.par.grid_power,M.par.grid_A,M.sol.Vw_remain_couple[idx],M_bef.sim.power[i,44],M_bef.sim.A[i,44])
            
            
        
         
        #Wife share of consumption
        event_time=np.arange(-5,20)
       
        
        baseb=(age>=np.maximum(age_marriage,age_initial)[:,None])& (M.sim.couple==1)   & (age<=age_final[:,None])  & (age_policy>=5+np.maximum(age_marriage,age_initial)-25)[:,None]#   & (M.sim.incw/(M.sim.incw+M.sim.incm)>0.5) 
        basea=(age>=np.maximum(age_marriage,age_initial)[:,None]) &  (M_bef.sim.couple==1)  & (age<=age_final[:,None]) & (age_policy>=5+np.maximum(age_marriage,age_initial)-25)[:,None]#  & (M_bef.sim.incw/(M_bef.sim.incw+M_bef.sim.incm)>0.5)
      
        
        sya=(basea) & (M.par.policy_init[:,None]-(age_marriage[:,None]-25)<15) 
        soa=(basea) & (M.par.policy_init[:,None]-(age_marriage[:,None]-25)>=15) 
        
        syb=(baseb) & (M.par.policy_init[:,None]-(age_marriage[:,None]-25)<15) 
        sob=(baseb) & (M.par.policy_init[:,None]-(age_marriage[:,None]-25)>=15) 
        
        
        
        ws_yb=np.array([(np.log(M_bef.sim.Cw)*0+M_bef.sim.Cw/(M_bef.sim.Cm+M_bef.sim.Cw))[(syb)    & (time_to_policy==i)].mean() for i in event_time])
        ws_ob=np.array([(np.log(M_bef.sim.Cw)*0+M_bef.sim.Cw/(M_bef.sim.Cm+M_bef.sim.Cw))[(sob)    & (time_to_policy==i)].mean() for i in event_time])
    
        ws_ya=np.array([(np.log(M.sim.Cw)*0+M.sim.Cw/(M.sim.Cm+M.sim.Cw))[(sya)    & (time_to_policy==i)].mean() for i in event_time])
        ws_oa=np.array([(np.log(M.sim.Cw)*0+M.sim.Cw/(M.sim.Cm+M.sim.Cw))[(soa)    & (time_to_policy==i)].mean() for i in event_time])
               
        effect=(ws_oa-ws_ob)-(ws_ya-ws_yb)
        plt.plot(event_time,effect)
        plt.show()
        
        Aws_yb=(M_bef.sim.Cw/(M_bef.sim.Cm+M_bef.sim.Cw))[(syb)    & (time_to_policy>=0)].mean() 
        Aws_ob=(M_bef.sim.Cw/(M_bef.sim.Cm+M_bef.sim.Cw))[(sob)    & (time_to_policy>=0)].mean()
    
        Aws_ya=(M.sim.Cw/(M.sim.Cm+M.sim.Cw))[(sya)    & (time_to_policy>=0)].mean()
        Aws_oa=(M.sim.Cw/(M.sim.Cm+M.sim.Cw))[(soa)    & (time_to_policy>=0)].mean()
               
             
        print("aggregate eff is {}".format((Aws_oa-Aws_ob)-(Aws_ya-Aws_yb)))
      
        
        SA=(age>=np.maximum(age_marriage,age_initial)[:,None]) & (age<=age_final[:,None]) & (M.sim.couple==1)  & (M_bef.sim.couple==1)
        SB=(age>=np.maximum(age_marriage,age_initial)[:,None]) & (age<=age_final[:,None]) & (M.sim.couple==1)  & (M_bef.sim.couple==1)
        
        Swa=M.sim.Vcw-M.sim.Vsw < 0.01
        Sma=M.sim.Vcm-M.sim.Vsm < 0.01
        
        Swb=M_bef.sim.Vcw-M_bef.sim.Vsw < 0.01
        Smb=M_bef.sim.Vcm-M_bef.sim.Vsm < 0.01
        
        plt.plot(np.mean(Swa,where=SA,axis=0),label="after")
        plt.plot(np.mean(Swb,where=SB,axis=0),label="before")
        plt.legend()
        plt.show()
        
        plt.plot(np.mean(M.sim.power,where=SA,axis=0),label="after")
        plt.plot(np.mean(M_bef.sim.power,where=SB,axis=0),label="before")
        plt.legend()
        plt.show()
        
    
        #Share binding by age
        sample_empla =  (age>=np.maximum(age_marriage,age_initial)[:,None]) & (age<=age_final[:,None]) & (M.sim.couple==1) & (M_bef.sim.couple==1)  & (M.par.policy_init[:,None]-(age_marriage[:,None]-25)>=15)
        sample_emplb =  (age>=np.maximum(age_marriage,age_initial)[:,None]) & (age<=age_final[:,None]) & (M_bef.sim.couple==1) & (M.sim.couple==1) & (M.par.policy_init[:,None]-(age_marriage[:,None]-25)>=15)
        
        
        bindwb=np.nanmean(Swb ,where=sample_emplb,axis=0)
        bindmb=np.nanmean(Smb ,where=sample_emplb,axis=0)
        
        bindw_eb=np.array([(Swb)[(sample_emplb)    & (time_to_policy==i)].mean() for i in event_time])
        bindm_eb=np.array([(Smb)[(sample_emplb)    & (time_to_policy==i)].mean() for i in event_time])
    
        bindwa=np.nanmean(Swa ,where=sample_empla,axis=0)
        bindma=np.nanmean(Sma,where=sample_empla,axis=0)
        
        bindw_ea=np.array([(Swa)[(sample_empla)    & (time_to_policy==i)].mean() for i in event_time])
        bindm_ea=np.array([(Sma)[(sample_empla)    & (time_to_policy==i)].mean() for i in event_time])
    
        
        BPa=np.array([(M.sim.Cw/(M.sim.Cw+M.sim.Cm))[(sample_empla)   & (time_to_policy==i)].mean() for i in event_time])
        BPb=np.array([(M_bef.sim.Cw/(M_bef.sim.Cw+M_bef.sim.Cm))[(sample_emplb)   & (time_to_policy==i)].mean() for i in event_time])
        
        plt.plot(bindma,label="Men")
        plt.plot(bindwa,label="Women")
        plt.ylabel('Percent binding')
        plt.legend()
        plt.show()
        
        plt.plot(event_time,bindm_ea-bindm_eb,label="Men")
        plt.plot(event_time,bindw_ea-bindw_eb,label="Women")
        plt.ylabel('Percent binding')
        plt.legend()
        plt.show()
        
        
        plt.plot(event_time,BPa,label="after")
        plt.plot(event_time,BPb,label="before")
        plt.ylabel('Womens power')
        plt.legend()
        plt.show()
        
        
        
        # plt.plot(np.mean(M_bef.sim.power<0,where=(M_bef.sim.power_lag>0),axis=0),label="before")
        # plt.plot(np.mean(M.sim.power<0,where=(M.sim.power_lag>0),axis=0),label="after")
        # plt.ylabel('Divrce')
        # plt.legend()
        # plt.show()
        
        #Compute samples relevant for moments
        
        # Sample used for divorce moments and employment/expenditures moments
        sample_div =  (age>=age_initial[:,None]-1) & (age<=age_final[:,None]) & (M.sim.couple_lag==1) 
        sample_empl =  (age>=age_initial[:,None]) & (age<=age_final[:,None]) & (M.sim.couple==1) 
    
        # Sample to be used for pass through from total to public good expenditures
        sample_pass= (age>age_initial[:,None]) & (age<=age_final[:,None]) & (M.sim.couple==1) & (M.sim.couple_lag==1)
        sample_pass_m1= np.roll(sample_pass,-1,axis=1)
        
        # This sample will be used for pass throughs regressions (if sample, BPP persistent will not work)
        sample_reg= (age>age_initial[:,None]) & (age<=age_final[:,None]) & (M.sim.couple_lag==1) #& (M.sim.couple==1) 
     
        wife_empl = np.mean(M.sim.WLP[sample_empl]>0)        
        divorce_rate=np.mean((M.sim.couple==0)[sample_div])
        divorce_rate_young=np.mean((M.sim.couple==0)[(sample_div) & (age<=40)])
        expenditure_x_share=np.mean((M.sim.dw/M.sim.C_tot)[sample_empl])
        wife_cons_share=np.mean((M.sim.Cw/(M.sim.Cw+M.sim.Cm))[sample_empl])# wife's share of private consumption, married couples
        wife_ratio=np.mean((M.sim.incwg/(M.sim.incmg+M.sim.incwg))[sample_empl][M.sim.WLP[sample_empl]>0])# mean wife/husband gross earnings ratio, working wives
        
    
        ΔC =np.log(M.sim.C_tot[sample_pass])#  -np.log(M.sim.C_tot[sample_pass_m1])
        Δd=np.log(M.sim.dw[sample_pass]) # -np.log(M.sim.dw[sample_pass_m1])     
        βdC=np.cov(ΔC,Δd)[0,1]/np.var(ΔC)
        
        # Average household income
        couple_assets = M.sim.A[sample_empl].mean()/M.sim.incmg[sample_empl].mean()
        
        # print(111)
        #AWE
        B=insurance(M,sample_reg)
        AWE=B['wlp']['all_m']
        
        print('AWE is {}'.format(AWE))
           
        ###################################
        # Non-targeted moments
        ###################################
        gender_gap_earnings=(M.sim.incwg[sample_empl][M.sim.WLP[sample_empl]>0]).mean()/M.sim.incmg[sample_empl].mean()
        share_full_time=(M.sim.WLP[sample_empl][(M.sim.WLP[sample_empl]>0) & (M.sim.couple[sample_empl]==1)]==(M.par.num_wlp-1)).mean()
        
    
        fit =((wife_empl-0.5879)/0.5879)**2+((policy_effect_wife_ratio-.0139)/.0139)**2+((divorce_rate-0.0103)/0.0103)**2+((expenditure_x_share-0.812)/0.812)**2+((βdC-.97899)/.97899)**2+((couple_assets-2.634)/2.634)**2+((wife_cons_share-0.322)/0.322)**2+((gender_gap_earnings-.3767)/.3767)**2#+((AWE+0.03)/0.03)**2
        print('Point is {}, fit is {}'.format(pt,fit))  
        print('Simulated moments are {}'.format([wife_empl,policy_effect_wife_ratio,divorce_rate,expenditure_x_share,βdC,couple_assets,wife_cons_share,gender_gap_earnings]))
        
      
        print('Simulated moments are {}'.format([gender_gap_earnings,share_full_time,policy_effect_wife_ratio]))
    
        
        # Function tables computes a lot of tables with results and fit. Should be activated only for the final solution
        if table:tables(M,sample_reg,pt,root,divorce_rate,policy_effect_wife_ratio,expenditure_x_share,wife_empl,βdC,couple_assets,gender_gap_earnings,share_full_time,wife_cons_share,gender_gap_earnings)
      
        #fitt=[((wife_empl-.567)/.567),((divorce_rate_young-.0109)/.0109),((divorce_rate-.0101)/.0101),((expenditure_x_share-.782)/.782),((βdC-.9)/.9)]   
        fitt=[((wife_empl-0.5879)/0.5879),((policy_effect_wife_ratio-.0139)/.0139),((divorce_rate-0.0103)/0.0103),((expenditure_x_share-.812)/.812),((βdC-.97899)/.97899),((couple_assets-2.634)/2.634),((wife_cons_share-0.322)/0.322),((gender_gap_earnings-.3767)/.3767)]#,(AWE+0.03)/0.03]

        if np.isnan(fitt).max():fitt=[10000.0,10000.0,10000.0,10000.0,10000.0,10000.0,10000.0,10000.0]#
        return fitt#fit#

    except:

        print("Global error! Point is {}".format(pt))
        return [10000.0,10000.0,10000.0,10000.0,10000.0,10000.0,10000.0,10000.0]
     
def tables(M,sample,pt,root,divorce_rate,policy_effect_wife_ratio,expenditure_x_share,wife_empl,βdC,couple_assets,gender_gap_earnings,share_full_time,wife_cons_share,wife_ratio):
    
    # Extract empirical pass throughs from Sara's files
    def simple_extract(filename):
        """Simplified extraction for the specific format"""
        with open(filename, 'r') as file:
            content = file.read()
        
        # Extract all numbers from the file
        import re
        numbers = re.findall(r'-?\d+\.\d+', content)
        
    
        #Pass-throughs when the shock hits the husband (pass_husband) or wife (pass_wife)
        pass_husband={'tot':numbers[0],'com':numbers[1],'hus':numbers[2],'wif':numbers[3],'wif_rel':numbers[4]}
        pass_wife   ={'tot':numbers[5],'com':numbers[6],'hus':numbers[7],'wif':numbers[8],'wif_rel':numbers[9]}
        
        return pass_husband,pass_wife

            
 
    
    #############################
    # Pass throughs
    ############################
    
    # Compute pass - throughs
    B=insurance(M,sample)
    
    
    #Tables with pass-throughs results below
    def p33(x): y=x;return str('%3.3f' % y)    
    def p42(x): return str('%4.2f' % x)  
    def p43(x): return str('%4.3f' % x)     
    def p40(x): return str('%4.0f' % x) 
     
    #% changes in consumption out of a 1 % change in all income
    table=r"...total income   & \textbf{"+p33(B['totc']['all'])+'} & '+p33(B['dins']['all'])+' & & &    \\\\ '+\
          r'...wife income    & '+p33(B['totc']['all_w'])+' & '+p33(B['dins']['all_w'])+'& '+' \\textbf{'+p33(B['indc']['all_w_m'])+'} &  \\textbf{'+p33(B['indc']['all_w_w'])+'} &  \\textbf{'+p33(B['w_sh']['all_w'])+'}    \\\\ '+\
          r'...husband income & '+p33(B['totc']['all_m'])+' &  '+p33(B['dins']['all_m'])+'& '+' \\textbf{'+p33(B['indc']['all_m_m'])+'} &  \\textbf{'+p33(B['indc']['all_m_w'])+'} &  \\textbf{'+p33(B['w_sh']['all_m'])+'}    \\\\\\bottomrule'
    with open(root+'/Output files/model/allinc.tex', 'w') as f: f.write(table); f.close() 
    
    #% changes in consumption out of a 1 % transitory change in income
    table=r"...total income   & \textbf{"+p33(B['totc']['tra'])+'} & '+p33(B['dins']['tra'])+' & & &    \\\\ '+\
          r'...wife income    & '+p33(B['totc']['tra_w'])+' & '+p33(B['dins']['tra_w'])+'& '+' \\textbf{'+p33(B['indc']['tra_w_m'])+'} &  \\textbf{'+p33(B['indc']['tra_w_w'])+'} &  \\textbf{'+p33(B['w_sh']['tra_w'])+'}    \\\\ '+\
          r'...husband income & '+p33(B['totc']['tra_m'])+' &  '+p33(B['dins']['tra_m'])+'& '+' \\textbf{'+p33(B['indc']['tra_m_m'])+'} &  \\textbf{'+p33(B['indc']['tra_m_w'])+'} &  \\textbf{'+p33(B['w_sh']['tra_m'])+'}    \\\\\\bottomrule'
    with open(root+'/Output files/model/trainc.tex', 'w') as f: f.write(table); f.close() 
    
    
    #% changes in consumption out of a 1 % persistent change in income
    table=r"...total income   & \textbf{"+p33(B['totc']['per'])+'} & '+p33(B['dins']['per'])+' & & &    \\\\ '+\
          r'...wife income    & '+p33(B['totc']['per_w'])+' & '+p33(B['dins']['per_w'])+'& '+' \\textbf{'+p33(B['indc']['per_w_m'])+'} &  \\textbf{'+p33(B['indc']['per_w_w'])+'} &  \\textbf{'+p33(B['w_sh']['per_w'])+'}    \\\\ '+\
          r'...husband income & '+p33(B['totc']['per_m'])+' &  '+p33(B['dins']['per_m'])+'& '+' \\textbf{'+p33(B['indc']['per_m_m'])+'} &  \\textbf{'+p33(B['indc']['per_m_w'])+'} &  \\textbf{'+p33(B['w_sh']['per_m'])+'}    \\\\\\bottomrule'
    with open(root+'/Output files/model/perinc.tex', 'w') as f: f.write(table); f.close() 
    
    
    #MPC tables
    table=r"...husband income & "+p33(B['BPP_MPC']['ym_tot'])+' & '+p33(B['BPP_MPC']['ym_d'])+' & '+p33(B['BPP_MPC']['ym_cm'])+' & '+p33(B['BPP_MPC']['ym_cw'])+'  \\\\ '+\
          r'...wife income    & '+p33(B['BPP_MPC']['yw_tot'])+' & '+p33(B['BPP_MPC']['yw_d'])+' & '+p33(B['BPP_MPC']['yw_cm'])+' & '+p33(B['BPP_MPC']['yw_cw'])+'  \\\\ '+\
          r'...total income   & '+p33(B['BPP_MPC']['al_tot'])+' & '+p33(B['BPP_MPC']['al_d'])+' & '+p33(B['BPP_MPC']['al_cm'])+' & '+p33(B['BPP_MPC']['al_cw'])+'  \\\\\\bottomrule'
    with open(root+'/Output files/model/BPP_MPC.tex', 'w') as f: f.write(table); f.close() 
    
    #BPP persistent tables
    table=r"...husband income & "+p33(B['BPP_PER']['ym_tot'])+' & '+p33(B['BPP_PER']['ym_d'])+' & '+p33(B['BPP_PER']['ym_cm'])+' & '+p33(B['BPP_PER']['ym_cw'])+'  \\\\ '+\
          r'...wife income    & '+p33(B['BPP_PER']['yw_tot'])+' & '+p33(B['BPP_PER']['yw_d'])+' & '+p33(B['BPP_PER']['yw_cm'])+' & '+p33(B['BPP_PER']['yw_cw'])+'  \\\\ '+\
          r'...total income   & '+p33(B['BPP_PER']['al_tot'])+' & '+p33(B['BPP_PER']['al_d'])+' & '+p33(B['BPP_PER']['al_cm'])+' & '+p33(B['BPP_PER']['al_cw'])+'  \\\\\\bottomrule '
    with open(root+'/Output files/model/BPP_PER.tex', 'w') as f: f.write(table); f.close() 
    
    #labor supply	
    table=r'  '+p33(B['wlp']['tra_w'])+' & '+p33(B['wlp']['tra_m'])+' & '+p33(B['wlp']['per_w'])+' & '+p33(B['wlp']['per_m'])+' & '+p33(B['wlp']['all_w'])+' & '+p33(B['wlp']['all_m'])+'  \\\\\\bottomrule '
    with open(root+'/Output files/model/WLP.tex', 'w') as f: f.write(table); f.close() 
    
    #level changes in consumption out of a level change in all income
    table=r"...total income   & \textbf{"+p33(B['totc']['level_s'])+'} & '+p33(B['dins']['level_s'])+' & & &    \\\\ '+\
          r'...wife income    & '+p33(B['totc']['level_w'])+' & '+p33(B['dins']['level_w'])+'& '+' \\textbf{'+p33(B['level']['all_w_m'])+'} &  \\textbf{'+p33(B['level']['all_w_w'])+'} &  \\textbf{'+p33(B['w_sh']['level_w'])+'}    \\\\ '+\
          r'...husband income & '+p33(B['totc']['level_m'])+' &  '+p33(B['dins']['level_m'])+'& '+' \\textbf{'+p33(B['level']['all_m_m'])+'} &  \\textbf{'+p33(B['level']['all_m_w'])+'} &  \\textbf{'+p33(B['w_sh']['level_m'])+'}    \\\\\\bottomrule'
    with open(root+'/Output files/model/level.tex', 'w') as f: f.write(table); f.close() 
    
    
    # Compares passthroughs in the data and in the model
    PHp,PWp=simple_extract(root+'/Empirical analysis/Tables/elasticity_persistent_earnings.txt')
    PHt,PWt=simple_extract(root+'/Empirical analysis/Tables/elasticity_transitory_earnings.txt')
    PHa,PWa=simple_extract(root+'/Empirical analysis/Tables/elasticity_all_earnings.txt')
    
    # Table with pass throughs in the data and in the model
    # Table with pass throughs in the data and in the model
    table=r'...persistent husband shocks & \textbf{'+p33(B['BPP_PER']['ym_cm'])+'}/\\textit{\\textcolor{orange}{0.429}} & \\textbf{'+p33(B['BPP_PER']['ym_cw'])+'}/\\textit{\\textcolor{orange}{0.175}} &  \\\\ '+\
          r'...persistent wife shocks    & \textbf{'+p33(B['BPP_PER']['yw_cm'])+'}/\\textit{\\textcolor{orange}{0.029}} & \\textbf{'+p33(B['BPP_PER']['yw_cw'])+'}/\\textit{\\textcolor{orange}{0.281}} & \\\\[1.5ex] '+\
          r'...transitory husband shocks & \textbf{'+p33(B['BPP_MPC']['ym_cm'])+'}/\\textit{\\textcolor{orange}{0.060}} & \\textbf{'+p33(B['BPP_MPC']['ym_cw'])+'}/\\textit{\\textcolor{orange}{0.055}} &  \\\\ '+\
          r'...transitory wife shocks    & \textbf{'+p33(B['BPP_MPC']['yw_cm'])+'}/\\textit{\\textcolor{orange}{0.006}} & \\textbf{'+p33(B['BPP_MPC']['yw_cw'])+'}/\\textit{\\textcolor{orange}{0.038}}   &   \\\\\\bottomrule '
    
    with open(root+'/Output files/model/elasticity_BPP_model_vs_data.tex', 'w') as f: f.write(table); f.close() 
    
    #############################
    # PARAMETERS
    ############################
    table=r'\begin{table}[H]\centering'+\
          r'\caption{Estimated structural parameters}'+\
          r'\label{table:structural_params}'+\
          r'\begin{tabular}{lccc} \toprule '+\
          r'Estimated Parameters &  & Value & Target Moment  \\ '+\
          r' \midrule '+\
          r'Match-quality shock (each spouse), s.d. & $\sigma_{\psi}$ & '+p42(pt[1])+' & Annual divorce rate'+' \\\\'+\
          r'Single--couple utility wedge, wife & $Wedge_w$       & '+p42(pt[4])+r" & Reform effect on wife's cons.\ share"+' \\\\'+\
          r'Single--couple utility wedge, husband & $Wedge_m$    & '+p42(pt[5])+r" & Wife's share of private consumption"+' \\\\'+\
          r'Female income trend, level         & $\iota_{0w}$    & '+p42(pt[7])+r" & Wife/husband earnings ratio, workers"+' \\\\'+\
          r'Risk aversion, private goods       & $\rho$          & '+p42(pt[3])+r' & Home-exp.\ elasticity to total cons.'+' \\\\'+\
          r'Weight on home goods               & $\alpha$        & '+p42(pt[2])+' & Expenditure share of home goods'+'  \\\\'+\
          r'Disutility of employment           & $\eta$          & '+p42(pt[0])+r' & Empl.\ rate of married women'+' \\\\'+\
          r'Discount factor                    & $\beta$         & '+p42(pt[6])+r" & Wealth/husband's earnings"+' \\\\'+\
          r' \bottomrule '+\
          r'\end{tabular}'+\
          r'\end{table}'
    
    #Write table to tex file 
    with open(root+'/Output files/model/params.tex', 'w') as f: 
        f.write(table) 
        f.close() 

    #############################
    # MODEL FIT
    ############################
    table=r'\begin{table}[H]\caption{Model fit and validation}\label{table:fit}\centering'+\
         r'\begin{tabular}{lcc}\toprule '+\
        r'Targeted Moments & Data  & Model  \\ \midrule '+\
        r'Annual divorce rate              & '+p43(0.010)+' & '+p43(divorce_rate)+'  \\\\'+\
        r"Effect of pension reform on wife's consumption share & "+p43(0.014)+' & '+p43(policy_effect_wife_ratio)+' \\\\'+\
        r'Pass-through of total consumption to home-good expenditure & '+p43(.979)+' & '+p43(βdC)+' \\\\'+\
        r'Employment rate of married women        & '+p43(0.588)+' & '+p43(wife_empl)+'  \\\\'+\
        r'Expenditure share of home goods         & '+p43(0.812)+' & '+p43(expenditure_x_share)+'  \\\\'+\
        r"Wealth to husband's earnings ratio      & "+p43(2.634)+' & '+p43(couple_assets)+'  \\\\'+\
        r"Wife's share of private consumption     & "+p43(0.322)+' & '+p43(wife_cons_share)+'  \\\\'+\
        r"Wife-to-husband earnings ratio, working wives & "+p43(0.46)+' & '+p43(wife_ratio)+'  \\\\'+\
        r'\midrule '+\
        r'Non-targeted Moments & Data  & Model \\'+\
        r'\midrule '+\
        r'Female-to-male earnings ratio, workers  & '+p43(0.532)+' & '+p43(gender_gap_earnings)+'\\\\'+\
        r"Wife's employment response to husband's income shocks & "+p43(-0.023)+' &  '+p43(B['wlp']['all_m'])+'\\\\'+\
        r'\bottomrule '+\
        r'\end{tabular}'+\
        r'\end{table}'
        
        #r'Share of household income going to working women               &  0.33  & '+p42(share_iw)+' \\\\'+\
    
    #Write table to tex file 
    with open(root+'/Output files/model/fit.tex', 'w') as f: 
        f.write(table) 
        f.close() 
            
        
 
import numpy as np 
 
 
if __name__ == '__main__': 
     
   
    if ESTIMATE:
        
        # computation_options = { "num_workers" : 9,        # use four processes in parallel 
        #                         "working_dir" : root # where to save results in progress (in case interrupted) 
        #                         } 
         
        # global_search_options = { "num_points" : 10}  # number of points in global pre-test 
         
        # local_search_options = {  "algorithm"    : "dfols", # local search algorithm 
        #                                                       # can be either BOBYQA from NLOPT or NelderMead from scipy 
        #                           "num_restarts" : 18,      # how many local searches to do 
        #                           "shrink_after" : 9,       # after the first [shrink_after] restarts we begin searching 
        #                                                       # near the best point we have found so far 
        #                           "xtol_rel"     : 1e-6,     # relative tolerance on x 
        #                           "ftol_rel"     : 1e-6     # relative tolerance on f 
        #                         } 
         
        # opt = TikTak.TTOptimizer(computation_options, global_search_options, local_search_options, skip_global=True) 
        # x,fx = opt.minimize(q,xl,xu) 
        # print(f'The minimizer is s{x}') 
        # print(f'The objective value at the min is {fx}') 
        
        # Estimate the model
        res=dfols.solve(q, xc, rhobeg = 0.1, rhoend=1e-5, maxfun=100, bounds=(xl,xu),  
                    npt=len(xc)+5,scaling_within_bounds=True,   
                    user_params={'tr_radius.gamma_dec':0.98,'tr_radius.gamma_inc':1.0,  
                                  'tr_radius.alpha1':0.9,'tr_radius.alpha2':0.95},  
                    objfun_has_noise=False,print_progress=True) 
        
        # res = pybobyqa.solve(q, xc, rhobeg = 0.1, rhoend=1e-5, maxfun=100, bounds=(xl,xu),  
        #              npt=len(xc)+5,scaling_within_bounds=True,   
        #              user_params={'tr_radius.gamma_dec':0.98,'tr_radius.gamma_inc':1.0,  
        #                            'tr_radius.alpha1':0.9,'tr_radius.alpha2':0.95},  
        #              objfun_has_noise=False,print_progress=True)
        
        #Obtain tables
        #q(res.x,table=True)
        
    else:
        
        # Commpute tables with with paramters, and BPP given parameter xc
        q(xc,table=True)