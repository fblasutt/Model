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



# Guess of internal parameters: [ω,σL,α,ρ,wedge,β]

xc=np.array([0.55, 0.1       , 0.85, 1.2, 0.929     ,1.        ])

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


#Function to minimize 
def q(pt,table=False): 
 
    
    #########################################################################################################################################
    #Solve and simulate the model for given parametrization, then compute moments and their distance distance from the data
    #######################################################################################################################################
    try:  
           
        ###################
        #Pre reform model
        ####################
        #tic=time.time()
        
        # Set up the model with the input parameters pt
        M_bef = model.copy(name='numba_new_copy')
       
        M_bef.par.ω=pt[0]
        M_bef.par.grid_lovew,M_bef.par.Πlw,M_bef.par.Πlw0= usr.rouw_nonst(M_bef.par.T,pt[1],M_bef.par.σL0,M_bef.par.num_lovew) 
        M_bef.par.grid_lovem,M_bef.par.Πlm,M_bef.par.Πlm0= usr.rouw_nonst(M_bef.par.T,pt[1],M_bef.par.σL0,M_bef.par.num_lovem) 
        
        
        M_bef.par.Πl=[np.kron(M_bef.par.Πlw[t],M_bef.par.Πlm[t]) for t in range(M_bef.par.T-1)] # couples trans matrix
        M_bef.par.Πl0=[np.kron(M_bef.par.Πlw0[t],M_bef.par.Πlm0[t]) for t in range(M_bef.par.T-1)] # couples trans matrix 
        M_bef.par.α=pt[2] 
        M_bef.par.ρ=pt[3]
        M_bef.par.wedge=pt[4]    
        M_bef.par.β=pt[5]
    
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
        M.par.grid_lovew,M.par.Πlw,M.par.Πlw0= usr.rouw_nonst(M.par.T,pt[1],M.par.σL0,M.par.num_lovew) 
        M.par.grid_lovem,M.par.Πlm,M.par.Πlw0= usr.rouw_nonst(M.par.T,pt[1],M.par.σL0,M.par.num_lovem) 
        
        
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
        M.par.β=pt[5]        
    
        # Solve and simulate the model
        M.solve() 
        M.simulate() 
        
        #toc=time.time()
        #print('Time elapsed for model solution is {}'.format(toc-tic))
        
        aaa=M_bef.sim.power[age==age_marriage[:,None]]
        print(np.mean(aaa==model.sim.init_power))
        
        #############################################
        #Event Study: policy and consumption ratio
        ############################################
      
        #Covariates
        time_to_policy= (np.cumsum(np.ones((M.par.simN,M.par.T)),axis=1)-1)-M.par.policy_init[:,None]
        event_time=time_to_policy.copy()
        idd=np.repeat(np.cumsum(np.ones(M.par.simN))[:,None],M.par.T,axis=1)   
        policy_init= np.repeat((M.par.policy_init)[:,None],M.par.T,axis=1) 
        agei=np.repeat((age_marriage)[:,None],M.par.T,axis=1) 
        time=age-agei
        
        #Event-study specific variables
        treat_group=np.repeat((M.par.policy_init>=15)[:,None],M.par.T,axis=1) 
        event_time[event_time<=-5]=-5
        event_time[event_time>=5]=5
        event_time_PER_treat=event_time*treat_group 
        wife_ratio=M.sim.Cw/(M.sim.Cm)

        #Sample
        subset=   (age>=age_marriage[:,None]) & (M.sim.power>=0) & (age_policy>age_marriage-20+5)[:,None] 
    
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
            "treat_group":treat_group[subset],
            "event_time_PER_treat":event_time_PER_treat[subset] 
        }) 
         
       
        reference_value=-1
        event_cats = sorted(df['event_time'].unique()) 
        if reference_value in event_cats: 
            event_cats.remove(reference_value) 
            event_cats = [reference_value] + event_cats 
             
 
        # Step 1: Create the fixed effects structure 
        fe_df = df[[ 'event_time','idd','age']].astype('category')

         
        # Step 2: Create the HDFE projector 
        hdfe = create(fe_df) 
         
        # Create categorical with this ordering 
        df['event_cat'] = pd.Categorical(df['event_time_PER_treat'], categories=event_cats) 
     
        # Create dummies, drop_first will now drop your reference group 
        event_dummies = pd.get_dummies(df['event_cat'], prefix='event', drop_first=True) 
         
        
        # Residualize both y and X 
        y_resid = hdfe.residualize(df[['wife_ratio']].values) 
        X_resid = hdfe.residualize(event_dummies.values) 
        #X2_resid = hdfe.residualize(df[['inter']].values) 
         
        # OLS on residuals 
        model_ = sm.OLS(y_resid, X_resid) 
        results = model_.fit() 
        
        params=np.insert(results.params, 4, 0)
        plt.plot(np.linspace(-4,4,9),params[1:-1])
        
        policy_effect_wife_ratio=params[5:-1].mean()#sm.OLS(y_resid, X2_resid).fit().params[0] 
        if np.isnan(policy_effect_wife_ratio):policy_effect_wife_ratio=10.0
         
        ######################################
        #Other moments here
        ######################################  
     
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
        

        ΔC =np.log(M.sim.C_tot[sample_pass])#  -np.log(M.sim.C_tot[sample_pass_m1])
        Δd=np.log(M.sim.dw[sample_pass]) # -np.log(M.sim.dw[sample_pass_m1])     
        βdC=np.cov(ΔC,Δd)[0,1]/np.var(ΔC)
        
        # Average household income
        couple_assets = M.sim.A[sample_empl].mean()/M.sim.incm[sample_empl].mean()
           
    
        fit =((wife_empl-0.5879)/0.5879)**2+((policy_effect_wife_ratio-.065)/.065)**2+((divorce_rate-0.0103)/0.0103)**2+((expenditure_x_share-0.812)/0.812)**2+((βdC-.97899)/.97899)**2+((couple_assets-2.967)/2.967)**2
        print('Point is {}, fit is {}'.format(pt,fit))  
        print('Simulated moments are {}'.format([wife_empl,policy_effect_wife_ratio,divorce_rate,expenditure_x_share,βdC,couple_assets]))
        
        ###################################
        # Non-targeted moments
        ###################################
        gender_gap_earnings=(M.sim.incwg[sample_empl][M.sim.WLP[sample_empl]>0]).mean()/M.sim.incmg[sample_empl].mean()
        share_full_time=(M.sim.WLP[sample_empl][(M.sim.WLP[sample_empl]>0) & (M.sim.couple[sample_empl]==1)]==(M.par.num_wlp-1)).mean()
        
        print('Simulated moments are {}'.format([gender_gap_earnings,share_full_time,policy_effect_wife_ratio]))

        
        # Function tables computes a lot of tables with results and fit. Should be activated only for the final solution
        if table:tables(M,sample_reg,pt,root,divorce_rate,policy_effect_wife_ratio,expenditure_x_share,wife_empl,βdC,couple_assets,gender_gap_earnings,share_full_time)
      
        #fitt=[((wife_empl-.567)/.567),((divorce_rate_young-.0109)/.0109),((divorce_rate-.0101)/.0101),((expenditure_x_share-.782)/.782),((βdC-.9)/.9)]   
        fitt=[((wife_empl-0.5879)/0.5879),((policy_effect_wife_ratio-.065)/.065),((divorce_rate-0.0103)/0.0103),((expenditure_x_share-.812)/.812),((βdC-.97899)/.97899),((couple_assets-2.967)/2.967)]   

        if np.isnan(fitt).max():fitt=[10000.0,10000.0,10000.0,10000.0,10000.0,10000.0]#   
        return fitt#fit#
    
    except:

        print("Global error! Point is {}".format(pt))
        return [10000.0,10000.0,10000.0,10000.0,10000.0,10000.0]     
    
     
def tables(M,sample,pt,root,divorce_rate,policy_effect_wife_ratio,expenditure_x_share,wife_empl,βdC,couple_assets,gender_gap_earnings,share_full_time):
    
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
    table=r'...any husband shocks & \textbf{'+p33(B['indc']['all_m_m'])+'}/\\textcolor{red}{'+PHa['hus']+'} & \\textbf{'+p33(B['indc']['all_m_w'])+'}/\\textcolor{red}{'+PHa['wif']+'} & \\textbf{'+p33(B['dins']['all_m'])+'}/\\textcolor{red}{'+PHa['com']+'}  & \\textbf{'+p33(B['w_sh']['all_m'])+'}/\\textcolor{red}{'+PHa['wif_rel']+'} \\\\ '+\
          r'...any wife shocks    & \textbf{'+p33(B['indc']['all_w_m'])+'}/\\textcolor{red}{'+PWa['hus']+'} & \\textbf{'+p33(B['indc']['all_w_w'])+'}/\\textcolor{red}{'+PWa['wif']+'} & \\textbf{'+p33(B['dins']['all_w'])+'}/\\textcolor{red}{'+PWa['com']+'}  & \\textbf{'+p33(B['w_sh']['all_w'])+'}/\\textcolor{red}{'+PWa['wif_rel']+'} \\\\[1.5ex] '+\
          r'...persistent husband shocks & \textbf{'+p33(B['BPP_PER']['ym_cm'])+'}/\\textcolor{red}{'+PHp['hus']+'} & \\textbf{'+p33(B['BPP_PER']['ym_cw'])+'}/\\textcolor{red}{'+PHp['wif']+'} &  \\\\ '+\
          r'...persistent wife shocks    & \textbf{'+p33(B['BPP_PER']['yw_cm'])+'}/\\textcolor{red}{'+PWp['hus']+'} & \\textbf{'+p33(B['BPP_PER']['yw_cw'])+'}/\\textcolor{red}{'+PWp['wif']+'} & \\\\[1.5ex] '+\
          r'...transitory husband shocks   & \textbf{'+p33(B['BPP_MPC']['ym_cm'])+'}/\\textcolor{red}{'+PHt['hus']+'} & \\textbf{'+p33(B['BPP_MPC']['ym_cw'])+'}/\\textcolor{red}{'+PHt['wif']+'} &  \\\\ '+\
          r'...transitory wife shocks    & \textbf{'+p33(B['BPP_MPC']['yw_cm'])+'}/\\textcolor{red}{'+PWt['hus']+'} & \\textbf{'+p33(B['BPP_MPC']['yw_cw'])+'}/\\textcolor{red}{'+PWt['wif']+'}   &   \\\\\\bottomrule '
    
    
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
          r'Match quality shock, St. dev.         & $\sigma_{\psi}$   & '+p42(pt[1])+' & Divorce rate, all women'+' \\\\'+\
          r'Single-Couple wedge                     & $Wedge$          & '+p42(pt[4])+' & Divorce rate, younger women'+'  \\\\'+\
          r'Private goods utility curvature                      & $\sigma$         & '+p42(pt[3])+' & Consumption to home goods pass-through'+' \\\\'+\
          r'Weight on home goods                              & $\alpha$          & '+p42(pt[2])+' & Women employment rate'+'  \\\\'+\
          r'Home input weight            & $\nu$            & '+p42(pt[0])+' &  Expenditure share on common goods'+' \\\\'+\
          r'Discount factor            & $\beta$            & '+p42(pt[5])+' &  Wealth to (husband) earnings ratio'+' \\\\'+\
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
         r'\begin{tabular}{lccc}\toprule '+\
        r'Target Moments & Data  & Model  \\ \midrule '+\
        r'Divorce rate, all women              & '+p43(0.010)+' & '+p43(divorce_rate)+'  \\\\'+\
        r'Reform effect on consumtion ratio               & '+p43(0.065)+' & '+p43(policy_effect_wife_ratio)+' \\\\'+\
        r'Total to public cons pass-through        & '+p43(.979)+' & '+p43(βdC)+' \\\\'+\
        r'Women employment rate                   & '+p43(0.588)+' & '+p43(wife_empl)+'  \\\\'+\
        r'Expenditure share on common goods                & '+p43(0.812)+' & '+p43(expenditure_x_share)+'  \\\\'+\
        r'Wealth to (husband) earnings ratio                & '+p43(2.967)+' & '+p43(couple_assets)+'  \\\\'+\
        r'\midrule '+\
        r'External Moments & Data  & Model \\'+\
        r'\midrule '+\
        r'Gender earnings gap                                       & '+p43(0.50)+' & '+p43(gender_gap_earnings)+'\\\\'+\
        r'Share of women working full-time                          & '+p43(0.443)+' &  '+p43(share_full_time)+'\\\\'+\
        r'Cross-elasticity of women employment                          & '+p43(-0.026)+' &  '+p43(B['wlp']['all_m'])+'\\\\'+\
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
        
        # Estimate the model
        res=dfols.solve(q, xc, rhobeg = 0.1, rhoend=1e-4, maxfun=100, bounds=(xl,xu),  
                    npt=len(xc)+5,scaling_within_bounds=True,   
                    user_params={'tr_radius.gamma_dec':0.98,'tr_radius.gamma_inc':1.0,  
                                  'tr_radius.alpha1':0.9,'tr_radius.alpha2':0.95},  
                    objfun_has_noise=False,print_progress=True) 
        
        # Obtain tables
        q(res.x,table=True)
        
    else:
        
        # Commpute tables with with paramters, and BPP given parameter xc
        q(xc,table=True)