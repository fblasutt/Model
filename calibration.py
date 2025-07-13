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

# Guess of internal parameters: [ν,σL,α,χ,wedge]
xc=np.array([0.35870364, 0.00596541, 0.89223739, 0.96875457, 0.71759143])


# Lower and higher bounds of parameters
xl=np.array([0.02,0.0001,0.1,0.8,0.5]) 
xu=np.array([0.8,0.2,0.99,2.0,1.2]) 

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
 
#Function to minimize 
def q(pt,table=False): 
 
    
    #########################################################################################################################################
    #Solve and simulate the model for given parametrization, then compute moments and their distance distance from the data
    #######################################################################################################################################
    try:  
           
        # Set up the model with the input parameters pt
        M = model.copy(name='numba_new_copy')    
        M.par.ν=pt[0] 
        M.par.grid_love,M.par.Πl,M.par.Πl0= usr.addaco_nonst(M.par.T,pt[1],pt[1],M.par.num_love)
        M.par.α=pt[2] 
        M.par.χ=pt[3]
        M.par.wedge=pt[4]        
    
        # Solve and simulate the model
        M.solve() 
        M.simulate() 
         
        #############################################
        #Sample selection in accordance with the data
        ############################################
        age=(np.cumsum(np.ones((M.par.simN,M.par.T)),axis=1)-1)+20#age of hh  
               
        # Sample used for divorce moments
        sample =  (age>=age_initial[:,None]) & (age<=age_final[:,None]) & (M.sim.couple_lag==1) 

        # Sample for all the other moments
        sample_c =  (age>=age_initial[:,None]) & (age<=age_final[:,None]) & (M.sim.couple_lag==1) & (M.sim.couple==1)  
        sample_c1=np.roll(sample_c,1,axis=1)
        
        # This sample will be used for pass throughs (if sample, BPP persistent will not work)
        sample_reg= (age>age_initial[:,None]) & (age<=age_final[:,None]) & (M.sim.couple_lag==1) 
        
         
        ######################################
        #Moments here
        ######################################       
        wife_empl = np.mean(M.sim.WLP[sample_c]>0)        
        divorce_rate=np.mean((M.sim.couple==0)[sample])
        divorce_rate_young=np.mean((M.sim.couple==0)[(sample) & (age<=35)])
        expenditure_x_share=np.mean((M.sim.dw/M.sim.C_tot)[sample_c])
        
        #construct regression of changes in consumption to changes in 
        ΔC =np.log(M.sim.C_tot[sample_c1])  -np.log(M.sim.C_tot[sample_c])
        Δd=np.log(M.sim.dw[sample_c1])  -np.log(M.sim.dw[sample_c])
        
        βdC=np.cov(ΔC,Δd)[0,1]/np.var(ΔC)
    
     
                        
        fit =((wife_empl-.567 )/.567)**2+((divorce_rate_young-.0115)/.0115)**2+((divorce_rate-.0101)/.0101)**2+((expenditure_x_share-.782)/.782)**2+((βdC-.8956945)/.8956945)**2
        print('Point is {}, fit is {}'.format(pt,fit))  
        print('Simulated moments are {}'.format([wife_empl,divorce_rate_young,divorce_rate,expenditure_x_share,βdC]))
        
        ###################################
        # Non-targeted moments
        ###################################
        gender_gap_earnings=(M.sim.incwg[sample][M.sim.WLP[sample]>0]).mean()/M.sim.incmg[sample].mean()
        share_full_time=(M.sim.WLP[sample][(M.sim.WLP[sample]>0) & (M.sim.couple[sample]==1)]==(M.par.num_wlp-1)).mean()
        
        print('Simulated moments are {}'.format([gender_gap_earnings,share_full_time]))
        
        
        # Function tables computes a lot of tables with results and fit. Should be activated only for the final solution
        if table:tables(M,sample_reg,pt,root,divorce_rate,divorce_rate_young,expenditure_x_share,wife_empl,βdC,gender_gap_earnings,share_full_time)
      
        return [((wife_empl-.567)/.567),((divorce_rate_young-.0115)/.0115),((divorce_rate-.0101)/.0101),((expenditure_x_share-.782)/.782),((βdC-.8956945)/.8956945)]   
     
    except:

        print("Global error! Point is {}".format(pt))
        return [10000.0,10000.0,10000.0,10000.0,10000.0]     
    
     
def tables(M,sample,pt,root,divorce_rate,divorce_rate_young,expenditure_x_share,wife_empl,βdC,gender_gap_earnings,share_full_time):
    
    # Extract empirical pass throughs from Sara's files
    def simple_extract(filename):
        """Simplified extraction for the specific format"""
        with open(filename, 'r') as file:
            content = file.read()
        
        # Extract all numbers from the file
        import re
        numbers = re.findall(r'-?\d+\.\d+', content)
        
    
        #Pass-throughs when the shock hits the husband (pass_husband) or wife (pass_wife)
        pass_husband={'tot':numbers[0],'com':numbers[1],'child':numbers[2],'hus':numbers[3],'wif':numbers[4],'wif_rel':numbers[5]}
        pass_wife   ={'tot':numbers[6],'com':numbers[7],'child':numbers[8],'hus':numbers[9],'wif':numbers[10],'wif_rel':numbers[11]}
        
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
    with open(root+'/Model/results/allinc.tex', 'w') as f: f.write(table); f.close() 
    
    #% changes in consumption out of a 1 % transitory change in income
    table=r"...total income   & \textbf{"+p33(B['totc']['tra'])+'} & '+p33(B['dins']['tra'])+' & & &    \\\\ '+\
          r'...wife income    & '+p33(B['totc']['tra_w'])+' & '+p33(B['dins']['tra_w'])+'& '+' \\textbf{'+p33(B['indc']['tra_w_m'])+'} &  \\textbf{'+p33(B['indc']['tra_w_w'])+'} &  \\textbf{'+p33(B['w_sh']['tra_w'])+'}    \\\\ '+\
          r'...husband income & '+p33(B['totc']['tra_m'])+' &  '+p33(B['dins']['tra_m'])+'& '+' \\textbf{'+p33(B['indc']['tra_m_m'])+'} &  \\textbf{'+p33(B['indc']['tra_m_w'])+'} &  \\textbf{'+p33(B['w_sh']['tra_m'])+'}    \\\\\\bottomrule'
    with open(root+'/Model/results/trainc.tex', 'w') as f: f.write(table); f.close() 
    
    
    #% changes in consumption out of a 1 % persistent change in income
    table=r"...total income   & \textbf{"+p33(B['totc']['per'])+'} & '+p33(B['dins']['per'])+' & & &    \\\\ '+\
          r'...wife income    & '+p33(B['totc']['per_w'])+' & '+p33(B['dins']['per_w'])+'& '+' \\textbf{'+p33(B['indc']['per_w_m'])+'} &  \\textbf{'+p33(B['indc']['per_w_w'])+'} &  \\textbf{'+p33(B['w_sh']['per_w'])+'}    \\\\ '+\
          r'...husband income & '+p33(B['totc']['per_m'])+' &  '+p33(B['dins']['per_m'])+'& '+' \\textbf{'+p33(B['indc']['per_m_m'])+'} &  \\textbf{'+p33(B['indc']['per_m_w'])+'} &  \\textbf{'+p33(B['w_sh']['per_m'])+'}    \\\\\\bottomrule'
    with open(root+'/Model/results/perinc.tex', 'w') as f: f.write(table); f.close() 
    
    
    #MPC tables
    table=r"...husband income & "+p33(B['BPP_MPC']['ym_tot'])+' & '+p33(B['BPP_MPC']['ym_d'])+' & '+p33(B['BPP_MPC']['ym_cm'])+' & '+p33(B['BPP_MPC']['ym_cw'])+'  \\\\ '+\
          r'...wife income    & '+p33(B['BPP_MPC']['yw_tot'])+' & '+p33(B['BPP_MPC']['yw_d'])+' & '+p33(B['BPP_MPC']['yw_cm'])+' & '+p33(B['BPP_MPC']['yw_cw'])+'  \\\\ '+\
          r'...total income   & '+p33(B['BPP_MPC']['al_tot'])+' & '+p33(B['BPP_MPC']['al_d'])+' & '+p33(B['BPP_MPC']['al_cm'])+' & '+p33(B['BPP_MPC']['al_cw'])+'  \\\\\\bottomrule'
    with open(root+'/Model/results/BPP_MPC.tex', 'w') as f: f.write(table); f.close() 
    
    #BPP persistent tables
    table=r"...husband income & "+p33(B['BPP_PER']['ym_tot'])+' & '+p33(B['BPP_PER']['ym_d'])+' & '+p33(B['BPP_PER']['ym_cm'])+' & '+p33(B['BPP_PER']['ym_cw'])+'  \\\\ '+\
          r'...wife income    & '+p33(B['BPP_PER']['yw_tot'])+' & '+p33(B['BPP_PER']['yw_d'])+' & '+p33(B['BPP_PER']['yw_cm'])+' & '+p33(B['BPP_PER']['yw_cw'])+'  \\\\ '+\
          r'...total income   & '+p33(B['BPP_PER']['al_tot'])+' & '+p33(B['BPP_PER']['al_d'])+' & '+p33(B['BPP_PER']['al_cm'])+' & '+p33(B['BPP_PER']['al_cw'])+'  \\\\\\bottomrule '
    with open(root+'/Model/results/BPP_PER.tex', 'w') as f: f.write(table); f.close() 
    
    #labor supply	
    table=r'  '+p33(B['wlp']['tra_w'])+' & '+p33(B['wlp']['tra_m'])+' & '+p33(B['wlp']['per_w'])+' & '+p33(B['wlp']['per_m'])+' & '+p33(B['wlp']['all_w'])+' & '+p33(B['wlp']['all_m'])+'  \\\\\\bottomrule '
    with open(root+'/Model/results/WLP.tex', 'w') as f: f.write(table); f.close() 
    
    #level changes in consumption out of a level change in all income
    table=r"...total income   & \textbf{"+p33(B['totc']['level_s'])+'} & '+p33(B['dins']['level_s'])+' & & &    \\\\ '+\
          r'...wife income    & '+p33(B['totc']['level_w'])+' & '+p33(B['dins']['level_w'])+'& '+' \\textbf{'+p33(B['level']['all_w_m'])+'} &  \\textbf{'+p33(B['level']['all_w_w'])+'} &  \\textbf{'+p33(B['w_sh']['level_w'])+'}    \\\\ '+\
          r'...husband income & '+p33(B['totc']['level_m'])+' &  '+p33(B['dins']['level_m'])+'& '+' \\textbf{'+p33(B['level']['all_m_m'])+'} &  \\textbf{'+p33(B['level']['all_m_w'])+'} &  \\textbf{'+p33(B['w_sh']['level_m'])+'}    \\\\\\bottomrule'
    with open(root+'/Model/results/level.tex', 'w') as f: f.write(table); f.close() 
    
    
    # Compares passthroughs in the data and in the model
    PHp,PWp=simple_extract(root+'/Tables/elasticity_persistent_earnings.txt')
    PHt,PWt=simple_extract(root+'/Tables/elasticity_transitory_earnings.txt')
    PHa,PWa=simple_extract(root+'/Tables/elasticity_all_earnings.txt')
    
    # Table with pass throughs in the data and in the model
    table=r'...any husband shocks & \textbf{'+p33(B['indc']['all_m_m'])+'}/\\textcolor{red}{'+PHa['hus']+'} & \\textbf{'+p33(B['indc']['all_m_w'])+'}/\\textcolor{red}{'+PHa['wif']+'} & \\textbf{'+p33(B['dins']['all_m'])+'}/\\textcolor{red}{'+PHa['com']+'}  & \\textbf{'+p33(B['w_sh']['all_m'])+'}/\\textcolor{red}{'+PHa['wif_rel']+'} \\\\ '+\
          r'...any wife shocks    & \textbf{'+p33(B['indc']['all_w_m'])+'}/\\textcolor{red}{'+PWa['hus']+'} & \\textbf{'+p33(B['indc']['all_w_w'])+'}/\\textcolor{red}{'+PWa['wif']+'} & \\textbf{'+p33(B['dins']['all_w'])+'}/\\textcolor{red}{'+PWa['com']+'}  & \\textbf{'+p33(B['w_sh']['all_w'])+'}/\\textcolor{red}{'+PWa['wif_rel']+'} \\\\[1.5ex] '+\
          r'...persistent husband shocks & \textbf{'+p33(B['BPP_PER']['ym_cm'])+'}/\\textcolor{red}{'+PHp['hus']+'} & \\textbf{'+p33(B['BPP_PER']['ym_cw'])+'}/\\textcolor{red}{'+PHp['wif']+'} &  \\\\ '+\
          r'...persistent wife shocks    & \textbf{'+p33(B['BPP_PER']['yw_cm'])+'}/\\textcolor{red}{'+PWp['hus']+'} & \\textbf{'+p33(B['BPP_PER']['yw_cw'])+'}/\\textcolor{red}{'+PWp['wif']+'} & \\\\[1.5ex] '+\
          r'...transitory husband shocks   & \textbf{'+p33(B['BPP_MPC']['ym_cm'])+'}/\\textcolor{red}{'+PHt['hus']+'} & \\textbf{'+p33(B['BPP_MPC']['ym_cw'])+'}/\\textcolor{red}{'+PHt['wif']+'} &  \\\\ '+\
          r'...transitory wife shocks    & \textbf{'+p33(B['BPP_MPC']['yw_cm'])+'}/\\textcolor{red}{'+PWt['hus']+'} & \\textbf{'+p33(B['BPP_MPC']['yw_cw'])+'}/\\textcolor{red}{'+PWt['wif']+'}   &   \\\\\\bottomrule '
    
    
    with open(root+'/Model/results/elasticity_BPP_model_vs_data.tex', 'w') as f: f.write(table); f.close() 
    
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
          r'Singl-Couple wedge                     & $Wedge$          & '+p42(pt[4])+' & Divorce rate, younger women'+'  \\\\'+\
          r'Home goods utility curvature                      & $\chi$         & '+p42(pt[3])+' & Consumption to home goods pass-through'+' \\\\'+\
          r'Weight on home goods                              & $\alpha$          & '+p42(pt[2])+' & Women employment rate'+'  \\\\'+\
          r'Home input weight            & $\nu$            & '+p42(pt[0])+' &  Expenditure share on common goods'+' \\\\'+\
          r' \bottomrule '+\
          r'\end{tabular}'+\
          r'\end{table}'
    
    #Write table to tex file 
    with open(root+'/Tables/params.tex', 'w') as f: 
        f.write(table) 
        f.close() 

    #############################
    # MODEL FIT
    ############################
    table=r'\begin{table}[H]\caption{Model fit and validation}\label{table:fit}\centering'+\
         r'\begin{tabular}{lccc}\toprule '+\
        r'Target Moments & Data  & Model  \\ \midrule '+\
        r'Divorce rate, all women              & '+p43(0.0101)+' & '+p43(divorce_rate)+'  \\\\'+\
        r'Divorce rate, younger women               & '+p43(0.0115)+' & '+p43(divorce_rate_young)+' \\\\'+\
        r'Priv. cons. share to hus. income pass-through        & '+p43(0.895)+' & '+p43(βdC)+' \\\\'+\
        r'Women employment rate                   & '+p43(0.567)+' & '+p43(wife_empl)+'  \\\\'+\
        r'Expenditure share on common goods                & '+p43(0.782)+' & '+p43(expenditure_x_share)+'  \\\\'+\
        r'\midrule '+\
        r'External Moments & Data  & Model \\'+\
        r'\midrule '+\
        r'Gender earnings gap                                       & '+p43(0.52)+' & '+p43(gender_gap_earnings)+'\\\\'+\
        r'Share of women working full-time                          & '+p43(0.356)+' &  '+p43(share_full_time)+'\\\\'+\
        r'Cross-elasticity of women employment                          & '+p43(-0.03)+' &  '+p43(B['wlp']['all_m'])+'\\\\'+\
        r'\bottomrule '+\
        r'\end{tabular}'+\
        r'\end{table}'
        
        #r'Share of household income going to working women               &  0.33  & '+p42(share_iw)+' \\\\'+\
    
    #Write table to tex file 
    with open(root+'/Tables/fit.tex', 'w') as f: 
        f.write(table) 
        f.close() 
            
        
 
import numpy as np 
 
 
if __name__ == '__main__': 
     
    # Estimate the model
    res=dfols.solve(q, xc, rhobeg = 0.05, rhoend=1e-3, maxfun=100, bounds=(xl,xu),  
                npt=len(xc)+5,scaling_within_bounds=True,   
                user_params={'tr_radius.gamma_dec':0.98,'tr_radius.gamma_inc':1.0,  
                              'tr_radius.alpha1':0.9,'tr_radius.alpha2':0.95},  
                objfun_has_noise=False,print_progress=True) 
    
    # Obtain tables
    q(res.x,table=True)