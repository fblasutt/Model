# -*- coding: utf-8 -*- 
""" 
Created on Fri Feb 23 15:31:56 2024 
 
@author: 32489 
""" 
 
import numpy as np 
import Bargaining_numba as brg  
import UserFunctions_numba as usr 
import dfols 
import pandas as pd
 
#Initialize seed 
np.random.seed(10) 
 
#Create sample with replacement 
N=10_000#sample size 

#Root
root='C:/Users/32489/Dropbox/Family Risk Sharing'


baseline_sample=np.array(pd.read_excel(root+'/Output files/data_sample.csv'))

pr=np.ones(baseline_sample.shape[0])/baseline_sample.shape[0]
indexes=np.array(np.random.choice(baseline_sample[:,0], size=N, p=pr, replace=True),dtype=np.int32)-1
final_sample= baseline_sample[:,1:][indexes] 

age_initial=final_sample[:,0]
age_final=final_sample[:,1]
married_initial=final_sample[:,2]

# POints: [θ,λ,σL,α2,γ]
xc=np.array([0.46874957, 0.05944867, 0.06511688, 0.78126218, 0.46638862])
xc=np.array([0.29749257, 0.11708597, 0.35,       0.85,       0.3 ])
xc=np.array([0.4264216 , 0.09909561, 0.18060578, 0.96689822, 0.23019184])
xl=np.array([0.20,0.004,0.0001,0.50,0.1]) 
xu=np.array([0.95,0.9,0.99,0.99,0.85]) 


#Parametrize the model 
par = {'simN':N,'θ': xc[0], 'meet':xc[1],'σL0':xc[2],'σL':xc[2],'α2':xc[3],'α1':1.0-xc[3],'γ':xc[4]} 
model = brg.HouseholdModelClass(par=par)  

 
#Function to minimize 
def q(pt,tables=False): 
 
 
    
    ########################################################
    #Solve and simulate the model for given parametrization 
    ######################################################
    try:  
        

        M = model.copy(name='numba_new_copy')    
        M.par.θ=pt[0] 
        M.par.meet=pt[1]
        M.par.λ_grid = np.ones(M.par.T)*pt[1] 
        M.par.grid_love,M.par.Πl,M.par.Πl0= usr.rouw_nonst(M.par.T,pt[2],pt[2],M.par.num_love)    
        M.par.α2=pt[3] 
        M.par.α1=1.0-pt[3]
        M.par.γ=pt[4]
        M.solve() 
        M.simulate() 
         

        ######################################
        #Sample selection
        ######################################
        
        Sage_initial=(np.arange(M.par.simN),age_initial)
        Sage_final=(np.arange(M.par.simN),age_final)
        
        age=(np.cumsum(np.ones((M.par.simN,M.par.T)),axis=1)-1)+20#age of hh 
        wasmarried=np.repeat(M.sim.couple[Sage_initial][:,None],M.par.T,axis=1)
        
        sample = (wasmarried==married_initial[:,None]) & (age>=age_initial[:,None]) & (age<=age_final[:,None])
        
         
        ######################################
        #Moments here
        ######################################
        wife_empl = np.mean(M.sim.WLP[sample][M.sim.couple[sample]==1]>0)        
        
        
        ismarried,isdivorced=np.zeros((2,M.par.simN,M.par.T),dtype=bool)
        ismarried[sample]=M.sim.couple[sample]
        isdivorced[sample]=(M.sim.couple[sample]==0) & (M.sim.couple_lag[sample]==1)
        
        ever_married = (np.cumsum(M.sim.couple,axis=1)>0)[Sage_final].mean()       
        ever_divorced = (np.cumsum(isdivorced,axis=1))[Sage_final].mean()      
        expenditure_wife_share=np.mean((M.sim.Cw/(M.sim.Cw+M.sim.Cm))[sample][M.sim.couple[sample]==1])
        expenditure_x_share=np.mean((M.sim.xw/M.sim.C_tot)[sample][M.sim.couple[sample]==1])
    
 
                        
        fit =((wife_empl-0.596 )/.596)**2+((ever_married-.768)/.768)**2+((ever_divorced-.0976)/.0976)**2+((expenditure_wife_share-0.347/.347))**2+((expenditure_x_share-0.785)/.785)**2
        print('Point is {}, fit is {}'.format(pt,fit))  
        print('Simulated moments are {}'.format([wife_empl,ever_married,ever_divorced,expenditure_wife_share,expenditure_x_share]))
        
        ###################################
        # Non-targeted moments
        ###################################
        gender_gap_earnings=((M.par.grid_wlp[M.sim.WLP[sample]]*M.sim.incw[sample])[M.sim.WLP[sample]>0]).mean()/M.sim.incm[sample].mean()
        share_full_time=(M.sim.WLP[sample][(M.sim.WLP[sample]>0) & (M.sim.couple[sample]==1)]==(M.par.num_wlp-1)).mean()
        
        print('Simulated moments are {}'.format([gender_gap_earnings,share_full_time]))
        
        if tables:tables(pt,root,ever_divorced,ever_married,expenditure_x_share,wife_empl,expenditure_wife_share,gender_gap_earnings,share_full_time)
      
        return [((wife_empl-.596)/.596),((ever_married-.768)/.768),((ever_divorced-.0976)/.0976),((expenditure_wife_share-0.347)/0.347),((expenditure_x_share-0.785)/0.785)]   
     
    except:

        print("Global error! Point is {}".format(pt))
        return [10000.0,10000.0,10000.0,10000.0,10000.0]     
     
     
def tables(pt,root,ever_divorced,ever_married,expenditure_x_share,wife_empl,expenditure_wife_share,gender_gap_earnings,share_full_time):
    
            
    def p42(x): return str('%4.2f' % x)  
    def p43(x): return str('%4.3f' % x)     
    def p40(x): return str('%4.0f' % x)  
    
    #############################
    # PARAMETERS
    ############################
    table=r'\begin{table}[H]\centering'+\
          r'\caption{Estimated structural parameters}'+\
          r'\label{table:structural_params}'+\
          r'\begin{tabular}{lccc} \toprule '+\
          r'Estimated Parameters &  & Value & Target Moment  \\ '+\
          r' \midrule '+\
          r'Standard deviation of match quality shock         & $\sigma_{\psi}$   & '+p42(pt[2])+' & Share of women ever divorced'+' \\\\'+\
          r'Probability of meeting a partner                  & $\lambda$         & '+p42(pt[1])+' & Share of women ever married'+' \\\\'+\
          r'Weight on home goods                              & $\alpha$          & '+p42(pt[3])+' & Share of non-private expenditures'+'  \\\\'+\
          r'Home production, weight on home inputs            & $\chi$            & '+p42(pt[0])+' & Married women employment rate'+' \\\\'+\
          r'Women weight, Nash bargaining                     & $\gamma$          & '+p42(pt[4])+' & Share of wives private consumption'+'  \\\\'+\
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
            r'Share of women who have ever divorced              & '+p42(0.0836)+' & '+p42(ever_divorced)+'  \\\\'+\
            r'Share of women who have ever married               & '+p42(0.718)+' & '+p42(ever_married)+' \\\\'+\
            r'Share of common and children’s expenditures        & '+p42(0.785)+' & '+p42(expenditure_x_share)+' \\\\'+\
            r'Married women’s employment rate                    & '+p42(0.596)+' & '+p42(wife_empl)+'  \\\\'+\
            r'Share of wife’s private consumption                & '+p42(0.347)+' & '+p42(expenditure_wife_share)+'  \\\\'+\
            r'\midrule '+\
            r'External Moments & Data  & Model \\'+\
            r'\midrule '+\
            r'Gender earnings gap                                       & '+p42(0.5055)+' & '+p42(gender_gap_earnings)+'\\\\'+\
            r'Share of women working full-time                          & '+p42(0.427490)+' &  '+p42(share_full_time)+'\\\\'+\
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
     
 


    res=dfols.solve(q, xc, rhobeg = 0.1, rhoend=1e-4, maxfun=100, bounds=(xl,xu),  
                npt=len(xc)+5,scaling_within_bounds=True,   
                user_params={'tr_radius.gamma_dec':0.98,'tr_radius.gamma_inc':1.0,  
                              'tr_radius.alpha1':0.9,'tr_radius.alpha2':0.95},  
                objfun_has_noise=False,print_progress=True) 