# -*- coding: utf-8 -*- 
""" 
Created on Fri Feb 23 15:31:56 2024 
 
@author: 32489 
""" 
 
import numpy as np 
import Bargaining_numba as brg  
import UserFunctions_numba as usr 
import dfols 
from consav import linear_interp


 
#Initialize seed 
np.random.seed(10) 
 
#Create sample with replacement 
N=10_000#sample size 


# POints: [θ,σL0,σL,α2,γ]
xc=np.array([0.49552882, 0.05758893, 0.10326143, 0.77329494, 0.46355014])
xl=np.array([0.20,0.004,0.0001,0.50,0.3]) 
xu=np.array([0.95,0.9,0.35,0.85,0.85]) 


#Parametrize the model 
par = {'simN':N,'θ': xc[0], 'meet':xc[1],'σL0':xc[2],'σL':xc[2],'α2':xc[3],'α1':1.0-xc[3],'γ':xc[4]} 
model = brg.HouseholdModelClass(par=par)  

 
#Function to minimize 
def q(pt): 
 
 
    
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
        #Moments here
        ######################################
        wife_empl = np.mean(M.sim.WLP[:,7:M.par.Tr][M.sim.couple[:,7:M.par.Tr]==1])
        share_wife_earnings=M.sim.incw[:,7:M.par.Tr][M.sim.WLP[:,7:M.par.Tr]==1].mean()/M.sim.incm[:,7:M.par.Tr].mean()
        
        ever_married = np.mean(np.cumsum(M.sim.couple[:,7:M.par.Tr]==1,axis=1)[:,14]>0) 
        ever_divorced = np.mean(np.cumsum((M.sim.couple_lag[:,7:M.par.Tr]==1) & (M.sim.couple[:,7:M.par.Tr]==0),axis=1)[:,14]>0) 
        
 
        expenditure_wife_share=np.mean((M.sim.Cw/(M.sim.Cw+M.sim.Cm))[:,7:M.par.Tr][M.sim.couple[:,7:M.par.Tr]==1])
        expenditure_x_share=np.mean((M.sim.xw/M.sim.C_tot)[:,7:M.par.Tr][M.sim.couple[:,7:M.par.Tr]==1])
    
                        
        fit =((wife_empl-0.596 )/.596)**2+((ever_married-.718)/.718)**2+((ever_divorced-.0836427)/.0836427)**2+((expenditure_wife_share-0.347/.347))**2+((expenditure_x_share-0.785)/.785)**2
        print('Point is {}, fit is {}'.format(pt,fit))  
        print('Simulated moments are {}'.format([wife_empl,ever_married,ever_divorced,expenditure_wife_share,expenditure_x_share,share_wife_earnings]))
      
        return [((wife_empl-.596)/.596),((ever_married-.718)/.718),((ever_divorced-.0836427)/.0836427),((expenditure_wife_share-0.347)/0.347),((expenditure_x_share-0.785)/0.785)]   
     
    except:

        print("Global error! Point is {}".format(pt))
        return [10000.0,10000.0,10000.0,10000.0,10000.0]     
     
     
 
 
import numpy as np 
 
 
if __name__ == '__main__': 
     
 


    res=dfols.solve(q, xc, rhobeg = 0.1, rhoend=1e-4, maxfun=100, bounds=(xl,xu),  
                npt=len(xc)+5,scaling_within_bounds=True,   
                user_params={'tr_radius.gamma_dec':0.98,'tr_radius.gamma_inc':1.0,  
                              'tr_radius.alpha1':0.9,'tr_radius.alpha2':0.95},  
                objfun_has_noise=False,print_progress=True) 