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
xc=np.array([0.59711125, 0.08309608, 0.07246665, 0.75837416, 0.50298593])
xl=np.array([0.20,0.004,0.0001,0.50,0.3]) 
xu=np.array([0.95,0.5,0.15,0.85,0.65]) 

#Parametrize the model 
par = {'simN':N,
       'θ': xc[0], 'meet':xc[1],'σL0':xc[2],'σL':xc[2],'α2':xc[3],'α1':1.0-xc[3],'γ':xc[4]} 
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
        M.par.grid_love,M.par.Πl,M.par.Πl0= usr.addaco_nonst(M.par.T,pt[2],pt[2],M.par.num_love)    
        M.par.α2=pt[3] 
        M.par.α1=1.0-pt[3]
        M.par.γ=pt[4]
        M.solve() 
        M.simulate() 
         
       
        #################################
        #Consumption  below
        ##################################
    
        #Women individual consumption
        poww=np.fmax(np.fmin(M.sim.power,0.999),0.001)
        CwP,CwNP,Cw=np.zeros((3,M.par.simT*M.par.simN))  
        linear_interp.interp_2d_vec(M.par.grid_power,M.par.grid_Ctot,M.sol.pre_Ctot_Cw_priv[1],poww.flatten(),M.sim.C_tot.flatten(),CwP)  
        linear_interp.interp_2d_vec(M.par.grid_power,M.par.grid_Ctot,M.sol.pre_Ctot_Cw_priv[0],poww.flatten(),M.sim.C_tot.flatten(),CwNP)  
        Cw = (M.sim.WLP.flatten()*CwP+(1-M.sim.WLP.flatten())*CwNP).reshape((M.par.simN,M.par.simT))  
        Cw[M.sim.couple==0]=np.nan 
    
        #Men individual consumption
        CmP,CmNP,Cm=np.zeros((3,M.par.simT*M.par.simN)) 
        linear_interp.interp_2d_vec(M.par.grid_power,M.par.grid_Ctot,M.sol.pre_Ctot_Cm_priv[1],poww.flatten(),M.sim.C_tot.flatten(),CmP) 
        linear_interp.interp_2d_vec(M.par.grid_power,M.par.grid_Ctot,M.sol.pre_Ctot_Cm_priv[0],poww.flatten(),M.sim.C_tot.flatten(),CmNP) 
        Cm = ((M.sim.WLP.flatten()*CmP+(1-M.sim.WLP.flatten())*CmNP).reshape((M.par.simN,M.par.simT)))#m.sim.C_tot-Cw#
        Cm[M.sim.couple==0]=np.nan 
        
        #Home good consumption
        Q=M.sim.C_tot-Cm-Cw
        
        
         
        ######################################
        #Moments here
        ######################################
        age=40
        WLP = np.mean(M.sim.WLP[:,age])     
        share_m = np.mean(np.cumsum(M.sim.couple==1,axis=1)[:,age]>0) 
        share_d = np.mean(np.cumsum((M.sim.couple_lag==1) & (M.sim.couple==0),axis=1)[:,age]>0) 
        MQ=np.mean((Q/M.sim.C_tot)[M.sim.couple==1])
        MCw=np.mean((Cw/M.sim.C_tot)[M.sim.couple==1])
        
       
        
    
                        
        fit =((WLP-.5956))**2+((share_m-0.92965754))**2+((share_d-.3936217))**2+((MQ-0.80))**2+((MCw-0.065))**2
        print('Point is {}, fit is {}'.format(pt,fit))  
        print('Simulated moments are {}'.format([WLP,share_m,share_d,MQ,MCw]))
      
        return [((WLP-.5956)),((share_m-0.92965754)),((share_d-.3936217)),((MQ-0.80)),((MCw-0.065))]   
     
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