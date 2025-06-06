# -*- coding: utf-8 -*-
"""
This file solves and simulate the model  and run regression to determine
the pass-through of income to (various types) of consumption 
added in the paper 
"""

import numpy as np
import Bargaining_numba as brg
from consav import linear_interp
import UserFunctions_numba as usr 

#Root 
root='C:/Users/32489/Dropbox/Family Risk Sharing'

def selection(m,par,name):
    
    #######################################################
    #Selection issues below: only use one model m
    ######################################################

    b=1;e=par.Tr-2

    #Select different samples depending of WLP and being in a couple
    sm= (m.sim.couple[:,b+1:e+1]==1) & (m.sim.couple[:,b:e]==1)
    smw= (sm) & (m.sim.WLP[:,b+1:e+1]==1) & (m.sim.WLP[:,b:e]==1) 
    
    sm1= (sm) & (m.sim.couple[:,b+2:e+2]==1)
    smw1= (smw) & (m.sim.couple[:,b+2:e+2]==1) & (m.sim.WLP[:,b+2:e+2]==1) 
    
    sm11= (sm1) & (m.sim.couple[:,b-1:e-1]==1)
    smw11= (smw1) & (m.sim.couple[:,b-1:e-1]==1) & (m.sim.WLP[:,b-1:e-1]==1) 


    #################################
    #Consumption  below
    ##################################

    #Women individual consumption 
    Cw = m.sim.Cw#  (m.sim.WLP.flatten()*CwP+(1-m.sim.WLP.flatten())*CwNP).reshape((par.simN,par.simT))  
    Cm = m.sim.Cm#((m.sim.WLP.flatten()*CmP+(1-m.sim.WLP.flatten())*CmNP).reshape((par.simN,par.simT)))#m.sim.C_tot-Cw#
    Q=m.sim.xw#m.sim.C_tot-Cm-Cw

  
    #Log consumption growth for wife, husband and share of wife's expenses
    ΔCm  =np.log(Cm[:,b+1:e+1])  -np.log(Cm[:,b:e])
    ΔCw  =np.log(Cw[:,b+1:e+1])  -np.log(Cw[:,b:e])  
    Δws =np.log(Cw[:,b+1:e+1]/(Cm[:,b+1:e+1]))  -np.log(Cw[:,b:e]/(Cm[:,b:e]))#poww[:,b+1:e+1]-poww[:,b:e]#np.log(Q[:,b+1:e+1]/(m.sim.C_tot[:,b+1:e+1]))  -np.log(Q[:,b:e]/(m.sim.C_tot[:,b:e]))#
    ΔC =np.log(m.sim.C_tot[:,b+1:e+1])-np.log(m.sim.C_tot[:,b:e])
    ΔQ=np.log(Q[:,b+1:e+1])-np.log(Q[:,b:e])
    
    #Now the equivalent changes, but in levels not in log
    ΔLCm  =Cm[:,b+1:e+1]  -Cm[:,b:e]
    ΔLCw  =Cw[:,b+1:e+1]  -Cw[:,b:e]
    ΔLws =Cw[:,b+1:e+1]/(Cm[:,b+1:e+1])  -Cw[:,b:e]/(Cm[:,b:e])#poww[:,b+1:e+1]-poww[:,b:e]#
    ΔLC =m.sim.C_tot[:,b+1:e+1]-m.sim.C_tot[:,b:e]
    ΔLQ=Q[:,b+1:e+1]-Q[:,b:e]

    #################################
    #Income Shocks below
    ################################

    #Build transitory (ϵ) and persistent (z) shocks
    zm,ϵm,zw,ϵw=np.zeros((4,par.simN,par.T))
    izm=m.sim.iz%par.num_zw
    izw=m.sim.iz//par.num_zm
    for t in range(par.T):
        for i in range(par.simN):
        
            zm[i,t]=par.grid_pw[t,izm[i,t],m.sim.ih[i,t]]
            ϵm[i,t]=par.grid_ϵw[t,izm[i,t],m.sim.ih[i,t]]
            zw[i,t]=par.grid_pw[t,izw[i,t],m.sim.ih[i,t]]
            ϵw[i,t]=par.grid_ϵw[t,izw[i,t],m.sim.ih[i,t]]
                
    Δzm=zm[:,b+1:e+1]-zm[:,b:e]
    Δϵm=ϵm[:,b+1:e+1]-ϵm[:,b:e]
    Δzw=zw[:,b+1:e+1]-zw[:,b:e]
    Δϵw=ϵw[:,b+1:e+1]-ϵw[:,b:e]   
    Δz=Δzm+Δzw
    Δϵ=Δϵm+Δϵw
    
    #Total shocks(transitory+persistent)
    Δtm=Δzm+Δϵm
    Δtw=Δzw+Δϵw

    #Log changes in income (husband m, wife w and total)
    ΔYm  =np.log(m.sim.incm[:,b+1:e+1])  -np.log(m.sim.incm[:,b:e])
    ΔYw  =np.log(m.sim.incw[:,b+1:e+1])  -np.log(m.sim.incw[:,b:e])
    ΔY   =np.log(m.sim.incm[:,b+1:e+1]+m.sim.WLP[:,b+1:e+1]*m.sim.incw[:,b+1:e+1])  -np.log(m.sim.incm[:,b:e]+m.sim.WLP[:,b:e]*m.sim.incw[:,b:e])

    #Level changes in income
    ΔLYm  =m.sim.incm[:,b+1:e+1]  -m.sim.incm[:,b:e]
    ΔLYw  =m.sim.incw[:,b+1:e+1]*m.sim.WLP[:,b+1:e+1]  -m.sim.incw[:,b:e]*m.sim.WLP[:,b:e]
    ΔLY   =ΔLYm+ΔLYw
       
    #changes in income one period ahed (t+1) (husband m, wife w and total)
    ΔY1m  =np.log(m.sim.incm[:,b+2:e+2])  -np.log(m.sim.incm[:,b+1:e+1])
    ΔY1w  =np.log(m.sim.incw[:,b+2:e+2])  -np.log(m.sim.incw[:,b+1:e+1])
    ΔY1   =np.log(m.sim.incm[:,b+2:e+2]+m.sim.WLP[:,b+2:e+2]*m.sim.incw[:,b+2:e+2])  -np.log(m.sim.incm[:,b+1:e+1]+m.sim.WLP[:,b+1:e+1]*m.sim.incw[:,b+1:e+1])

    #changes in income one period before (t-1) (husband m, wife w and total)
    ΔY_1m  =np.log(m.sim.incm[:,b:e])  -np.log(m.sim.incm[:,b-1:e-1])
    ΔY_1w  =np.log(m.sim.incw[:,b:e])  -np.log(m.sim.incw[:,b-1:e-1])
    ΔY_1   =np.log(m.sim.incm[:,b:e]+m.sim.WLP[:,b:e]*m.sim.incw[:,b:e])  -np.log(m.sim.incm[:,b-1:e-1]+m.sim.WLP[:,b-1:e-1]*m.sim.incw[:,b-1:e-1])

    #these variables below are used to construct the BPP moments
    ΔYYm  =ΔYm+ΔY1m+ΔY_1m 
    ΔYYw  =ΔYw+ΔY1w+ΔY_1w 
    ΔYY   =ΔY+ΔY1+ΔY1   
    
    #Change in WLP
    ΔWLP=m.sim.WLP[:,b+1:e+1]-m.sim.WLP[:,b:e]
    
    
     
    #Multinomial regression using men and women income shocks
    intercept=np.ones(ΔCm[smw].flatten().shape)
    X=np.hstack((intercept[:,None],ΔYm[smw].flatten()[:,None],ΔYw[smw].flatten()[:,None]))#explicative variables    
    βcm= np.linalg.inv(X.T @ X) @ (X.T @ ΔCm[smw].flatten()) 
    βcw= np.linalg.inv(X.T @ X) @ (X.T @ ΔCw[smw].flatten()) 
    
    print("Cm insruance with mult reg: Ym {}, Yw {}, univ reg:  Ym {}, Yw {}".format(βcm[1],βcm[2],
                                                                                     np.cov(ΔYm[smw],ΔCm[smw])[0,1]/np.var(ΔYm[smw]),
                                                                                     np.cov(ΔYw[smw],ΔCm[smw])[0,1]/np.var(ΔYw[smw])))
    
    print("Cw insruance with mult reg: Ym {}, Yw {}, univ reg:  Ym {}, Yw {}".format(βcw[1],βcw[2],
                                                                                     np.cov(ΔYm[smw],ΔCw[smw])[0,1]/np.var(ΔYm[smw]),
                                                                                     np.cov(ΔYw[smw],ΔCw[smw])[0,1]/np.var(ΔYw[smw])))
    
    ##################################
    #Compute and store pass-through   
    ##################################
    
    #Notation: all: means all income Y (transitory + persistent); per: persistent shock, tra: transitory shock
    #all_x_y means shock of x's income on m's expenditure
    indc={'all_m_m':np.cov(ΔYm[sm],ΔCm[sm])[0,1]/np.var(ΔYm[sm]),
         'all_m_w':np.cov(ΔYm[sm],ΔCw[sm])[0,1]/np.var(ΔYm[sm]),
         'all_w_m':np.cov(ΔYw[smw],ΔCm[smw])[0,1]/np.var(ΔYw[smw]),
         'all_w_w':np.cov(ΔYw[smw],ΔCw[smw])[0,1]/np.var(ΔYw[smw]),        
         'per_m_m':np.cov(Δzm[sm],ΔCm[sm])[0,1]/np.var(Δzm[sm]),
         'per_m_w':np.cov(Δzm[sm],ΔCw[sm])[0,1]/np.var(Δzm[sm]),
         'per_w_m':np.cov(Δzw[smw],ΔCm[smw])[0,1]/np.var(Δzw[smw]),
         'per_w_w':np.cov(Δzw[smw],ΔCw[smw])[0,1]/np.var(Δzw[smw]),
         'tra_m_m':np.cov(Δϵm[sm],ΔCm[sm])[0,1]/np.var(Δϵm[sm]),
         'tra_m_w':np.cov(Δϵm[sm],ΔCw[sm])[0,1]/np.var(Δϵm[sm]),
         'tra_w_m':np.cov(Δϵw[smw],ΔCm[smw])[0,1]/np.var(Δϵw[smw]),
         'tra_w_w':np.cov(Δϵw[smw],ΔCw[smw])[0,1]/np.var(Δϵw[smw])}
    
    level={'all_m_m':np.cov(ΔLYm[sm],ΔLCm[sm])[0,1]/np.var(ΔLYm[sm]),
          'all_m_w':np.cov(ΔLYm[sm],ΔLCw[sm])[0,1]/np.var(ΔLYm[sm]),
          'all_w_m':np.cov(ΔLYw[sm],ΔLCm[sm])[0,1]/np.var(ΔLYw[sm]),
          'all_w_w':np.cov(ΔLYw[sm],ΔLCw[sm])[0,1]/np.var(ΔLYw[sm])}
         
    
    #Pass-through on women's consumption share
    w_sh={'all_m':np.cov(ΔYm[sm],Δws[sm])[0,1]/np.var(ΔYm[sm]),         
          'all_w':np.cov(ΔYw[smw],Δws[smw])[0,1]/np.var(ΔYw[smw]),
          'per_m':np.cov(Δzm[sm],Δws[sm])[0,1]/np.var(Δzm[sm]),
          'per_w':np.cov(Δzw[smw],Δws[smw])[0,1]/np.var(Δzw[smw]),
          'tra_m':np.cov(Δϵm[sm],Δws[sm])[0,1]/np.var(Δϵm[sm]),
          'tra_w':np.cov(Δϵw[smw],Δws[smw])[0,1]/np.var(Δϵw[smw]),
          'level_m':np.cov(ΔLYm[sm],ΔLws[sm])[0,1]/np.var(ΔLYm[sm]),         
          'level_w':np.cov(ΔLYw[sm],ΔLws[sm])[0,1]/np.var(ΔLYw[sm])}
    
    
    #Effect of income shocks on WLP (in percentage points)
    wlp= {'all_m':np.cov(Δtm[sm],ΔWLP[sm])[0,1]/np.var(Δtm[sm]),         
          'all_w':np.cov(Δtw[sm],ΔWLP[sm])[0,1]/np.var(Δtw[sm]),
          'per_m':np.cov(Δzm[sm],ΔWLP[sm])[0,1]/np.var(Δzm[sm]),
          'per_w':np.cov(Δzw[sm],ΔWLP[sm])[0,1]/np.var(Δzw[sm]),
          'tra_m':np.cov(Δϵm[sm],ΔWLP[sm])[0,1]/np.var(Δϵm[sm]),
          'tra_w':np.cov(Δϵw[sm],ΔWLP[sm])[0,1]/np.var(Δϵw[sm])}
    
    

    
    #Pass-throughs of various components of income on total consumption
    totc={'all':np.cov(ΔY[sm],ΔC[sm])[0,1]/np.var(ΔY[sm]),         
          'per':np.cov(Δz[sm],ΔC[sm])[0,1]/np.var(Δz[sm]),
          'tra':np.cov(Δϵ[sm],ΔC[sm])[0,1]/np.var(Δϵ[sm]),          
          'all_m':np.cov(ΔYm[sm],ΔC[sm])[0,1]/np.var(ΔYm[sm]),         
          'per_m':np.cov(Δzm[sm],ΔC[sm])[0,1]/np.var(Δzm[sm]),
          'tra_m':np.cov(Δϵm[sm],ΔC[sm])[0,1]/np.var(Δϵm[sm]),
          'all_w':np.cov(ΔYw[smw],ΔC[smw])[0,1]/np.var(ΔYw[smw]),         
          'per_w':np.cov(Δzw[smw],ΔC[smw])[0,1]/np.var(Δzw[smw]),
          'tra_w':np.cov(Δϵw[smw],ΔC[smw])[0,1]/np.var(Δϵw[smw]),
          'level_s':np.cov(ΔLY[sm],ΔLC[sm])[0,1]/np.var(ΔLY[sm]),   
          'level_m':np.cov(ΔLYm[sm],ΔLC[sm])[0,1]/np.var(ΔLYm[sm]),     
          'level_w':np.cov(ΔLYw[sm],ΔLC[sm])[0,1]/np.var(ΔLYw[sm])} 
    
    #Pass-throughs of various components of income on public ependiture
    Qins={'all':np.cov(ΔY[sm],ΔQ[sm])[0,1]/np.var(ΔY[sm]),         
          'per':np.cov(Δz[sm],ΔQ[sm])[0,1]/np.var(Δz[sm]),
          'tra':np.cov(Δϵ[sm],ΔQ[sm])[0,1]/np.var(Δϵ[sm]),
          'all_m':np.cov(ΔYm[sm],ΔQ[sm])[0,1]/np.var(ΔYm[sm]),         
          'all_w':np.cov(ΔYw[smw],ΔQ[smw])[0,1]/np.var(ΔYw[smw]),
          'per_m':np.cov(Δzm[sm],ΔQ[sm])[0,1]/np.var(Δzm[sm]),
          'per_w':np.cov(Δzw[smw],ΔQ[smw])[0,1]/np.var(Δzw[smw]),
          'tra_m':np.cov(Δϵm[sm],ΔQ[sm])[0,1]/np.var(Δϵm[sm]),
          'tra_w':np.cov(Δϵw[smw],ΔQ[smw])[0,1]/np.var(Δϵw[smw]),
          'level_s':np.cov(ΔLY[sm],ΔLQ[sm])[0,1]/np.var(ΔLY[sm]),   
          'level_m':np.cov(ΔLYm[sm],ΔLQ[sm])[0,1]/np.var(ΔLYm[sm]),     
          'level_w':np.cov(ΔLYw[sm],ΔLQ[sm])[0,1]/np.var(ΔLYw[sm])} 
    
    #BPP moment for MPC
    BPP_MPC={'al_tot':np.cov(ΔY1[sm1],ΔC[sm1])[0,1]/np.cov(ΔY[sm1],ΔY1[sm1])[0,1],
             'al_Q':np.cov(ΔY1[sm1],ΔQ[sm1])[0,1]/np.cov(ΔY[sm1],ΔY1[sm1])[0,1],
             'al_cm':np.cov(ΔY1[sm1],ΔCm[sm1])[0,1]/np.cov(ΔY[sm1],ΔY1[sm1])[0,1],
             'al_cw':np.cov(ΔY1[sm1],ΔCw[sm1])[0,1]/np.cov(ΔY[sm1],ΔY1[sm1])[0,1],             
             'ym_tot':np.cov(ΔY1m[sm1],ΔC[sm1])[0,1]/np.cov(ΔYm[sm1],ΔY1m[sm1])[0,1],
             'ym_Q':np.cov(ΔY1m[sm1],ΔQ[sm1])[0,1]/np.cov(ΔYm[sm1],ΔY1m[sm1])[0,1],
             'ym_cm':np.cov(ΔY1m[sm1],ΔCm[sm1])[0,1]/np.cov(ΔYm[sm1],ΔY1m[sm1])[0,1],
             'ym_cw':np.cov(ΔY1m[sm1],ΔCw[sm1])[0,1]/np.cov(ΔYm[sm1],ΔY1m[sm1])[0,1],             
             'yw_tot':np.cov(ΔY1w[smw1],ΔC[smw1])[0,1]/np.cov(ΔYw[smw1],ΔY1w[smw1])[0,1],
             'yw_Q':np.cov(ΔY1w[smw1],ΔQ[smw1])[0,1]/np.cov(ΔYw[smw1],ΔY1w[smw1])[0,1],
             'yw_cm':np.cov(ΔY1w[smw1],ΔCm[smw1])[0,1]/np.cov(ΔYw[smw1],ΔY1w[smw1])[0,1],
             'yw_cw':np.cov(ΔY1w[smw1],ΔCw[smw1])[0,1]/np.cov(ΔYw[smw1],ΔY1w[smw1])[0,1]}
             
    #BPP moment describing inurance to persistent income shocks
    BPP_PER={'al_tot':np.cov(ΔYY[sm11],ΔC[sm11])[0,1]/np.cov(ΔY[sm11],ΔYY[sm11])[0,1],
             'al_Q':np.cov(ΔYY[sm11],ΔQ[sm11])[0,1]/np.cov(ΔY[sm11],ΔYY[sm11])[0,1],
             'al_cm':np.cov(ΔYY[sm11],ΔCm[sm11])[0,1]/np.cov(ΔY[sm11],ΔYY[sm11])[0,1],
             'al_cw':np.cov(ΔYY[sm11],ΔCw[sm11])[0,1]/np.cov(ΔY[sm11],ΔYY[sm11])[0,1],             
             'ym_tot':np.cov(ΔYYm[sm11],ΔC[sm11])[0,1]/np.cov(ΔYm[sm11],ΔYYm[sm11])[0,1],
             'ym_Q':np.cov(ΔYYm[sm11],ΔQ[sm11])[0,1]/np.cov(ΔYm[sm11],ΔYYm[sm11])[0,1],
             'ym_cm':np.cov(ΔYYm[sm11],ΔCm[sm11])[0,1]/np.cov(ΔYm[sm11],ΔYYm[sm11])[0,1],
             'ym_cw':np.cov(ΔYYm[sm11],ΔCw[sm11])[0,1]/np.cov(ΔYm[sm11],ΔYYm[sm11])[0,1],             
             'yw_tot':np.cov(ΔYYw[smw11],ΔC[smw11])[0,1]/np.cov(ΔYw[smw11],ΔYYw[smw11])[0,1],
             'yw_Q':np.cov(ΔYYw[smw11],ΔQ[smw11])[0,1]/np.cov(ΔYw[smw11],ΔYYw[smw11])[0,1],
             'yw_cm':np.cov(ΔYYw[smw11],ΔCm[smw11])[0,1]/np.cov(ΔYw[smw11],ΔYYw[smw11])[0,1],
             'yw_cw':np.cov(ΔYYw[smw11],ΔCw[smw11])[0,1]/np.cov(ΔYw[smw11],ΔYYw[smw11])[0,1]}
    
    
    
             

    
    print("Average shares for all couple: Q {}, Cm {}, Cw {} ".format(np.mean((Q/m.sim.C_tot)[m.sim.couple==1]),
                                                                      np.mean((Cm/m.sim.C_tot)[m.sim.couple==1]),
                                                                      np.mean((Cw/m.sim.C_tot)[m.sim.couple==1])))
   
    print("Average shares for all couple with working wife: Q {}, Cm {}, Cw {} ".format(np.mean((Q/m.sim.C_tot)[(m.sim.couple==1) & (m.sim.WLP==1)]),
                                                                      np.mean((Cm/m.sim.C_tot)[(m.sim.couple==1) & (m.sim.WLP==1)]),
                                                                      np.mean((Cw/m.sim.C_tot)[(m.sim.couple==1) & (m.sim.WLP==1)])))
  
    return {'indc':indc,'w_sh':w_sh,'totc':totc,'Qins':Qins,'BPP_MPC':BPP_MPC,'BPP_PER':BPP_PER,'wlp':wlp,'level':level}

##################################################
#Solve and simulate the benchmark model
##################################################

#Initialize seed 
np.random.seed(10) 
 
#Create sample with replacement 
N=10_000#sample size 

xc=np.array([0.46874957, 0.05944867, 0.06511688, 0.78126218, 0.46638862])


par = {'simN':N,'θ': xc[0], 'meet':xc[1],'σL0':xc[2],'σL':xc[2],'α2':xc[3],'α1':1.0-xc[3],'γ':xc[4]} 
MB = brg.HouseholdModelClass(par=par)  
MB.solve()
MB.simulate()





        
##################################################
#Run the selection code for both kind of models:
#################################################
B=selection(MB,MB.par,'MB')


######################################
#Tables with results below
######################################
def p33(x): y=x;return str('%3.3f' % y)     
 
#% changes in consumption out of a 1 % change in all income
table=r"...total income   & \textbf{"+p33(B['totc']['all'])+'} & '+p33(B['Qins']['all'])+' & & &    \\\\ '+\
      r'...wife income    & '+p33(B['totc']['all_w'])+' & '+p33(B['Qins']['all_w'])+'& '+' \\textbf{'+p33(B['indc']['all_w_m'])+'} &  \\textbf{'+p33(B['indc']['all_w_w'])+'} &  \\textbf{'+p33(B['w_sh']['all_w'])+'}    \\\\ '+\
      r'...husband income & '+p33(B['totc']['all_m'])+' &  '+p33(B['Qins']['all_m'])+'& '+' \\textbf{'+p33(B['indc']['all_m_m'])+'} &  \\textbf{'+p33(B['indc']['all_m_w'])+'} &  \\textbf{'+p33(B['w_sh']['all_m'])+'}    \\\\\\bottomrule'
with open(root+'/Model/results/allinc.tex', 'w') as f: f.write(table); f.close() 

#% changes in consumption out of a 1 % transitory change in income
table=r"...total income   & \textbf{"+p33(B['totc']['tra'])+'} & '+p33(B['Qins']['tra'])+' & & &    \\\\ '+\
      r'...wife income    & '+p33(B['totc']['tra_w'])+' & '+p33(B['Qins']['tra_w'])+'& '+' \\textbf{'+p33(B['indc']['tra_w_m'])+'} &  \\textbf{'+p33(B['indc']['tra_w_w'])+'} &  \\textbf{'+p33(B['w_sh']['tra_w'])+'}    \\\\ '+\
      r'...husband income & '+p33(B['totc']['tra_m'])+' &  '+p33(B['Qins']['tra_m'])+'& '+' \\textbf{'+p33(B['indc']['tra_m_m'])+'} &  \\textbf{'+p33(B['indc']['tra_m_w'])+'} &  \\textbf{'+p33(B['w_sh']['tra_m'])+'}    \\\\\\bottomrule'
with open(root+'/Model/results/trainc.tex', 'w') as f: f.write(table); f.close() 


#% changes in consumption out of a 1 % persistent change in income
table=r"...total income   & \textbf{"+p33(B['totc']['per'])+'} & '+p33(B['Qins']['per'])+' & & &    \\\\ '+\
      r'...wife income    & '+p33(B['totc']['per_w'])+' & '+p33(B['Qins']['per_w'])+'& '+' \\textbf{'+p33(B['indc']['per_w_m'])+'} &  \\textbf{'+p33(B['indc']['per_w_w'])+'} &  \\textbf{'+p33(B['w_sh']['per_w'])+'}    \\\\ '+\
      r'...husband income & '+p33(B['totc']['per_m'])+' &  '+p33(B['Qins']['per_m'])+'& '+' \\textbf{'+p33(B['indc']['per_m_m'])+'} &  \\textbf{'+p33(B['indc']['per_m_w'])+'} &  \\textbf{'+p33(B['w_sh']['per_m'])+'}    \\\\\\bottomrule'
with open(root+'/Model/results/perinc.tex', 'w') as f: f.write(table); f.close() 


#MPC tables
table=r"...husband income & "+p33(B['BPP_MPC']['ym_tot'])+' & '+p33(B['BPP_MPC']['ym_Q'])+' & '+p33(B['BPP_MPC']['ym_cm'])+' & '+p33(B['BPP_MPC']['ym_cw'])+'  \\\\ '+\
      r'...wife income    & '+p33(B['BPP_MPC']['yw_tot'])+' & '+p33(B['BPP_MPC']['yw_Q'])+' & '+p33(B['BPP_MPC']['yw_cm'])+' & '+p33(B['BPP_MPC']['yw_cw'])+'  \\\\ '+\
      r'...total income   & '+p33(B['BPP_MPC']['al_tot'])+' & '+p33(B['BPP_MPC']['al_Q'])+' & '+p33(B['BPP_MPC']['al_cm'])+' & '+p33(B['BPP_MPC']['al_cw'])+'  \\\\\\bottomrule'
with open(root+'/Model/results/BPP_MPC.tex', 'w') as f: f.write(table); f.close() 

#BPP persistent tables
table=r"...husband income & "+p33(B['BPP_PER']['ym_tot'])+' & '+p33(B['BPP_PER']['ym_Q'])+' & '+p33(B['BPP_PER']['ym_cm'])+' & '+p33(B['BPP_PER']['ym_cw'])+'  \\\\ '+\
      r'...wife income    & '+p33(B['BPP_PER']['yw_tot'])+' & '+p33(B['BPP_PER']['yw_Q'])+' & '+p33(B['BPP_PER']['yw_cm'])+' & '+p33(B['BPP_PER']['yw_cw'])+'  \\\\ '+\
      r'...total income   & '+p33(B['BPP_PER']['al_tot'])+' & '+p33(B['BPP_PER']['al_Q'])+' & '+p33(B['BPP_PER']['al_cm'])+' & '+p33(B['BPP_PER']['al_cw'])+'  \\\\\\bottomrule '
with open(root+'/Model/results/BPP_PER.tex', 'w') as f: f.write(table); f.close() 

#labor supply	
table=r'  '+p33(B['wlp']['tra_w'])+' & '+p33(B['wlp']['tra_m'])+' & '+p33(B['wlp']['per_w'])+' & '+p33(B['wlp']['per_m'])+' & '+p33(B['wlp']['all_w'])+' & '+p33(B['wlp']['all_m'])+'  \\\\\\bottomrule '
with open(root+'/Model/results/WLP.tex', 'w') as f: f.write(table); f.close() 

#level changes in consumption out of a level change in all income
table=r"...total income   & \textbf{"+p33(B['totc']['level_s'])+'} & '+p33(B['Qins']['level_s'])+' & & &    \\\\ '+\
      r'...wife income    & '+p33(B['totc']['level_w'])+' & '+p33(B['Qins']['level_w'])+'& '+' \\textbf{'+p33(B['level']['all_w_m'])+'} &  \\textbf{'+p33(B['level']['all_w_w'])+'} &  \\textbf{'+p33(B['w_sh']['level_w'])+'}    \\\\ '+\
      r'...husband income & '+p33(B['totc']['level_m'])+' &  '+p33(B['Qins']['level_m'])+'& '+' \\textbf{'+p33(B['level']['all_m_m'])+'} &  \\textbf{'+p33(B['level']['all_m_w'])+'} &  \\textbf{'+p33(B['w_sh']['level_m'])+'}    \\\\\\bottomrule'
with open(root+'/Model/results/level.tex', 'w') as f: f.write(table); f.close() 




#plt.plot(np.var(np.log(MB.sim.incm),axis=0))