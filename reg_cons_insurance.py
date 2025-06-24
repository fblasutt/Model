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

    b=5;e=25#par.Tr-2

    #Select different samples depending of WLP and being in a couple
    sm= (m.sim.couple[:,b+1:e+1]==1) & (m.sim.couple[:,b:e]==1)
    smw= (sm) & (m.sim.WLP[:,b+1:e+1]>0) & (m.sim.WLP[:,b:e]>0) 
    
    sm1= (sm) & (m.sim.couple[:,b+2:e+2]==1)
    smw1= (smw) & (m.sim.couple[:,b+2:e+2]==1) & (m.sim.WLP[:,b+2:e+2]>0) 
    
    sm11= (sm1) & (m.sim.couple[:,b-1:e-1]==1)
    smw11= (smw1) & (m.sim.couple[:,b-1:e-1]==1) & (m.sim.WLP[:,b-1:e-1]>0) 


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
        
            zm[i,t]=par.grid_pm[t,izm[i,t],m.sim.ih[i,t]]
            ϵm[i,t]=par.grid_ϵm[t,izm[i,t],m.sim.ih[i,t]]
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
    
    #Labor supply
    LS=m.par.grid_wlp[m.sim.WLP[:,b:e]]*0+1
    LS1=m.par.grid_wlp[m.sim.WLP[:,b+1:e+1]]*0+1
    LS2=m.par.grid_wlp[m.sim.WLP[:,b+2:e+2]]*0+1
    LS_1=m.par.grid_wlp[m.sim.WLP[:,b-1:e-1]]*0+1
    

    #Log changes in income (husband m, wife w and total)
    ΔYm  =np.log(m.sim.incmg[:,b+1:e+1])  -np.log(m.sim.incmg[:,b:e])
    ΔYw  =np.log(m.sim.incwg[:,b+1:e+1]*LS1)  -np.log(m.sim.incwg[:,b:e]*LS)
    ΔY   =np.log(m.sim.incmg[:,b+1:e+1]+m.sim.incwg[:,b+1:e+1]*LS1)  -np.log(m.sim.incmg[:,b:e]+m.sim.incwg[:,b:e]*LS)

    #Level changes in income
    ΔLYm  =m.sim.incmg[:,b+1:e+1]  -m.sim.incmg[:,b:e]
    ΔLYw  =m.sim.incwg[:,b+1:e+1]*LS1  -m.sim.incwg[:,b:e]*LS
    ΔLY   =ΔLYm+ΔLYw
       
    #changes in income one period ahed (t+1) (husband m, wife w and total)
    ΔY1m  =np.log(m.sim.incmg[:,b+2:e+2])  -np.log(m.sim.incmg[:,b+1:e+1])
    ΔY1w  =np.log(m.sim.incwg[:,b+2:e+2]*LS2)  -np.log(m.sim.incwg[:,b+1:e+1]*LS1)
    ΔY1   =np.log(m.sim.incmg[:,b+2:e+2]+m.sim.incwg[:,b+2:e+2]*LS2)  -np.log(m.sim.incmg[:,b+1:e+1]+m.sim.incwg[:,b+1:e+1]*LS1)

    #changes in income one period before (t-1) (husband m, wife w and total)
    ΔY_1m  =np.log(m.sim.incmg[:,b:e])  -np.log(m.sim.incmg[:,b-1:e-1])
    ΔY_1w  =np.log(m.sim.incwg[:,b:e]*LS)  -np.log(m.sim.incwg[:,b-1:e-1]*LS_1)
    ΔY_1   =np.log(m.sim.incmg[:,b:e]+m.sim.incwg[:,b:e]*LS)  -np.log(m.sim.incmg[:,b-1:e-1]+m.sim.incwg[:,b-1:e-1]*LS_1)

    #these variables below are used to construct the BPP moments
    ΔYYm  =ΔYm+ΔY1m+ΔY_1m 
    ΔYYw  =ΔYw+ΔY1w+ΔY_1w 
    ΔYY   =ΔY+ΔY_1+ΔY1   
    
    #Change in WLP
    ΔWLP=LS1-LS#m.sim.WLP[:,b+1:e+1]-m.sim.WLP[:,b:e]
    
    
   
    ##########################################
    # Compact OLS regression
    ##########################################
    def ols(indep,dep,cond):
        
        intercept=np.ones(indep.shape)[cond]
        age=(np.cumsum(np.ones(indep.shape),axis=1)-1)[cond]
        age2=age**2
        
        
        
        X=np.hstack((intercept[:,None],indep[cond][:,None],age[:,None],age2[:,None]))#explicative variables 
     
        try:  
            #OLS on untreated observations 
            β = (np.linalg.inv(X.T @ X) @ (X.T @ dep[cond].flatten()))[1]
            
        except:
            
            β=100000.0
         
        return β
    
    ##################################
    #Compute and store pass-through   
    ##################################
    
    #Notation: all: means all income Y (transitory + persistent); per: persistent shock, tra: transitory shock
    #all_x_y means shock of x's income on m's expenditure
    indc={'all_m_m':ols(ΔYm,ΔCm,sm),#[0,1]/np.var(ΔYm,sm),
         'all_m_w':ols(ΔYm,ΔCw,sm),#[0,1]/np.var(ΔYm,sm),
         'all_w_m':ols(ΔYw,ΔCm,smw),#[0,1]/np.var(ΔYw,smw),
         'all_w_w':ols(ΔYw,ΔCw,smw),#[0,1]/np.var(ΔYw,smw),        
         'per_m_m':ols(Δzm,ΔCm,sm),#[0,1]/np.var(Δzm,sm),
         'per_m_w':ols(Δzm,ΔCw,sm),#[0,1]/np.var(Δzm,sm),
         'per_w_m':ols(Δzw,ΔCm,smw),#[0,1]/np.var(Δzw,smw),
         'per_w_w':ols(Δzw,ΔCw,smw),#[0,1]/np.var(Δzw,smw),
         'tra_m_m':ols(Δϵm,ΔCm,sm),#[0,1]/np.var(Δϵm,sm),
         'tra_m_w':ols(Δϵm,ΔCw,sm),#[0,1]/np.var(Δϵm,sm),
         'tra_w_m':ols(Δϵw,ΔCm,smw),#[0,1]/np.var(Δϵw,smw),
         'tra_w_w':ols(Δϵw,ΔCw,smw)}#[0,1]/np.var(Δϵw,smw)}
    
    level={'all_m_m':ols(ΔLYm,ΔLCm,sm),#[0,1]/np.var(ΔLYm,sm),
          'all_m_w':ols(ΔLYm,ΔLCw,sm),#[0,1]/np.var(ΔLYm,sm),
          'all_w_m':ols(ΔLYw,ΔLCm,sm),#[0,1]/np.var(ΔLYw,sm),
          'all_w_w':ols(ΔLYw,ΔLCw,sm)}#[0,1]/np.var(ΔLYw,sm)}
         
    
    #Pass-through on women's consumption share
    w_sh={'all_m':ols(ΔYm,Δws,sm),#[0,1]/np.var(ΔYm,sm),         
          'all_w':ols(ΔYw,Δws,smw),#[0,1]/np.var(ΔYw,smw),
          'per_m':ols(Δzm,Δws,sm),#[0,1]/np.var(Δzm,sm),
          'per_w':ols(Δzw,Δws,smw),#[0,1]/np.var(Δzw,smw),
          'tra_m':ols(Δϵm,Δws,sm),#[0,1]/np.var(Δϵm,sm),
          'tra_w':ols(Δϵw,Δws,smw),#[0,1]/np.var(Δϵw,smw),
          'level_m':ols(ΔLYm,ΔLws,sm),#[0,1]/np.var(ΔLYm,sm),         
          'level_w':ols(ΔLYw,ΔLws,sm)}#[0,1]/np.var(ΔLYw,sm)}
    
    
    #Effect of income shocks on WLP (in percentage points)
    wlp= {'all_m':ols(Δtm,ΔWLP,sm),#[0,1]/np.var(Δtm,sm),         
          'all_w':ols(Δtw,ΔWLP,sm),#[0,1]/np.var(Δtw,sm),
          'per_m':ols(Δzm,ΔWLP,sm),#[0,1]/np.var(Δzm,sm),
          'per_w':ols(Δzw,ΔWLP,sm),#[0,1]/np.var(Δzw,sm),
          'tra_m':ols(Δϵm,ΔWLP,sm),#[0,1]/np.var(Δϵm,sm),
          'tra_w':ols(Δϵw,ΔWLP,sm)}#[0,1]/np.var(Δϵw,sm)}
    
    

    
    #Pass-throughs of various components of income on total consumption
    totc={'all':ols(ΔY,ΔC,sm),#[0,1]/np.var(ΔY,sm),         
          'per':ols(Δz,ΔC,sm),#[0,1]/np.var(Δz,sm),
          'tra':ols(Δϵ,ΔC,sm),#[0,1]/np.var(Δϵ,sm),          
          'all_m':ols(ΔYm,ΔC,sm),#[0,1]/np.var(ΔYm,sm),         
          'per_m':ols(Δzm,ΔC,sm),#[0,1]/np.var(Δzm,sm),
          'tra_m':ols(Δϵm,ΔC,sm),#[0,1]/np.var(Δϵm,sm),
          'all_w':ols(ΔYw,ΔC,smw),#[0,1]/np.var(ΔYw,smw),         
          'per_w':ols(Δzw,ΔC,smw),#[0,1]/np.var(Δzw,smw),
          'tra_w':ols(Δϵw,ΔC,smw),#[0,1]/np.var(Δϵw,smw),
          'level_s':ols(ΔLY,ΔLC,sm),#[0,1]/np.var(ΔLY,sm),   
          'level_m':ols(ΔLYm,ΔLC,sm),#[0,1]/np.var(ΔLYm,sm),     
          'level_w':ols(ΔLYw,ΔLC,sm)}#[0,1]/np.var(ΔLYw,sm)} 
    
    #Pass-throughs of various components of income on public ependiture
    Qins={'all':ols(ΔY,ΔQ,sm),#[0,1]/np.var(ΔY,sm),         
          'per':ols(Δz,ΔQ,sm),#[0,1]/np.var(Δz,sm),
          'tra':ols(Δϵ,ΔQ,sm),#[0,1]/np.var(Δϵ,sm),
          'all_m':ols(ΔYm,ΔQ,sm),#[0,1]/np.var(ΔYm,sm),         
          'all_w':ols(ΔYw,ΔQ,smw),#[0,1]/np.var(ΔYw,smw),
          'per_m':ols(Δzm,ΔQ,sm),#[0,1]/np.var(Δzm,sm),
          'per_w':ols(Δzw,ΔQ,smw),#[0,1]/np.var(Δzw,smw),
          'tra_m':ols(Δϵm,ΔQ,sm),#[0,1]/np.var(Δϵm,sm),
          'tra_w':ols(Δϵw,ΔQ,smw),#[0,1]/np.var(Δϵw,smw),
          'level_s':ols(ΔLY,ΔLQ,sm),#[0,1]/np.var(ΔLY,sm),   
          'level_m':ols(ΔLYm,ΔLQ,sm),#[0,1]/np.var(ΔLYm,sm),     
          'level_w':ols(ΔLYw,ΔLQ,sm)}#[0,1]/np.var(ΔLYw,sm)} 
    
    #BPP moment for MPC
    BPP_MPC={'al_tot':np.mean(ΔY1[sm1]*ΔC[sm1])/np.mean(ΔY[sm1]*ΔY1[sm1]),
             'al_Q':np.mean(ΔY1[sm1]*ΔQ[sm1])/np.mean(ΔY[sm1]*ΔY1[sm1]),
             'al_cm':np.mean(ΔY1[sm1]*ΔCm[sm1])/np.mean(ΔY[sm1]*ΔY1[sm1]),
             'al_cw':np.mean(ΔY1[sm1]*ΔCw[sm1])/np.mean(ΔY[sm1]*ΔY1[sm1]),             
             'ym_tot':np.mean(ΔY1m[sm1]*ΔC[sm1])/np.mean(ΔYm[sm1]*ΔY1m[sm1]),
             'ym_Q':np.mean(ΔY1m[sm1]*ΔQ[sm1])/np.mean(ΔYm[sm1]*ΔY1m[sm1]),
             'ym_cm':np.mean(ΔY1m[sm1]*ΔCm[sm1])/np.mean(ΔYm[sm1]*ΔY1m[sm1]),
             'ym_cw':np.mean(ΔY1m[sm1]*ΔCw[sm1])/np.mean(ΔYm[sm1]*ΔY1m[sm1]),             
             'yw_tot':np.mean(ΔY1w[smw1]*ΔC[smw1])/np.mean(ΔYw[smw1]*ΔY1w[smw1]),
             'yw_Q':np.mean(ΔY1w[smw1]*ΔQ[smw1])/np.mean(ΔYw[smw1]*ΔY1w[smw1]),
             'yw_cm':np.mean(ΔY1w[smw1]*ΔCm[smw1])/np.mean(ΔYw[smw1]*ΔY1w[smw1]),
             'yw_cw':np.mean(ΔY1w[smw1]*ΔCw[smw1])/np.mean(ΔYw[smw1]*ΔY1w[smw1])}
             
    #BPP moment describing inurance to persistent income shocks
    BPP_PER={'al_tot':np.mean(ΔYY[sm11]*ΔC[sm11])/np.mean(ΔY[sm11]*ΔYY[sm11]),
             'al_Q':np.mean(ΔYY[sm11]*ΔQ[sm11])/np.mean(ΔY[sm11]*ΔYY[sm11]),
             'al_cm':np.mean(ΔYY[sm11]*ΔCm[sm11])/np.mean(ΔY[sm11]*ΔYY[sm11]),
             'al_cw':np.mean(ΔYY[sm11]*ΔCw[sm11])/np.mean(ΔY[sm11]*ΔYY[sm11]),             
             'ym_tot':np.mean(ΔYYm[sm11]*ΔC[sm11])/np.mean(ΔYm[sm11]*ΔYYm[sm11]),
             'ym_Q':np.mean(ΔYYm[sm11]*ΔQ[sm11])/np.mean(ΔYm[sm11]*ΔYYm[sm11]),
             'ym_cm':np.mean(ΔYYm[sm11]*ΔCm[sm11])/np.mean(ΔYm[sm11]*ΔYYm[sm11]),
             'ym_cw':np.mean(ΔYYm[sm11]*ΔCw[sm11])/np.mean(ΔYm[sm11]*ΔYYm[sm11]),             
             'yw_tot':np.mean(ΔYYw[smw11]*ΔC[smw11])/np.mean(ΔYw[smw11]*ΔYYw[smw11]),
             'yw_Q':np.mean(ΔYYw[smw11]*ΔQ[smw11])/np.mean(ΔYw[smw11]*ΔYYw[smw11]),
             'yw_cm':np.mean(ΔYYw[smw11]*ΔCm[smw11])/np.mean(ΔYw[smw11]*ΔYYw[smw11]),
             'yw_cw':np.mean(ΔYYw[smw11]*ΔCw[smw11])/np.mean(ΔYw[smw11]*ΔYYw[smw11])}
    
    
    
             

    
    print("Average shares for all couple: Q {}, Cm {}, Cw {} ".format(np.mean((Q/m.sim.C_tot)[m.sim.couple==1]),
                                                                      np.mean((Cm/m.sim.C_tot)[m.sim.couple==1]),
                                                                      np.mean((Cw/m.sim.C_tot)[m.sim.couple==1])))
   
    print("Average shares for all couple with working wife: Q {}, Cm {}, Cw {} ".format(np.mean((Q/m.sim.C_tot)[(m.sim.couple==1) & (m.sim.WLP==1)]),
                                                                      np.mean((Cm/m.sim.C_tot)[(m.sim.couple==1) & (m.sim.WLP==1)]),
                                                                      np.mean((Cw/m.sim.C_tot)[(m.sim.couple==1) & (m.sim.WLP==1)])))
  
    return {'indc':indc,'w_sh':w_sh,'totc':totc,'Qins':Qins,'BPP_MPC':BPP_MPC,'BPP_PER':BPP_PER,'wlp':wlp,'level':level}


##########################################
# Extract file with 
##########################################
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


##################################################
#Solve and simulate the benchmark model
##################################################

#Initialize seed 
np.random.seed(10) 
 
#Create sample with replacement 
N=10_000#sample size 

xc=np.array([0.46874957, 0.05944867, 0.06511688, 0.78126218, 0.46638862])
xc=np.array([0.42746862, 0.05855591, 0.07982651, 0.79583438, 0.45299326])

xc=np.array([0.3891439,  0.10542822, 0.24982279, 0.80776906, 0.43745421])

xc=np.array([0.41061764, 0.10059446, 0.1299048,  0.79572175, 0.46274672])

xc=np.array([0.38681067, 0.08642281, 0.27674663, 0.80784814, 0.45397164])

xc=np.array([0.39201713, 0.08418509, 0.29896431, 0.75446918, 0.48368333])

xc=np.array([0.41210637, 0.0857334 , 0.23632749, 0.99      , 0.16526502])

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


#compares passthroughs in the data and in the model
PHp,PWp=simple_extract(root+'/Tables/elasticity_persistent_earnings.txt')
PHt,PWt=simple_extract(root+'/Tables/elasticity_transitory_earnings.txt')



table=r'...persistent husband shocks & \textbf{'+p33(B['BPP_PER']['ym_cm'])+'}/\\textcolor{red}{'+PHp['hus']+'} & \\textbf{'+p33(B['BPP_PER']['ym_cw'])+'}/\\textcolor{red}{'+PHp['wif']+'} & \\\\ '+\
      r'...persistent wife shocks    & \textbf{'+p33(B['BPP_PER']['yw_cm'])+'}/\\textcolor{red}{'+PWp['hus']+'} & \\textbf{'+p33(B['BPP_PER']['yw_cw'])+'}/\\textcolor{red}{'+PWp['wif']+'}  &\\\\ '+\
      r'&  &  &   \\\\[0.5ex] '+\
      r'...transitory husband shocks   & \textbf{'+p33(B['BPP_MPC']['ym_cm'])+'}/\\textcolor{red}{'+PHt['hus']+'} & \\textbf{'+p33(B['BPP_MPC']['ym_cw'])+'}/\\textcolor{red}{'+PHt['wif']+'} & \\\\ '+\
      r'...transitory wife shocks    & \textbf{'+p33(B['BPP_MPC']['yw_cm'])+'}/\\textcolor{red}{'+PWt['hus']+'} & \\textbf{'+p33(B['BPP_MPC']['yw_cw'])+'}/\\textcolor{red}{'+PWt['wif']+'}  & \\\\\\bottomrule '


with open(root+'/Model/results/elasticity_BPP_model_vs_data.tex', 'w') as f: f.write(table); f.close() 





