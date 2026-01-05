# -*- coding: utf-8 -*-
"""

Not stand along. Run this within a given parametrization in calibration.py 
to check whether the effect of the pension using actual rules and the 
approximation used in the paper lead to similar gains/drops in pension
for men and women. The core idea is to compare the pension you would get
by staying divorced or divorcing in the period before retirement before and
after the implementation of the pension policy. 

Parameter 0.715 rescales the women's pension to account for the fact that
many women do not work throughout life. 0.715 is the avg employment rate
among ALL women, single, divorced and married

@author: 32489
"""

# Store parameters here
par=M_bef.par

# Potential (if working 1 unit of time) gross income of men and women
YM=par.grid_zm[np.arange(par.T),0,M_bef.sim.iz,0]
YW=par.grid_zw[np.arange(par.T),0,M_bef.sim.iz,0]

# Actual labor income of women, taking into account their labor supply
YWa=YW*par.grid_wlp[M_bef.sim.WLP]
YMa=YM.copy()

# Dummy for being in a couple
iscouple=M_bef.sim.couple.copy()
iscouple[(age<age_initial[:,None])]=False #before age_initial men and women are single


###############################################################################################################
# 1. CALCULATION OF PENSION BASED OF EXACT RULES
# E_... stands for "Exact"
###############################################################################################################

# Compute the shared wage part of pension, while couples are married
E_shared_pension=np.mean((YMa[:,:M_bef.par.Tr]+YWa[:,:M_bef.par.Tr])/2,where=iscouple[:,:M_bef.par.Tr],axis=1)
E_shared_pension[np.isnan(E_shared_pension)]=0.0#do not stay married

# Compute wage part of pension while not in a couple
E_ind_pension_m=np.mean(YMa[:,:M_bef.par.Tr],where=(~iscouple[:,:M_bef.par.Tr]),axis=1)
E_ind_pension_w=np.mean(YWa[:,:M_bef.par.Tr],where=(~iscouple[:,:M_bef.par.Tr]),axis=1)

# Compute the relative weight of shared vs. individual pension based on time spent together
E_w=np.mean(iscouple[:,:M_bef.par.Tr],axis=1)

# Pension after the reform
E_pension_m_aft=M_bef.par.p_b+M_bef.par.κ*(E_w*E_shared_pension+(1-E_w)*E_ind_pension_m)
E_pension_w_aft=M_bef.par.p_b+M_bef.par.κ*(E_w*E_shared_pension+(1-E_w)*E_ind_pension_w)

# Pension before the reform
E_pension_m_bef=M_bef.par.p_b+M_bef.par.κ*(np.mean(YMa[:,:M_bef.par.Tr],axis=1))
E_pension_w_bef=M_bef.par.p_b+M_bef.par.κ*(np.mean(YWa[:,:M_bef.par.Tr],axis=1))



#######################################################################################
# 2. COMPUTE A_w, the approximated time spent into a relationship give age at divorce
######################################################################################

##
#First big effort it to create wcc, AN ESTIMATION of the share time spent single and in a couple
#based only on the age at divorce. Idea: for each possible date when the marriage is formed,
#compute a share of time spent in a couple. Weight each one of the possible dates depending
#on their empirical distribution, taking into account that you cannot divorce if not married
#yet (this possibilities should get weight 0)
##

# Create array "Fage" with age at divorce or retirement out
idx = ((M_bef.sim.couple==0) & (M_bef.sim.couple_lag==1) & (age>=age_initial[:,None])).argmax(axis=1)
Fage = age[np.arange(age.shape[0]), idx]-20
Fage[Fage<=0]=M_bef.par.Tr-1


# Finally compute the estimated (relative) time spent in a relationship A_w
A_w=M_bef.par.PW[np.array(Fage//par.Dper,dtype=np.int_)]


###############################################################################################################
# 3. CALCULATION OF PENSION BASED OF APPROXIMATED RULES
# A_... stands for "Approximate"
###############################################################################################################
    

# Compute the approximated share of pension
A_shared_pension=((YMa+YWa*0.565)/2)[:,M_bef.par.Tr-1]


# Compute the approximated pension before the reform. 71.5% is the overall WLP across single and married women in the data
A_pension_m_bef=M_bef.par.p_b+M_bef.par.κ*YM[:,M_bef.par.Tr-1]
A_pension_w_bef=M_bef.par.p_b+M_bef.par.κ*YW[:,M_bef.par.Tr-1]*0.715*par.grid_wlp[-1]

# Compute the approximate pension after the reform. 56.5% is WLP amonng married women in the data
A_pension_m_aft=M_bef.par.p_b+M_bef.par.κ* ((1.0-A_w)*YM[:,M_bef.par.Tr-1]                 +(A_w)*A_shared_pension)
A_pension_w_aft=M_bef.par.p_b+M_bef.par.κ* ((1.0-A_w)*YW[:,M_bef.par.Tr-1]*par.grid_wlp[-1]+(A_w)*A_shared_pension)

###############################################################################################################
# 4. DIAGNOSTICS: COMPARE EXACT PENSIONS WITH APPROXIMATIONS
###############################################################################################################
 
# Relative time spend in a couple
print("Rel. time spent in a couple: correlation for all {}, just actual divorces {}".format(np.corrcoef(A_w,E_w)[0,1],np.corrcoef(A_w[Fage<44],E_w[Fage<44])[0,1]))


#Ratio of pension before and after the reform
print("E: Pension now/ pension before: M {}, F {}".format(
    E_pension_m_aft.mean()/E_pension_m_bef.mean(),
    E_pension_w_aft.mean()/E_pension_w_bef.mean()
    ))

print("A: Pension now/ pension before: M {}, F {}".format(
    A_pension_m_aft.mean()/A_pension_m_bef.mean(),
    A_pension_w_aft.mean()/A_pension_w_bef.mean()
    ))

print("E actual divorce: Pension now/ pension before: M {}, F {}".format(
    E_pension_m_aft[Fage<44].mean()/E_pension_m_bef[Fage<44].mean(),
    E_pension_w_aft[Fage<44].mean()/E_pension_w_bef[Fage<44].mean()
    ))

print("A actual divorce: Pension now/ pension before: M {}, F {}".format(
    A_pension_m_aft[Fage<44].mean()/A_pension_m_bef[Fage<44].mean(),
    A_pension_w_aft[Fage<44].mean()/A_pension_w_bef[Fage<44].mean()
    ))


#Correlations between pensions
print("Correlation E,A: bef m {}, aft m {}, bef w {}, aft w {}".format(np.corrcoef(E_pension_m_bef,A_pension_m_bef)[0,1],
                                                                       np.corrcoef(E_pension_m_aft,A_pension_m_aft)[0,1],
                                                                       np.corrcoef(E_pension_w_bef,A_pension_w_bef)[0,1],
                                                                       np.corrcoef(E_pension_w_aft,A_pension_w_aft)[0,1]))

print("Cor E,A divorce: bef m {}, aft m {}, bef w {}, aft w {}".format(np.corrcoef(E_pension_m_bef[Fage<44],A_pension_m_bef[Fage<44])[0,1],
                                                                       np.corrcoef(E_pension_m_aft[Fage<44],A_pension_m_aft[Fage<44])[0,1],
                                                                       np.corrcoef(E_pension_w_bef[Fage<44],A_pension_w_bef[Fage<44])[0,1],
                                                                       np.corrcoef(E_pension_w_aft[Fage<44],A_pension_w_aft[Fage<44])[0,1]))

#Average pensions:
print('Avg pensions bef: E m {}, E w {},  A m {}, A w {}'.format(E_pension_m_bef.mean(),
                                                                 E_pension_w_bef.mean(),
                                                                 A_pension_m_bef.mean(),
                                                                 A_pension_w_bef.mean()))

print('Avg pensions aft: E m {}, E w {},  A m {}, A w {}'.format(E_pension_m_aft.mean(),
                                                                 E_pension_w_aft.mean(),
                                                                 A_pension_m_aft.mean(),
                                                                 A_pension_w_aft.mean()))

print('Avg pens div bef: E m {}, E w {},  A m {}, A w {}'.format(E_pension_m_bef[Fage<44].mean(),
                                                                 E_pension_w_bef[Fage<44].mean(),
                                                                 A_pension_m_bef[Fage<44].mean(),
                                                                 A_pension_w_bef[Fage<44].mean()))

print('Avg pens div aft: E m {}, E w {},  A m {}, A w {}'.format(E_pension_m_aft[Fage<44].mean(),
                                                                 E_pension_w_aft[Fage<44].mean(),
                                                                 A_pension_m_aft[Fage<44].mean(),
                                                                 A_pension_w_aft[Fage<44].mean()))