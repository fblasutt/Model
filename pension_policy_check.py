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


def income_single(par,t,iD,ih,iz,assets,women=True): 
    """"
    This gives gross and net labor income of singles income
    """ 
    
    ws=1.0 if t>=par.Tr else par.grid_wlp[-1]
    labor_income =  par.grid_zws[t,iD,iz,ih]*ws if women else par.grid_zms[t,iD,iz,ih]#without HC! 
   
    tax_income = (labor_income) -par.Λ*(labor_income)**(1-par.τ)#taxes(labor_income,s=True)# 
  
    
    
    if women: return labor_income-tax_income+par.alimony,labor_income+par.alimony,tax_income
    else:    return  labor_income-tax_income-par.alimony,labor_income-par.alimony,tax_income
    
    
    
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
iscouple[(age<age_marriage[:,None])]=False #before age_initial men and women are single


###############################################################################################################
# 1. CALCULATION OF PENSION BASED OF EXACT RULES
# E_... stands for "Exact"
###############################################################################################################

# # Compute the shared wage part of pension, while couples are married
# E_shared_pension=np.mean((YMa[:,:M_bef.par.Tr]+YWa[:,:M_bef.par.Tr])/2,where=iscouple[:,:M_bef.par.Tr],axis=1)
# E_shared_pension[np.isnan(E_shared_pension)]=0.0#do not stay married

# # Compute wage part of pension while not in a couple
# E_ind_pension_m=np.mean(YM[:,:M_bef.par.Tr],where=(~iscouple[:,:M_bef.par.Tr]),axis=1)
# E_ind_pension_w=np.mean(YW[:,:M_bef.par.Tr]*par.grid_wlp[-1],where=(~iscouple[:,:M_bef.par.Tr]),axis=1)

# # Compute the relative weight of shared vs. individual pension based on time spent together
# E_w=np.mean(iscouple[:,:M_bef.par.Tr],axis=1)

# # Pension after the reform
# E_pension_m_aft=M_bef.par.p_b+M_bef.par.κ*(E_w*E_shared_pension+(1-E_w)*E_ind_pension_m)
# E_pension_w_aft=M_bef.par.p_b+M_bef.par.κ*(E_w*E_shared_pension+(1-E_w)*E_ind_pension_w)

# # Pension before the reform
# E_pension_m_bef=M_bef.par.p_b+M_bef.par.κ*(np.mean(YMa[:,:M_bef.par.Tr],axis=1))
# E_pension_w_bef=M_bef.par.p_b+M_bef.par.κ*(np.mean(YWa[:,:M_bef.par.Tr],axis=1))






# Compute the shared wage part of pension, while couples are married
E_shared_pensionm=np.array([(iscouple[:,:i]*(YMa[:,:i]+YWa[:,:i])/2+(1-iscouple[:,:i])*YMa[:,:i]).mean(axis=1)*(i)/(M.par.Tr-1)
                                   +YMa[:,i:M.par.Tr].mean(axis=1)*(M.par.Tr-1-i)/(M.par.Tr-1) for i in range(M.par.Tr)])

E_shared_pensionw=np.array([(iscouple[:,:i]*(YMa[:,:i]+YWa[:,:i])/2+(1-iscouple[:,:i])*YWa[:,:i]).mean(axis=1)*(i)/(M.par.Tr-1)
                                   +(YW[:,i:M.par.Tr]*par.grid_wlp[-1]).mean(axis=1)*(M.par.Tr-1-i)/(M.par.Tr-1) for i in range(M.par.Tr)])


# Pension after the reform
E_pension_m_aft=M_bef.par.p_b+M_bef.par.κ*E_shared_pensionm
E_pension_w_aft=M_bef.par.p_b+M_bef.par.κ*E_shared_pensionw

# Pension before the reform
E_pension_m_bef=np.array([M_bef.par.p_b+M_bef.par.κ*(np.mean(YMa[:,:M_bef.par.Tr],axis=1)) for i in range(M.par.Tr)])
E_pension_w_bef=np.array([M_bef.par.p_b+M_bef.par.κ*(YWa[:,:i].mean(axis=1)*(i)/(M.par.Tr-1)+(YW[:,i:M.par.Tr].mean(axis=1)*par.grid_wlp[-1])*(M.par.Tr-1-i)/(M.par.Tr-1)) for i in range(M.par.Tr)])






#######################################################################################
# 2. COMPUTE A_w, the approximated time spent into a relationship give age at divorce
######################################################################################

#Loop over age and compute what would have been the pension of men and women in the 
#model if they split in a given year after the reform
A_pension_m_aft,A_pension_w_aft=np.zeros((2,par.simN,par.Tr))


for i in range(par.simN):
    for t in range(par.Tr):
        
        A_pension_m_aft[i,t]=income_single(M.par,par.Tr,t//par.Dper,M.sim.ih[i,par.Tr],M.sim.iz[i,par.Tr],M.sim.Am[i,par.Tr],women=False)[1]
        A_pension_w_aft[i,t]=income_single(M.par,par.Tr,t//par.Dper,M.sim.ih[i,par.Tr],M.sim.iz[i,par.Tr],M.sim.Aw[i,par.Tr],women=True)[1]
        
        
        
        
A_pension_m_bef=np.repeat(M_bef.sim.incmg[:,par.Tr][:,None],par.Tr,axis=1)
A_pension_w_bef=np.repeat(M_bef.sim.incwg[:,par.Tr][:,None],par.Tr,axis=1)



###############################################################################################################
# 4. DIAGNOSTICS: COMPARE EXACT PENSIONS WITH APPROXIMATIONS
###############################################################################################################


#Ratio of pension before and after the reform
print("E: Pension now/ pension before: M {}, F {}".format(
    E_pension_m_aft.mean(axis=1)/E_pension_m_bef.mean(axis=1),
    E_pension_w_aft.mean(axis=1)/E_pension_w_bef.mean(axis=1)
    ))

E_ratio_m=E_pension_m_aft.mean(axis=1)/E_pension_m_bef.mean(axis=1)
E_ratio_w=E_pension_w_aft.mean(axis=1)/E_pension_w_bef.mean(axis=1)

print("A: Pension now/ pension before: M {}, F {}".format(
    A_pension_m_aft.mean(axis=0)/A_pension_m_bef.mean(axis=0),
    A_pension_w_aft.mean(axis=0)/A_pension_w_bef.mean(axis=0)
    ))

A_ratio_m=A_pension_m_aft.mean(axis=0)/A_pension_m_bef.mean(axis=0)
A_ratio_w=A_pension_w_aft.mean(axis=0)/A_pension_w_bef.mean(axis=0)


plt.plot(np.arange(par.Tr),E_ratio_m,np.arange(par.Tr),A_ratio_m)

plt.plot(np.arange(par.Tr),E_ratio_w,np.arange(par.Tr),A_ratio_w)



