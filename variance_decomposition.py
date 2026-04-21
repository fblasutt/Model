# -*- coding: utf-8 -*- 
""" 
Created on Fri Feb 23 15:31:56 2024 
 
@author: 32489 
""" 
 
import numpy as np 
import Bargaining_numba as brg  
import pandas as pd
from reg_cons_insurance import insurance
import getpass


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


 
###################
#Pre reform model
####################
#tic=time.time()

#Model - limited commitment
m_LC = model.copy(name='numba_new_copy')    
m_LC.solve() 
m_LC.simulate() 

sample_LC= (age>age_initial[:,None]) & (age<=age_final[:,None]) & (m_LC.sim.couple_lag==1) #& (M.sim.couple==1) 

#Model - full commitment
m_FC = model.copy(name='numba_new_copy')    
m_FC.par.full=True
m_FC.solve() 
m_FC.simulate() 

sample_FC= (age>age_initial[:,None]) & (age<=age_final[:,None]) & (m_FC.sim.couple_lag==1) #& (M.sim.couple==1) 



##########################################################
# Build LaTeX rows for the variance-decomposition table
##########################################################
def vardec_row(B, spouse, label):
    """
    Return TWO tabular rows for one case (one spouse × one regime) from a
    results dict B returned by insurance(...):
      - row 1: variance values (×100) for the four columns;
      - row 2: share of total (integer %) in scriptsize, in parentheses,
        pulled close to row 1 with a negative \\\\[-0.5ex] separator.
    Between each numeric value is an empty cell (" & & ") for the narrow
    operator column that holds = or + in the LaTeX header. No trailing
    \\\\ on row 2 — caller adds it.

    Columns expected in the LaTeX template (8 total):
      label | Individual | op | Household private | op | Own share | op | Covariance
    """
    d = B['vardec_w'] if spouse == 'w' else B['vardec_m']
    # Near-zero guard: avoid "-0.00" / "(-0%)" when Δlog s ≡ 0 (full commitment)
    def v(x):  return '%.2f' % (100.0*x) if abs(x) > 1e-6 else '0.00'
    def p(x):  return '%.0f' % (100.0*x) if abs(x) > 1e-3 else '0'
    # Two plain tabular rows per case: values on top, scriptsize percentages below.
    # A negative \\[-0.5ex] pulls the % row close to the value row.
    pct = lambda x: r'{\scriptsize (' + p(x) + r'\%)}'
    # Empty cells (" & & ") sit in the narrow c-columns that hold = and + in the header.
    row1 = (label + ' & ' +
            v(d['V_total']) + ' & & ' +
            v(d['V_agg'])   + ' & & ' +
            v(d['V_reb'])   + ' & & ' +
            v(d['2Cov']))
    row2 = (' & ' + pct(1.0)         + ' & & ' +
                    pct(d['sh_agg']) + ' & & ' +
                    pct(d['sh_reb']) + ' & & ' +
                    pct(d['sh_cov']))
    return row1 + r' \\[-0.5ex]' + '\n' + row2


# Usage: after running the model under LC and FC to get B_LC and B_FC
# ------------------------------------------------------------------
B_LC = insurance(m_LC, sample_LC, ...)
B_FC = insurance(m_FC, sample_FC, ...)

table = '\n'.join([
    r'\textit{A. Wife} & & & & & & & \\',
    r'\addlinespace',
    vardec_row(B_LC, 'w', 'Limited commitment') + r' \\',
    vardec_row(B_FC, 'w', 'Full commitment')    + r' \\',
    r'\addlinespace',
    r'\textit{B. Husband} & & & & & & & \\',
    r'\addlinespace',
    vardec_row(B_LC, 'm', 'Limited commitment') + r' \\',
    vardec_row(B_FC, 'm', 'Full commitment'),     # no trailing \\
])
with open(root+'/Output files/model/vardec.tex', 'w') as f: f.write(table)