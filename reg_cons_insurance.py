import numpy as np
from matplotlib import pyplot as plt

#Root 
root='C:/Users/32489/Dropbox/Family Risk Sharing'

def insurance(m,sample,shock_type='permanent',shock_gender='Male',consumption_gender='Male',name_file='baseline',name_line='Baseline'):
    
    """
    This function takes an input model m (with solution and simulations) and considers
    a sample "sample" to compute pass-throughs from income shocks (all, transitory and persistent)
    expenditures (total, private, and public), both usins model-observed shocks and the BPP methodology.
    
    Finally, given the information of provided variables in the optional arguments, it computes a
    decomposition of the insurance arising from the chosen shock, for the chosen type of private consumption  
    """
    
    # Store parameters
    par=m.par
    
    #######################################################
    #Selection issues below: only use one model m
    ######################################################

    # Lagged and leads of sample
    sample1  = np.roll(sample,1,axis=1)
    sample2  = np.roll(sample,2,axis=1)
    sample_1 = np.roll(sample,-1,axis=1)

    # Select different samples depending of WLP and being in a couple
    sm   = (m.sim.couple[sample1]==1) & (m.sim.couple[sample]==1)
    smw  = (sm) & (m.sim.WLP[sample1]>0) & (m.sim.WLP[sample]>0) 
    sm1  = (sm) & (m.sim.couple[sample2]==1)
    smw1 = (smw) & (m.sim.couple[sample2]==1) & (m.sim.WLP[sample2]>0) 
    sm11 = (sm1) & (m.sim.couple[sample_1]==1)
    smw11= (smw1) & (m.sim.couple[sample_1]==1) & (m.sim.WLP[sample_1]>0) 


    #################################
    #Consumption computation
    ##################################

    # Consumption levels
    cw = m.sim.Cw  # Wife private consumption
    cm = m.sim.Cm  # Husband private consumption
    d  = m.sim.dw  # Public exppenditures
    cp = cw+cm     # Total private consumption
    C  = cp+d      # Total consumption
  
    # Log consumption growth  for variout types of consumption
    Δcm  = np.log(cm[sample1]/cm[sample])                            # Husband ¨Private consumption Cm
    Δcw  = np.log(cw[sample1]/cw[sample])                            # Wife     Private consumption Cw
    Δcp  = np.log((cw[sample1]+cm[sample1])/(cw[sample]+cm[sample])) # Husband+ wife Private consumption Cm+Cw
    Δd   = np.log(d[sample1]/d[sample])                              # Home good expenditures d
    ΔC   = np.log(m.sim.C_tot[sample1]/m.sim.C_tot[sample])          # Total consumption (Cw+Cm+d)
    
    # Log consumption growth ofconsumption ratios
    Δs  = np.log((cw[sample1]/cp[sample1])/(cw[sample]/cp[sample]))  # cw/cp
    Δs1 = np.log((cm[sample1]/cp[sample1])/(cm[sample]/cp[sample]))  # cm/cp
    Δsp = np.log((cp[sample1]/C[sample1])/(cp[sample]/C[sample]))    # cp/C 
    Δws =np.log((cw[sample1]/cm[sample1])/(cw[sample]/cm[sample]))   # cw/cm
      
    #Now the equivalent changes, but in levels not in log
    ΔLCm  = cm[sample1]  -cm[sample]
    ΔLCw  = cw[sample1]  -cw[sample]
    ΔLws  = cw[sample1]/(cm[sample1])  -cw[sample]/(cm[sample])
    ΔLC   = m.sim.C_tot[sample1]-m.sim.C_tot[sample]
    ΔLd   = d[sample1]-d[sample]
    
    ################################
    #Income Shocks below
    ################################

    #Build transitory (ϵ) and persistent (z) shocks
    zm,ϵm,zw,ϵw=np.zeros((4,par.simN,par.T))
    izm=m.sim.iz%par.num_zw
    izw=m.sim.iz//par.num_zm
    ID=m.sim.ID
    for t in range(par.T):
        for i in range(par.simN):
        
            zm[i,t]=par.grid_pm[t,ID[i,t],izm[i,t],m.sim.ih[i,t]]
            ϵm[i,t]=par.grid_ϵm[t,ID[i,t],izm[i,t],m.sim.ih[i,t]]
            zw[i,t]=par.grid_pw[t,ID[i,t],izw[i,t],m.sim.ih[i,t]]
            ϵw[i,t]=par.grid_ϵw[t,ID[i,t],izw[i,t],m.sim.ih[i,t]]
                
    Δzm=zm[sample1]-zm[sample] # persisten shocks husband
    Δϵm=ϵm[sample1]-ϵm[sample] # transitory shocks husband
    Δzw=zw[sample1]-zw[sample] # persistent shocks wife
    Δϵw=ϵw[sample1]-ϵw[sample] # transitory shocks wife
        
    # Total shocks, aggregated by type (transitory+persistent) or within couple
    Δz=Δzm+Δzw
    Δϵ=Δϵm+Δϵw
    Δtm=Δzm+Δϵm
    Δtw=Δzw+Δϵw
    
    # Log income growth. It is gross, unless _net, denoting net income, is attached
    ΔYm    = np.log(m.sim.incmg[sample1]/m.sim.incmg[sample]) # husband gross income
    ΔYw    = np.log(m.sim.incwg[sample1]/m.sim.incwg[sample]) # wife gross income
    ΔY     = np.log((m.sim.incmg[sample1]+m.sim.incwg[sample1])/(m.sim.incmg[sample]+m.sim.incwg[sample])) # household gross income
    ΔY_net = np.log((m.sim.incm[sample1]+m.sim.incw[sample1])/(m.sim.incm[sample]+m.sim.incw[sample]))     # household net income
    
    # Changes in the Level of income
    ΔLYm  =m.sim.incmg[sample1]  -m.sim.incmg[sample]
    ΔLYw  =m.sim.incwg[sample1]  -m.sim.incwg[sample]
    ΔLY   =ΔLYm+ΔLYw
        
    # Changes in income one period ahed (t+1) (husband m, wife w and total)
    ΔY1m  =np.log(m.sim.incmg[sample2]/m.sim.incmg[sample1])
    ΔY1w  =np.log(m.sim.incwg[sample2]/m.sim.incwg[sample1])
    ΔY1   =np.log((m.sim.incmg[sample2]+m.sim.incwg[sample2])/(m.sim.incmg[sample1]+m.sim.incwg[sample1]))

    # Changes in income one period before (t-1) (husband m, wife w and total)
    ΔY_1m  =np.log(m.sim.incmg[sample]/m.sim.incmg[sample_1])
    ΔY_1w  =np.log(m.sim.incwg[sample]/m.sim.incwg[sample_1])
    ΔY_1   =np.log((m.sim.incmg[sample]+m.sim.incwg[sample])/(m.sim.incmg[sample_1]+m.sim.incwg[sample_1]))

    # These variables below are used to construct the BPP moments
    ΔYYm  =ΔYm+ΔY1m+ΔY_1m 
    ΔYYw  =ΔYw+ΔY1w+ΔY_1w 
    ΔYY   =ΔY +ΔY1 +ΔY_1   
    
    # Change in WLP
    ΔWLP=m.par.grid_wlp[m.sim.WLP][sample1]-m.par.grid_wlp[m.sim.WLP][sample]
    
    #Love shock changes
    lovw,lovm=np.zeros((2,m.par.simN,m.par.T))
    
    for i in range(par.T):lovw[:,i]=par.grid_lovew[i][m.sim.love[:,i]//par.num_lovem]
    for i in range(par.T):lovm[:,i]=par.grid_lovem[i][m.sim.love[:,i]%par.num_lovew]
    
    Δlovw=lovw[sample1]-lovw[sample]
    Δlovm=lovm[sample1]-lovm[sample]
    

    ##########################################################
    # Variance decomposition of individual consumption volatility
    ##########################################################
    # Identity: Δlog c^g = Δlog C + Δlog s^g, where C = cw+cm (total private) and s^g = c^g/C.
    # =>  Var(Δlog c^g) = Var(Δlog C) + Var(Δlog s^g) + 2 Cov(Δlog C, Δlog s^g).
    # Under limited commitment Δlog s^g ≠ 0 only on rebargaining periods, so Var(Δlog s^g)
    # isolates rebargaining. The covariance captures how rebargaining co-moves with aggregate
    # HH private consumption (negative = within-household insurance; positive = amplification).

    def _vardec(dC, ds, dc, cond):
        """Variance decomposition of Var(Δlog c) into aggregate + rebargaining + covariance."""
        dC_, ds_, dc_ = dC[cond], ds[cond], dc[cond]
        V_total = dc_.var(ddof=1)
        V_agg   = dC_.var(ddof=1)
        V_reb   = ds_.var(ddof=1)
        K       = np.cov(dC_, ds_, ddof=1)[0, 1]
        return {
            # Variance components (levels)
            'V_total':        V_total,
            'V_agg':          V_agg,            # aggregate HH private consumption
            'V_reb':          V_reb,            # rebargaining (share change)
            '2Cov':           2.0*K,
            # Raw shares of total volatility (sum to 1, with 2·Cov as separate term)
            'sh_agg':         V_agg / V_total,
            'sh_reb':         V_reb / V_total,
            'sh_cov':         2.0*K / V_total,
            # Identity residual (should be ≈ 0; catches bugs)
            'identity_resid': V_total - (V_agg + V_reb + 2.0*K),
        }

    # Δcp = Δlog(cw+cm) total private, Δs = Δlog(cw/cp) wife share, Δs1 = Δlog(cm/cp) husband share
    vardec_wife    = _vardec(Δcp, Δs,  Δcw, sm)
    vardec_husband = _vardec(Δcp, Δs1, Δcm, sm)     
    

    ##########################################
    # Compact OLS regression
    ##########################################
    def ols(indep,dep,cond,take=1,cov=None):
        """
        Perform Ordinary Least Squares (OLS) regression on a filtered subsample of 1D arrays.
        
        Parameters:
        -----------
        indep : ndarray
            Main independent variable (1D array).
        dep : ndarray
            Dependent variable (1D array).
        cond : ndarray (boolean)
            Boolean array used to select observations (e.g. based on a condition).
        take : int, optional (default=1)
            Index of the coefficient to return. 
            - 0 corresponds to the intercept,
            - 1 corresponds to the main independent variable,
            - 2+ are for additional covariates (if any).
        cov : list of ndarray, optional
            List of additional 1D covariate arrays to include in the regression, stored in a tuple
        
        Returns:
        --------
        β : float
            The `take`-th OLS coefficient from the regression. 

        """
        
        intercept=np.ones(sample.shape)[sample][cond]
        
        X=np.hstack((intercept[:,None],indep[cond][:,None]))#explicative variables 
        
        if (cov!=None):
         for j in range(len(cov)):
             
             X=np.hstack((X,cov[j][cond][:,None]))
     
        try:  
            #OLS on untreated observations 
            β = (np.linalg.inv(X.T @ X) @ (X.T @ dep[cond].flatten()))[take]
            
        except:
            
            β=100000.0
         
        return β
    
    ##################################
    #Compute and store pass-through   
    ##################################
    
    #Notation: all: means all income Y; per: persistent shock, tra: transitory shock
    #all_x_y means shock of x's income on m's expenditure
    indc={'all_m_m':ols(ΔYm,Δcm,sm),
         'all_m_w':ols(ΔYm,Δcw,sm),
         'all_w_m':ols(ΔYw,Δcm,smw),
         'all_w_w':ols(ΔYw,Δcw,smw),     
         'per_m_m':ols(Δzm,Δcm,sm),
         'per_m_w':ols(Δzm,Δcw,sm),
         'per_m_p':ols(Δzm,Δcp,sm),
         'per_w_p':ols(Δzw,Δcp,sm),
         'per_w_m':ols(Δzw,Δcm,smw),
         'per_w_w':ols(Δzw,Δcw,smw),
         'tra_m_m':ols(Δϵm,Δcm,sm),
         'tra_m_w':ols(Δϵm,Δcw,sm),
         'tra_w_m':ols(Δϵw,Δcm,smw),
         'tra_w_w':ols(Δϵw,Δcw,smw)}
    
    # Similar to above but in levels
    level={'all_m_m':ols(ΔLYm,ΔLCm,sm),
           'all_m_w':ols(ΔLYm,ΔLCw,sm),
           'all_w_m':ols(ΔLYw,ΔLCm,sm),
           'all_w_w':ols(ΔLYw,ΔLCw,sm)}
         
    
    #Pass-through on women's consumption ratio
    w_sh={'all_m':ols(ΔYm,Δws,sm),    
          'all_w':ols(ΔYw,Δws,smw),
          'per_m':ols(Δzm,Δws,sm),
          'per_w':ols(Δzw,Δws,smw),
          'tra_m':ols(Δϵm,Δws,sm),
          'tra_w':ols(Δϵw,Δws,smw),
          'level_m':ols(ΔLYm,ΔLws,sm),        
          'level_w':ols(ΔLYw,ΔLws,sm)}
    
    #Effect of income shocks on WLP (in percentage points)
    wlp= {'all_m':ols(Δtm,ΔWLP,sm),
          'all_w':ols(Δtw,ΔWLP,sm),
          'per_m':ols(Δzm,ΔWLP,sm),
          'per_w':ols(Δzw,ΔWLP,sm),
          'tra_m':ols(Δϵm,ΔWLP,sm),
          'tra_w':ols(Δϵw,ΔWLP,sm)}

    #Pass-throughs of various components of income on total consumption
    totc={'all':ols(ΔY,ΔC,sm),  
          'per':ols(Δz,ΔC,sm),
          'tra':ols(Δϵ,ΔC,sm),         
          'all_m':ols(ΔYm,ΔC,sm),         
          'per_m':ols(Δzm,ΔC,sm),
          'tra_m':ols(Δϵm,ΔC,sm),
          'all_w':ols(ΔYw,ΔC,smw),         
          'per_w':ols(Δzw,ΔC,smw),
          'tra_w':ols(Δϵw,ΔC,smw),
          'level_s':ols(ΔLY,ΔLC,sm),
          'level_m':ols(ΔLYm,ΔLC,sm),     
          'level_w':ols(ΔLYw,ΔLC,sm)}
    
    #Pass-throughs of various components of income on public ependiture
    dins={'all':ols(ΔY,Δd,sm),  
          'per':ols(Δz,Δd,sm),
          'tra':ols(Δϵ,Δd,sm),
          'all_m':ols(ΔYm,Δd,sm),        
          'all_w':ols(ΔYw,Δd,smw),
          'per_m':ols(Δzm,Δd,sm),
          'per_w':ols(Δzw,Δd,smw),
          'tra_m':ols(Δϵm,Δd,sm),
          'tra_w':ols(Δϵw,Δd,smw),
          'level_s':ols(ΔLY,ΔLd,sm),  
          'level_m':ols(ΔLYm,ΔLd,sm),   
          'level_w':ols(ΔLYw,ΔLd,sm)}
    
    #BPP moment for MPC
    BPP_MPC={'al_tot':np.mean(ΔY1[sm1]*ΔC[sm1])/np.mean(ΔY[sm1]*ΔY1[sm1]),
             'al_d':np.mean(ΔY1[sm1]*Δd[sm1])/np.mean(ΔY[sm1]*ΔY1[sm1]),
             'al_cm':np.mean(ΔY1[sm1]*Δcm[sm1])/np.mean(ΔY[sm1]*ΔY1[sm1]),
             'al_cw':np.mean(ΔY1[sm1]*Δcw[sm1])/np.mean(ΔY[sm1]*ΔY1[sm1]),             
             'ym_tot':np.mean(ΔY1m[sm1]*ΔC[sm1])/np.mean(ΔYm[sm1]*ΔY1m[sm1]),
             'ym_d':np.mean(ΔY1m[sm1]*Δd[sm1])/np.mean(ΔYm[sm1]*ΔY1m[sm1]),
             'ym_cm':np.mean(ΔY1m[sm1]*Δcm[sm1])/np.mean(ΔYm[sm1]*ΔY1m[sm1]),
             'ym_cw':np.mean(ΔY1m[sm1]*Δcw[sm1])/np.mean(ΔYm[sm1]*ΔY1m[sm1]),             
             'yw_tot':np.mean(ΔY1w[smw1]*ΔC[smw1])/np.mean(ΔYw[smw1]*ΔY1w[smw1]),
             'yw_d':np.mean(ΔY1w[smw1]*Δd[smw1])/np.mean(ΔYw[smw1]*ΔY1w[smw1]),
             'yw_cm':np.mean(ΔY1w[smw1]*Δcm[smw1])/np.mean(ΔYw[smw1]*ΔY1w[smw1]),
             'yw_cw':np.mean(ΔY1w[smw1]*Δcw[smw1])/np.mean(ΔYw[smw1]*ΔY1w[smw1])}
             
    #BPP moment describing inurance to persistent income shocks
    BPP_PER={'al_tot':np.mean(ΔYY[sm11]*ΔC[sm11])/np.mean(ΔY[sm11]*ΔYY[sm11]),
             'al_d':np.mean(ΔYY[sm11]*Δd[sm11])/np.mean(ΔY[sm11]*ΔYY[sm11]),
             'al_cm':np.mean(ΔYY[sm11]*Δcm[sm11])/np.mean(ΔY[sm11]*ΔYY[sm11]),
             'al_cw':np.mean(ΔYY[sm11]*Δcw[sm11])/np.mean(ΔY[sm11]*ΔYY[sm11]),             
             'ym_tot':np.mean(ΔYYm[sm11]*ΔC[sm11])/np.mean(ΔYm[sm11]*ΔYYm[sm11]),
             'ym_d':np.mean(ΔYYm[sm11]*Δd[sm11])/np.mean(ΔYm[sm11]*ΔYYm[sm11]),
             'ym_cm':np.mean(ΔYYm[sm11]*Δcm[sm11])/np.mean(ΔYm[sm11]*ΔYYm[sm11]),
             'ym_cw':np.mean(ΔYYm[sm11]*Δcw[sm11])/np.mean(ΔYm[sm11]*ΔYYm[sm11]),             
             'yw_tot':np.mean(ΔYYw[smw11]*ΔC[smw11])/np.mean(ΔYw[smw11]*ΔYYw[smw11]),
             'yw_d':np.mean(ΔYYw[smw11]*Δd[smw11])/np.mean(ΔYw[smw11]*ΔYYw[smw11]),
             'yw_cm':np.mean(ΔYYw[smw11]*Δcm[smw11])/np.mean(ΔYw[smw11]*ΔYYw[smw11]),
             'yw_cw':np.mean(ΔYYw[smw11]*Δcw[smw11])/np.mean(ΔYw[smw11]*ΔYYw[smw11])}
    

    ####################################################
    # Graph to make sure BPP captures well true shocks
    #################################################
    plt.scatter(BPP_PER['ym_cm'], indc['per_m_m'], c='blue', marker='o', s=100,zorder=2)
    plt.scatter(BPP_PER['ym_cm'], indc['per_m_m'], c='red', marker='o', s=30,zorder=2)    
    plt.scatter(BPP_PER['ym_cw'], indc['per_m_w'], c='blue', marker='o', s=100,zorder=2)
    plt.scatter(BPP_PER['ym_cw'], indc['per_m_w'], c='blue', marker='o', s=30,zorder=2)    
    plt.scatter(BPP_PER['yw_cm'], indc['per_w_m'], c='red', marker='o', s=100,zorder=2)
    plt.scatter(BPP_PER['yw_cm'], indc['per_w_m'], c='red', marker='o', s=30,zorder=2)
    plt.scatter(BPP_PER['yw_cw'], indc['per_w_w'], c='red', marker='o', s=100,zorder=2)
    plt.scatter(BPP_PER['yw_cw'], indc['per_w_w'], c='blue', marker='o', s=30,zorder=2)
    plt.scatter(BPP_MPC['ym_cm'], indc['tra_m_m'], c='blue', marker='^', s=100,zorder=2)
    plt.scatter(BPP_MPC['ym_cm'], indc['tra_m_m'], c='red', marker='^', s=30,zorder=2)
    plt.scatter(BPP_MPC['ym_cw'], indc['tra_m_w'], c='blue', marker='^', s=100,zorder=2)
    plt.scatter(BPP_MPC['yw_cm'], indc['tra_w_m'], c='red', marker='^', s=100,zorder=2)
    plt.scatter(BPP_MPC['yw_cm'], indc['tra_w_m'], c='red', marker='^', s=30,zorder=2) 
    plt.scatter(BPP_MPC['yw_cw'], indc['tra_w_w'], c='red', marker='^', s=100,zorder=2)
    plt.scatter(BPP_MPC['yw_cw'], indc['tra_w_w'], c='blue', marker='^', s=30,zorder=2)
    plt.plot([-0.2, 1], [-0.2, 1], color = 'black', linestyle='--', linewidth = 2,zorder=1)
    plt.xlabel("BPP") 
    plt.ylabel("True") 
    plt.savefig(root+'/Output files/model/BPP_true.eps', format='eps', bbox_inches="tight")  
    plt.show()
    
    # Innovation in consumption
    ξ=ΔC-(Δzm*totc['per_m']+Δzw*totc['per_w']+Δϵm*totc['tra_m']+Δϵw.var()*totc['tra_w'])
    
    
    
    #Variance decomposition
    

    ####################################################    
    # Decomposition of hh private consumption growth
    ##################################################
    
    # Which shock should I consider?
    if    (shock_type=='permanent') & (shock_gender=='Male'): SHOCK = Δzm; CONTROLS = (Δzw,Δϵm,Δϵw)
    elif  (shock_type=='permanent') & (shock_gender!='Male'): SHOCK = Δzw; CONTROLS = (Δzm,Δϵm,Δϵw)
    elif  (shock_type!='permanent') & (shock_gender=='Male'): SHOCK = Δϵm; CONTROLS = (Δzw,Δzm,Δϵw)
    elif  (shock_type!='permanent') & (shock_gender!='Male'): SHOCK = Δϵw; CONTROLS = (Δzw,Δϵm,Δzm)  
       
    # Which consumption should I consider?
    if   consumption_gender=='Male': ΔCONS= Δcm;SHARE=Δs1
    elif consumption_gender!='Male': ΔCONS= Δcw;SHARE=Δs
    
    ### A - household consumption decomposition like in Wu and Krueger AEJ Macro (same notation for the Ks)

    # Pass-trough SHOCK to husband, wife and total earnings
    κymp=ols(SHOCK,ΔYm,sm,cov=CONTROLS,take=1)
    κywp=ols(SHOCK,ΔYw,smw,cov=CONTROLS,take=1)
    κyp=ols(SHOCK,ΔY,sm,cov=CONTROLS,take=1)
    
    # Pass through from gross to net household earnings
    κynug=ols(ΔY,ΔY_net,sm)
    
    # Share of earnings going to husband
    sYm=np.mean(m.sim.incmg[sample][sm]/(m.sim.incmg[sample][sm]+m.sim.incwg[sample][sm]))
    
    # Find the different sources of insurance
    K1=κymp
    K2=sYm
    K3=1+((1-sYm)*κywp)/(sYm*κymp)
    K4=κyp/(K1*K2*K3)
    K5=κynug
    K6=ols(SHOCK,ΔC,sm,cov=CONTROLS,take=1)/(κyp*K5)
    
    # Finally the decomposition of household insurance   
    Passive_insurance = 1-K1*K2
    Active_insurance  = 1-K1*K2*K3*K4-(1-K1*K2)
    Taxes             = 1-K1*K2*K3*K4*K5-(1-K1*K2*K3*K4)
    Self_insurance    = 1-K1*K2*K3*K4*K5*K6-(1-K1*K2*K3*K4*K5)
       
    ### B from hosehold to individual conusmption insurance
    
    κs1mp=ols(SHOCK,SHARE,sm,cov=CONTROLS,take=1) # pass-through from SHOCK to individual share of conusmption
    κspmp=ols(SHOCK,Δsp,sm,cov=CONTROLS,take=1)   # pass-through from SHOCK to private to total share of consumtion
    
    # Finally the decomposition of individual insurance
    HH_insurance=1-K1*K2*K3*K4*K5*K6
    private_shift=(1-HH_insurance-κspmp)-(1-HH_insurance)
    bargaining_shift=(1-HH_insurance-κspmp-κs1mp)-(1-HH_insurance-κspmp)
    
    # Total private consumption insurance
    ind_con_ins=1-ols(SHOCK,ΔCONS,sm,cov=CONTROLS,take=1)

    ### C Provide a latex code line with results
    
    # Clean the raw strings
    def p31(x): return str('%3.1f' % (x*100)) 
    
    # Create a line with the decomposition in latex and save it
    table=name_line+'& '+p31(Passive_insurance)+' & '+p31(Active_insurance)+'& '+p31(Taxes)+'& '+p31(Self_insurance)+'& '+p31(private_shift)+'& '+p31(bargaining_shift)+'& '+p31(ind_con_ins)
    with open(root+'/Output files/model/'+name_file+'exp.tex', 'w') as f: f.write(table); f.close() 
    
    # Store the decomposition of insurance
    ins_dec={ 'Passive_insurance':Passive_insurance,     
              'Active_insuranc':Active_insurance,
              'Taxes':Taxes,
              'Self_insurance':Self_insurance,         
              'HH_insurance':HH_insurance,
              'private_shift':private_shift,
              'bargaining_shift':bargaining_shift,
              'ind_con_ins':ind_con_ins}
    
  

    return {'indc':indc,'w_sh':w_sh,'totc':totc,'dins':dins,'BPP_MPC':BPP_MPC,'BPP_PER':BPP_PER,'wlp':wlp,'level':level,'ins_dec':ins_dec,
            'vardec_w':vardec_wife,'vardec_m':vardec_husband}