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
    
    #Age
    age=(np.cumsum(np.ones((m.par.simN,m.par.T)),axis=1)-1)+25#age of hh  
    
    #######################################################
    #Selection issues below: only use one model m
    ######################################################

    # Lagged and leads of sample
    sample1  = np.roll(sample,1,axis=1)
    sample2  = np.roll(sample,2,axis=1)
    sample_1 = np.roll(sample,-1,axis=1)

    # Select different samples depending of WLP and being in a couple
    sm   = (m.sim.couple[sample1]==1) & (m.sim.couple[sample]==1)
    sw  =  (sm) &  (m.sim.WLP[sample]>0) 
    smw  = (sm) & (m.sim.WLP[sample1]>0) & (m.sim.WLP[sample]>0) 
    sm1  = (sm) & (m.sim.couple[sample2]==1)
    smw1 = (smw) & (m.sim.couple[sample2]==1) & (m.sim.WLP[sample2]>0) 
    sm11 = (sm1) & (m.sim.couple[sample_1]==1)
    smw11= (smw1) & (m.sim.couple[sample_1]==1) & (m.sim.WLP[sample_1]>0) 


    ##########################################################
    # BPP first stage: residualize LOG LEVELS on observables
    ##########################################################
    # As in BPP (2008) the first stage is run on log LEVELS: each (log) panel
    # variable y[i,t] is regressed on observables (age polynomial) over couple
    # observations, and all growth variables below are FIRST DIFFERENCES OF THE
    # RESIDUALS. Non-finite entries (log of 0) are left as-is and get masked by
    # the regression conditions downstream.
    _aget  = np.broadcast_to((np.arange(par.T)/par.T)[None,:],(par.simN,par.T)).astype(float)
    # NB: no employment dummy in the controls -- WLP is an endogenous choice that
    # responds to the very shocks whose pass-through we measure; controlling for it
    # would absorb part of the behavioral response (age polynomial only, as in
    # simulated-BPP exercises a la Kaplan-Violante).
    _Zp    = np.stack([np.ones_like(_aget),_aget,_aget**2,_aget**3],axis=-1)
    _Zflat = _Zp.reshape(-1,_Zp.shape[-1])
    _fitok = (m.sim.couple==1).ravel()                                # fit on couples

    def _residualize(y):
        """Panel y (simN,T) minus its OLS projection on observables (fit on couples, finite obs)."""
        y=np.asarray(y,dtype=float); yf=y.ravel().copy()
        ok=np.isfinite(yf); fit=ok&_fitok
        if fit.sum()>_Zflat.shape[1]:
            b=np.linalg.lstsq(_Zflat[fit],yf[fit],rcond=None)[0]
            yf[ok]=yf[ok]-_Zflat[ok]@b
        return yf.reshape(y.shape)

    #################################
    #Consumption computation
    ##################################

    # Consumption levels
    cw = m.sim.Cw  # Wife private consumption
    cm = m.sim.Cm  # Husband private consumption
    d  = m.sim.dw  # Public exppenditures
    cp = cw+cm     # Total private consumption
    C  = cp+d      # Total consumption
  
    # BPP first stage on consumption: residualized log-level panels
    lcm=_residualize(np.log(cm)); lcw=_residualize(np.log(cw)); lcp=_residualize(np.log(cp))
    ldp=_residualize(np.log(d));  lC =_residualize(np.log(m.sim.C_tot)); lCc=_residualize(np.log(C))

    # Log consumption growth = first differences of residualized log levels
    Δcm  = lcm[sample1]-lcm[sample]  # Husband  private consumption Cm
    Δcw  = lcw[sample1]-lcw[sample]  # Wife     private consumption Cw
    Δcp  = lcp[sample1]-lcp[sample]  # Husband+wife private consumption Cm+Cw
    Δd   = ldp[sample1]-ldp[sample]  # Home good expenditures d
    ΔC   = lC[sample1] -lC[sample]   # Total consumption (Cw+Cm+d)
    
    # Log growth of consumption ratios (residualization is linear, so ratio
    # growth = difference of the residualized log growths)
    Δs  = Δcw-Δcp                          # cw/cp
    Δs1 = Δcm-Δcp                          # cm/cp
    Δsp = Δcp-(lCc[sample1]-lCc[sample])   # cp/C
    Δws = Δcw-Δcm                          # cw/cm
      
    #Now the equivalent changes, but in levels not in log (residualized level panels)
    Lcm=_residualize(cm); Lcw=_residualize(cw); LCt=_residualize(m.sim.C_tot)
    Ld =_residualize(d);  Lws=_residualize(cw/cm)
    ΔLCm  = Lcm[sample1]-Lcm[sample]
    ΔLCw  = Lcw[sample1]-Lcw[sample]
    ΔLws  = Lws[sample1]-Lws[sample]
    ΔLC   = LCt[sample1]-LCt[sample]
    ΔLd   = Ld[sample1] -Ld[sample]
    
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
                
    # BPP first stage on the shock panels (log-level components)
    zm=_residualize(zm); ϵm=_residualize(ϵm); zw=_residualize(zw); ϵw=_residualize(ϵw)

    Δzm=zm[sample1]-zm[sample] # persistent innovation husband (= contemporaneous persistent shock)
    Δzw=zw[sample1]-zw[sample] # persistent innovation wife

    # Transitory shock needs TWO non-interchangeable versions:
    #  * contemporaneous level ε_{t+1} (var σϵ²): the NEW shock -> pass-through/response
    #    regressions (this is what BPP_MPC estimates).
    #  * first difference ε_{t+1}-ε_t (var 2σϵ²): how it enters income growth ΔY ->
    #    Δtm/Δtw, the shock-variance decomposition, and ξ.
    ϵm_c=ϵm[sample1]            # contemporaneous transitory shock husband (level)
    ϵw_c=ϵw[sample1]            # contemporaneous transitory shock wife
    Δϵm=ϵm[sample1]-ϵm[sample]  # transitory CHANGE husband (difference)
    Δϵw=ϵw[sample1]-ϵw[sample]  # transitory CHANGE wife
        
    # Total shocks, aggregated by type (transitory+persistent) or within couple
    Δz=Δzm+Δzw
    Δϵ=ϵm_c+ϵw_c              # total contemporaneous transitory shock (pass-throughs only)
    Δtm=Δzm+Δϵm              # husband income-growth shock content (difference transitory)
    Δtw=Δzw+Δϵw              # wife income-growth shock content
    
    # BPP first stage on income: residualized log-level panels
    lym=_residualize(np.log(m.sim.incmg)); lyw=_residualize(np.log(m.sim.incwg))
    ly =_residualize(np.log(m.sim.incmg+m.sim.incwg))
    lyn=_residualize(np.log(m.sim.incm+m.sim.incw))

    # Log income growth. It is gross, unless _net, denoting net income, is attached
    ΔYm    = lym[sample1]-lym[sample] # husband gross income
    ΔYw    = lyw[sample1]-lyw[sample] # wife gross income
    ΔY     = ly[sample1] -ly[sample]  # household gross income
    ΔY_net = lyn[sample1]-lyn[sample] # household net income
    
    # Changes in the Level of income (residualized level panels)
    Lym=_residualize(m.sim.incmg); Lyw=_residualize(m.sim.incwg)
    ΔLYm  = Lym[sample1]-Lym[sample]
    ΔLYw  = Lyw[sample1]-Lyw[sample]
    ΔLY   = ΔLYm+ΔLYw
        
    # Changes in income one period ahed (t+1) (husband m, wife w and total)
    ΔY1m  = lym[sample2]-lym[sample1]
    ΔY1w  = lyw[sample2]-lyw[sample1]
    ΔY1   = ly[sample2] -ly[sample1]

    # Changes in income one period before (t-1) (husband m, wife w and total)
    ΔY_1m  = lym[sample]-lym[sample_1]
    ΔY_1w  = lyw[sample]-lyw[sample_1]
    ΔY_1   = ly[sample] -ly[sample_1]

    # These variables below are used to construct the BPP moments
    ΔYYm  =ΔYm+ΔY1m+ΔY_1m
    ΔYYw  =ΔYw+ΔY1w+ΔY_1w
    ΔYY   =ΔY +ΔY1 +ΔY_1

    ##########################################################
    # NET (after-tax) income versions of the BPP income variables.
    # m.sim.incm / m.sim.incw are individual net incomes (couples' joint
    # taxation already split inside resources_couple); household net = sum.
    ##########################################################
    lym_n=_residualize(np.log(m.sim.incm)); lyw_n=_residualize(np.log(m.sim.incw))

    # Net log income growth (household ΔY_net defined above from lyn)
    ΔYm_net = lym_n[sample1]-lym_n[sample] # husband net income
    ΔYw_net = lyw_n[sample1]-lyw_n[sample] # wife net income

    # Net income growth one period ahead (t+1) and before (t-1)
    ΔY1m_net  = lym_n[sample2]-lym_n[sample1]
    ΔY1w_net  = lyw_n[sample2]-lyw_n[sample1]
    ΔY1_net   = lyn[sample2] -lyn[sample1]
    ΔY_1m_net = lym_n[sample]-lym_n[sample_1]
    ΔY_1w_net = lyw_n[sample]-lyw_n[sample_1]
    ΔY_1_net  = lyn[sample] -lyn[sample_1]

    # BPP sums on net income
    ΔYYm_net  = ΔYm_net+ΔY1m_net+ΔY_1m_net
    ΔYYw_net  = ΔYw_net+ΔY1w_net+ΔY_1w_net
    ΔYY_net   = ΔY_net+ΔY1_net+ΔY_1_net
    
    # Change in WLP
    ΔWLP=np.array([(m.par.grid_wlp[m.sim.WLP][sample1]>0)],dtype=np.float64)[0]-np.array([(m.par.grid_wlp[m.sim.WLP][sample]>0)],dtype=np.float64)[0]
    
    #Love shock changes
    lovw,lovm=np.zeros((2,m.par.simN,m.par.T))
    
    for i in range(par.T):lovw[:,i]=par.grid_lovew[i][m.sim.love[:,i]//par.num_lovem]
    for i in range(par.T):lovm[:,i]=par.grid_lovem[i][m.sim.love[:,i]%par.num_lovew]
    
    # BPP first stage on love panels, then difference
    lovw=_residualize(lovw); lovm=_residualize(lovm)
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
    indc={'all_m_m':ols(ΔYm,Δcm,sm11),
         'all_m_w':ols(ΔYm,Δcw,sm11),
         'all_w_m':ols(ΔYw,Δcm,smw11),
         'all_w_w':ols(ΔYw,Δcw,smw11),     
         'per_m_m':ols(Δzm,Δcm,sm11, cov = (Δzw,ϵm_c,ϵw_c)),
         'per_m_w':ols(Δzm,Δcw,sm11, cov = (Δzw,ϵm_c,ϵw_c)),
         'per_m_p':ols(Δzm,Δcp,sm11),
         'per_w_p':ols(Δzw,Δcp,smw11, cov = (Δzm,ϵm_c,ϵw_c)),
         'per_w_m':ols(Δzw,Δcm,smw11, cov = (Δzm,ϵm_c,ϵw_c)),
         'per_w_w':ols(Δzw,Δcw,smw11, cov = (Δzm,ϵm_c,ϵw_c)),
         'tra_m_m':ols(ϵm_c,Δcm,sm1),
         'tra_m_w':ols(ϵm_c,Δcw,sm1),
         'tra_w_m':ols(ϵw_c,Δcm,smw1),
         'tra_w_w':ols(ϵw_c,Δcw,smw1)}
    
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
          'tra_m':ols(ϵm_c,Δws,sm),
          'tra_w':ols(ϵw_c,Δws,smw),
          'level_m':ols(ΔLYm,ΔLws,sm),        
          'level_w':ols(ΔLYw,ΔLws,sm)}
    
    #Effect of income shocks on WLP (in percentage points)
    wlp= {'all_m':ols(ΔYm,ΔWLP,sm),
          'all_w':ols(ΔYm,ΔWLP,sm),
          'per_m':ols(Δzm,ΔWLP,sm),
          'per_w':ols(Δzw,ΔWLP,sm),
          'tra_m':ols(ϵm_c,ΔWLP,sm),
          'tra_w':ols(ϵw_c,ΔWLP,sm)}

    #Pass-throughs of various components of income on total consumption
    totc={'all':ols(ΔY,ΔC,sm),  
          'per':ols(Δz,ΔC,sm),
          'tra':ols(Δϵ,ΔC,sm),         
          'all_m':ols(ΔYm,ΔC,sm),         
          'per_m':ols(Δzm,ΔC,sm),
          'tra_m':ols(ϵm_c,ΔC,sm),
          'all_w':ols(ΔYw,ΔC,smw),         
          'per_w':ols(Δzw,ΔC,smw),
          'tra_w':ols(ϵw_c,ΔC,smw),
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
          'tra_m':ols(ϵm_c,Δd,sm),
          'tra_w':ols(ϵw_c,Δd,smw),
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

    #Same BPP moments computed on NET (after-tax) income
    BPP_MPC_net={'al_tot':np.mean(ΔY1_net[sm1]*ΔC[sm1])/np.mean(ΔY_net[sm1]*ΔY1_net[sm1]),
             'al_d':np.mean(ΔY1_net[sm1]*Δd[sm1])/np.mean(ΔY_net[sm1]*ΔY1_net[sm1]),
             'al_cm':np.mean(ΔY1_net[sm1]*Δcm[sm1])/np.mean(ΔY_net[sm1]*ΔY1_net[sm1]),
             'al_cw':np.mean(ΔY1_net[sm1]*Δcw[sm1])/np.mean(ΔY_net[sm1]*ΔY1_net[sm1]),
             'ym_tot':np.mean(ΔY1m_net[sm1]*ΔC[sm1])/np.mean(ΔYm_net[sm1]*ΔY1m_net[sm1]),
             'ym_d':np.mean(ΔY1m_net[sm1]*Δd[sm1])/np.mean(ΔYm_net[sm1]*ΔY1m_net[sm1]),
             'ym_cm':np.mean(ΔY1m_net[sm1]*Δcm[sm1])/np.mean(ΔYm_net[sm1]*ΔY1m_net[sm1]),
             'ym_cw':np.mean(ΔY1m_net[sm1]*Δcw[sm1])/np.mean(ΔYm_net[sm1]*ΔY1m_net[sm1]),
             'yw_tot':np.mean(ΔY1w_net[smw1]*ΔC[smw1])/np.mean(ΔYw_net[smw1]*ΔY1w_net[smw1]),
             'yw_d':np.mean(ΔY1w_net[smw1]*Δd[smw1])/np.mean(ΔYw_net[smw1]*ΔY1w_net[smw1]),
             'yw_cm':np.mean(ΔY1w_net[smw1]*Δcm[smw1])/np.mean(ΔYw_net[smw1]*ΔY1w_net[smw1]),
             'yw_cw':np.mean(ΔY1w_net[smw1]*Δcw[smw1])/np.mean(ΔYw_net[smw1]*ΔY1w_net[smw1])}

    BPP_PER_net={'al_tot':np.mean(ΔYY_net[sm11]*ΔC[sm11])/np.mean(ΔY_net[sm11]*ΔYY_net[sm11]),
             'al_d':np.mean(ΔYY_net[sm11]*Δd[sm11])/np.mean(ΔY_net[sm11]*ΔYY_net[sm11]),
             'al_cm':np.mean(ΔYY_net[sm11]*Δcm[sm11])/np.mean(ΔY_net[sm11]*ΔYY_net[sm11]),
             'al_cw':np.mean(ΔYY_net[sm11]*Δcw[sm11])/np.mean(ΔY_net[sm11]*ΔYY_net[sm11]),
             'ym_tot':np.mean(ΔYYm_net[sm11]*ΔC[sm11])/np.mean(ΔYm_net[sm11]*ΔYYm_net[sm11]),
             'ym_d':np.mean(ΔYYm_net[sm11]*Δd[sm11])/np.mean(ΔYm_net[sm11]*ΔYYm_net[sm11]),
             'ym_cm':np.mean(ΔYYm_net[sm11]*Δcm[sm11])/np.mean(ΔYm_net[sm11]*ΔYYm_net[sm11]),
             'ym_cw':np.mean(ΔYYm_net[sm11]*Δcw[sm11])/np.mean(ΔYm_net[sm11]*ΔYYm_net[sm11]),
             'yw_tot':np.mean(ΔYYw_net[smw11]*ΔC[smw11])/np.mean(ΔYw_net[smw11]*ΔYYw_net[smw11]),
             'yw_d':np.mean(ΔYYw_net[smw11]*Δd[smw11])/np.mean(ΔYw_net[smw11]*ΔYYw_net[smw11]),
             'yw_cm':np.mean(ΔYYw_net[smw11]*Δcm[smw11])/np.mean(ΔYw_net[smw11]*ΔYYw_net[smw11]),
             'yw_cw':np.mean(ΔYYw_net[smw11]*Δcw[smw11])/np.mean(ΔYw_net[smw11]*ΔYYw_net[smw11])}


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
    ξ=ΔC-(Δzm*totc['per_m']+Δzw*totc['per_w']+Δϵm*totc['tra_m']+Δϵw*totc['tra_w'])
    
    
    
    ##########################################################
    # Shock-based variance decomposition of Var(Δlog target)
    ##########################################################
    # Project each target y ∈ {Δcw,Δcm,Δcp,ΔC,Δs,Δs1} on the six structural
    # shocks x ∈ {Δzw,Δzm,Δϵw,Δϵm,Δlovw,Δlovm} and attribute variance:
    #
    #   Var(Δlog y) = Σ_x κ²_{y,x} · Var(x)
    #               + 2 Σ_{x<x'} κ_{y,x} κ_{y,x'} · Cov(x, x')
    #               + Var(η_y),
    #
    # where κ_{y,x} is the partial pass-through of shock x to target y (OLS
    # with all other shocks as controls), and η_y is the residual. Cross
    # terms are non-negligible only for ϵ^w - ϵ^m (which are correlated by
    # construction); the others are near-zero in expectation but we include
    # all of them so the identity holds exactly in-sample.

    # List of (shock_name, innovation_array) pairs.
    _shock_list = [('zeta_w', Δzw), ('zeta_m', Δzm),
                   ('eps_w',  Δϵw), ('eps_m',  Δϵm),
                   ('psi_w',  Δlovw), ('psi_m', Δlovm)]

    def _shockdec(target, cond, shocks=_shock_list):
        """
        Decompose Var(target | cond) into own + cross + residual contributions
        using OLS partial pass-throughs and sample moments of the shocks.
        """
        names = [s[0] for s in shocks]
        arrs  = [s[1] for s in shocks]

        # Partial pass-throughs κ_x : coefficient on x when target is regressed
        # on all shocks simultaneously (other shocks as OLS controls).
        κ = {}
        for i, (nm, arr) in enumerate(shocks):
            controls = tuple(a for j, a in enumerate(arrs) if j != i)
            κ[nm] = ols(arr, target, cond, cov=controls, take=1)

        # Sample second moments of shocks on the same cond
        σ2    = {nm: float(np.var(arr[cond], ddof=1))     for nm, arr in shocks}
        σcov  = {(nm_i, nm_j): float(np.cov(a_i[cond], a_j[cond], ddof=1)[0, 1])
                 for i, (nm_i, a_i) in enumerate(shocks)
                 for j, (nm_j, a_j) in enumerate(shocks) if j > i}

        # Variance contributions
        own   = {nm: κ[nm]**2 * σ2[nm] for nm in names}
        cross = {(i, j): 2.0 * κ[i] * κ[j] * σcov[(i, j)] for (i, j) in σcov}

        V_total     = float(target[cond].var(ddof=1))
        V_explained = sum(own.values()) + sum(cross.values())
        V_resid     = V_total - V_explained

        return {
            'κ':           κ,            # pass-throughs
            'σ2':          σ2,           # shock variances
            'σ_cross':     σcov,         # shock pairwise covariances
            'own':         own,          # κ² · σ² per shock
            'cross':       cross,        # 2·κ·κ' · σ_{xy} per pair
            'V_total':     V_total,      # total Var(target)
            'V_explained': V_explained,  # sum(own) + sum(cross)
            'V_resid':     V_resid,      # Var(η_y)
            # Shares of total variance (sum to 1 including residual)
            'sh_own':      {nm: v / V_total for nm, v in own.items()},
            'sh_cross':    {k:  v / V_total for k, v in cross.items()},
            'sh_resid':    V_resid / V_total,
            'R2':          V_explained / V_total,
        }

    # Run the decomposition for each target of interest
    shockdec = {
        'Cw'    : _shockdec(Δcw, sm),   # wife private consumption
        'Cm'    : _shockdec(Δcm, sm),   # husband private consumption
        'Cpriv' : _shockdec(Δcp, sm),   # within-couple private (cw+cm)
        'Ctot'  : _shockdec(ΔC,  sm),   # total (private + public)
        'sw'    : _shockdec(Δs,  sm),   # wife's private share
        'sm'    : _shockdec(Δs1, sm),   # husband's private share
    }



    ####################################################    
    # Decomposition of hh private consumption growth
    ##################################################
    
    # Which shock should I consider?
    if    (shock_type=='permanent') & (shock_gender=='Male'): SHOCK = Δzm;  CONTROLS = (Δzw,ϵm_c,ϵw_c)
    elif  (shock_type=='permanent') & (shock_gender!='Male'): SHOCK = Δzw;  CONTROLS = (Δzm,ϵm_c,ϵw_c)
    elif  (shock_type!='permanent') & (shock_gender=='Male'): SHOCK = ϵm_c; CONTROLS = (Δzw,Δzm,ϵw_c)
    elif  (shock_type!='permanent') & (shock_gender!='Male'): SHOCK = ϵw_c; CONTROLS = (Δzw,ϵm_c,Δzm)
       
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
    
    # Share of earnings going to husband  and WLP
    sYm=np.mean(m.sim.incmg[sample][sm]/(m.sim.incmg[sample][sm]+m.sim.incwg[sample][sm]))
    wpart=(m.sim.WLP[sample][sm]>0).mean()
    
    #Changes in women earnings reltive to household income
    ΔsYw=np.mean((m.sim.incwg[sample1][sw]-m.sim.incwg[sample][sw])/(m.sim.incmg[sample][sw]+m.sim.incwg[sample][sw]))
    
    if shock_gender=='Male':
        
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
           
        
    else:
        
        K1=κywp
        K3=1-sYm
        temp = (sYm*κymp)/((1-sYm)*κywp)
        K2=κyp/(K1*K3)#κyp/(K1*K3)-temp
        
        K4=1#1+temp/K2
        K5=κynug
        K6=ols(SHOCK,ΔC,sm,cov=CONTROLS,take=1)/(κyp*K5)
        
        # Finally the decomposition of household insurance   
        Active_insurance =  1-K1*K2
        Passive_insurance = 1-K1*K2*K3*K4-(1-K1*K2)
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
    
  

    return {'indc':indc,'w_sh':w_sh,'totc':totc,'dins':dins,'BPP_MPC':BPP_MPC,'BPP_PER':BPP_PER,
            'BPP_MPC_net':BPP_MPC_net,'BPP_PER_net':BPP_PER_net,'wlp':wlp,'level':level,'ins_dec':ins_dec,
            'vardec_w':vardec_wife,'vardec_m':vardec_husband,
            'shockdec':shockdec}