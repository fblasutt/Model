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
    _Zp    = np.stack([np.ones_like(_aget),_aget,_aget**2],axis=-1)
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
    DH=40*np.array([(m.par.grid_wlp[m.sim.WLP][sample1])],dtype=np.float64)[0]-40*np.array([(m.par.grid_wlp[m.sim.WLP][sample])],dtype=np.float64)[0]
    #Love shock changes
    lovw,lovm=np.zeros((2,m.par.simN,m.par.T))
    
    # grid_lovew/grid_lovem are FLAT joint-index arrays (length num_love =
    # num_lovew*num_lovem, built by repeat/tile in setup_grids), so they are
    # indexed with the joint love state directly.
    for i in range(par.T):lovw[:,i]=par.grid_lovew[i][m.sim.love[:,i]]
    for i in range(par.T):lovm[:,i]=par.grid_lovem[i][m.sim.love[:,i]]
    
    # BPP first stage on love panels, then difference
    lovw=_residualize(lovw); lovm=_residualize(lovm)
    Δlovw=lovw[sample1]-lovw[sample]
    Δlovm=lovm[sample1]-lovm[sample]

    # Human-capital depreciation shock (wife only; XHm=0 in labor_income).
    # grid_h[ih] is the ADDITIVE log-earnings component of the h state — it is
    # NOT part of grid_pw (labor_income returns XPw = RW component only), so
    # without this series depreciation events would load on the shockdec
    # residual. Its first difference is the one-time depreciation event (-mu).
    hw=np.zeros((par.simN,par.T))
    for t in range(par.T): hw[:,t]=par.grid_h[m.sim.ih[:,t]]
    hw=_residualize(hw)
    Δhw=hw[sample1]-hw[sample]

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
    wlp= {'all_m':ols(ΔYm,DH,(sm) & (age[sample]<40)),
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
    # 8 pass-throughs, encoded as: SHAPE = shock type (circle persistent,
    # triangle transitory), FILL = shocked earner (blue husband, red wife),
    # LABEL = whose consumption responds. Points on the 45-degree line mean
    # the BPP estimator recovers the true model pass-through.
    _blue, _red = '#1f77b4', '#d62728'
    # last tuple element: label offset in display points (NE for the husband's
    # persistent shocks; SE for the wife's persistent and ALL transitory ones)
    _pts = [
        (BPP_PER['ym_cm'], indc['per_m_m'], 'o', _blue, r'$y^m\!\rightarrow\!c^m$', ( 8,   4)),
        (BPP_PER['ym_cw'], indc['per_m_w'], 'o', _blue, r'$y^m\!\rightarrow\!c^w$', ( 8,   4)),
        (BPP_PER['yw_cm'], indc['per_w_m'], 'o', _red,  r'$y^w\!\rightarrow\!c^m$', ( 8, -11)),
        (BPP_PER['yw_cw'], indc['per_w_w'], 'o', _red,  r'$y^w\!\rightarrow\!c^w$', ( 8, -11)),
        (BPP_MPC['ym_cm'], indc['tra_m_m'], '^', _blue, r'$y^m\!\rightarrow\!c^m$', ( 8, -11)),
        (BPP_MPC['ym_cw'], indc['tra_m_w'], '^', _blue, r'$y^m\!\rightarrow\!c^w$', ( 8, -11)),
        (BPP_MPC['yw_cm'], indc['tra_w_m'], '^', _red,  r'$y^w\!\rightarrow\!c^m$', ( 8, -11)),
        (BPP_MPC['yw_cw'], indc['tra_w_w'], '^', _red,  r'$y^w\!\rightarrow\!c^w$', ( 8, -11)),
    ]
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    _lo = min(min(x for x, *_ in _pts), min(y for _, y, *_ in _pts)) - 0.05
    _hi = 0.8   # axes extended up to 0.8
    ax.plot([_lo, _hi], [_lo, _hi], color='0.35', ls='--', lw=1.2, zorder=1)
    for _x, _y, _mk, _cl, _lab, _off in _pts:
        ax.scatter(_x, _y, marker=_mk, s=95, facecolor=_cl,
                   edgecolor='black', linewidth=0.7, zorder=3)
        ax.annotate(_lab, (_x, _y), textcoords='offset points',
                    xytext=_off, fontsize=9, zorder=4)
    from matplotlib.lines import Line2D
    _handles = [
        Line2D([], [], marker='o', ls='', mfc='0.75', mec='black', ms=9, label='Persistent shock'),
        Line2D([], [], marker='^', ls='', mfc='0.75', mec='black', ms=9, label='Transitory shock'),
        Line2D([], [], marker='s', ls='', mfc=_blue,  mec='black', ms=9, label="Husband's earnings shock"),
        Line2D([], [], marker='s', ls='', mfc=_red,   mec='black', ms=9, label="Wife's earnings shock"),
    ]
    ax.legend(handles=_handles, loc='upper left', frameon=False, fontsize=9)
    ax.set_xlim(_lo, _hi); ax.set_ylim(_lo, _hi); ax.set_aspect('equal')
    ax.set_xlabel('BPP-estimated pass-through')
    ax.set_ylabel('True pass-through (model)')
    ax.grid(True, linewidth=0.4, alpha=0.35); ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(root+'/Output files/model/BPP_true.eps', format='eps', bbox_inches='tight')
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
                   ('psi_w',  Δlovw), ('psi_m', Δlovm),
                   ('dep_w',  Δhw)]   # wife's human-capital depreciation event

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
        'Y'     : _shockdec(ΔY,     sm),   # gross household earnings
        'Ynet'  : _shockdec(ΔY_net, sm),   # net (after-tax) household earnings
        'Cw'    : _shockdec(Δcw, sm),   # wife private consumption
        'Cm'    : _shockdec(Δcm, sm),   # husband private consumption
        'Cpriv' : _shockdec(Δcp, sm),   # within-couple private (cw+cm)
        'Ctot'  : _shockdec(ΔC,  sm),   # total (private + public)
        'sw'    : _shockdec(Δs,  sm),   # wife's private share
        'sm'    : _shockdec(Δs1, sm),   # husband's private share
    }


    ##########################################################
    # Potential household income (top of the volatility ladder)
    ##########################################################
    # grid_zw = exp(trend + z + eps + h): the wife's potential-earnings grid
    # INCLUDES the human-capital component, so potential income risk includes
    # DEPRECIATION risk. resources_couple sets incwg = grid_zw*grid_wlp[wlp]
    # pre-retirement, so for a working wife potential (at working hours) and
    # actual earnings coincide EXACTLY — the wedges below are then pure
    # composition / participation-change effects.
    Yw_pot=np.zeros((par.simN,par.T))
    for t in range(par.T):
        fac=par.grid_wlp[par.num_wlp-1] if t<par.Tr else 1.0
        for i in range(par.simN):
            Yw_pot[i,t]=par.grid_zw[t,ID[i,t],m.sim.iz[i,t],m.sim.ih[i,t]]*fac

    lypot=_residualize(np.log(m.sim.incmg+Yw_pot))
    ΔY_pot=lypot[sample1]-lypot[sample]     # potential household income growth

    # Fixed-participation income: the wife's earnings enter with her
    # PREVIOUS-period participation status (post-retirement pension income is
    # always included, matching the actual income definition). At the base
    # period of each growth cell this panel equals actual household income,
    # so ΔY_fp = log Y^{fp}_{t+1} − log Y_t holds participation fixed at its
    # base-period status. (Base value from the actual-income panel: the
    # residualization-profile difference between the two panels is second
    # order in growth.)
    Plag=np.zeros((par.simN,par.T))
    Plag[:,1:]=(m.sim.WLP>0)[:,:-1]
    Plag[:,0] =(m.sim.WLP>0)[:,0]
    Plag[:,par.Tr:]=1.0
    lyfp=_residualize(np.log(m.sim.incmg+Plag*Yw_pot))
    ΔY_fp=lyfp[sample1]-ly[sample]


    ##########################################################
    # Volatility ladder: consumption volatility by INSURANCE CHANNEL
    ##########################################################
    # Walk the variance of growth down the budget flow, ALL rungs exact sample
    # variances (of genuine log-of-sum panels, no share-weighting) on
    # IDENTICAL cells:
    #   V_pot  = Var(ΔY_pot)  potential household income (wife at working
    #                          hours regardless of status; incl. depreciation)
    #   V_fp   = Var(ΔY_fp)   earnings at base-period participation
    #   V_Y    = Var(ΔY)      actual gross household earnings growth
    #   V_Ynet = Var(ΔY_net)  after taxes
    #   V_C    = Var(ΔC)      after self-insurance (total consumption)
    #   V_cp   = Var(Δcp)     after the private/public expenditure split
    #   V_cg   = Var(Δc^g)    after intra-household allocation, per spouse
    # Contributions telescope EXACTLY:
    #   compos = V_pot−V_fp (non-participation: single-earner couples lose
    #            income diversification, so this is typically NEGATIVE),
    #   partchg = V_fp−V_Y (participation entries/exits: added-worker
    #            responses absorb, churn adds),
    #   taxes = V_Y−V_Ynet, savings = V_Ynet−V_C, pubpriv = V_C−V_cp,
    #   alloc_g = V_cp−V_cg, and V_pot − Σ contributions = V_cg.
    # Each step equals −[Var(new wedge) + 2Cov(upstream flow, new wedge)]: a
    # smoothing channel is positive through its negative covariance with
    # upstream risk; NEGATIVE entries are informative (channel adds
    # volatility, e.g. renegotiation for the disfavored spouse, or love-shock
    # risk arriving mid-ladder in the savings rung).

    def _vol_ladder(cond):
        # identical cells at every rung: require all pieces finite
        ok = cond & np.isfinite(ΔY_pot) & np.isfinite(ΔY_fp) & np.isfinite(ΔY) \
                  & np.isfinite(ΔY_net) & np.isfinite(ΔC) \
                  & np.isfinite(Δcp) & np.isfinite(Δcm) & np.isfinite(Δcw)
        V = lambda x: float(np.var(x[ok], ddof=1))
        V_pot, V_fp = V(ΔY_pot), V(ΔY_fp)
        V_Y, V_Ynet, V_C = V(ΔY), V(ΔY_net), V(ΔC)
        V_cp, V_cm, V_cw = V(Δcp), V(Δcm), V(Δcw)
        out = {
            'V_pot': V_pot, 'V_fp': V_fp,
            'V_Y': V_Y, 'V_Ynet': V_Ynet, 'V_C': V_C,
            'V_cp': V_cp, 'V_cm': V_cm, 'V_cw': V_cw,
            'compos':  V_pot  - V_fp,     # non-participation (risk concentration)
            'partchg': V_fp   - V_Y,      # participation changes (entries/exits)
            'taxes':   V_Y    - V_Ynet,   # progressive taxation
            'savings': V_Ynet - V_C,      # self-insurance (net income -> consumption)
            'pubpriv': V_C    - V_cp,     # private/public expenditure shift (C -> cw+cm)
            'alloc_m': V_cp   - V_cm,     # intra-HH (bargaining) allocation, husband
            'alloc_w': V_cp   - V_cw,     # intra-HH (bargaining) allocation, wife
            'n_cells': int(ok.sum()),
        }
        # Exact telescoping check
        chain = out['compos']+out['partchg']+out['taxes']+out['savings']+out['pubpriv']
        out['resid_m'] = V_pot - (chain+out['alloc_m']) - V_cm
        out['resid_w'] = V_pot - (chain+out['alloc_w']) - V_cw
        return out

    vol_ladder = _vol_ladder(sm)



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

        # Wu-Krueger appendix A, FEMALE-shock chain: K1 = her intensive
        # margin, K2 = her extensive margin (+ interactions, residual),
        # K3 = her earnings share (composition), K4 = male intensive
        # response. The share dilution is applied BEFORE the spouse step, so
        # the active increment is measured on the diluted chain (the old
        # ordering measured it pre-dilution, mechanically giving negative
        # active and >100% passive insurance).
        K1=κywp                                   # her intensive margin
        K3=1-sYm                                  # composition: her earnings share
        # residual BEFORE the spouse step: her extensive margin + interactions
        # (κyp - sYm*κymp = the part of the household response due to HER earnings)
        K2=(κyp-sYm*κymp)/(K1*K3)
        K4=κyp/(K1*K2*K3)                         # male intensive response (spousal insurance)
        K5=κynug
        K6=ols(SHOCK,ΔC,sm,cov=CONTROLS,take=1)/(κyp*K5)

        # Same grouping as the male branch: Passive = own responses +
        # dilution, Active = the SPOUSE's labor-supply increment
        Passive_insurance = 1-K1*K2*K3
        Active_insurance  = 1-K1*K2*K3*K4-(1-K1*K2*K3)
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
            'shockdec':shockdec,'vol_ladder':vol_ladder}

def vol_stack_figure(models, samples, labels, name_file, mode='3seg'):
    """
    Stacked-bar figure of private-consumption volatility for a list of model
    versions (e.g. the policy variants of an experiment). Bars are GROUPED BY
    SPOUSE: all variants' wife bars first, then all variants' husband bars.
    Each bar decomposes the RAW variance of one-year log-consumption growth
    (couples present in t and t+1, x100), based on the exact identity
    Var(dlog c^g) = Var(dlog C_priv) + Var(dlog s^g) + 2Cov, with `mode`:

      '3seg'    exact 3-segment stack: V_s (blue) starts at zero, V_C stacks
                on top, 2Cov stacks above when positive and HANGS BELOW ZERO
                when negative; a black OUTLINED RECTANGLE from 0 to the NET
                total marks Var(dlog c^g)
      'fold'    2 segments, 2Cov folded into the household part:
                [V_C + 2Cov | V_s]  (exact adding-up preserved)
      'rescale' 2 segments rescaled to the total:
                [V_cg*V_C/(V_C+V_s) | V_cg*V_s/(V_C+V_s)]

    Saved to root+'/Output files/model/'+name_file+'.eps'.
    """
    col_C, col_s, col_cov = '#fdae6b', '#6baed6', '#bdbdbd'

    def _raw_vars(m, sample):
        s1 = np.roll(sample, 1, axis=1)
        smm = (m.sim.couple[s1] == 1) & (m.sim.couple[sample] == 1)
        def V(x):
            dx = np.log(x[s1]/x[sample])[smm]
            return dx.var(ddof=1)
        cp = m.sim.Cw+m.sim.Cm
        return {'V_C': V(cp),
                'w': (V(m.sim.Cw), V(m.sim.Cw/cp)),
                'm': (V(m.sim.Cm), V(m.sim.Cm/cp))}

    lab_C  = r'Var$(\Delta\log C_t)$'
    lab_Cf = r'Var$(\Delta\log C_t)+2\,$Cov'
    lab_s  = r'Var$(\Delta\log s^g_t)$'
    lab_cv = r'$2\,$Cov'
    lab_T  = r'Var$(\Delta\log c^g_t)$'

    D = [_raw_vars(m, sample) for m, sample in zip(models, samples)]
    n = len(models)
    fig, ax = plt.subplots(figsize=(0.9+0.65*2*n, 3.6))
    seen = set()
    xs_all, ticklabs = [], []
    for j, g in enumerate(('w', 'm')):           # wife group first, then husband
        for i, (d, lab) in enumerate(zip(D, labels)):
            x = j*(n+0.8) + i*1.0
            V_cg, V_s = d[g]
            V_C = d['V_C']
            cov2 = V_cg - V_C - V_s
            if   mode == '3seg':
                segs = ((lab_s, V_s, col_s), (lab_C, V_C, col_C), (lab_cv, cov2, col_cov))
            elif mode == 'fold':
                segs = ((lab_s, V_s, col_s), (lab_Cf, V_C + cov2, col_C))
            elif mode == 'rescale':
                tot = V_C + V_s
                segs = ((lab_s, V_cg*V_s/tot, col_s), (lab_C, V_cg*V_C/tot, col_C))
            else:
                raise ValueError(f"Unknown mode: {mode!r}")
            cum = 0.0
            for slab, val, col in segs:
                if mode == '3seg' and slab == lab_cv and val < 0.0:
                    # negative covariance hangs below zero instead of being
                    # buried inside the stack
                    ax.bar(x, 100*val, bottom=0.0, width=0.8, color=col,
                           edgecolor='white', linewidth=0.8, zorder=2)
                else:
                    ax.bar(x, 100*val, bottom=100*cum, width=0.8, color=col,
                           edgecolor='white', linewidth=0.8, zorder=2)
                    cum += val
                seen.add(slab)
            if mode == '3seg':
                # outlined rectangle from 0 to the NET total Var(dlog c^g)
                from matplotlib.patches import Rectangle
                ax.add_patch(Rectangle((x-0.4, 0.0), 0.8, 100*V_cg,
                                       fill=False, edgecolor='black',
                                       linewidth=1.4, zorder=3))
            xs_all.append(x); ticklabs.append(lab)
        ax.text(j*(n+0.8) + (n-1)/2, -0.30,
                'Wife ($c^w$)' if g == 'w' else 'Husband ($c^m$)',
                ha='center', va='top', fontsize=12,
                transform=ax.get_xaxis_transform())
    ax.set_xticks(xs_all)
    ax.set_xticklabels(ticklabs, fontsize=10, rotation=30, ha='right')
    ax.tick_params(axis='y', labelsize=10)
    ax.set_ylabel(r'Volatility ($\times 100$)', fontsize=12)
    ax.set_ylim(-0.2, 0.9)                   # common scale across experiments
    ax.axhline(0.0, color='0.4', linewidth=0.8)
    ax.grid(True, axis='y', linewidth=0.4, alpha=0.35); ax.set_axisbelow(True)
    # explicit legend (top left), one entry per volatility object with symbol
    from matplotlib.patches import Patch, Rectangle as _Rect
    handles = [Patch(facecolor=col_s, edgecolor='white', label=lab_s),
               Patch(facecolor=col_C, edgecolor='white',
                     label=lab_Cf if mode == 'fold' else lab_C)]
    if mode == '3seg':
        handles += [Patch(facecolor=col_cov, edgecolor='white', label=lab_cv),
                    _Rect((0, 0), 1, 1, fill=False, edgecolor='black',
                          linewidth=1.4, label=lab_T)]
    ax.legend(handles=handles, loc='upper left', frameon=False, fontsize=9.5,
              ncol=2, columnspacing=1.0, handlelength=1.4,
              handletextpad=0.5)             # two rows (2x2) for legibility
    fig.tight_layout()
    fig.savefig(root+'/Output files/model/'+name_file+'.eps',
                format='eps', bbox_inches='tight')
    plt.show()


def share_var_decomposition(models, samples, labels):
    """
    Decompose the (uncentered) variance of LOG share growth, per variant and
    spouse, into SIZE of share changes vs LEVEL at which they occur:
        E[(dlog s)^2] = freq * E[dS^2|ren] * E[M^2|ren] * R
    dS   = change of the OWN share in LEVELS (same magnitude for the two
           spouses, so freq and E[dS^2|ren] are common: the entire spouse
           asymmetry is in M and R)
    M    = dlog s / dS, the exact 'effective 1/share' at which the change is
           evaluated (mean-value form)
    ren  = cells with dS != 0; freq = their frequency
    R    = E[dS^2 M^2|ren] / (E[dS^2|ren] E[M^2|ren]), size-level interaction
    ren  = RENEGOTIATION cells (power != power_lag): off those cells the share
           still moves microscopically (interpolation of the intra-period
           allocation over the power/resources grids), so the identity holds
           for the renegotiation PART of E[y^2]; the residual non-reneg noise
           share is reported in the last column ('nonren%', should be tiny).
    Factors are log-additive across variants.
    """
    print()
    print('=== Share-growth variance: size vs level decomposition ===')
    print('    (tow.w / tow.m: renegotiation frequency by DIRECTION -- '
          'power toward the wife / toward the husband)')
    print(f"{'variant':18s} {'sp':>3s} {'E[y2]x100':>10s} {'freq':>7s} "
          f"{'tow.w':>7s} {'tow.m':>7s} "
          f"{'E[dS2|r]x100':>13s} {'E[M2|r]':>9s} {'R':>7s} {'nonren%':>8s}")
    for m, sample, lab in zip(models, samples, labels):
        s1 = np.roll(sample, 1, axis=1)
        smm = np.roll(sample, -1, axis=1)[sample]
        cp = m.sim.Cw + m.sim.Cm
        pw_now, pw_lag = m.sim.power[s1][smm], m.sim.power_lag[s1][smm]
        ren = pw_now != pw_lag
        f_up = (ren & (pw_now > pw_lag)).mean()    # renegotiation toward the wife
        f_dn = (ren & (pw_now < pw_lag)).mean()    # renegotiation toward the husband
        for g in ('w', 'm'):
            sh = (m.sim.Cw/cp) if g == 'w' else (m.sim.Cm/cp)
            X = (sh[s1]-sh[sample])[smm]
            y = np.log(sh[s1]/sh[sample])[smm]
            freq = ren.mean()
            Ey2 = np.mean(y**2)
            nonren = np.mean((y**2)*(~ren))/Ey2 if Ey2 > 0 else 0.0
            r_ok = ren & (np.abs(X) > 1e-14)
            if not r_ok.any():
                print(f'{lab:18s} {g:>3s} {100*Ey2:10.4f} {freq:7.4f} '
                      f'{f_up:7.4f} {f_dn:7.4f}'
                      f'            -         -       - {100*nonren:8.2f}')
                continue
            EX2 = np.mean(X[r_ok]**2)
            M = y[r_ok]/X[r_ok]
            EM2 = np.mean(M**2)
            R = np.mean((X[r_ok]**2)*(M**2))/(EX2*EM2)
            # identity check on the renegotiation part of E[y^2]
            Ey2_ren = np.mean((y**2)*r_ok)
            gap = abs(r_ok.mean()*EX2*EM2*R - Ey2_ren)/max(Ey2_ren, 1e-16)
            flag = '' if gap < 1e-6 else f'  [identity gap {gap:.1e}]'
            print(f'{lab:18s} {g:>3s} {100*Ey2:10.4f} {freq:7.4f} '
                  f'{f_up:7.4f} {f_dn:7.4f} '
                  f'{100*EX2:13.5f} {EM2:9.2f} {R:7.3f} {100*nonren:8.2f}'+flag)
