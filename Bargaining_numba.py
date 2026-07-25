import numpy as np
from EconModel import EconModelClass, jit
from consav.grids import nonlinspace
from consav import linear_interp,upperenvelope
from numba import njit,prange,config
import UserFunctions_numba as usr
import setup
from quantecon.optimize.root_finding import bisect 


# Store upper envelope algorithm for singles and couples
upper_envelope=usr.create(usr.couple_time_utility)
upp_env_single = upperenvelope.create(usr.single_time_util)

# General configuratiion and glabal variables (common across files)
config.DISABLE_JIT = setup.nojit;parallel=setup.parallel;cache=setup.cache
woman=setup.woman;man=setup.man

class HouseholdModelClass(EconModelClass):
    
    def settings(self):self.namespaces = []#needed to make class work, otherwise useless
                   
    def setup(self):
        par = self.par
        
        ####################
        # Parameters below
        #####################
        
        par.full=False #dummy for full/limited commitment. if full commitment: mutual consent divorce
        par.pens_reform=False #dummy telling whether the pension reform is in place or not
       
        
        # Demographics
        par.T = 65 # periods; t=0 is AGE 25, so last period is age 89. terminal age: https://www.mortality.org/File/GetDocument/hmd.v6/JPN/STATS/fltper_1x1.txt 
        par.Tr = 40 # retirement period: t=40 is age 65
         
        # Prices
        par.R = 1.0121#1.035#1+ interest rate Source:FM.M.JP.JPY.4F.BB.R_JP10YT_RR.YLDA
               
        # Preferences
        par.β = 1.0   # Discount factor
        par.ρ = 1.5     # Risk avresion private goods
        par.χ = 2.0    # Risk aversion home goods
        par.α = 0.35    # Weight on home good
        par.wedge=0.0#35   #Single-couple utility wedge
        par.Ω=0.0
        
        # Income processes: tren and shocks (sd of persistent (σpi), transitory (σϵi), initial (σ0i) income shocks)
        # Trends rebased to an AGE-25 origin: original age-20-based estimates were
        # ι0+ι1·s+ι2·s²; substituting s=t+5 gives the coefficients below, so the
        # profile at model age t equals the old profile at t+5.
        par.ι0m= 0.0;       par.ι1m=.0497715; par.ι2m=-.0010579 #trend for husband
        par.ι0w=  -.2895476;     par.ι1w=.0497715; par.ι2w=-.0010579 #trend for wife
 
        # par.ι0m= -0.224+5*0.046+25*(-0.00075858); par.ι1m=0.046+10*(-0.00075858); par.ι2m=-0.00075858 #trend for husband
        # par.ι0w= -0.591+5*0.046+25*(-0.00075858); par.ι1w=0.046+10*(-0.00075858); par.ι2w=-0.00075858 #trend for wife


        # σ0 absorbs 5 periods of persistent innovations (age-25 start): the
        # cross-sectional dispersion at t=0 now equals what the age-20 model
        # implied at age 25, so income grids/transitions match the old t+5 ones.
        par.σzm=.0102609**0.5  ;par.σϵm=.0100159 **0.5;par.σ0m= 0.15**0.5; #shock size husband
        par.σzw=.0251831**0.5;  par.σϵw=.01069**0.5 ;par.σ0w= 0.15**0.5; #shock size wife
        
        # par.σzm=0.0082**0.5  ;par.σϵm= 0.0125**0.5;par.σ0m=  (0.0338+5*0.0082)**0.5; #shock size husband
        # par.σzw= 0.00978**0.5;par.σϵw=0.0137**0.5 ;par.σ0w= (0.1198+5*0.00978)**0.5; #shock size wife
        
        par.σϵwm=0.0#0.00289 #correlation of transitory shocks
        
        # Pension parameters
        par.p_b=0.23082 #basic pension
        par.κ=0.219    #proportional part of pension
        
        # Depreciation of human capital
        par.μ = 0.85599 #1.195/5        # Human capital depreciation drift
        par.p_μ = 1/20.0#5.0/40.0   # Probability that  human capital depreciates
        
        # Home good production:  Q = (θ·home_time^ν + (1-θ)·x^ν)^(1/ν)
        par.ν = 0.5#-0.6   # CES substitution parameter in home production
        par.θ = 0.08 # Weight on home_time (vs. money) in home production
        par.η = 0.08   # Labor disutility from working
        par.ϕ = 0.326  # Time spend on public goods by singles
        par.px = 1.42# Price of durables for couples (equivalence scale on d_pub)
        
        # Taxes
        par.Λ=0.8471                              # 1- tax level
        par.τ=0.09188                              # Tax progressivity        
        par.d0=0.1697;par.d1=0.01349;par.d2=-0.55 # Husband deduction parameters
                
        # Post-divorce transfers
        par.alimony=0.0       #Alimony used for experiments
        par.div_A_share = 0.5 # Asset share to wife at divorce
        
        # Meeting probability
        par.meet = 0.0#probability of meeting a partner if single
        
        ##########################################
        # Grid-related parameters and state space
        ##########################################
        
        #Divorce period and grid
        par.num_perdiv = 40 # one divorce period per working year (t=0 is age 25 -> Tr=40). MUST divide Tr
        par.Dper = int(par.Tr/par.num_perdiv)
        assert par.num_perdiv<=par.Tr and par.Tr%par.num_perdiv==0, 'num_perdiv must divide Tr (Dper=Tr/num_perdiv would be 0 or leave years unmapped)'
        
        # Wealth
        par.num_A = 15;par.max_A = 75.0
        
        # Bargaining power
        par.num_power = 11
        par.power_min=1e-3;par.power_max=1.0-par.power_min
        
        # Women's human capital states
        par.num_h = 1
        par.num_h = 2

        # love/match quality
        par.num_lovew = 3;par.num_lovem = 3;
        par.num_love=par.num_lovem*4
        par.σL = 0.1; par.σL0 = 0.00001
        
        # productivity of men and women: gridpoints
        par.num_ϵw=2;par.num_ϵm=2#transitory
        par.num_pw=3;par.num_pm=3#persistent
        par.num_zw=par.num_pw*par.num_ϵw;par.num_zm=par.num_pm*par.num_ϵw#total by gender
        par.num_z=par.num_zm*par.num_zw#total, couple
        
        # pre-computation of consumption Ctot=cf+cm+d, used in the intra-period maximization
        par.num_Ctot = 150;par.max_Ctot = par.max_A*2

        
        ##########################################
        # Simulations parameters
        ##########################################
        par.seed = 9211;par.simT = par.T;par.simN = 10_000
              
        # Simulations Grids
        par.women =np.ones(par.simN)#0: simumate men, 1 women
        par.sample_init=np.zeros(par.simN,dtype=np.int32)#in which period do we start simulating the sample?
        par.policy_init=np.zeros(par.simN,dtype=np.int32)#when does the pension policy (if pens_reform=True) kicks in?
      
        
    def setup_grids(self):
        par = self.par
    
        


        #Grid for the pr. of meeting a partner in each t
        par.λ_grid = np.ones(par.T)*par.meet
        for t in range(par.Tr,par.T):par.λ_grid[t]=0.0
        
        # Assets grid. Single grids are such to avoid interpolation
        par.grid_A = np.append(nonlinspace(0.0,par.max_A,par.num_A-1,2.1),par.max_A*10)
        par.grid_Aw =  par.grid_A * par.div_A_share; par.grid_Am =  par.grid_A*(1.0-par.div_A_share)

        # Women's labor supply grids
        par.grid_wlp=np.array([0.0,0.698])#np.array([0.0,0.823])
        par.num_wlp=len(par.grid_wlp)
        
        # Match quality shock grid and transition matrices  
        # Love grids pre-widened by the 5 unmodeled years (age-25 start): initial sd
        # = sqrt(σL0²+5σL²) makes grids/transitions at t equal the old age-20 model's
        # at t+5 exactly (love still starts at 0 = middle grid point at marriage).
        par.grid_love_,par.Πl_,par.Πl0_= usr.rouw_nonst(par.T,par.σL,par.σL0,par.num_lovew)
        
  
        disagw=np.array([-par.Ω,-par.Ω, par.Ω,par.Ω])
        disagm=np.array([-par.Ω, par.Ω,-par.Ω,par.Ω])
        par.trans_love=np.array([[0.25,0.25,0.25,0.25],[0.25,0.25,0.25,0.25],[0.25,0.25,0.25,0.25],[0.25,0.25,0.25,0.25]])
        
        par.grid_lovew=[(par.grid_love_[t][:,None]+disagw[None,:]).ravel() for t in range(par.T)]
        par.grid_lovem=[(par.grid_love_[t][:,None]+disagm[None,:]).ravel() for t in range(par.T)]
        
        par.Πl=[np.kron(par.Πl_[t],par.trans_love)  for t in range(par.T-1)]
        par.Πl0=[np.kron(par.Πl0_[t],par.trans_love)  for t in range(par.T-1)]
 
        
        # Bargaining power grid. non-linear grid with more mass in both tails.        
        par.grid_power = usr.grid_fat_tails(par.power_min,par.power_max,par.num_power)
        
        # Women's human capital grid plus transition matrices for working full time (Πh_pt) and not working (Πh_pt)       
        par.grid_h = np.flip(np.linspace(-par.num_h*par.μ,0.0,par.num_h)) if par.num_h>1 else np.zeros(par.num_h)#0 position is best

        Πh_pt = np.eye(par.num_h) if par.num_h>1 else np.ones(par.num_h) # Perfect transition if full-time participation...
        Πh_nt = np.array([[1-par.p_μ, par.p_μ], [0, 1]]).T  if par.num_h>1 else np.ones(par.num_h)  # ...depreciation otherwise
        # Here a fix that works with more than 2 human capital states, TOCHECK
        # OLF transition: from state i stay with prob 1-p_μ, drop one step (i -> i+1) with prob p_μ; bottom rung absorbing.                                                                                                           
        # Built in [from,to] convention then transposed to the [to,from] convention used downstream.               
        if par.num_h>1:                                                                                            
            Πh_nt_ft = (1.0-par.p_μ)*np.eye(par.num_h)                                                             
            for _i in range(par.num_h-1):                                                                          
                Πh_nt_ft[_i, _i+1] = par.p_μ                                                                       
            Πh_nt_ft[-1, -1] = 1.0                                                                                 
            Πh_nt = Πh_nt_ft.T                                                                                     
        else:                                                                                                      
            Πh_nt = np.ones(par.num_h)  # ...depreciation otherwise       

        par.Πh_t = np.array([w*Πh_pt + (1-w)*Πh_nt for w in par.grid_wlp])  if par.num_h>1 else np.array([np.eye(par.num_h) for w in par.grid_wlp])# Work-hour weighted transitions
        identity_block = np.tile(np.eye(par.num_h), (par.num_wlp, 1, 1))   #stops depreciating at retirement
        par.Πh = [par.Πh_t if t < par.Tr else identity_block for t in range(par.T)]
        
        # Grid of total consumption, used in the intra-period problem
        par.grid_Ctot = nonlinspace(1.0e-3,par.max_Ctot,par.num_Ctot,1.4)
        
        # % of shared pension in case of policy change, for given divorce period
        par.PW=usr.pension_share(par)
 
        # Income shocks grids: couples
        par.grid_zw,par.grid_ϵw,par.grid_pw,par.Π_zw0, \
            par.grid_zm,par.grid_ϵm,par.grid_pm,par.Π_zm0, \
                                        par.Π=usr.labor_income(par)                                       
                                        
        # Income shocks grids: singles
        par.grid_zws,par.grid_ϵw,par.grid_pw,par.Π_zw0, \
            par.grid_zms,par.grid_ϵm,par.grid_pm,par.Π_zm0, \
                                                par.Πs=usr.labor_income(par,single=True,pens_reform=par.pens_reform) 
                                                


        
    def allocate(self):
        par = self.par;sol = self.sol;sim = self.sim;self.setup_grids()

        # setup grids
        par.simT = par.T
        
        # Intra period problem: given total consumption, how much public and private expenditures
        shape_pre = (2,par.num_wlp,par.num_power,par.num_Ctot)#first dimension if for not retired/retired
        sol.pre_Cw_priv = np.nan + np.ones(shape_pre)   #wife private cf
        sol.pre_Cm_priv = np.nan + np.ones(shape_pre)   #husband private cm
        sol.pre_d_pub = np.nan + np.ones(shape_pre)     #public consumption d
           
        # Marginal utility arrays to be used in the EGM, filled in the intra-period problem 
        par.grid_inv_marg_u = np.flip(par.grid_Ctot)
        par.grid_marg_u = np.nan + np.ones(shape_pre)         # couples
        par.grid_marg_uw = np.nan + np.ones(shape_pre)        # couples
        par.grid_marg_um = np.nan + np.ones(shape_pre)        # couples
        par.grid_marg_u_for_inv = np.nan + np.ones(shape_pre) # couples
        par.grid_cpriv_s =  np.nan + np.ones((par.num_Ctot,par.num_wlp,2))# singles
        par.grid_marg_u_s = np.nan + np.ones((par.num_Ctot,par.num_wlp,2))# singles
         
        # Singles: value functions (vf), consumption, marg util
        shape_single = (par.T,par.num_perdiv,par.num_h,par.num_z,par.num_A)
        sol.Vw_single = np.nan + np.ones(shape_single)     # vw in t
        sol.Vm_single = np.nan + np.ones(shape_single)     # vm in t
        sol.Cw_tot_single = np.nan + np.ones(shape_single) # = cw+d
        sol.Cm_tot_single = np.nan + np.ones(shape_single) # = cm+d
        sol.wlp_s_w = np.ones(shape_single,dtype=np.int32)-1
        
        sol.wlp_s_m = np.ones(shape_single,dtype=np.int32)-1

        # Couples: value functions (vf), consumption, marg util, bargaining power
        shape_couple = (par.T,par.num_h,par.num_z,par.num_power,par.num_love,par.num_A)                 
        shape_couple_wls = (par.T,par.num_wlp,par.num_h,par.num_z,par.num_power,par.num_love,par.num_A) 
        
        sol.Vw_couple = np.nan + np.ones(shape_couple)                # vw in t
        sol.Vm_couple = np.nan + np.ones(shape_couple)                # vm in t
        sol.Vw_remain_couple = np.nan + np.ones(shape_couple)         # vw|couple
        sol.Vm_remain_couple = np.nan + np.ones(shape_couple)         # vm|couple
        sol.i_Vw_remain_couple = np.nan + np.ones(shape_couple_wls)   # vw|couple|w lab supp.
        sol.i_Vm_remain_couple = np.nan + np.ones(shape_couple_wls)   # vm|couple|w lab supp. 
        sol.i_C_tot_remain_couple = np.nan + np.ones(shape_couple_wls)# cons|couple
        sol.remain_WLP = np.ones(shape_couple_wls)                    # 0/1 indicator of chosen WLP
        sol.power =  np.nan +np.zeros(shape_couple)                   # barg power of wife θ

        # Simulation arrays
        shape_sim = (par.simN,par.simT)
        sim.C_tot = np.nan + np.ones(shape_sim)         # total consumption cw+cm+dw+dm
        sim.Cw_tot = np.nan + np.ones(shape_sim)        # total consumption cw+d
        sim.Cm_tot = np.nan + np.ones(shape_sim)        # total consumption cm+d
        
        sim.Cm = np.nan + np.ones(shape_sim)        # private consumption m
        sim.Cw = np.nan + np.ones(shape_sim)        # private consumption w
        sim.dm = np.nan + np.ones(shape_sim)        # public expenditure m
        sim.dw = np.nan + np.ones(shape_sim)        # public expenditure w
        
        sim.Vsw = np.nan + np.ones(shape_sim)       # value function if divorce w
        sim.Vsm = np.nan + np.ones(shape_sim)       # value function if divorce m
        sim.Vcw = np.nan + np.ones(shape_sim)       # before-ren value function w
        sim.Vcm = np.nan + np.ones(shape_sim)       # before-ren value function m
        
        sim.iz = np.ones(shape_sim,dtype=np.int32)   # index of income shocks 
        sim.ID = np.zeros(shape_sim,dtype=np.int32) #index period at divorce
        sim.A = np.zeros(shape_sim)                 # total assets (m+w)
        sim.Aw = np.zeros(shape_sim)                # w's assets
        sim.Am = np.zeros(shape_sim)                # m's assets
        sim.couple = np.ones(shape_sim,dtype=bool)        # In a couple? True/False
        sim.couple_lag = np.ones(shape_sim,dtype=bool)    # In a couple previous period? True/False
        sim.power = -100.0*np.ones(shape_sim)             # Bargaining power θ
        sim.power_lag = -100.0*np.ones(shape_sim)         # Bargaining power θ previous period
        sim.love = np.ones(shape_sim,dtype=np.int32)       # Match quality
        sim.incw = np.nan + np.ones(shape_sim)            # w's net income
        sim.incm = np.nan + np.ones(shape_sim)            # m's net income
        sim.incwg = np.nan + np.ones(shape_sim)           # w's gross income
        sim.incmg = np.nan + np.ones(shape_sim)           # m's gross income
        sim.WLP = np.ones(shape_sim,dtype=np.int32)        # w's labor supply index
        sim.ih = np.zeros(shape_sim,dtype=np.int32)         # w's human capital 
        sim.tax = np.zeros(shape_sim)                     # Taxes paid by the couple or divorces (sum w+m)

        # Shocks
        np.random.seed(par.seed)
        sim.shock_love = np.random.random_sample((par.simN,par.simT)) # Match quality
        sim.shock_iz=np.random.random_sample((par.simN,2))            # Initial labor income index 
        sim.shock_z=np.random.random_sample((par.simN,par.simT))      # Labor income shocks
        sim.shock_h=np.random.random_sample((par.simN,par.simT))      # Human capital draws

        # Initial distribution (this will be overwritten by user input)
        sim.init_ih = np.zeros(par.simN,dtype=np.int32)                  # Initial w's human capital
        sim.init_couple = np.ones(par.simN,dtype=bool)                  # State (couple=1/single=0)
        sim.init_power =  np.random.random_sample(par.simN)             # Barg power 
        sim.init_A =  np.zeros(par.simN)                               # Assets 
        sim.init_lovew = np.ones(par.simN,dtype=np.int32)*par.num_lovew//2#w's initial love 
        sim.init_lovem = np.ones(par.simN,dtype=np.int32)*par.num_lovem//2#m's initial love 
        sim.init_love = sim.init_lovew*par.num_lovem+sim.init_lovem          #initial love 
        sim.init_z  = np.zeros(par.simN,dtype=np.int32)                  # Initial income index

                       
    def solve(self):

        with jit(self) as model:#This allows passing sol and par to jiit functions  
            
            #Import parameters and arrays for solution
            par = model.par; sol = model.sol
            
            # precompute the optimal intra-temporal consumption allocation given total consumpotion
            solve_intraperiod(sol,par)
            
            # loop backwards and obtain policy functions
            for t in reversed(range(par.T)):
                
                # choose EGM or vhi method to solve the single's problem
                solve_single_egm(sol,par,t) 
                
                # solve the couple's problem (EGM vs. vfi done later)
                solve_couple(sol,par,t)
    
                     
    def simulate(self):
        
        with jit(self) as model:    
            
            #Import parameter, policy functions and simulations arrays
            par = model.par; sol = model.sol; sim = model.sim
            
            #Call routing performing the simulation
            simulate_lifecycle(sim,sol,par)
             
####################################################
# INTRAPERIOD OPTIMIZATION FOR SINGLES AND COUPLES #
####################################################
@njit(parallel=parallel)
def solve_intraperiod(sol,par):
        
    # unpack to help numba (horrible)
    d_pub,  Cw_priv, Cm_priv, grid_marg_u, grid_marg_u_for_inv, grid_marg_u_s, grid_cpriv_s, grid_marg_uw, grid_marg_um =\
        sol.pre_d_pub, sol.pre_Cw_priv, sol.pre_Cm_priv, par.grid_marg_u, par.grid_marg_u_for_inv, par.grid_marg_u_s,\
        par.grid_cpriv_s, par.grid_marg_uw, par.grid_marg_um
        
    pars=(par.ρ,par.χ,par.α,par.ν,par.θ,par.η,par.ϕ,par.wedge,par.px)
    ϵ = 1e-8# delta increase in xs to compute numerical deratives

    ################ Singles part #####################
    for i,C_tot in enumerate(par.grid_Ctot):
        for iwlp,wlp in enumerate(par.grid_wlp): 
            for g in range(2):
            
                if g==0:#women
                    home= 1-wlp
                    female=True
                else:#men
                    home = 1.0 if wlp==0 else 0.0
                    female=False
                    
                
                pars_sex=(par.ρ,par.χ,par.α,par.ν,par.θ,par.η,par.ϕ,par.wedge,par.px,0.0,0.0,home,female)
                
                # optimize to get util from total consumption(m<->C_tot)=private cons(c)+public cons(m-c)
                grid_cpriv_s[i,iwlp,g] = usr.optimizer(lambda c,m,p:-usr.util(c,m-c,*p),ϵ,C_tot-ϵ,args=(C_tot,pars_sex))[0]
                
                # numerical derivative of util wrt total consumption C_tot, using envelope thm
                share_priv=grid_cpriv_s[i,iwlp,g]/C_tot
                forward  = usr.util(share_priv*(C_tot+ϵ),(1.0-share_priv)*(C_tot+ϵ),*pars_sex)
                backward = usr.util(share_priv*(C_tot-ϵ),(1.0-share_priv)*(C_tot-ϵ),*pars_sex)
                grid_marg_u_s[i,iwlp,g] = (forward - backward)/(2*ϵ)
               
    for iP in prange(par.num_power):  
        for ret in range(2):
            for iwlp,wlp in enumerate(par.grid_wlp):  
                for i,C_tot in enumerate(par.grid_Ctot):  
                      
                     
                    # initialize bounds and bargaining power  
                    power=par.grid_power[iP]  
                    mult = power**(1/par.ρ)/(power**(1/par.ρ)+(1-power)**(1/par.ρ)) 
                     
                    parss=(par.ρ,par.χ,par.α,par.ν,par.θ,par.ϕ,par.wedge,par.px)
                   
                    home_time=2.0 if (ret==1) else 1-wlp

                    # Endpoint evaluations of the FOC. Under CES, Q is bounded
                    # away from 0 even at x=0 (positive home_time), so the
                    # FOC may have the same sign at both endpoints — meaning
                    # the optimum is a corner. Detect this and avoid calling
                    # bisect (which requires opposite signs).
                    fa = usr.couple_root(1e-12,       C_tot, power, *parss, home_time)
                    fb = usr.couple_root(C_tot-1e-12, C_tot, power, *parss, home_time)
                    if fa * fb < 0.0:
                        ress = bisect(usr.couple_root, 1e-12, C_tot-1e-12,
                                      args=(C_tot, power, *parss, home_time))[0]
                    elif fa > 0.0:
                        # FOC>0 throughout: more to private, corner at x → 0.
                        ress = 1e-12
                    else:
                        # FOC<0 throughout: more to public, corner at x → C_tot.
                        ress = C_tot - 1e-12
                    d_pub[ret,iwlp,iP,i]  = ress
                     
                    Cw_priv[ret,iwlp,iP,i] = (C_tot-d_pub[ret,iwlp,iP,i])*mult 
                    Cm_priv[ret,iwlp,iP,i] = (C_tot-d_pub[ret,iwlp,iP,i])*(1-mult) 
                    res = np.array([Cw_priv[ret,iwlp,iP,i],Cm_priv[ret,iwlp,iP,i]]) 
                    
     
                    # numerical derivative of util wrt total consumption C_tot, using envelope thm  
                    _,forw_w,forw_m = usr.couple_util(res/(C_tot)*(C_tot+ϵ),C_tot+ϵ,power,1.0-wlp,*pars)  
                    _,bakw_w,bakw_m = usr.couple_util(res/(C_tot)*(C_tot-ϵ),C_tot-ϵ,power,1.0-wlp,*pars)  
                    grid_marg_uw[ret,iwlp,iP,i] = (forw_w - bakw_w)/(2*ϵ);grid_marg_um[ret,iwlp,iP,i] = (forw_m - bakw_m)/(2*ϵ) 
                                       
                #Create grid of couple's marginal util and inverse marginal utility   
                grid_marg_u[ret,iwlp,iP,:] = power*grid_marg_uw[ret,iwlp,iP,:]+(1.0-power)*grid_marg_um[ret,iwlp,iP,:]  
                grid_marg_u_for_inv[ret,iwlp,iP,:]=np.flip(par.grid_marg_u[ret,iwlp,iP,:])   
                
  
#######################
# SOLUTIONS - SINGLES #
#######################



@njit(parallel=parallel)
def integrate_single(sol, par, t):
    """
    Compute the expected values of being single, 
    by integrating over all possible next-period states.

    Original version looped over all (iA, iz, ih, jz) and performed
    non-contiguous 1D dot products in the innermost loop.  
    This version restructures the computation to:
        - Remove the jz loop from the hot path.
        - Use contiguous arrays for BLAS-friendly GEMV operations, where 
        BLAS is Basic Linear Algebra Subprograms and GEMV stands for GEneral Matrix–Vector multiplication
        - Minimize creation of temporaries inside loops.
    """
    
    # A single continuation value per state (no labor-choice dimension): singles
    # of both genders work FULL TIME before retirement by assumption (the
    # non-working option is killed in loop_savings_singles), so the human-capital
    # transition is always the full-time/identity slot -- a single woman's HC
    # never depreciates, and a divorced man's expectation over the ex-wife's HC
    # (which enters his pension under the sharing reform) is frozen at its
    # current level. If the singles' labor choice is ever re-enabled, this
    # integration must be made iwlp-specific again (Hw = par.Πh[t][iwlp]).
    Ew_nomeet  = np.zeros((par.num_perdiv, par.num_h, par.num_z, par.num_A))
    Em_nomeet  = np.zeros((par.num_perdiv, par.num_h, par.num_z, par.num_A))

    # Single income transition (Fortran-order = contiguous columns) and the
    # labor-independent human-capital transition (full-time / identity slot).
    S = np.asfortranarray(par.Πs[t])              # (num_z, num_z)
    H = np.asfortranarray(par.Πh[t][-1, :, :])    # identity (full-time slot)

    # Parallelize across iA
    for iA in prange(par.num_A):
        for iD in range(par.num_perdiv):

            # Ensure fast row access for V* (C-order = contiguous rows)
            Vw = np.ascontiguousarray(sol.Vw_single[t+1,iD, :, :, iA])  # (num_h, num_z)
            Vm = np.ascontiguousarray(sol.Vm_single[t+1,iD, :, :, iA])  # (num_h, num_z)

            Ew_nomeet[iD, :, :, iA] = (Vw.T @ H).T @ S
            Em_nomeet[iD, :, :, iA] = (Vm.T @ H).T @ S

    return Ew_nomeet, Em_nomeet
    
@njit(parallel=parallel)
def solve_single_egm(sol,par,t):

    #Integrate to get continuation value unless if you are in the last period
    Ew,Em=np.zeros((2,par.num_perdiv,par.num_h,par.num_zw,par.num_A))
    if t<par.T-1:Ew,Em = integrate_single(sol,par,t) #if t<par.T-1 else
             
    #Pre-define outcomes (if update .sol directly, parallelization go crazy)
    ciw,cim,viw,vim,cwt,Ewt,cwp,cmt,Emt,cmp=np.ones((10,2,par.num_perdiv,par.num_h,par.num_z,par.num_A))
    
    
    #function to find optimal savings, called for both men and women below
    def loop_savings_singles(par,grid_Ai,ci,Ei,cit,Eit,cip,vi,women,divorce):
        
        
        g=0 if women else 1
        
        # iwlp=0 if t>=par.Tr else 0
        # home=1.0 if ret==1 else 0.0
        # home=1.0-par.grid_wlp[-1] if ((women) & (ret==0)) else 0.0
        
        
        
     
        
        for iz in range(par.num_z):
            for ih in range(par.num_h):
                for iD in range(par.num_perdiv):

                    if t==(par.T-1): 
                        
                        iwlp=0
                        home=1.0
                        
                        pars=(par.ρ,par.χ,par.α,par.ν,par.θ,par.η,par.ϕ,par.wedge,par.px,0.0,0.0,home,women)
                        
                        resi = par.R*grid_Ai+usr.income_single(par,t,iwlp,iD,ih,iz,grid_Ai,women)[0]
                        ci[iwlp,iD,ih,iz,:] = resi.copy() #consume all resources
                        linear_interp.interp_1d_vec(par.grid_Ctot,par.grid_cpriv_s[:,iwlp,g],ci[iwlp,iD,ih,iz,:],cip[iwlp,iD,ih,iz,:])#private cons
                        vi[iwlp,iD,ih,iz,:]=usr.util(cip[iwlp,iD,ih,iz,:],ci[iwlp,iD,ih,iz,:]-cip[iwlp,iD,ih,iz,:],*pars)#util
                        
                    else: #before T-1 make consumption saving choices
                    
                        
                    
                        #Choice conditional on employment
                        for iwlp in range(par.num_wlp): 
                            
                            resi = par.R*grid_Ai+usr.income_single(par,t,iwlp,iD,ih,iz,grid_Ai,women)[0]
                            
                            wlp=par.grid_wlp[iwlp]
                            
                            if g==0:#women
                                home= 1-wlp
                            else:#men
                                home = 1.0 if wlp==0 else 0.0
                            
                            pars=(par.ρ,par.χ,par.α,par.ν,par.θ,par.η,par.ϕ,par.wedge,par.px,0.0,0.0,home,women)
                        
                            # marginal utility of assets next period (continuation is
                            # labor-choice independent: full-time/identity HC transition)
                            βEid=par.β*usr.deriv(grid_Ai,Ei[iD,ih,iz,:])
                            
                            # first get toatl -consumption out of grid using FOCs
                            linear_interp.interp_1d_vec(np.flip(par.grid_marg_u_s[:,iwlp,g]),par.grid_inv_marg_u,βEid,cit[iwlp,iD,ih,iz,:])
                            
                            # use budget constraint to get current resources
                            Ri_now = grid_Ai.flatten() + cit[iwlp,iD,ih,iz,:]
                                   
                            # use the upper envelope algorithm to get optimal consumption and util
                            upp_env_single(grid_Ai,Ri_now,cit[iwlp,iD,ih,iz,:],par.β*Ei[iD,ih,iz,:],resi,ci[iwlp,iD,ih,iz,:],vi[iwlp,iD,ih,iz,:],*pars)
                
                    # Singles of BOTH genders work full time before retirement (kill the
                    # non-working option). This keeps the model consistent: the simulation
                    # forces single women full-time and divorced men integrate the ex-wife's
                    # HC with the identity (full-time) transition.
                    if (t<par.Tr) : vi[0,iD,ih,iz,:]=-10000000
                    if (t>=par.Tr): vi[1,iD,ih,iz,:]=-10000000
                    
                    #if (t<par.Tr)                : vi[1,iD,ih,iz,:,4,4]=-10000000 
                    
                  
                    
                    
    #loop_savings_singles(par,par.grid_Aw,sol.Cw_tot_single[t],Ew,cwt,Ewt,cwp,sol.Vw_single[t],True,False) #savings
    #loop_savings_singles(par,par.grid_Am,sol.Cm_tot_single[t],Em,cmt,Emt,cmp,sol.Vm_single[t],False,False)#savings
    
    loop_savings_singles(par,par.grid_Aw,ciw,Ew,cwt,Ewt,cwp,viw,True,False) #savings
    loop_savings_singles(par,par.grid_Am,cim,Em,cmt,Emt,cmp,vim,False,False)#savings
   
    #Employment choice:
    for iA in range(par.num_A):
        for iz in range(par.num_z):
            for ih in range(par.num_h):
                for iD in range(par.num_perdiv):
                    
                    sol.wlp_s_w[t,iD,ih,iz,iA]      = np.argmax(viw[:,iD,ih,iz,iA])                  
                    sol.Cw_tot_single[t,iD,ih,iz,iA]= ciw[sol.wlp_s_w[t,iD,ih,iz,iA],iD,ih,iz,iA]
                    sol.Vw_single[t,iD,ih,iz,iA]    = viw[sol.wlp_s_w[t,iD,ih,iz,iA],iD,ih,iz,iA]
                    
                    
                    sol.wlp_s_m[t,iD,ih,iz,iA]      = np.argmax(vim[:,iD,ih,iz,iA])               
                    sol.Cm_tot_single[t,iD,ih,iz,iA]= cim[sol.wlp_s_m[t,iD,ih,iz,iA],iD,ih,iz,iA]
                    sol.Vm_single[t,iD,ih,iz,iA]    = vim[sol.wlp_s_m[t,iD,ih,iz,iA],iD,ih,iz,iA]
                    
                    
                    
                    
               

#################################################
# SOLUTION - COUPLES
################################################

def solve_couple(sol,par,t):#Solve the couples's problem, choose EGM of VFI techniques
 
    # solve the couple's problem: choose your fighter
    tuple_with_outcomes =     solve_remain_couple_egm(par,sol,t)
              
    #Store above outcomes into solution
    store(*tuple_with_outcomes,par,sol,t)
        
@njit(parallel=parallel)
def integrate_couple(par,sol,t): 
     
    EVw,EVm=np.zeros((2,par.num_wlp,par.num_h,par.num_z,par.num_power,par.num_love,par.num_A)) 

    #kroneker product of uncertainty in love, income, human capital
    Π=[np.kron(np.kron(par.Πh[t][wlp],par.Π[t]),par.Πl[t]) for wlp in range(par.num_wlp)]
    to_pass=(Π[0]==0.0) & (Π[1]==0.0)#whether kroneker product is 0 and does not contribute to EV
    
    for iL in prange(par.num_love): 
        for ih in range(par.num_h):  
            for iP in range(par.num_power):     
                for iA in range(par.num_A): 
                    for iz in range(par.num_z): 
                        for jL in range(par.num_love): 
                            for jh in range(par.num_h): 
                                for jz in range(par.num_z):
                                    
                                    zdx = ih*par.num_love*par.num_z + iz*par.num_love+iL
                                    zjdx =jh*par.num_love*par.num_z + jz*par.num_love+jL
                                    
                                    if to_pass[zjdx,zdx]:continue 
                                    
                                    for wlp in range(par.num_wlp):
                                    
                                        idx=(wlp,ih,iz,iP,iL,iA);jdx=(t+1,jh,jz,iP,jL,iA)
                                        
                                        EVw[idx]+= sol.Vw_couple[jdx]*Π[wlp][zjdx,zdx]
                                        EVm[idx]+= sol.Vm_couple[jdx]*Π[wlp][zjdx,zdx]
                                      
    return EVw,EVm


@njit(parallel=parallel) 
def solve_remain_couple_egm(par,sol,t): 
               
    #Integration if not last period
    if t<(par.T-1): EVw,EVm = integrate_couple(par,sol,t)


    # initialize 
    i_Vw,i_Vm,i_Vc,i_C_tot,wls=np.zeros((5,par.num_wlp,par.num_h,par.num_z,par.num_power,par.num_love,par.num_A)) 
    Vw,Vm=np.zeros((2,par.num_h,par.num_z,par.num_power,par.num_love,par.num_A)) 
        
    pars=(par.ρ,par.χ,par.α,par.ν,par.θ,par.η,par.ϕ,par.wedge,par.px)
    for iL in prange(par.num_love): 
        for ih in range(par.num_h):
            for iz in range(par.num_z):
                for iP in range(par.num_power):
                
                    # indexes
                    idx=(ih,iz,iP,iL,slice(None))
                                      
                    # resources depending on women labor supply
                    resources,a,b,c,d,e=usr.resources_couple(par,t,ih,iz,par.grid_A) 
                    
                    #love shocks
                    love = (par.grid_lovew[t][iL], par.grid_lovem[t][iL]) 
                    
                    
                    # continuation values 
                    if t==(par.T-1):#last period 
                        
                        #Get consumption then utilities (assume no labor participation). Note: no savings!
                        Vw[idx],Vm[idx]=usr.couple_time_utility(resources[0],par,sol,1,iP,0,love,pars)            
                        wls[0,*idx]=1.0;wls[1:,*idx]=0.0;i_Vm[1:,*idx]=i_Vw[1:,*idx]=-1e10;i_Vw[:,*idx]=Vw[idx];i_Vm[:,*idx]=Vm[idx];i_C_tot[0,*idx] = resources[0].copy() 
                                            
                    else:#periods before the last 
                                 
                        # compute consumption* and util given labor supply wlp. last 4 arguments below are output at iz,iL,iP
                        for wlp in range(par.num_wlp):
                            compute_couple(par,sol,t,idx,pars,EVw[wlp],EVm[wlp],wlp,resources[wlp],i_C_tot[wlp],i_Vw[wlp],i_Vm[wlp],i_Vc[wlp],love) # participation 
                     
                        if (t>=par.Tr):i_Vw[1:,*idx]=i_Vm[1:,*idx]=i_Vc[1:,*idx]=-1e10 # after retirement no labor participation 
                                                   
                        # deterministic labor part. choice (wls 0/1 indicator) + util Vw and Vm at the chosen option
                        choose_labor(par,i_Vc,i_Vw,i_Vm,idx,wls,Vw,Vm)
                        
              
                #if (t<par.Tr):  #Eventual rebargaining + separation decisions happen below, *if not retired* 
                    #Eventual rebargaining happens below
           
    if (t<par.Tr):
        for iL in prange(par.num_love): 
            for ih in range(par.num_h):
                for iz in range(par.num_z):
                                    
                        for iA in range(par.num_A):        
                            
                            tt=np.minimum(t//par.Dper, (par.Tr-1)//par.Dper)
                            idx_s = (t,tt,ih,iz,iA)
                            idxx = [(t,ih,iz,i,iL,iA) for i in range(par.num_power)]               
                            list_couple = (sol.Vw_couple, sol.Vm_couple)                 #couple        list
                            list_raw    = (Vw[ih,iz,:,iL,iA],Vm[ih,iz,:,iL,iA])          #remain-couple list
                            list_single = (sol.Vw_single[idx_s],sol.Vm_single[idx_s])    #single        list
                            iswomen     = (True,False)                                   #iswomen? in   list
                            
                            check_participation_constraints(par,sol.power,par.grid_power,list_raw,list_single,idxx,list_couple,iswomen)   
                           
    if (t>=par.Tr):sol.Vw_couple[t] = Vw.copy(); sol.Vm_couple[t]= Vm.copy() #copy utility if retired                                                   
    return (Vw,Vm,i_Vw,i_Vm,i_C_tot,wls) # return a tuple
       
@njit    
def compute_couple(par,sol,t,idx,pars2,EVw,EVm,wls,res,C_tot,Vw,Vm,Vc,love): 
 
    # indexes & initialization 
    ret=1 if t>=par.Tr else 0
    idz=idx[:-1];iP=idx[2];iL=idx[3];power = par.grid_power[iP]
    C_pd,βEw,βEm,Vwd,Vmd,_= np.ones((6,par.num_A));pars=(par,sol,ret,iP,wls,love,pars2)  
    
                  
    # discounted expected marginal utility from t+1, wrt assets
    βEVd=par.β*usr.deriv(par.grid_A,power*EVw[idz]+(1.0-power)*EVm[idz])

    # get consumption out of grid using FOCs (i) + use budget constraint to get current resources (ii)  
    linear_interp.interp_1d_vec(par.grid_marg_u_for_inv[ret,wls,iP,:],par.grid_inv_marg_u,βEVd,C_pd) #(i) 
    A_now =  par.grid_A.flatten() + C_pd    
            
    #Apply upper envelope for optimal consumption and C-tot and Vx,Vm,Vc
    upper_envelope(par.grid_A,A_now,C_pd,par.β*EVw[idz],par.β*EVm[idz],power,res,C_tot[idx],Vw[idx],Vm[idx],Vc[idx],*pars) 
        
       
@njit
def choose_labor(par,i_Vc,i_Vw,i_Vm,idx,wls,Vw,Vm):

    # deterministic labor participation (no taste shocks): for each asset level pick
    # the WLP option that maximizes the couple objective i_Vc. wls becomes a 0/1
    # indicator of the chosen option, and each spouse gets the individual value of
    # the chosen alternative.
    i_idx=(slice(None),*idx)
    for iA in range(par.num_A):
        k=np.argmax(i_Vc[*i_idx[:-1],iA])
        wls[*i_idx[:-1],iA]=0.0
        wls[k,*idx[:-1],iA]=1.0
        Vw[*idx[:-1],iA]=i_Vw[k,*idx[:-1],iA]
        Vm[*idx[:-1],iA]=i_Vm[k,*idx[:-1],iA]
    
    
@njit
def check_participation_constraints(par,solpower,gridpower,list_raw,list_single,idx,
                                    list_couple=(np.zeros((1,1)),),iswomen=(True,),nosim=True):
                 
    # surplus of marriage, then its min and max given states
    Sw = list_raw[0] - list_single[0] 
    Sm = list_raw[1] - list_single[1] 
    min_Sw = np.min(Sw);min_Sm = np.min(Sm)
    max_Sw = np.max(Sw);max_Sm = np.max(Sm) 


    power_at_0_w = linear_interp.interp_1d(Sw      ,par.grid_power      ,0.0)   
    power_at_0_m = linear_interp.interp_1d(Sm[::-1],par.grid_power[::-1],0.0)          

    Sm_at_0_w = linear_interp.interp_1d(par.grid_power, Sm, power_at_0_w)   
    Sw_at_0_m = linear_interp.interp_1d(par.grid_power, Sw, power_at_0_m)   
      
        
    ##################################################################
    # For a given power, find out if marriage, divorce or rebargaining
    # Then, update power and (if no simulation) update value functions
    #################################################################
    for iP,power in enumerate(gridpower):

      
        if par.full:#Full commiment case
        
            
            
            if (Sw[iP]>0) | (Sm[iP]>0):#If at least one is happy married stay married...
                
                solpower[idx[iP]] = power #update power, below update value function
                if nosim:no_power_change(list_couple,list_raw,idx,iP,power)
                
            else:#...otherwise, divorce!
                
                solpower[idx[iP]] = -100.0 #update power, below update value function
                if nosim:divorce(list_couple,list_single,idx[iP])
            
        else:#Limited commitment case

            #1) all iP values are consistent with marriage
            if ((min_Sw >= 0.0) & (min_Sm >= 0.0)): 
                solpower[idx[iP]] = power #update power, below update value function
                if nosim:no_power_change(list_couple,list_raw,idx,iP,power)
                         
            #2) no iP values consistent with marriage
            elif (max_Sw < 0.0) | (max_Sm < 0.0) : 
                solpower[idx[iP]] = -100.0 #update power, below update value function
                if nosim:divorce(list_couple,list_single,idx[iP])
                    
            #3) some iP are (invidivually) consistent with marriage: try rebargaining
            else:             
                # 3.1) woman wants to leave &  man happy to shift some bargaining power
                if (power<power_at_0_w) & (Sm_at_0_w > 0):  
                    solpower[idx[iP]] = power_at_0_w #update power, below update value function
                    if nosim:do_power_change(par,list_couple,list_raw,idx,iP,power_at_0_w)
                                                                              
                # 3.2) man wants to leave & woman happy to shift some bargaining power
                elif (power>power_at_0_m) & (Sw_at_0_m > 0): 
                    solpower[idx[iP]] = power_at_0_m #update power, below update value function
                    if nosim:do_power_change(par,list_couple,list_raw,idx,iP,power_at_0_m)
                                            
                # 3.3) divorce: men (women) wants to leave & woman (men) not happy to shift some bargaining power
                elif ((power<power_at_0_w) & (Sm_at_0_w <=0)) | ((power>power_at_0_m) & (Sw_at_0_m <=0)):
                    solpower[idx[iP]] = -100.0  #update power, below update value function
                    if nosim:divorce(list_couple,list_single,idx[iP])
                    
                # 3.4) no-one wants to leave
                else: 
                    solpower[idx[iP]] = power #update power, belowe update value function
                    if nosim:no_power_change(list_couple,list_raw,idx,iP,power)
 
@njit
def no_power_change(list_couple,list_raw,idx,iP,power):
    for i,key in enumerate(list_couple): key[idx[iP]] = list_raw[i][iP]        
@njit
def divorce(list_couple,list_single,idx):    
    for i,key in enumerate(list_couple): key[idx]=list_single[i]
              
@njit
def do_power_change(par,list_couple,list_raw,idx,iP,power_at_0_i):    
    for i,key in enumerate(list_couple): 
        key[idx[iP]] = linear_interp.interp_1d(par.grid_power,list_raw[i],power_at_0_i)                             
        
@njit
def store(Vw,Vm,i_Vw,i_Vm,i_C_tot,wls,par,sol,t):    
                
    sol.i_C_tot_remain_couple[t] = i_C_tot
    sol.Vw_remain_couple[t] = Vw
    sol.Vm_remain_couple[t] = Vm
    sol.i_Vw_remain_couple[t] = i_Vw
    sol.i_Vm_remain_couple[t] = i_Vm
    sol.remain_WLP[t] = wls
                 
##################################
#        SIMULATIONS
#################################

@njit(parallel=parallel)
def simulate_lifecycle(sim,sol,par):
    
    # unpacking some values to help numba optimize (horrible)
    A=sim.A;Aw=sim.Aw;Am=sim.Am;couple=sim.couple;power=sim.power;C_tot=sim.C_tot;Cm_tot=sim.Cm_tot;Cw_tot=sim.Cw_tot;couple_lag=sim.couple_lag;power_lag=sim.power_lag
    love=sim.love;shock_love=sim.shock_love;iz=sim.iz;wlp=sim.WLP;incw=sim.incw;incm=sim.incm;ih=sim.ih;iD=sim.ID;incwg=sim.incwg;incmg=sim.incmg
    dw=sim.dw;dm=sim.dm;Cw=sim.Cw;Cm=sim.Cm;Vsm=sim.Vsm;Vsw=sim.Vsw;Vcm=sim.Vcm;Vcw=sim.Vcw;tax=sim.tax

    initial=sim.init_love.copy()
    
    for i in prange(par.simN):
        for t in range(par.simT):
 
            #Iterate only if in the sample...
            if t<par.sample_init[i]:continue
            
            #..and, if policy is in action, only if the policy was enacted already
            if (par.pens_reform) & (t<par.policy_init[i]):continue
            
            #BELOW YOUACTICATE HETEROGENEITY IN INITIAL MATCH QUALITY
            if t==par.sample_init[i]:
    
                #Initial condition for assets
                A[i,t] = sim.init_A[i]; Aw[i,t] =  par.div_A_share * A[i,t];  Am[i,t] =  (1.0-par.div_A_share) * A[i,t]
                
                #Initial love shock: common love is central value, treansitory shocks are drawn
                initial[i]=par.num_lovem//2*par.num_love//par.num_lovem+usr.mc_simulate(0,par.trans_love,shock_love[i,t])#

            # Copy variables from t-1 or initial condition. Initial (t>0) assets: preamble (later in the simulation) 
            # copy determines when to copy from previous period or use initial condition. This matters because
            # when the policy changed, we want to copy the values in t-1 when reform was not it place
    
            if (par.pens_reform): copy = True if par.policy_init[i]>par.sample_init[i] else t>par.sample_init[i] 
            else:                 copy = t>par.sample_init[i]
                
            
            #Initial condition for other variables
            Π = par.Πh[t][wlp[i,t-1]]                                                if t>0 else par.Πh[t][-1]
            ih[i,t] = usr.mc_simulate(ih[i,t-1],Π,sim.shock_h[i,t])                  if copy else sim.init_ih[i]            
            couple_lag[i,t] = couple[i,t-1]                                          if copy else sim.init_couple[i]
            power_lag[i,t] = power[i,t-1]                                            if copy else sim.init_power[i]      
            Πz=par.Π[t-1]                                                            if (couple[i,t-1]==1) else par.Πs[t-1]
            iz[i,t] = usr.mc_simulate(iz[i,t-1],Πz,sim.shock_z[i,t])                 if copy else sim.init_z[i]
            love[i,t] = usr.mc_simulate(love[i,t-1],par.Πl[t-1],shock_love[i,t])     if copy else initial[i]#sim.init_love[i]#
           
            # Indices of resources
            idx = (t,ih[i,t],iz[i,t],slice(None),love[i,t])
            
            # first check if they want to remain together and what the bargaining power will be if they do.
            if (couple_lag[i,t]) & (t<par.Tr):# do rebargaining power and divorce choice ifin a couple and not retired                 

                # Store before renegotiations utilities
                Vsw__=linear_interp.interp_1d(par.grid_Aw,sol.Vw_single[t,t//par.Dper,ih[i,t],iz[i,t]],Aw[i,t])
                Vsm__=linear_interp.interp_1d(par.grid_Am,sol.Vm_single[t,t//par.Dper,ih[i,t],iz[i,t]],Am[i,t])

                # Value of transitioning into singlehood
                list_single = (Vsw__,Vsm__)

                # Value of being ina  couple with given bargaining power
                list_raw    = (np.array([linear_interp.interp_1d(par.grid_A,sol.Vw_remain_couple[idx][iP],A[i,t]) for iP in range(par.num_power)]),
                               np.array([linear_interp.interp_1d(par.grid_A,sol.Vm_remain_couple[idx][iP],A[i,t]) for iP in range(par.num_power)]))

                # Rebargainings happens here
                check_participation_constraints(par,power,np.array([power_lag[i,t]]),list_raw,list_single,[(i,t)],nosim=False)
                couple[i,t] = False if power[i,t] <= -10.0 else True # partnership status: divorce is coded as -100
                
                #If divorce, update period at divorce
                if power[i,t] <= -10.0: iD[i,:]=t//par.Dper
                    
            else: #divorce is an absorbing state
                
                couple[i,t] = couple_lag[i,t]; power[i,t] = power[i,t-1]#stay single or copy relationship if retired 
                            
            # update behavior
            if couple[i,t]:
                
                if (t<par.Tr):
                    
                    # Store before renegotiations utilities
                    Vsw[i,t]=linear_interp.interp_1d(par.grid_Aw,sol.Vw_single[t,t//par.Dper,ih[i,t],iz[i,t]],Aw[i,t])
                    Vsm[i,t]=linear_interp.interp_1d(par.grid_Am,sol.Vm_single[t,t//par.Dper,ih[i,t],iz[i,t]],Am[i,t])
                    Vcw[i,t]=linear_interp.interp_2d(par.grid_power,par.grid_A,sol.Vw_remain_couple[idx],power[i,t],A[i,t])
                    Vcm[i,t]=linear_interp.interp_2d(par.grid_power,par.grid_A,sol.Vm_remain_couple[idx],power[i,t],A[i,t])
                    
                # Labor participation: deterministic choice of the WLP option with the
                # highest interpolated 0/1 choice indicator (no taste shocks)
                part_i=np.array([linear_interp.interp_2d(par.grid_power,par.grid_A,sol.remain_WLP[t,wls,*idx[1:]],power[i,t],A[i,t]) for wls in range(par.num_wlp)])
                wlp[i,t]=np.argmax(part_i)
             
                # Optimal total consumption allocation if couple (note use of the updated index)
                sol_C_tot = sol.i_C_tot_remain_couple[t,wlp[i,t],*idx[1:]]
                C_tot[i,t] = linear_interp.interp_2d(par.grid_power,par.grid_A,sol_C_tot,power[i,t],A[i,t])

                # Obtain household resources
                M_resources_raw, incmt,incwt,incmgt,incwgt,taxc = usr.resources_couple(par,t,ih[i,t],iz[i,t],A[i,t])
                incm[i,t]=incmt[wlp[i,t]];incw[i,t]=incwt[wlp[i,t]];incmg[i,t]=incmgt;incwg[i,t]=incwgt[wlp[i,t]];tax[i,t]=taxc[wlp[i,t]]
                M_resources= M_resources_raw[wlp[i,t]] 
                
                if t< par.simT-1:A[i,t+1] = M_resources - C_tot[i,t]#
                if t< par.simT-1:Aw[i,t+1] =       par.div_A_share * A[i,t+1]# in case of divorce 
                if t< par.simT-1:Am[i,t+1] = (1.0-par.div_A_share) * A[i,t+1]# in case of divorce 
                
                # Obtain public and private consumption given total consumption Ctot
                ret=1 if t>=par.Tr else 0
                Cw[i,t]=linear_interp.interp_2d(par.grid_power,par.grid_Ctot,sol.pre_Cw_priv[ret,wlp[i,t]],sim.power[i,t],sim.C_tot[i,t])
                Cm[i,t]=linear_interp.interp_2d(par.grid_power,par.grid_Ctot,sol.pre_Cm_priv[ret,wlp[i,t]],sim.power[i,t],sim.C_tot[i,t])
                dw[i,t]=sim.C_tot[i,t]-Cm[i,t]-Cw[i,t]
                dm[i,t]=sim.C_tot[i,t]-Cm[i,t]-Cw[i,t]
                
            else: # single
               
                # pick relevant solution for single
                sol_single_w = sol.Cw_tot_single[t,iD[i,t],ih[i,t],iz[i,t]]
                sol_single_m = sol.Cm_tot_single[t,iD[i,t],ih[i,t],iz[i,t]]
                
                #Store before renegotiations utilities
                Vsw[i,t]=linear_interp.interp_1d(par.grid_Aw,sol.Vw_single[t,iD[i,t],ih[i,t],iz[i,t]],Aw[i,t])
                Vsm[i,t]=linear_interp.interp_1d(par.grid_Am,sol.Vm_single[t,iD[i,t],ih[i,t],iz[i,t]],Am[i,t])

                # optimal consumption allocations
                Cw_tot[i,t] = linear_interp.interp_1d(par.grid_Aw,sol_single_w,Aw[i,t])
                Cm_tot[i,t] = linear_interp.interp_1d(par.grid_Am,sol_single_m,Am[i,t])   
                C_tot[i,t]  = Cw_tot[i,t] + Cm_tot[i,t]
                              
                home=1 if t>=par.Tr else 0
                Cw[i,t],dw[i,t] = usr.intraperiod_allocation_single(Cw_tot[i,t],par.ρ,par.χ,par.α,par.ν,par.θ,par.η,par.ϕ,par.wedge,par.px,0.0,0.0,home)
                Cm[i,t],dm[i,t] = usr.intraperiod_allocation_single(Cm_tot[i,t],par.ρ,par.χ,par.α,par.ν,par.θ,par.η,par.ϕ,par.wedge,par.px,0.0,0.0,home)

                #Labor supply
                wlp[i,t]=par.num_wlp-1 if t<par.Tr else 0
                
                #resources
                incw[i,t],incwg[i,t],taxsw=usr.income_single(par,t,wlp[i,t],iD[i,t],ih[i,t],iz[i,t],Aw[i,t],women=True)
                incm[i,t],incmg[i,t],taxsm=usr.income_single(par,t,1       ,iD[i,t],ih[i,t],iz[i,t],Am[i,t],women=False)
                tax[i,t]=taxsw+taxsm
                
                # update end-of-period states
                Mw = par.R*Aw[i,t] + incw[i,t] # total resources woman
                Mm = par.R*Am[i,t] + incm[i,t] # total resources man

                if t< par.simT-1: 
                    #if par.women[i]: Aw[i,t+1] = Mw - Cw_tot[i,t]; Am[i,t+1] = Aw[i,t+1]*par.div_A_share
                    #else:            Am[i,t+1] = Mm - Cm_tot[i,t]; Aw[i,t+1] = Am[i,t+1]*par.div_A_share
                    Aw[i,t+1] = Mw - Cw_tot[i,t]#; Am[i,t+1] = Aw[i,t+1]*par.div_A_share
                    Am[i,t+1] = Mm - Cm_tot[i,t]#; Aw[i,t+1] = Am[i,t+1]*par.div_A_share
                    A[i,t+1]  = Aw[i,t+1] + Am[i,t+1] 
                    