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
        
        # Demographics
        par.T = 63+5 # terminal age: https://www.mortality.org/File/GetDocument/hmd.v6/JPN/STATS/fltper_1x1.txt 
        par.Tr = 40+5 # age at retirement
         
        # Prices
        par.R = 1.03068#1+ interest rate
               
        # Preferences
        par.β = 0.98    # Discount factor
        par.ρ = 1.5     # Risk avresion private goods
        par.χ = 1.75    # Risk aversion home goods
        par.α = 0.35    # Weight on home good
        par.σ = 0.00015 # Taste shock for employment decitions. !!! We might drop this
        par.wedge=0.1   #Single-couple utility wedge
        
        # Income processes: tren and shocks (sd of persistent (σpi), transitory (σϵi), initial (σ0i) income shocks)
        par.ι0m= -0.224; par.ι1m=0.046   ;par.ι2m=-0.00075858 #trend for husband
        par.ι0w=-0.591  ;par.ι1w=0.046   ;par.ι2w=-0.00075858 #trend for wife
 
        par.σzm=0.0082**0.5  ;par.σϵm= 0.0125**0.5;par.σ0m=  0.0338**0.5; #shock size husband
        par.σzw= 0.00978**0.5;par.σϵw=0.0137**0.5 ;par.σ0w= 0.1198**0.5; #shock size wife
        par.σϵwm=0.0#0.00289 #correlation of transitory shocks
        
        # Pension parameters
        par.p_b=0.3578 #basic pension
        par.κ=0.219    #proportional part of pension
        
        # Depreciation of human capital
        par.μ = 1.195        # Human capital depreciation drift
        par.p_μ = 1.0/40.0   # Probability that  human capital depreciates
        
        # Home good production
        par.ν = 0.21  # Weight on money vs. time to produce home good
        par.ϕ = 0.326 # Time spend on public goods by singles
        
        # Taxes
        par.Λ=0.92                              # 1- tax level
        par.τ=0.08                              # Tax progressivity        
        par.d0=0.172;par.d1=0.0132;par.d2=-0.56 # Husband deduction parameters
                
        # Post-divorce transfers
        par.alimony=0.0       #Alimony used for experiments
        par.div_A_share = 0.5 # Asset share to wife at divorce
        
        ##########################################
        # Grid-related parameters and state space
        ##########################################
        
        # Wealth
        par.num_A = 15;par.max_A = 75.0
        
        # Bargaining power
        par.num_power = 15
        par.power_min=1e-3;par.power_max=1.0-par.power_min
        
        # Women's human capital states
        par.num_h = 2

        # love/match quality
        par.num_love =7
        par.σL = 0.1; par.σL0 = 0.1
        
        # productivity of men and women: gridpoints
        par.num_ϵw=3;par.num_ϵm=3#transitory
        par.num_pw=3;par.num_pm=3#persistent
        par.num_zw=par.num_pw*par.num_ϵw;par.num_zm=par.num_pm*par.num_ϵw#total by gender
        par.num_z=par.num_zm*par.num_zw#total, couple
        
        # pre-computation of consumption Ctot=cf+cm+d, used in the intra-period maximization
        par.num_Ctot = 150;par.max_Ctot = par.max_A*2
        
        ##########################################
        # Simulations parameters
        ##########################################
        par.seed = 9211;par.simT = par.T;par.simN = 100_000
       
        
    def setup_grids(self):
        par = self.par
        
        # Assets grid. Single grids are such to avoid interpolation
        par.grid_A = np.append(nonlinspace(0.0,par.max_A,par.num_A-1,2.1),par.max_A*10)
        par.grid_Aw =  par.grid_A * par.div_A_share; par.grid_Am =  par.grid_A*(1.0-par.div_A_share)

        # Women's labor supply grids
        par.grid_wlp=np.array([0.0,0.561,0.823]);par.num_wlp=len(par.grid_wlp)
        
        # Match quality shock grid and transition matrices    
        par.grid_love,par.Πl,par.Πl0= usr.addaco_nonst(par.T,par.σL,par.σL0,par.num_love)
        
        # Bargaining power grid. non-linear grid with more mass in both tails.        
        par.grid_power = usr.grid_fat_tails(par.power_min,par.power_max,par.num_power)
        
        # Women's human capital grid plus transition matrices for working full time (Πh_pt) and not working (Πh_pt)       
        par.grid_h = np.flip(np.linspace(-par.num_h*par.μ,0.0,par.num_h))#0 position is best

        Πh_pt = np.eye(par.num_h)  # Perfect transition if full-time participation...
        Πh_nt = np.array([[1-par.p_μ, par.p_μ], [0, 1]]).T  # ...depreciation otherwise
        par.Πh_t = np.array([w*Πh_pt + (1-w)*Πh_nt for w in par.grid_wlp]) # Work-hour weighted transitions
        identity_block = np.tile(np.eye(par.num_h), (par.num_wlp, 1, 1)) #stops depreciating at retirement
        par.Πh = [par.Πh_t if t < par.Tr else identity_block for t in range(par.T)]
        
        # Grid of total consumption, used in the intra-period problem
        par.grid_Ctot = nonlinspace(1.0e-6,par.max_Ctot,par.num_Ctot,1.1)
 
        # Income shocks grids: couples
        par.grid_zw,par.grid_ϵw,par.grid_pw,par.Π_zw0, \
            par.grid_zm,par.grid_ϵm,par.grid_pm,par.Π_zm0, \
                                        par.Π=usr.labor_income(par)                                       
                                        
        # Income shocks grids: singles
        par.grid_zw,par.grid_ϵw,par.grid_pw,par.Π_zw0, \
            par.grid_zm,par.grid_ϵm,par.grid_pm,par.Π_zm0, \
                                                par.Πs=usr.labor_income(par,single=True) 
              
        # Simulation arrays
        par.women =np.ones(par.simN)#0: simumate men, 1 women
        par.sample_init=np.zeros(par.simN)#in which period do we start simulating the sample?
        
    def allocate(self):
        par = self.par;sol = self.sol;sim = self.sim;self.setup_grids()

        # setup grids
        par.simT = par.T
        
        # Intra period problem: given total consumption, how much public and private expenditures
        shape_pre = (par.num_wlp,par.num_power,par.num_Ctot)
        sol.pre_Cw_priv = np.nan + np.ones(shape_pre)   #wife private cf
        sol.pre_Cm_priv = np.nan + np.ones(shape_pre)   #husband private cm
        sol.pre_d_pub = np.nan + np.ones(shape_pre)     #public consumption d
           
        # Marginal utility arrays to be used in the EGM, filled in the intra-period problem 
        par.grid_inv_marg_u = np.flip(par.grid_Ctot)
        par.grid_marg_u = np.nan + np.ones(shape_pre)         # couples
        par.grid_marg_uw = np.nan + np.ones(shape_pre)        # couples
        par.grid_marg_um = np.nan + np.ones(shape_pre)        # couples
        par.grid_marg_u_for_inv = np.nan + np.ones(shape_pre) # couples
        par.grid_cpriv_s =  np.nan + np.ones((par.num_Ctot,2))# singles
        par.grid_marg_u_s = np.nan + np.ones((par.num_Ctot,2))# singles
         
        # Singles: value functions (vf), consumption, marg util
        shape_single = (par.T,par.num_h,par.num_z,par.num_A)
        sol.Vw_single = np.nan + np.ones(shape_single)     # vw in t
        sol.Vm_single = np.nan + np.ones(shape_single)     # vm in t
        sol.Cw_tot_single = np.nan + np.ones(shape_single) # = cw+d
        sol.Cm_tot_single = np.nan + np.ones(shape_single) # = cm+d

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
        sol.remain_WLP = np.ones(shape_couple_wls)                    # pr. of chosing WLP   
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
        
        sim.iz = np.ones(shape_sim,dtype=np.int_)   # index of income shocks 
        sim.A = np.zeros(shape_sim)                 # total assets (m+w)
        sim.Aw = np.zeros(shape_sim)                # w's assets
        sim.Am = np.zeros(shape_sim)                # m's assets
        sim.couple = np.ones(shape_sim,dtype=bool)        # In a couple? True/False
        sim.couple_lag = np.ones(shape_sim,dtype=bool)    # In a couple previous period? True/False
        sim.power = -100.0*np.ones(shape_sim)             # Bargaining power θ
        sim.power_lag = -100.0*np.ones(shape_sim)         # Bargaining power θ previous period
        sim.love = np.ones(shape_sim,dtype=np.int_)       # Match quality
        sim.incw = np.nan + np.ones(shape_sim)            # w's net income
        sim.incm = np.nan + np.ones(shape_sim)            # m's net income
        sim.incwg = np.nan + np.ones(shape_sim)           # w's gross income
        sim.incmg = np.nan + np.ones(shape_sim)           # m's gross income
        sim.WLP = np.ones(shape_sim,dtype=np.int_)        # w's labor supply index
        sim.ih = np.ones(shape_sim,dtype=np.int_)         # w's human capital 
        sim.tax = np.zeros(shape_sim)                     # Taxes paid by the couple or divorces (sum w+m)

        # Shocks
        np.random.seed(par.seed)
        sim.shock_love = np.random.random_sample((par.simN,par.simT)) # Match quality
        sim.shock_iz=np.random.random_sample((par.simN,2))            # Initial labor income index 
        sim.shock_z=np.random.random_sample((par.simN,par.simT))      # Labor income shocks
        sim.shock_taste=np.random.random_sample((par.simN,par.simT))  # Taste shock (linked to σ)
        sim.shock_h=np.random.random_sample((par.simN,par.simT))      # Human capital draws

        # Initial distribution (this will be overwritten by user input)
        sim.init_ih = np.zeros(par.simN,dtype=np.int_)                  # Initial w's human capital
        sim.init_couple = np.ones(par.simN,dtype=bool)                  # State (couple=1/single=0)
        sim.init_power =  np.random.random_sample(par.simN)             # Barg power 
        sim.init_love = np.ones(par.simN,dtype=np.int_)*par.num_love//2 # Initial match quality    
        sim.init_z  = np.zeros(par.simN,dtype=np.int_)                  # Initial income index
        
                       
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
        
    pars=(par.ρ,par.χ,par.α,par.ν,par.ϕ,par.wedge)  
    ϵ = 1e-8# delta increase in xs to compute numerical deratives

    ################ Singles part #####################
    for i,C_tot in enumerate(par.grid_Ctot):
        for sex in range(2):
        
            home= 1-par.grid_wlp[-1] if sex==0 else 0.0
            pars_sex=(par.ρ,par.χ,par.α,par.ν,par.ϕ,par.wedge,0.0,0.0,home)  
            
            # optimize to get util from total consumption(m<->C_tot)=private cons(c)+public cons(m-c)
            grid_cpriv_s[i,sex] = usr.optimizer(lambda c,m,p:-usr.util(c,m-c,*p),ϵ,C_tot-ϵ,args=(C_tot,pars_sex))[0]
            
            # numerical derivative of util wrt total consumption C_tot, using envelope thm
            share_priv=grid_cpriv_s[i,sex]/C_tot
            forward  = usr.util(share_priv*(C_tot+ϵ),(1.0-share_priv)*(C_tot+ϵ),*pars_sex)
            backward = usr.util(share_priv*(C_tot-ϵ),(1.0-share_priv)*(C_tot-ϵ),*pars_sex)
            grid_marg_u_s[i,sex] = (forward - backward)/(2*ϵ)

            
    
    for iP in prange(par.num_power):       
        for iwlp,wlp in enumerate(par.grid_wlp):  
            for i,C_tot in enumerate(par.grid_Ctot):  
                  
                 
                # initialize bounds and bargaining power  
                power=par.grid_power[iP]  
                mult = power**(1/par.ρ)/(power**(1/par.ρ)+(1-power)**(1/par.ρ)) 
                 
                parss=(par.ρ,par.χ,par.α,par.ν,par.ϕ,par.wedge)
                 
                ress=bisect(usr.couple_root,1e-12,C_tot-1e-12, args=(C_tot,power,*parss,1-wlp))[0] 
                d_pub[iwlp,iP,i]  = ress 
                 
                Cw_priv[iwlp,iP,i] = (C_tot-d_pub[iwlp,iP,i])*mult 
                Cm_priv[iwlp,iP,i] = (C_tot-d_pub[iwlp,iP,i])*(1-mult) 
                res = np.array([Cw_priv[iwlp,iP,i],Cm_priv[iwlp,iP,i]]) 
                
 
                # numerical derivative of util wrt total consumption C_tot, using envelope thm  
                _,forw_w,forw_m = usr.couple_util(res/(C_tot)*(C_tot+ϵ),C_tot+ϵ,power,1.0-wlp,*pars)  
                _,bakw_w,bakw_m = usr.couple_util(res/(C_tot)*(C_tot-ϵ),C_tot-ϵ,power,1.0-wlp,*pars)  
                grid_marg_uw[iwlp,iP,i] = (forw_w - bakw_w)/(2*ϵ);grid_marg_um[iwlp,iP,i] = (forw_m - bakw_m)/(2*ϵ) 
                                   
            #Create grid of couple's marginal util and inverse marginal utility   
            grid_marg_u[iwlp,iP,:] = power*grid_marg_uw[iwlp,iP,:]+(1.0-power)*grid_marg_um[iwlp,iP,:]  
            grid_marg_u_for_inv[iwlp,iP,:]=np.flip(par.grid_marg_u[iwlp,iP,:])   
            
  
#######################
# SOLUTIONS - SINGLES #
#######################

# @njit(parallel=parallel)
# def integrate_single(sol,par,t):
#     Ew_nomeet,Em_nomeet=np.zeros((2,par.num_h,par.num_z,par.num_A)) 
     
#     # 1. Expected value if not meeting a partner
#     for iA in prange(par.num_A):
#         for iz in range(par.num_z):
#             for ih in range(par.num_h):
#                 for jz in range(par.num_z):
                                      
#                     Ew_nomeet[ih,iz,iA] += sol.Vw_single[t+1,:,jz,iA] @ par.Πh[t][-1,:,ih] * par.Πs[t][jz,iz]
#                     Em_nomeet[ih,iz,iA] += sol.Vm_single[t+1,:,jz,iA] @ par.Πh[t][-1,:,ih] * par.Πs[t][jz,iz]                    
       
#     # 2. If we ever add probaility of meeting partners, it should be here
                            
#     # 3. Return expected value given meeting probabilities                                              
#     return Ew_nomeet,Em_nomeet

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
    
    # Output arrays: Ew_nomeet[ih, iz, iA] and Em_nomeet[ih, iz, iA]
    Ew_nomeet  = np.zeros((par.num_h, par.num_z, par.num_A))
    Em_nomeet  = np.zeros((par.num_h, par.num_z, par.num_A))

    # -------------------------
    # Precompute contiguous versions of Πh and Πs for fast column access
    # -------------------------

    # Make column access fast once per call (Fortran-order = contiguous columns)
    H = np.asfortranarray(par.Πh[t][-1, :, :])   # (2, 2)
    S = np.asfortranarray(par.Πs[t])             # (81, 81)

    # Parallelize across iA
    for iA in prange(par.num_A):
        # Ensure fast row access for V* (C-order = contiguous rows)
        Vw = np.ascontiguousarray(sol.Vw_single[t+1, :, :, iA])  # (2, 81)
        Vm = np.ascontiguousarray(sol.Vm_single[t+1, :, :, iA])  # (2, 81)

        # Step (a): collapse h (2) -> produce per-z vectors for each ih in one go
        # V*.T: (81, 2) @ H: (2, 2) -> Uw/Um: (81, 2)
        Uw = Vw.T @ H
        Um = Vm.T @ H

        # Step (b): collapse z with one GEMM
        # Uw.T: (2, 81) @ S: (81, 81) -> (2, 81)
        Ew_nomeet[:, :, iA] = Uw.T @ S
        Em_nomeet[:, :, iA] = Um.T @ S

    return Ew_nomeet, Em_nomeet
    
#@njit(parallel=parallel)
def solve_single_egm(sol,par,t):

    #Integrate to get continuation value unless if you are in the last period
    Ew,Em=np.zeros((2,par.num_h,par.num_zw,par.num_A))
    if t<par.T-1:Ew,Em = integrate_single(sol,par,t) #if t<par.T-1 else 
             
    #Pre-define outcomes (if update .sol directly, parallelization go crazy)
    cwt,Ewt,cwp,cmt,Emt,cmp=np.ones((6,par.num_h,par.num_z,par.num_A))
    
    
    #function to find optimal savings, called for both men and women below
    def loop_savings_singles(par,grid_Ai,ci,Ei,cit,Eit,cip,vi,women,divorce):
        
        home=1.0-par.grid_wlp[-1] if women else 0.0
        pars=(par.ρ,par.χ,par.α,par.ν,par.ϕ,par.wedge,0.0,0.0,home)
        sex=0 if women else 1
        
        for iz in range(par.num_z):
            for ih in range(par.num_h):

                resi = par.R*grid_Ai+usr.income_single(par,t,ih,iz,grid_Ai,women)[0]
                
                if t==(par.T-1): 
                    
                    ci[ih,iz,:] = resi.copy() #consume all resources
                    linear_interp.interp_1d_vec(par.grid_Ctot,par.grid_cpriv_s[:,sex],ci[ih,iz,:],cip[ih,iz,:])#private cons
                    vi[ih,iz,:]=usr.util(cip[ih,iz,:],ci[ih,iz,:]-cip[ih,iz,:],*pars)#util
                    
                else: #before T-1 make consumption saving choices
                    
                    # marginal utility of assets next period
                    βEid=par.β*usr.deriv(grid_Ai,Ei[ih,iz,:])
                    
                    # first get toatl -consumption out of grid using FOCs
                    linear_interp.interp_1d_vec(np.flip(par.grid_marg_u_s[:,sex]),par.grid_inv_marg_u,βEid,cit[ih,iz,:])
                    
                    # use budget constraint to get current resources
                    Ri_now = grid_Ai.flatten() + cit[ih,iz,:]
                           
                    # use the upper envelope algorithm to get optimal consumption and util
                    upp_env_single(grid_Ai,Ri_now,cit[ih,iz,:],par.β*Ei[ih,iz,:],resi,ci[ih,iz,:],vi[ih,iz,:],*pars)

    loop_savings_singles(par,par.grid_Aw,sol.Cw_tot_single[t],Ew,cwt,Ewt,cwp,sol.Vw_single[t],True,False) #savings
    loop_savings_singles(par,par.grid_Am,sol.Cm_tot_single[t],Em,cmt,Emt,cmp,sol.Vm_single[t],False,False)#savings
               

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
        
    pars=(par.ρ,par.χ,par.α,par.ν,par.ϕ,par.wedge)     
    for iL in prange(par.num_love): 
        for ih in range(par.num_h):
            for iz in range(par.num_z):
                for iP in range(par.num_power):
                
                    # indexes
                    idx=(ih,iz,iP,iL,slice(None))
                                      
                    # resources depending on women labor supply
                    resources,a,b,c,d,e=usr.resources_couple(par,t,ih,iz,par.grid_A) 
                    
                    # continuation values 
                    if t==(par.T-1):#last period 
                        
                        #Get consumption then utilities (assume no labor participation). Note: no savings!
                        Vw[idx],Vm[idx]=usr.couple_time_utility(resources[0],par,sol,iP,0,par.grid_love[t][iL],pars)            
                        wls[0,*idx]=1.0;wls[1:,*idx]=0.0;i_Vm[1:,*idx]=i_Vw[1:,*idx]=-1e10;i_Vw[:,*idx]=Vw[idx];i_Vm[:,*idx]=Vm[idx];i_C_tot[0,*idx] = resources[0].copy() 
                                            
                    else:#periods before the last 
                                 
                        # compute consumption* and util given labor supply wlp. last 4 arguments below are output at iz,iL,iP
                        for wlp in range(par.num_wlp):
                            compute_couple(par,sol,t,idx,pars,EVw[wlp],EVm[wlp],wlp,resources[wlp],i_C_tot[wlp],i_Vw[wlp],i_Vm[wlp],i_Vc[wlp]) # participation 
                     
                        if (t>=par.Tr):i_Vw[1:,*idx]=i_Vm[1:,*idx]=i_Vc[1:,*idx]=-1e10 # after retirement no labor participation 
                                                   
                        # compute the Pr. of of labor part. (wls) + before-taste-shock util Vw and Vm
                        before_taste_shock(par,i_Vc,i_Vw,i_Vm,idx,wls,Vw,Vm)
                        
              
                if (t<par.Tr):  #Eventual rebargaining + separation decisions happen below, *if not retired* 
                    #Eventual rebargaining happens below
                    for iA in range(par.num_A):        
                        
                        idx_s = (t,ih,iz,iA)
                        idxx = [(t,ih,iz,i,iL,iA) for i in range(par.num_power)]               
                        list_couple = (sol.Vw_couple, sol.Vm_couple)                 #couple        list
                        list_raw    = (Vw[ih,iz,:,iL,iA],Vm[ih,iz,:,iL,iA])          #remain-couple list
                        list_single = (sol.Vw_single[idx_s],sol.Vm_single[idx_s])    #single        list
                        iswomen     = (True,False)                                   #iswomen? in   list
                        
                        check_participation_constraints(par,sol.power,par.grid_power,list_raw,list_single,idxx,list_couple,iswomen)   
                       
    if (t>=par.Tr):sol.Vw_couple[t] = Vw.copy(); sol.Vm_couple[t]= Vm.copy() #copy utility if retired                                                  
    return (Vw,Vm,i_Vw,i_Vm,i_C_tot,wls) # return a tuple
       
@njit    
def compute_couple(par,sol,t,idx,pars2,EVw,EVm,wls,res,C_tot,Vw,Vm,Vc): 
 
    # indexes & initialization 
    idz=idx[:-1];iP=idx[2];iL=idx[3];love=par.grid_love[t][iL];power = par.grid_power[iP]
    C_pd,βEw,βEm,Vwd,Vmd,_= np.ones((6,par.num_A));pars=(par,sol,iP,wls,love,pars2)  
                  
    # discounted expected marginal utility from t+1, wrt assets
    βEVd=par.β*usr.deriv(par.grid_A,power*EVw[idz]+(1.0-power)*EVm[idz])

    # get consumption out of grid using FOCs (i) + use budget constraint to get current resources (ii)  
    linear_interp.interp_1d_vec(par.grid_marg_u_for_inv[wls,iP,:],par.grid_inv_marg_u,βEVd,C_pd) #(i) 
    A_now =  par.grid_A.flatten() + C_pd    
            
    #Apply upper envelope for optimal consumption and C-tot and Vx,Vm,Vc
    upper_envelope(par.grid_A,A_now,C_pd,par.β*EVw[idz],par.β*EVm[idz],power,res,C_tot[idx],Vw[idx],Vm[idx],Vc[idx],*pars) 
        
       
@njit 
def before_taste_shock(par,i_Vc,i_Vw,i_Vm,idx,wls,Vw,Vm):
 
    # get the probabilit of employment type in wls, based on couple utility choices
    i_idx=(slice(None),*idx)
    c=np.array([np.max(i_Vc[*i_idx[:-1],iA])/par.σ for iA in range(par.num_A)])# constant to avoid overflow
    v_couple=par.σ*np.euler_gamma+par.σ*(c+np.log(np.sum(np.exp(i_Vc[i_idx]/par.σ-c),axis=0)))
    wls[i_idx]=np.exp(i_Vc[i_idx]/par.σ-(v_couple-par.σ*np.euler_gamma)/par.σ) 
    
    # now the value of making the choice: see Shepard (2019), page 11
    Vw[idx]=v_couple+(1.0-par.grid_power[idx[2]])*np.sum(wls[i_idx]*(i_Vw[i_idx]-i_Vm[i_idx]),axis=0)
    Vm[idx]=v_couple+    (par.grid_power[idx[2]])*np.sum(wls[i_idx]*(i_Vm[i_idx]-i_Vw[i_idx]),axis=0)
    
    
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
            if ((min_Sw >= 0.0) & (min_Sm >= 0.0)) | (par.full): 
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
    love=sim.love;shock_love=sim.shock_love;iz=sim.iz;wlp=sim.WLP;incw=sim.incw;incm=sim.incm;ih=sim.ih;incwg=sim.incwg;incmg=sim.incmg
    dw=sim.dw;dm=sim.dm;Cw=sim.Cw;Cm=sim.Cm;Vsm=sim.Vsm;Vsw=sim.Vsw;Vcm=sim.Vcm;Vcw=sim.Vcw;tax=sim.tax

    for i in prange(par.simN):
        for t in range(par.simT):
 
            #Decide whether to iterate or not
            if t<par.sample_init[i]:continue
            
            # Copy variables from t-1 or initial condition. Initial (t>0) assets: preamble (later in the simulation)   
            Π = par.Πh[t][wlp[i,t-1]]                                                if t>0 else par.Πh[t][-1]
            ih[i,t] = usr.mc_simulate(ih[i,t-1],Π,sim.shock_h[i,t])                  if t>par.sample_init[i] else sim.init_ih[i]            
            couple_lag[i,t] = couple[i,t-1]                                          if t>par.sample_init[i] else sim.init_couple[i]
            power_lag[i,t] = power[i,t-1]                                            if t>par.sample_init[i] else sim.init_power[i]      
            Πz=par.Π[t-1]                                                            if (couple[i,t-1]==1) else par.Πs[t-1]
            iz[i,t] = usr.mc_simulate(iz[i,t-1],Πz,sim.shock_z[i,t])                 if t>par.sample_init[i] else sim.init_z[i]
            love[i,t] = usr.mc_simulate(love[i,t-1],par.Πl[t-1],shock_love[i,t])     if t>par.sample_init[i] else sim.init_love[i]
           
            # Indices of resources
            idx = (t,ih[i,t],iz[i,t],slice(None),love[i,t])
            
            # first check if they want to remain together and what the bargaining power will be if they do.
            if (couple_lag[i,t]) & (t<par.Tr):# do rebargaining power and divorce choice ifin a couple and not retired                 

                # Store before renegotiations utilities
                Vsw[i,t]=linear_interp.interp_1d(par.grid_Aw,sol.Vw_single[t,ih[i,t],iz[i,t]],Aw[i,t])
                Vsm[i,t]=linear_interp.interp_1d(par.grid_Am,sol.Vm_single[t,ih[i,t],iz[i,t]],Am[i,t])

                # Value of transitioning into singlehood
                list_single = (Vsw[i,t],Vsm[i,t])

                # Value of being ina  couple with given bargaining power
                list_raw    = (np.array([linear_interp.interp_1d(par.grid_A,sol.Vw_remain_couple[idx][iP],A[i,t]) for iP in range(par.num_power)]),
                                np.array([linear_interp.interp_1d(par.grid_A,sol.Vm_remain_couple[idx][iP],A[i,t]) for iP in range(par.num_power)]))

                # Rebargainings happens here
                check_participation_constraints(par,power,np.array([power_lag[i,t]]),list_raw,list_single,[(i,t)],nosim=False)
                couple[i,t] = False if power[i,t] <= -100.0 else True # partnership status: divorce is coded as -100
                    
            else: #divorce is an absorbing state
                
                couple[i,t] = couple_lag[i,t]; power[i,t] = power[i,t-1]#stay single or copy relationship if retired 
                            
            # update behavior
            if couple[i,t]:
                
                # Store before renegotiations utilities
                Vsw[i,t]=linear_interp.interp_1d(par.grid_Aw,sol.Vw_single[t,ih[i,t],iz[i,t]],Aw[i,t])
                Vsm[i,t]=linear_interp.interp_1d(par.grid_Am,sol.Vm_single[t,ih[i,t],iz[i,t]],Am[i,t])
                Vcw[i,t]=linear_interp.interp_2d(par.grid_power,par.grid_A,sol.Vw_remain_couple[idx],power[i,t],A[i,t])
                Vcm[i,t]=linear_interp.interp_2d(par.grid_power,par.grid_A,sol.Vm_remain_couple[idx],power[i,t],A[i,t])
                
                # First decide about labor participation, given employment probabilities part_i and draw from [0,1] uniform shock_taste
                part_i=np.array([linear_interp.interp_2d(par.grid_power,par.grid_A,sol.remain_WLP[t,wls,*idx[1:]],power[i,t],A[i,t]) for wls in range(par.num_wlp)])
                wlp[i,t]=usr.binary_search_event(part_i, sim.shock_taste[i,t])            
             
                # Optimal total consumption allocation if couple (note use of the updated index)
                sol_C_tot = sol.i_C_tot_remain_couple[t,wlp[i,t],*idx[1:]]
                C_tot[i,t] = linear_interp.interp_2d(par.grid_power,par.grid_A,sol_C_tot,power[i,t],A[i,t])

                # Obtain household resources
                M_resources_raw, incmt,incwt,incmgt,incwgt,taxc = usr.resources_couple(par,t,ih[i,t],iz[i,t],A[i,t])
                incm[i,t]=incmt[wlp[i,t]];incw[i,t]=incwt[wlp[i,t]];incmg[i,t]=incmgt;incwg[i,t]=incwgt[wlp[i,t]];tax[i,t]=taxc[wlp[i,t]]
                M_resources= M_resources_raw[wlp[i,t]] 
                
                if t< par.simT-1:A[i,t+1] = M_resources - C_tot[i,t]#
                if t< par.simT-1:Aw[i,t+1] =       par.div_A_share * A[i,t]# in case of divorce 
                if t< par.simT-1:Am[i,t+1] = (1.0-par.div_A_share) * A[i,t]# in case of divorce 
                
                # Obtain public and private consumption given total consumption Ctot
                Cw[i,t]=linear_interp.interp_2d(par.grid_power,par.grid_Ctot,sol.pre_Cw_priv[wlp[i,t]],sim.power[i,t],sim.C_tot[i,t])
                Cm[i,t]=linear_interp.interp_2d(par.grid_power,par.grid_Ctot,sol.pre_Cm_priv[wlp[i,t]],sim.power[i,t],sim.C_tot[i,t])
                dw[i,t]=sim.C_tot[i,t]-Cm[i,t]-Cw[i,t]
                dm[i,t]=sim.C_tot[i,t]-Cm[i,t]-Cw[i,t]
                
            else: # single
               
                # pick relevant solution for single
                sol_single_w = sol.Cw_tot_single[t,ih[i,t],iz[i,t]]
                sol_single_m = sol.Cm_tot_single[t,ih[i,t],iz[i,t]]
                
                #Store before renegotiations utilities
                Vsw[i,t]=linear_interp.interp_1d(par.grid_Aw,sol_single_w,Aw[i,t])
                Vsm[i,t]=linear_interp.interp_1d(par.grid_Am,sol_single_m,Am[i,t])

                # optimal consumption allocations
                Cw_tot[i,t] = linear_interp.interp_1d(par.grid_Aw,sol_single_w,Aw[i,t])
                Cm_tot[i,t] = linear_interp.interp_1d(par.grid_Am,sol_single_m,Am[i,t])   
                C_tot[i,t]  = Cw_tot[i,t] + Cm_tot[i,t]
                              
                Cw[i,t],dw[i,t] = usr.intraperiod_allocation_single(Cw_tot[i,t],par.ρ,par.χ,par.α,par.ν,par.ϕ,par.wedge)
                Cm[i,t],dm[i,t] = usr.intraperiod_allocation_single(Cm_tot[i,t],par.ρ,par.χ,par.α,par.ν,par.ϕ,par.wedge)

                #Labor supply
                wlp[i,t]=par.num_wlp-1 if t<par.Tr else 0
                
                #resources
                incw[i,t],incwg[i,t],taxsw=usr.income_single(par,t,ih[i,t],iz[i,t],Aw[i,t],women=True);incm[i,t],incmg[i,t],taxsm=usr.income_single(par,t,ih[i,t],iz[i,t],Am[i,t],women=False)
                tax[i,t]=taxsw+taxsm
                
                # update end-of-period states
                Mw = par.R*Aw[i,t] + incw[i,t] # total resources woman
                Mm = par.R*Am[i,t] + incm[i,t] # total resources man

                if t< par.simT-1: 
                    if par.women[i]: Aw[i,t+1] = Mw - Cw_tot[i,t]; Am[i,t+1] = Aw[i,t+1]*par.div_A_share
                    else:            Am[i,t+1] = Mm - Cm_tot[i,t]; Aw[i,t+1] = Am[i,t+1]*par.div_A_share
                    A[i,t+1]  = Aw[i,t+1] + Am[i,t+1] 
                    