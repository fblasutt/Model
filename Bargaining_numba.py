import numpy as np#import autograd.numpy as np#
from EconModel import EconModelClass, jit
from consav.grids import nonlinspace
from consav import linear_interp,upperenvelope
from numba import njit,prange,config
import UserFunctions_numba as usr
from quantecon.optimize.nelder_mead import nelder_mead
import setup


upper_envelope=usr.create(usr.couple_time_utility)
upp_env_single = upperenvelope.create(usr.single_time_util)

#general configuratiion and glabal variables (common across files)
config.DISABLE_JIT = setup.nojit;parallel=setup.parallel;cache=setup.cache
woman=setup.woman;man=setup.man

class HouseholdModelClass(EconModelClass):
    
    def settings(self):self.namespaces = []#needed to make class work, otherwise useless
                   
    def setup(self):
        par = self.par
        
        par.R = 1.03
        par.β = 0.98# Discount factor
        
        par.div_A_share = 0.5 # divorce share of wealth to wife
        
        par.full=False #dummy for full/limited commitment. if full commitment: renegotiation/divorce is illegal

        # Utility: CES aggregator or additively linear
        par.ρ = 2.0#        # CRRA      
        par.α1 = 0.65
        par.α2 = 0.35
        par.ϕ1 = 0.43
        par.ϕ2 = (1.0-par.ρ)/par.ϕ1
        
        # production of home good
        par.θ = 0.21 #weight on money vs. time to produce home good
        par.λ = 0.19 #elasticity betwen money and time in public good
        par.tb = 0.191 #time spend on public goods by singles
        
        #Taste shock
        par.σ = 0.0002 #taste shock applied to working/not working
        
        par.γ=0.5#woemn's relative power during nash bargaining
        

        
        ####################
        # state variables
        #####################
        
        par.T = 62 # terminal age
        par.Tr = 46 # age at retirement
        
        # wealth
        par.num_A = 15;par.max_A = 75.0
        
        # bargaining power
        par.num_power = 15
        par.power_min=1e-3
        par.power_max=1.0-par.power_min
        
        #women's human capital states
        par.num_h = 2
        par.drift = 0.0#1.84 #1.44716#1.84 #human capital depreciation drift
        par.pr_h_change = 0.0#2/46# 0.025 # probability that  human capital depreciates

        # love/match quality
        par.num_love = 7
        par.σL = 0.1; par.σL0 = 0.1
        
        # productivity of men and women: gridpoints
        par.num_ϵw=3;par.num_ϵm=3#transitory
        par.num_pw=3;par.num_pm=3#persistent
        par.num_zw=par.num_pw*par.num_ϵw;par.num_zm=par.num_pm*par.num_ϵw#total by gender
        par.num_z=par.num_zm*par.num_zw#total, couple
        
        # income of men and women: parameters of the age log-polynomial
        par.t0m=0.000;par.t1m=0.07095764 ;par.t2m= -0.00186095  ;par.t3m= 0.00000941  ;
        par.t0w=-0.728 ;par.t1w= 0.07851440  ;par.t2w=-0.00191743   ;par.t3w=0.00000941  ;
        #par.t0w=0.000;par.t1w=0.07095764 ;par.t2w= -0.00186095  ;par.t3w= 0.00000941  ;
        # productivity of men and women: sd of persistent (σpi), transitory (σϵi), initial (σ0i) income shocks
        par.σpw=0.16435;par.σϵw=0.08*1.5;par.σ0w= 0.37913;
        par.σpm=0.14162;par.σϵm=0.08*1.5;par.σ0m=0.38338
        #par.σpw=0.14162;par.σϵw=0.08*1.5;par.σ0w=0.38338
       
        
        

        # pre-computation fo consumption
        par.num_Ctot = 150;par.max_Ctot = par.max_A*2
        
        par.meet = 0.4#probability of meeting a partner if single
        par.relw = 1.0#1.125#ratio of men over women's assets at meeting
        par.ϕ = 0.5-(par.relw-1)/(par.relw+1)#given relw, share of womens assets at meeting
        
        # simulation
        par.seed = 9211;par.simT = par.T;par.simN = 100_000
       
        
    def setup_grids(self):
        par = self.par
        
        #Grid for the pr. of meeting a partner in each t
        par.λ_grid = np.ones(par.T)*par.meet
        for t in range(par.Tr,par.T):par.λ_grid[t]=0.0

        # wealth. Single grids are such to avoid interpolation
        par.grid_A = np.append(nonlinspace(0.0,par.max_A,par.num_A-1,2.1),par.max_A*10)#nonlinspace(0.0,par.max_A,par.num_A,1.1)
        par.grid_Aw =  par.grid_A * par.ϕ; par.grid_Am =  par.grid_A*(1.0-par.ϕ)

        # women's human capital grid plus transition if w working (p) or not (n)
        par.grid_h = np.flip(np.linspace(-par.num_h*par.drift,0.0,par.num_h))
        par.Πh_pt  =  np.eye(par.num_h)                                            #assumes 2 states for h
        par.Πh_nt  = np.array([[1.0-par.pr_h_change,par.pr_h_change],[0.0,1.0]]).T #assumes 2 states for h
        
        #No depreciation at retirement
        par.Πh_p = [par.Πh_pt*(t<par.Tr)+(t>=par.Tr)*np.eye(par.num_h)  for t in range(par.T)] 
        par.Πh_n = [par.Πh_nt*(t<par.Tr)+(t>=par.Tr)*np.eye(par.num_h)  for t in range(par.T)] 
        
        # bargaining power. non-linear grid with more mass in both tails.        
        par.grid_power = usr.grid_fat_tails(par.power_min,par.power_max,par.num_power)

        # love grid and shock    
        par.grid_love,par.Πl,par.Πl0= usr.addaco_nonst(par.T,par.σL,par.σL0,par.num_love)
        
        # grid for women's labor supply
        par.grid_wlp=np.array([0.0,1.0]);par.num_wlp=len(par.grid_wlp)
         
        # pre-computation of total consumption
        par.grid_Ctot = nonlinspace(1.0e-6,par.max_Ctot,par.num_Ctot,1.1)

        # marginal utility grids for EGM 
        par.grid_inv_marg_u = np.flip(par.grid_Ctot)
        par.grid_marg_u = np.nan + np.ones((par.num_wlp,par.num_power,par.num_Ctot))# couples
        par.grid_marg_uw = np.nan + np.ones((par.num_wlp,par.num_power,par.num_Ctot))# couples
        par.grid_marg_um = np.nan + np.ones((par.num_wlp,par.num_power,par.num_Ctot))# couples
        par.grid_marg_u_for_inv = np.nan + np.ones(par.grid_marg_u.shape)# couples
        par.grid_cpriv_s =  np.nan + np.ones(par.num_Ctot)# singles
        par.grid_marg_u_s = np.nan + np.ones(par.num_Ctot)# singles
   
        # income shocks grids: singles and couples
        par.grid_zw,par.grid_ϵw,par.grid_pw,par.Π_zw, par.Π_zw0 =\
            usr.labor_income(par.t0w,par.t1w,par.t2w,par.t3w,par.T,par.Tr,par.σpw,par.σϵw,par.σ0w,par.num_pw,par.num_ϵw,par.grid_h,women=True) 
        
        par.grid_zm,par.grid_ϵm,par.grid_pm,par.Π_zm, par.Π_zm0=\
            usr.labor_income(par.t0m,par.t1m,par.t2m,par.t3m,par.T,par.Tr,par.σpm,par.σϵw,par.σ0m,par.num_pm,par.num_ϵw,par.grid_h,women=False)
        
        par.Π=[np.kron(par.Π_zw[t],par.Π_zm[t]) for t in range(par.T-1)] # couples trans matrix, possible to change this and make shocks correlated

        #Simulation
        par.women =np.ones(par.simN)#0: simumate men 1 women
        
    def allocate(self):
        par = self.par;sol = self.sol;sim = self.sim;self.setup_grids()

        # setup grids
        par.simT = par.T
         
        # singles: value functions (vf), consumption, marg util
        shape_single = (par.T,par.num_h,par.num_z,par.num_A)
        sol.Vw_single = np.nan + np.ones(shape_single) #vf in t
        sol.Vm_single = np.nan + np.ones(shape_single) #vf in t
        sol.Cw_tot_single = np.nan + np.ones(shape_single) #priv+tot cons
        sol.Cm_tot_single = np.nan + np.ones(shape_single) #priv+tot cons

        # couples: value functions (vf), consumption, marg util, bargaining power
        shape_couple = (par.T,par.num_h,par.num_z,par.num_power,par.num_love,par.num_A) # 2 is for men/women
        sol.Vw_couple = np.nan + np.ones(shape_couple) #vf in t
        sol.Vm_couple = np.nan + np.ones(shape_couple) #vf in t
        sol.Vw_remain_couple = np.nan + np.ones(shape_couple) #vf|couple
        sol.Vm_remain_couple = np.nan + np.ones(shape_couple) #vf|couple
        sol.p_Vw_remain_couple = np.nan + np.ones(shape_couple)#vf|(couple+part)
        sol.p_Vm_remain_couple = np.nan + np.ones(shape_couple)#vf|(couple+part)
        sol.n_Vw_remain_couple = np.nan + np.ones(shape_couple)#vf|(couple+no part)
        sol.n_Vm_remain_couple = np.nan + np.ones(shape_couple)#vf|(couple+no part)      
        sol.p_C_tot_remain_couple = np.nan + np.ones(shape_couple)#cons|(couple+part)
        sol.n_C_tot_remain_couple = np.nan + np.ones(shape_couple)#cons|(couple+nopart)
        sol.remain_WLP = np.ones(shape_couple)#pr. of participation|couple   
        sol.power =  np.nan +np.zeros(shape_couple)                  #barg power value

        # pre-compute optimal consumption allocation: private and public
        shape_pre = (par.num_wlp,par.num_power,par.num_Ctot)
        sol.pre_Ctot_Cw_priv = np.nan + np.ones(shape_pre)
        sol.pre_Ctot_Cm_priv = np.nan + np.ones(shape_pre)
        sol.pre_Ctot_C_pub = np.nan + np.ones(shape_pre)
        
        # simulation
        shape_sim = (par.simN,par.simT)
        sim.C_tot = np.nan + np.ones(shape_sim)         # total consumption
        sim.Cw_tot = np.nan + np.ones(shape_sim)        # total consumption w
        sim.Cm_tot = np.nan + np.ones(shape_sim)        # total consumption m
        
        sim.Cm = np.nan + np.ones(shape_sim)        # private consumption m
        sim.Cw = np.nan + np.ones(shape_sim)        # private consumption w
        sim.xm = np.nan + np.ones(shape_sim)        # public expenditure m
        sim.xw = np.nan + np.ones(shape_sim)        # public expenditure w
        
        sim.Vsw = np.nan + np.ones(shape_sim)       # value function if divorce w
        sim.Vsm = np.nan + np.ones(shape_sim)       # value function if divorce m
        sim.Vcw = np.nan + np.ones(shape_sim)       # before-ren value function w
        sim.Vcm = np.nan + np.ones(shape_sim)       # before-ren value function m
        
        sim.iz = np.ones(shape_sim,dtype=np.int_)       # index of income shocks 
        sim.A = np.nan + np.ones(shape_sim)             # total assets (m+w)
        sim.Aw = np.nan + np.ones(shape_sim)            # w's assets
        sim.Am = np.nan + np.ones(shape_sim)            # m's assets
        sim.couple = np.zeros(shape_sim,dtype=bool)        # In a couple? True/False
        sim.couple_lag = np.zeros(shape_sim,dtype=bool)    # In a couple? True/False
        sim.power = -100.0*np.ones(shape_sim)         # barg power
        sim.power_lag = -100.0*np.ones(shape_sim)              # barg power index
        sim.love = np.ones(shape_sim,dtype=np.int_)     # love
        sim.incw = np.nan + np.ones(shape_sim)          # w's income
        sim.incm = np.nan + np.ones(shape_sim)          # m's income
        sim.WLP = np.ones(shape_sim,dtype=np.int_)      # w's labor participation
        sim.ih = np.ones(shape_sim,dtype=np.int_)       # w's human capital

        # shocks
        np.random.seed(par.seed)
        sim.shock_love = np.random.random_sample((par.simN,par.simT))#love
        sim.shock_iz=np.random.random_sample((par.simN,2))           #initial income 
        sim.shock_z=np.random.random_sample((par.simN,par.simT))     #income
        sim.shock_taste=np.random.random_sample((par.simN,par.simT)) #taste shock
        sim.shock_h=np.random.random_sample((par.simN,par.simT))     #human capital
        sim.shock_meet=np.random.random_sample((par.simN,par.simT))  #meeting
        
        # initial distribution
        sim.init_ih = np.zeros(par.simN,dtype=np.int_)                  #initial w's human capital
        sim.A[:,0] = par.grid_A[0] + np.zeros(par.simN)                 #total assetes
        sim.Aw[:,0] = par.div_A_share * sim.A[:,0]                      #w's assets
        sim.Am[:,0] = (1.0 - par.div_A_share) * sim.A[:,0]              #m's assets
        sim.init_couple = np.zeros(par.simN,dtype=bool)                  #state (couple=1/single=0)
        sim.init_power =  0.11373989**np.ones(par.simN)                 #barg power 
        sim.init_love = np.ones(par.simN,dtype=np.int_)**par.num_love//2#initial love    
        sim.init_zw =np.array([usr.mc_simulate(par.num_zw//2,par.Π_zw0[0],sim.shock_iz[i,0]) for i in range(par.simN)])
        sim.init_zm =np.array([usr.mc_simulate(par.num_zm//2,par.Π_zm0[0],sim.shock_iz[i,1]) for i in range(par.simN)])
        sim.init_z  = sim.init_zw*par.num_zm+sim.init_zm                #    initial income
        
           
            
     
                        
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
    C_pub,  Cw_priv, Cm_priv, grid_marg_u, grid_marg_u_for_inv, grid_marg_u_s, grid_cpriv_s, grid_marg_uw, grid_marg_um =\
        sol.pre_Ctot_C_pub, sol.pre_Ctot_Cw_priv, sol.pre_Ctot_Cm_priv, par.grid_marg_u, par.grid_marg_u_for_inv, par.grid_marg_u_s,\
        par.grid_cpriv_s, par.grid_marg_uw, par.grid_marg_um
        
    pars=(par.ρ,par.ϕ1,par.ϕ2,par.α1,par.α2,par.θ,par.λ,par.tb)  
    ϵ = 1e-8# delta increase in xs to compute numerical deratives

    ################ Singles part #####################
    for i,C_tot in enumerate(par.grid_Ctot):
        
        # optimize to get util from total consumption(m<->C_tot)=private cons(c)+public cons(m-c)
        grid_cpriv_s[i] = usr.optimizer(lambda c,m,p:-usr.util(c,m-c,*p),ϵ,C_tot-ϵ,args=(C_tot,pars))[0]
        
        # numerical derivative of util wrt total consumption C_tot, using envelope thm
        share_priv=grid_cpriv_s[i]/C_tot
        forward  = usr.util(share_priv*(C_tot+ϵ),(1.0-share_priv)*(C_tot+ϵ),*pars)
        backward = usr.util(share_priv*(C_tot-ϵ),(1.0-share_priv)*(C_tot-ϵ),*pars)
        grid_marg_u_s[i] = (forward - backward)/(2*ϵ)
  
    ################ Couples part ##########################   
    icon=np.array([0.33,0.33])#initial condition, to be overwritten 
    for iP in prange(par.num_power):      
        for iwlp,wlp in enumerate(par.grid_wlp): 
            for i,C_tot in enumerate(par.grid_Ctot): 
                 
                # initialize bounds and bargaining power 
                bounds=np.array([[0.0,C_tot],[0.0,C_tot]]);power=par.grid_power[iP] 
                 
                # estimate optima private and public cons, unpack, update initial condition
                res = nelder_mead(lambda c,p:usr.couple_util(c,*p)[0],icon*C_tot,bounds=bounds,args=((C_tot,power,1.0-wlp,*pars),)) 
                Cw_priv[iwlp,iP,i]= res.x[0];Cm_priv[iwlp,iP,i]= res.x[1];C_pub[iwlp,iP,i] = C_tot - res.x.sum()             
                icon=res.x/C_tot if i<par.num_Ctot-1 else np.array([0.33,0.33]) 
       
                # numerical derivative of util wrt total consumption C_tot, using envelope thm 
                _,forw_w,forw_m = usr.couple_util(res.x/(C_tot)*(C_tot+ϵ),C_tot+ϵ,power,1.0-wlp,*pars) 
                _,bakw_w,bakw_m = usr.couple_util(res.x/(C_tot)*(C_tot-ϵ),C_tot-ϵ,power,1.0-wlp,*pars) 
                grid_marg_uw[iwlp,iP,i] = (forw_w - bakw_w)/(2*ϵ);grid_marg_um[iwlp,iP,i] = (forw_m - bakw_m)/(2*ϵ)
                                  
            #Create grid of couple's marginal util and inverse marginal utility  
            grid_marg_u[iwlp,iP,:] = power*grid_marg_uw[iwlp,iP,:]+(1.0-power)*grid_marg_um[iwlp,iP,:] 
            grid_marg_u_for_inv[iwlp,iP,:]=np.flip(par.grid_marg_u[iwlp,iP,:])   
        
#######################
# SOLUTIONS - SINGLES #
#######################

@njit(parallel=parallel)
def integrate_single(sol,par,t):
    Ew_nomeet,Em_nomeet,Ew_meet,Em_meet=np.zeros((4,par.num_h,par.num_z,par.num_A)) 
     
    # 1. Expected value if not meeting a partner
    for iA in prange(par.num_A):
        for iz in range(par.num_z):
            for ih in range(par.num_h):
                for jz in range(par.num_z):
                                      
                    Ew_nomeet[ih,iz,iA] += sol.Vw_single[t+1,:,jz,iA] @ par.Πh_p[t][:,ih] * par.Π[t][jz,iz]
                    Em_nomeet[ih,iz,iA] += sol.Vm_single[t+1,:,jz,iA] @ par.Πh_p[t][:,ih] * par.Π[t][jz,iz]                    
                    
    # 2. Expected value if meeting a partner: Π=kroneker product of love, wage, HC risk.
    Π=usr.rescale_matrix(np.kron(np.kron(par.Πh_p[t],par.Π[t]),par.Πl0[t])) 
    for iA in prange(par.num_A):
        for jz in range(par.num_z):
            for jh in range(par.num_h):
                for jL in range(par.num_love):
                    
                   #indices (mean-zero love shock for w and m: assumes symmetry)
                    iL = par.num_love//2;cjx=(t+1,jh,jz,slice(None),jL,iA)                   
                    a_cjx =jh*par.num_love*par.num_z + jz*par.num_love+jL;sjx = (t+1,jh,jz,iA);
                    
                    #w and m value of max(partner,single) for cjx,sjx
                    vwt,vmt,_,_=marriage_mkt(par,sol.Vw_remain_couple[cjx],sol.Vm_remain_couple[cjx],
                                                 sol.Vw_single[sjx],sol.Vm_single[sjx])                  
                    for iz in range(par.num_z):
                        for ih in range(par.num_h):
                                                      
                            a_cix =   ih*par.num_love*par.num_z + iz*par.num_love+iL
                            
                            Ew_meet[ih,iz,iA]+= vwt * Π[a_cjx,a_cix]
                            Em_meet[ih,iz,iA]+= vmt * Π[a_cjx,a_cix]
                            
    # 3. Return expected value given meeting probabilities                                              
    return par.λ_grid[t]*Ew_meet+(1.0-par.λ_grid[t])*Ew_nomeet, par.λ_grid[t]*Em_meet+(1.0-par.λ_grid[t])*Em_nomeet

@njit
def marriage_mkt(par,vcw,vcm,vsw,vsm):
       
    wp = vcw - vsw; mp = vcm - vsm # surplus of being in a couple by pareto weight and gender
              
    if (wp[-1]<0) | (mp[0]<0): return vsw,vsm,-100.0, False # negative surplus for any b. power
    else:

        θmin = np.maximum(linear_interp.interp_1d(vcw,      par.grid_power,      vsw),par.power_min)
        θmax = np.minimum(linear_interp.interp_1d(vcm[::-1],par.grid_power[::-1],vsm),par.power_max)
        
        if θmin>θmax: return vsw,vsm,-100.0,False #again, negative surplus for any b. power
        else:#find the the b. powdr that maximizes symmetric nash bargaining
        
            θ, v = usr.optimizer(nash_bargaining,θmin,θmax,args=(par.grid_power,wp,mp,par.γ))
            vcwi = linear_interp.interp_1d(par.grid_power,vcw,θ)
            vcmi = linear_interp.interp_1d(par.grid_power,vcm,θ)  
            
                
            
            return vcwi,vcmi,θ,True

@njit
def nash_bargaining(x,xgrid,wp,mp,γ):
    
   wpθ=linear_interp.interp_1d(xgrid,wp,x)
   mpθ=linear_interp.interp_1d(xgrid,mp,x)
   return  -wpθ**γ*mpθ**(1.0-γ)#-γ*np.log(wpθ)-(1.0-γ)*np.log(mpθ)#
    
@njit(parallel=parallel)
def solve_single_egm(sol,par,t):

    #Integrate to get continuation value unless if you are in the last period
    Ew,Em=np.zeros((2,par.num_h,par.num_zw,par.num_A))
    if t<par.T-1:Ew,Em = integrate_single(sol,par,t) #if t<par.T-1 else 
             
    #Pre-define outcomes (if update .sol directly, parallelization go crazy)
    cwt,Ewt,cwp,cmt,Emt,cmp=np.ones((6,par.num_h,par.num_z,par.num_A))
    pars=(par.ρ,par.ϕ1,par.ϕ2,par.α1,par.α2,par.θ,par.λ,par.tb)
    
    #function to find optimal savings, called for both men and women below
    def loop_savings_singles(par,grid_Ai,ci,Ei,cit,Eit,cip,vi,women,divorce):
        
        for iz in prange(par.num_z):
            for ih in range(par.num_h):

                resi = par.R*grid_Ai+usr.income_single(par,t,ih,iz,grid_Ai,women)
                
                if t==(par.T-1): 
                    
                    ci[ih,iz,:] = resi.copy() #consume all resources
                    linear_interp.interp_1d_vec(par.grid_Ctot,par.grid_cpriv_s,ci[ih,iz,:],cip[ih,iz,:])#private cons
                    vi[ih,iz,:]=usr.util(cip[ih,iz,:],ci[ih,iz,:]-cip[ih,iz,:],*pars)#util
                    
                else: #before T-1 make consumption saving choices
                    
                    # marginal utility of assets next period
                    βEid=par.β*usr.deriv(grid_Ai,Ei[ih,iz,:])
                    
                    # first get toatl -consumption out of grid using FOCs
                    linear_interp.interp_1d_vec(np.flip(par.grid_marg_u_s),par.grid_inv_marg_u,βEid,cit[ih,iz,:])
                    
                    # use budget constraint to get current resources
                    Ri_now = grid_Ai.flatten() + cit[ih,iz,:]
                           
                    # use the upper envelope algorithm to get optimal consumption and util
                    upp_env_single(grid_Ai,Ri_now,cit[ih,iz,:],par.β*Ei[ih,iz,:],resi,ci[ih,iz,:],vi[ih,iz,:],*pars)

    loop_savings_singles(par,par.grid_Aw,sol.Cw_tot_single[t],Ew,cwt,Ewt,cwp,sol.Vw_single[t],True,False) #savings
    loop_savings_singles(par,par.grid_Am,sol.Cm_tot_single[t],Em,cmt,Emt,cmp,sol.Vm_single[t],False,False)#savings
               
    #loop_savings_singles(par,par.grid_Aw,sol.Cw_tot_divorce[t],Ew,cwt,Ewt,cwp,sol.Vw_divorce[t],True,True) #savings 
    #loop_savings_singles(par,par.grid_Am,sol.Cm_tot_divorce[t],Em,cmt,Emt,cmp,sol.Vm_divorce[t],False,True)#savings 
    
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
     
    pEVw,pEVm,nEVw,nEVm=np.zeros((4,par.num_h,par.num_z,par.num_power,par.num_love,par.num_A)) 

    #kroneker product of uncertainty in love, income, human capital
    Πp=np.kron(np.kron(par.Πh_p[t],par.Π[t]),par.Πl[t]);Πn=np.kron(np.kron(par.Πh_n[t],par.Π[t]),par.Πl[t])
    to_pass=(Πp==0.0) & (Πn==0.0)#whether kroneker product is 0 and does not contribute to EV
    
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
                                    
                                    idx=(ih,iz,iP,iL,iA);jdx=(t+1,jh,jz,iP,jL,iA)
                                        
                                    pEVw[idx]+= sol.Vw_couple[jdx]*Πp[zjdx,zdx]
                                    nEVw[idx]+= sol.Vw_couple[jdx]*Πn[zjdx,zdx]
                                    pEVm[idx]+= sol.Vm_couple[jdx]*Πp[zjdx,zdx]
                                    nEVm[idx]+= sol.Vm_couple[jdx]*Πn[zjdx,zdx]                     
    return pEVw,pEVm,nEVw,nEVm

@njit(parallel=parallel) 
def solve_remain_couple_egm(par,sol,t): 
               
    #Integration if not last period
    if t<(par.T-1): pEVw,pEVm,nEVw,nEVm = integrate_couple(par,sol,t)


    # initialize 
    Vw,Vm,p_Vw,p_Vm,n_Vw,n_Vm,p_Vc,n_Vc,p_C_tot,n_C_tot,wlp\
        =np.zeros((11,par.num_h,par.num_z,par.num_power,par.num_love,par.num_A)) 
    pars=(par.ρ,par.ϕ1,par.ϕ2,par.α1,par.α2,par.θ,par.λ,par.tb)     
    for iL in prange(par.num_love): 
        for ih in range(par.num_h):
            for iz in range(par.num_z):
                for iP in range(par.num_power):
                
                    # indexes
                    idx=(ih,iz,iP,iL,slice(None))
                                      
                    # resources if no participation (n_) or participation (_p)
                    n_res, p_res,_,_=usr.resources_couple(par,t,ih,iz,par.grid_A) 
                    
                    # continuation values 
                    if t==(par.T-1):#last period 
                        
                        #Get consumption then utilities (assume no labor participation). Note: no savings!
                        Vw[idx],Vm[idx]=usr.couple_time_utility(n_res,par,sol,iP,0,par.grid_love[t][iL],pars)            
                        wlp[idx]=0.0;p_Vm[idx]=p_Vw[idx]=-1e10;n_Vw[idx],n_Vm[idx]=Vw[idx],Vm[idx];n_C_tot[idx] = n_res.copy() 
                                            
                    else:#periods before the last 
                                 
                        # compute consumption* and util given partecipation (0/1). last 4 arguments below are output at iz,iL,iP
                        compute_couple(par,sol,t,idx,pars,pEVw,pEVm,1,p_res,p_C_tot,p_Vw,p_Vm,p_Vc) # participation 
                        compute_couple(par,sol,t,idx,pars,nEVw,nEVm,0,n_res,n_C_tot,n_Vw,n_Vm,n_Vc) # no participation 
                        if (t>=par.Tr):p_Vw[idx]=p_Vm[idx]=p_Vc[idx]=-1e10 # before retirement no labor participation 
                                                   
                        # compute the Pr. of of labor part. (wlp) + before-taste-shock util Vw and Vm
                        before_taste_shock(par,p_Vc,n_Vc,p_Vw,n_Vw,p_Vm,n_Vm,idx,wlp,Vw,Vm)
        
                if (t<par.Tr):  #Eventual rebargaining + separation decisions happen below, *if not retired* 
                    #Eventual rebargaining happens below
                    for iA in range(par.num_A):        
                        
                        idx_s = (t,ih,iz,iA)
                        idxx = [(t,ih,iz,i,iL,iA) for i in range(par.num_power)]               
                        list_couple = (sol.Vw_couple, sol.Vm_couple)                 #couple        list
                        list_raw    = (Vw[ih,iz,:,iL,iA],Vm[ih,iz,:,iL,iA])          #remain-couple list
                        list_single = (sol.Vw_single[idx_s],sol.Vm_single[idx_s])#single        list
                        iswomen     = (True,False)                                   #iswomen? in   list
                        
                        check_participation_constraints(par,sol.power,par.grid_power,list_raw,list_single,idxx,list_couple,iswomen)   
                        
    if (t>=par.Tr):sol.Vw_couple[t] = Vw.copy(); sol.Vm_couple[t]= Vm.copy() #copy utility if retired                                                  
    return (Vw,Vm,p_Vw,p_Vm,n_Vw,n_Vm,p_C_tot,n_C_tot,wlp) # return a tuple
       
@njit    
def compute_couple(par,sol,t,idx,pars2,EVw,EVm,part,res,C_tot,Vw,Vm,Vc): 
 
    # indexes & initialization 
    idz=idx[:-1];iP=idx[2];iL=idx[3];love=par.grid_love[t][iL];power = par.grid_power[iP]
    C_pd,βEw,βEm,Vwd,Vmd,_= np.ones((6,par.num_A));pars=(par,sol,iP,part,love,pars2)  
                  
    # discounted expected marginal utility from t+1, wrt assets
    βEVd=par.β*usr.deriv(par.grid_A,power*EVw[idz]+(1.0-power)*EVm[idz])

    # get consumption out of grid using FOCs (i) + use budget constraint to get current resources (ii)  
    linear_interp.interp_1d_vec(par.grid_marg_u_for_inv[part,iP,:],par.grid_inv_marg_u,βEVd,C_pd) #(i) 
    A_now =  par.grid_A.flatten() + C_pd                                                         #(ii) 
 
    if np.any(np.diff(A_now)<0):#apply upperenvelope + enforce no borrowing constraint 
 
        upper_envelope(par.grid_A,A_now,C_pd,par.β*EVw[idz],par.β*EVm[idz],power,res,C_tot[idx],Vw[idx],Vm[idx],Vc[idx],*pars) 
        
    else:#upperenvelope not necessary: enforce no borrowing constraint 
     
        # interpolate onto common beginning-of-period asset grid to get consumption 
        linear_interp.interp_1d_vec(A_now,C_pd,res,C_tot[idx]) 
        C_tot[idx] = np.minimum(C_tot[idx] , res) #...+ apply borrowing constraint 
     
        # compute the value function 
        Cw_priv, Cm_priv, C_pub =\
            usr.intraperiod_allocation(C_tot[idx],par.grid_Ctot,sol.pre_Ctot_Cw_priv[part,iP],sol.pre_Ctot_Cm_priv[part,iP])  
             
        linear_interp.interp_1d_vec(par.grid_A,par.β*EVw[idz],res-C_tot[idx],βEw) 
        linear_interp.interp_1d_vec(par.grid_A,par.β*EVm[idz],res-C_tot[idx],βEm)     
        Vw[idx] = usr.util(Cw_priv,C_pub,*pars2,love,True,1.0-par.grid_wlp[part])+βEw                         
        Vm[idx] = usr.util(Cm_priv,C_pub,*pars2,love,True,1.0-par.grid_wlp[part])+βEm 
        Vc[idx] = power*Vw[idx]+(1.0-power)*Vm[idx]
       
@njit
def before_taste_shock(par,p_Vc,n_Vc,p_Vw,n_Vw,p_Vm,n_Vm,idx,wlp,Vw,Vm):
 
    # get the probabilit of workin wlp, based on couple utility choices
    c=np.maximum(p_Vc[idx],n_Vc[idx])/par.σ # constant to avoid overflow
    v_couple=par.σ*(c+np.log(np.exp(p_Vc[idx]/par.σ-c)+np.exp(n_Vc[idx]/par.σ-c)))
    wlp[idx]=np.exp(p_Vc[idx]/par.σ-v_couple/par.σ)
    
    # now the value of making the choice: see Shepard (2019), page 11
    Δp=p_Vw[idx]-p_Vm[idx];Δn=n_Vw[idx]-n_Vm[idx]
    Vw[idx]=v_couple+(1.0-par.grid_power[idx[2]])*(wlp[idx]*Δp+(1.0-wlp[idx])*Δn)
    Vm[idx]=v_couple+(-par.grid_power[idx[2]])   *(wlp[idx]*Δp+(1.0-wlp[idx])*Δn) 
    
@njit
def check_participation_constraints(par,solpower,gridpower,list_raw,list_single,idx,
                                    list_couple=(np.zeros((1,1)),),iswomen=(True,),nosim=True):
                 
    # surplus of marriage, then its min and max given states
    Sw = list_raw[0] - list_single[0] 
    Sm = list_raw[1] - list_single[1] 
    min_Sw = np.min(Sw);min_Sm = np.min(Sm)
    max_Sw = np.max(Sw);max_Sm = np.max(Sm) 

    # if expect rebargaining, interpolate the surplus of each member at indifference points
    #if ~((min_Sw >= 0.0) & (min_Sm >= 0.0)) & ~((max_Sw < 0.0) | (max_Sm < 0.0)):             

    power_at_0_w = linear_interp.interp_1d(Sw      ,par.grid_power      ,0.0)   
    power_at_0_m = linear_interp.interp_1d(Sm[::-1],par.grid_power[::-1],0.0)          

    Sm_at_0_w = linear_interp.interp_1d(par.grid_power, Sm, power_at_0_w)   
    Sw_at_0_m = linear_interp.interp_1d(par.grid_power, Sw, power_at_0_m)   
      
        
    ##################################################################
    # For a given power, find out if marriage, divorce or rebargaining
    # Then, update power and (if no simulation) update value functions
    #################################################################
    for iP,power in enumerate(gridpower):


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
def store(Vw,Vm,p_Vw,p_Vm,n_Vw,n_Vm,p_C_tot,n_C_tot, wlp,par,sol,t):
                
    sol.p_C_tot_remain_couple[t] = p_C_tot
    sol.n_C_tot_remain_couple[t] = n_C_tot
    sol.Vw_remain_couple[t] = Vw
    sol.Vm_remain_couple[t] = Vm
    sol.p_Vw_remain_couple[t] = p_Vw
    sol.p_Vm_remain_couple[t] = p_Vm
    sol.n_Vw_remain_couple[t] = n_Vw
    sol.n_Vm_remain_couple[t] = n_Vm
    sol.remain_WLP[t] = wlp
                 
##################################
#        SIMULATIONS
#################################

@njit(parallel=parallel)
def simulate_lifecycle(sim,sol,par):
    
    # unpacking some values to help numba optimize
    A=sim.A;Aw=sim.Aw;Am=sim.Am;couple=sim.couple;power=sim.power;C_tot=sim.C_tot;Cm_tot=sim.Cm_tot;Cw_tot=sim.Cw_tot;couple_lag=sim.couple_lag;power_lag=sim.power_lag
    love=sim.love;shock_love=sim.shock_love;iz=sim.iz;wlp=sim.WLP;incw=sim.incw;incm=sim.incm;ih=sim.ih;
    xw=sim.xw;xm=sim.xm;Cw=sim.Cw;Cm=sim.Cm;
    Vsm=sim.Vsm;Vsw=sim.Vsw;Vcm=sim.Vcm;Vcw=sim.Vcw;
    
    interp2d = lambda a,b,c : linear_interp.interp_2d(par.grid_power,par.grid_A,a,b,c)
    interp1d = lambda a,b   : linear_interp.interp_1d(par.grid_A,               a,b) 
    
    for i in prange(par.simN):
        for t in range(par.simT):
              
            # Copy variables from t-1 or initial condition. Initial (t>0) assets: preamble (later in the simulation)   
            Π = [par.Πh_p[t],par.Πh_n[t]][1- wlp[i,t-1]]                             if t>0  else par.Πh_p[t]
            ih[i,t] = usr.mc_simulate(ih[i,t-1],Π,sim.shock_h[i,t])                  if t>0 else sim.init_ih[i]            
            couple_lag[i,t] = couple[i,t-1]                                          if t>0 else sim.init_couple[i]
            power_lag[i,t] = power[i,t-1]                                            if t>0 else sim.init_power[i]           
            iz[i,t] = usr.mc_simulate(iz[i,t-1],par.Π[t-1],sim.shock_z[i,t])         if t>0 else sim.init_z[i]
            love[i,t] = usr.mc_simulate(love[i,t-1],par.Πl[t-1],shock_love[i,t])     if t>0 else sim.init_love[i]
           


            # indices and resources
            idx = (t,ih[i,t],iz[i,t],slice(None),love[i,t])
            incw[i,t]=usr.income_single(par,t,ih[i,t],iz[i,t],Aw[i,t],women=True);incm[i,t]=usr.income_single(par,t,ih[i,t],iz[i,t],Am[i,t],women=False)          
            M_resourcesNW,M_resourcesW, incmt,incwt = usr.resources_couple(par,t,ih[i,t],iz[i,t],A[i,t])
            
            # first check if they want to remain together and what the bargaining power will be if they do.
            if (couple_lag[i,t]) & (t<par.Tr):# do rebargaining power and divorce choice ifin a couple and not retired                 

                #Store before renegotiations utilities
                Vsw[i,t]=linear_interp.interp_1d(par.grid_Aw,sol.Vw_single[t,ih[i,t],iz[i,t]],Aw[i,t])
                Vsm[i,t]=linear_interp.interp_1d(par.grid_Am,sol.Vm_single[t,ih[i,t],iz[i,t]],Am[i,t])


                # value of transitioning into singlehood
                list_single = (Vsw[i,t],Vsm[i,t])

                list_raw    = (np.array([interp1d(sol.Vw_remain_couple[idx][iP],A[i,t]) for iP in range(par.num_power)]),
                               np.array([interp1d(sol.Vm_remain_couple[idx][iP],A[i,t]) for iP in range(par.num_power)]))

                check_participation_constraints(par,power,np.array([power_lag[i,t]]),list_raw,list_single,[(i,t)],nosim=False)
                
                
                couple[i,t] = False if power[i,t] <= -100.0 else True # partnership status: divorce is coded as -100
                    
            else: #meet a partner if single, eventually enter relationship
                
                
                if (sim.shock_meet[i,t]>par.λ_grid[t]): # same status as t-1 if meeting did not happen
                
                    couple[i,t] = couple_lag[i,t]; power[i,t] = power[i,t-1]#stay single or copy relationship if retired 
                    
                else:  #while meetin happens, continue below

                    # Utility as a single and as a couple for main individual and potential partner
                    Vsw_ = linear_interp.interp_1d(par.grid_Aw,sol.Vw_single[t,ih[i,t],iz[i,t]],Aw[i,t])
                    Vsm_ = linear_interp.interp_1d(par.grid_Am,sol.Vm_single[t,ih[i,t],iz[i,t]],Am[i,t])
                       
                    Vcw_ =  np.array([interp2d(sol.Vw_remain_couple[idx],powe,A[i,t]) for powe in par.grid_power])
                    Vcm_ =  np.array([interp2d(sol.Vm_remain_couple[idx],powe,A[i,t]) for powe in par.grid_power])
                    
                    #Do Nash bargaining to decide whether to marry and eventually set initial Pareto wight
                    vcwi,vcmi ,power[i,t],couple[i,t] = marriage_mkt(par,Vcw_,Vcm_,Vsw_,Vsm_)
                    


            # update behavior
            if couple[i,t]:
                
                #Store before renegotiations utilities
                Vsw[i,t]=linear_interp.interp_1d(par.grid_Aw,sol.Vw_single[t,ih[i,t],iz[i,t]],Aw[i,t])
                Vsm[i,t]=linear_interp.interp_1d(par.grid_Am,sol.Vm_single[t,ih[i,t],iz[i,t]],Am[i,t])
                Vcw[i,t]=interp2d(sol.Vw_remain_couple[idx],power[i,t],A[i,t])
                Vcm[i,t]=interp2d(sol.Vm_remain_couple[idx],power[i,t],A[i,t])
                

                # first decide about labor participation (if not taste shocks this should be different) 
                part_i=interp2d(sol.remain_WLP[idx],power[i,t],A[i,t])#
                wlp[i,t]=(part_i>sim.shock_taste[i,t])
                               
                # optimal consumption allocation if couple (note use of the updated index)
                sol_C_tot = sol.p_C_tot_remain_couple[idx] if wlp[i,t] else sol.n_C_tot_remain_couple[idx] 
                C_tot[i,t] = interp2d(sol_C_tot,power[i,t],A[i,t])

                # update end-of-period states
                M_resources= M_resourcesW if  wlp[i,t] else  M_resourcesNW 
                if t< par.simT-1:A[i,t+1] = M_resources - C_tot[i,t]#
                if t< par.simT-1:Aw[i,t+1] =       par.div_A_share * A[i,t]# in case of divorce 
                if t< par.simT-1:Am[i,t+1] = (1.0-par.div_A_share) * A[i,t]# in case of divorce 
                
                Cw[i,t]=linear_interp.interp_2d(par.grid_power,par.grid_Ctot,sol.pre_Ctot_Cw_priv[wlp[i,t]],sim.power[i,t],sim.C_tot[i,t])
                Cm[i,t]=linear_interp.interp_2d(par.grid_power,par.grid_Ctot,sol.pre_Ctot_Cm_priv[wlp[i,t]],sim.power[i,t],sim.C_tot[i,t])
                xw[i,t]=sim.C_tot[i,t]-Cm[i,t]-Cw[i,t]
                xm[i,t]=sim.C_tot[i,t]-Cm[i,t]-Cw[i,t]
      
               
            else: # single
               
                # pick relevant solution for single
                sol_single_w = sol.Cw_tot_single[t,ih[i,t],iz[i,t]]
                sol_single_m = sol.Cm_tot_single[t,ih[i,t],iz[i,t]]

                # optimal consumption allocations
                Cw_tot[i,t] = linear_interp.interp_1d(par.grid_Aw,sol_single_w,Aw[i,t])
                Cm_tot[i,t] = linear_interp.interp_1d(par.grid_Am,sol_single_m,Am[i,t])   
                C_tot[i,t] = Cw_tot[i,t] + Cm_tot[i,t]
                              
                Cw[i,t],xw[i,t] = usr.intraperiod_allocation_single(Cw_tot[i,t],par.ρ,par.ϕ1,par.ϕ2,par.α1,par.α2,par.θ,par.λ,par.tb)
                Cm[i,t],xw[i,t] = usr.intraperiod_allocation_single(Cm_tot[i,t],par.ρ,par.ϕ1,par.ϕ2,par.α1,par.α2,par.θ,par.λ,par.tb)


                # update end-of-period states
                Mw = par.R*Aw[i,t] + incw[i,t] # total resources woman
                Mm = par.R*Am[i,t] + incm[i,t] # total resources man

                if t< par.simT-1: 
                    if par.women[i]: Aw[i,t+1] = Mw - Cw_tot[i,t]; Am[i,t+1] = Aw[i,t+1]*par.relw   
                    else:            Am[i,t+1] = Mm - Cm_tot[i,t]; Aw[i,t+1] = Am[i,t+1]/par.relw
                    A[i,t+1]  = Aw[i,t+1] + Am[i,t+1] 
                    
  