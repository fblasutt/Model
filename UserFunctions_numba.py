from numba import njit
from numba_stats import norm
import numpy as np
from numba import config 
from consav import linear_interp
from consav.grids import nonlinspace
from consav.markov import rouwenhorst
from scipy.stats import multivariate_normal
from scipy.integrate import quad

import setup

#general configuratiion and glabal variables  (common across files)
config.DISABLE_JIT = setup.nojit;parallel=setup.parallel;cache=setup.cache
woman=setup.woman;man=setup.man

############################
# User-specified functions #
############################
@njit(cache=cache)
def home_good(x,θ,λ,tb,couple=0.0,ishom=0.0):
    home_time=(2*tb+ishom*(1-tb)) if couple else tb
    return (θ*x**λ+(1.0-θ)*home_time**λ)**(1.0/λ)

@njit(cache=cache)
def util(c_priv,c_pub,ρ,ϕ1,ϕ2,α1,α2,θ,λ,tb,love=0.0,couple=0.0,ishom=0.0):
    homegood=home_good(c_pub,θ,λ,tb,couple=couple,ishom=ishom)
    return ((α1*c_priv**ϕ1 + α2*homegood**ϕ1)**ϕ2)/(1.0-ρ)+love

 
@njit(cache=cache)  
def resources_couple(par,t,ih,iz,assets):      
      
   
    izw=iz//par.num_zm;izm=iz%par.num_zw  

    #women earnings
    yw_w = par.grid_zw[t,izw,ih]*par.grid_wlp[1] 
    yw_n = par.grid_zw[t,izw,ih]*par.grid_wlp[0] 
    
    #spousal deduction based on womens earnings
    SpDed_w=np.maximum(par.d0+par.d1*yw_w+par.d2*yw_w**2,0.0)#spousal deduction if women works
    SpDed_n=np.maximum(par.d0+par.d1*yw_n+par.d2*yw_n**2,0.0)#spousal deduction if women does not work
    
    #men income and taxable income
    yh = par.grid_zm[t,izm,ih]
    yh_taxable_w =   yh -SpDed_w
    yh_taxable_n =   yh -SpDed_n
    
    #nota that taxation is individual 
    tax_work =     yh_taxable_w+yw_w -  par.Λ*(yh_taxable_w)**(1-par.τ)- par.Λ*(yw_w)**(1-par.τ)
    tax_not_work = yh_taxable_n+yw_n -  par.Λ*(yh_taxable_n)**(1-par.τ)- par.Λ*(yw_n)**(1-par.τ)
    
    
     
    #resources depending on employment  
    res_not_work = par.R*assets + (yh +yw_n-tax_not_work)
    res_work     = par.R*assets + (yh +yw_w-tax_work) 
      
    # change resources if retired: women should not work!   
    if t>=par.Tr: return res_work,     res_work,yh,yw_w 
    else:         return res_not_work, res_work,yh,yw_w 


@njit(cache=cache)  
def income_single(par,t,ih,iz,assets,women=True): 
     
    
    iz_i=iz//par.num_zm if women else iz%par.num_zw 
    
     
    labor_income =  par.grid_zw[t,iz_i,ih] if women else par.grid_zm[t,iz_i,ih]#without HC! 
   
    tax_income = (labor_income) -par.Λ*(labor_income)**(1-par.τ)#taxes(labor_income,s=True)# 
  
     
    return labor_income-tax_income
    
    
@njit(cache=cache)
def couple_util(Cpriv,Ctot,power,ishom,ρ,ϕ1,ϕ2,α1,α2,θ,λ,tb):#function to minimize
    """
        Couple's utility given private (Cpriv np.array(float,float)) 
        and total consumption Ctot (float). Note that love does
        not matter here, as this fun is used for intra-period 
        allocation of private and home consumption
    """
    Cpub=Ctot-np.sum(Cpriv) if Ctot>np.sum(Cpriv) else 1e-15
    Vw=util(Cpriv[0],Cpub,ρ,ϕ1,ϕ2,α1,α2,θ,λ,tb,love=0.0,couple=True,ishom=ishom)
    Vm=util(Cpriv[1],Cpub,ρ,ϕ1,ϕ2,α1,α2,θ,λ,tb,love=0.0,couple=True,ishom=ishom)
    
    return np.array([power*Vw +(1.0-power)*Vm, Vw, Vm])

@njit(cache=cache) 
def single_time_util(Ctot,ρ,ϕ1,ϕ2,α1,α2,θ,λ,tb,love=0.0,couple=0.0,ishom=0.0,γ=0.0): 
     
    c_priv,c_pub = intraperiod_allocation_single(Ctot,ρ,ϕ1,ϕ2,α1,α2,θ,λ,tb) 
     
    return util(c_priv,c_pub,ρ,ϕ1,ϕ2,α1,α2,θ,λ,tb,love,couple,ishom) 
     

@njit(cache=cache)
def marg_util(C_tot,ρ,ϕ1,ϕ2,α1,α2,θ,λ,tb):
    
    share = 1.0/(1.0 + (α2/α1)**(1.0/(1.0-ϕ1)))
    constant = α1*share**ϕ1+α2*(1.0-share)**ϕ1
    return ϕ1*C_tot**((1.0-ρ)*ϕ1 -1.0)*constant**(1.0 - ρ)
    
    
    
@njit(cache=cache)
def couple_time_utility(Ctot,par,sol,iP,part,love,pars2):
    """
        Couple's utility given total consumption Ctot (float)
    """    
    p_Cw_priv, p_Cm_priv, p_C_pub =\
        intraperiod_allocation(Ctot,par.grid_Ctot,sol.pre_Ctot_Cw_priv[part,iP],sol.pre_Ctot_Cm_priv[part,iP]) 
        
    vw_new = util(p_Cw_priv,p_C_pub,*pars2,love,True,1.0-par.grid_wlp[part])                       
    vm_new = util(p_Cm_priv,p_C_pub,*pars2,love,True,1.0-par.grid_wlp[part])
     
    return vw_new, vm_new


    
@njit(cache=cache)
def intraperiod_allocation(C_tot,grid_Ctot,pre_Ctot_Cw_priv,pre_Ctot_Cm_priv):
 
    #if vector: # interpolate pre-computed solution if C_tot is vector      
    lenn=1 if np.isscalar(C_tot) else len(C_tot)
    Cw_priv,Cm_priv=np.ones((2,lenn))    
    linear_interp.interp_1d_vec(grid_Ctot,pre_Ctot_Cw_priv,C_tot,Cw_priv)
    linear_interp.interp_1d_vec(grid_Ctot,pre_Ctot_Cm_priv,C_tot,Cm_priv)

    return Cw_priv, Cm_priv, C_tot - Cw_priv - Cm_priv #returns numpy arrays
        

@njit(cache=cache)
def intraperiod_allocation_single(C_tot,ρ,ϕ1,ϕ2,α1,α2,θ,λ,tb):
    
    #find private and public expenditure to max util
    args=(ρ,ϕ1,ϕ2,α1,α2,θ,λ,tb)
    C_priv = optimizer(lambda x,y,args:-util(x,y-x,*args),1.0e-6, C_tot - 1.0e-6,args=(C_tot,args))[0]
    
    return C_priv,C_tot - C_priv#=C_pub

def labor_income(par,single=False): 
     
    ###################
    # Grids here
    ##################
    
    
    # Persistent shocks
    Pw, PiPw, Pi0Pw =rouw_nonst(par.T,par.σpw,par.σ0w,par.num_pw) 
    Pm, PiPm, Pi0Pm =rouw_nonst(par.T,par.σpm,par.σ0m,par.num_pm) 
    
    # Transitory shocks
    ρ=par.σϵwm/(par.σϵw*par.σϵm)#correlation between trasitory shocks
    
    gridw,transw,_,_,_=rouwenhorst(0.0,0.0,par.σϵw,n=par.num_ϵw)
    gridm,transm,_,_,_=rouwenhorst(0.0,0.0,par.σϵm,n=par.num_ϵm)
        
    if ((ρ<1e-6) | (single)): #uncorrelated shocks
        Πϵ=np.kron(transw.T,transm.T)
    else:#correlated shocks        
        gridw,gridm,Πϵ=var_discretization(0.0,0.0,par.σϵw,par.σϵm, 0.0,par.num_ϵw,par.num_ϵm)

    
    #Store here all the components of earnings
    XXw,XPw,XTw,XHw,XDw=np.zeros((5,par.T,par.num_pw*par.num_ϵw,len(par.grid_h)))
    XXm,XPm,XTm,XHm,XDm=np.zeros((5,par.T,par.num_pm*par.num_ϵm,len(par.grid_h)))
    
    for t in range(par.T):
        for i in range(len(par.grid_h)):
            
            XTw[t,:,i]=(np.zeros(Pw[t].shape)[:,None]+gridw[None,:]).flatten()
            XPw[t,:,i]=(Pw[t][:,None]+np.zeros(gridw.shape)[None,:]).flatten()
            XHw[t,:,i] = par.grid_h[i] 
            XDw[t,:,i] = par.t0w+par.t1w*t+par.t2w*t**2
            
            XTm[t,:,i]=(np.zeros(Pm[t].shape)[:,None]+gridm[None,:]).flatten()
            XPm[t,:,i]=(Pm[t][:,None]+np.zeros(gridm.shape)[None,:]).flatten()
            XHm[t,:,i] =  0.0
            XDm[t,:,i] = par.t0m+par.t1m*t+par.t2m*t**2
            

    XXw=np.exp(XTw+XPw+XHw+XDw)
    XXm=np.exp(XTm+XPm+XHm+XDm)
    
    for t in range(par.Tr,par.T):
        for i in range(len(par.grid_h)):
            XXw[t,:,i]=pens(XXw[par.Tr-1,:,i],par.p_b,par.κ)
            XXm[t,:,i]=pens(XXm[par.Tr-1,:,i],par.p_b,par.κ)
    
    #####################
    # Transition matrices
    ####################

    #Joint transition matrix by gender, initial period
    Pi0w=[np.kron(Pi0Pw[t],transw.T) for t in range(par.T-1)]
    Pi0m=[np.kron(Pi0Pm[t],transm.T) for t in range(par.T-1)]
    
    #Create the joint transition matrix, first using the kroneker product and then reshaping
    Π=[np.kron(np.kron(PiPw[t],PiPm[t]),Πϵ) for t in range(par.T-1)] 
    
    n_total=par.num_ϵw*par.num_ϵm*par.num_pw*par.num_pm
    for t in range(par.T-1):
        Π[t]=Π[t].reshape(par.num_pw,par.num_pm,par.num_ϵw,par.num_ϵm,par.num_pw,par.num_pm,par.num_ϵw,par.num_ϵm)\
            .transpose(0,2,1,3,4,6,5,7).reshape(n_total, n_total)
        
        
    #No more income shocks after retirement
    for t in range(par.Tr-1,par.T-1): Π[t][:]=np.eye(par.num_pm*par.num_ϵm*par.num_pw*par.num_ϵw) 
    
 
    return XXw, XTw,XPw, Pi0w,XXm, XTm,XPm, Pi0m,Π

def build_directly(Pia, Pib, Pic, Pid):
    """
    Use einsum to directly construct the ordering you would obtain by running
        
        Pi1_orig = np.kron(Pia, Pib)
        Pi2_orig = np.kron(Pic, Pid)
        Pi_original = np.kron(Pi1_orig, Pi2_orig)
        

    """
    result_tensor = np.einsum('ai,bj,ck,dl->abcdijkl', Pia, Pib, Pic, Pid)
    
    # Reshape to matrix: (a,b,c,d) x (a,b,c,d) ordering
    n_a, n_b, n_c, n_d = Pia.shape[0], Pib.shape[0], Pic.shape[0], Pid.shape[0]
    total_states = n_a * n_b * n_c * n_d

    
    return result_tensor.reshape(total_states, total_states)

   
#function to compute pension
def pens(value,p_b,κ):
    
 
    #Matters only income below threshold 
    valuef=value.copy()

    
    return p_b+κ*valuef


###########################
# Uncertainty below       #
###########################
 
def sd_rw(T,sigma_persistent,sigma_init):
    
    if isinstance(sigma_persistent,np.ndarray):
        return np.sqrt([sigma_init**2 + t*sigma_persistent[t]**2 for t in range(T)])
    else:
        return np.sqrt(sigma_init**2 + np.arange(0,T)*(sigma_persistent**2))
    
def sd_rw_trans(T,sigma_persistent,sigma_init,sigma_transitory):
    return sd_rw(T, sigma_persistent, sigma_init)

    
    
def normcdf_tr(z,nsd=5):
        
        z = np.minimum(z, nsd*np.ones_like(z))
        z = np.maximum(z,-nsd*np.ones_like(z))
            
        pup = norm.cdf(nsd,0.0,1.0)
        pdown = norm.cdf(-nsd,0.0,1.0)
        const = pup - pdown
        
        return (norm.cdf(z,0.0,1.0)-pdown)/const
    
    
def normcdf_ppf(z): return norm.ppf(z,0.0,1.0)       
        
def addaco_nonst(T=40,sigma_persistent=0.05,sigma_init=0.2,npts=50,mean=np.array([0.0],)):
  
    if mean.shape[0]!=T-1:mean=np.zeros(T-1) 
    # start with creating list of points
    sd_z = sd_rw(T,sigma_persistent,sigma_init)
    sd_z0 = np.array([np.sqrt(sd_z[t]**2-sigma_init**2) for t in range(T)])
        
    Pi = list();Pi0 = list();X = list();Int=list()


    #Probabilities per period
    Pr=normcdf_ppf((np.cumsum(np.ones(npts+1))-1)/npts)
    
    #Create interval limits
    for t in range(0,T):Int = Int + [Pr*sd_z[t]]
        
    
    #Create gridpoints
    for t in range(T):
        line=np.zeros(npts)
        for i in range(npts):
            line[i]= sd_z[t]*npts*(norm.pdf(Int[t][i]  /sd_z[t],0.0,1.0)-\
                                   norm.pdf(Int[t][i+1]/sd_z[t],0.0,1.0))
            
        X = X + [line]

    def integrand(x,e,e1,sd,sds,mean):
        return np.exp(-(x**2)/(2*sd**2))*(norm.cdf((e1-x-mean)/sds,0.0,1.0)-\
                                          norm.cdf((e- x-mean)/sds,0.0,1.0))
            
            
    #Fill probabilities
    for t in range(1,T):
        Pi_here = np.zeros([npts,npts]);Pi_here0 = np.zeros([npts,npts])
        for i in range(npts):
            for jj in range(npts):
                
                Pi_here[i,jj]=npts/np.sqrt(2*np.pi*sd_z[t-1]**2)\
                    *quad(integrand,Int[t-1][i],Int[t-1][i+1],
                     args=(Int[t][jj],Int[t][jj+1],sd_z[t],sigma_persistent,mean[t-1]))[0]
                
                Pi_here0[i,jj]= norm.cdf(Int[t][jj+1],0.0,sigma_init)-\
                                   norm.cdf(Int[t][jj],0.0,sigma_init)
                                   
            #Adjust probabilities to get exactly 1: the integral is an approximation
            Pi_here[i,:]=Pi_here[i,:]/np.sum(Pi_here[i,:])
            Pi_here0[i,:]=Pi_here0[i,:]/np.sum(Pi_here0[i,:])
                
        Pi = Pi + [Pi_here.T]
        Pi0 = Pi0 + [Pi_here0.T]
        
    return X, Pi, Pi0   

def rouw_nonst(T=40,sigma_persistent=0.05,sigma_init=0.2,npts=10): 
    
    sd_z = sd_rw(T,sigma_persistent,sigma_init) 
    sd_z0 = np.array([np.sqrt(sd_z[t]**2-sigma_init**2) for t in range(T)]) 
        
    Pi = list();Pi0 = list();X = list() 
 
    for t in range(0,T): 
        nsd = np.sqrt(npts-1) 
        X = X + [np.linspace(-nsd*sd_z[t],nsd*sd_z[t],num=npts)] 
         
        if t >= 1: Pi = Pi +   [rouw_nonst_one(sd_z[t-1],sd_z[t] ,npts).T] 
        if t >= 1: Pi0 = Pi0 + [rouw_nonst_one(sd_z0[t-1],sd_z[t-1],npts).T] 
        
    return X, Pi, Pi0  


def rouw_nonst_one(sd0,sd1,npts):
   
    # this generates one-period Rouwenhorst transition matrix
    assert(npts>=2)
    pi0 = 0.5*(1+(sd0/sd1))
    Pi = np.array([[pi0,1-pi0],[1-pi0,pi0]])
    assert(pi0<1)
    assert(pi0>0)
    for n in range(3,npts+1):
        A = np.zeros([n,n])
        A[0:(n-1),0:(n-1)] = Pi
        B = np.zeros([n,n])
        B[0:(n-1),1:n] = Pi
        C = np.zeros([n,n])
        C[1:n,1:n] = Pi
        D = np.zeros([n,n])
        D[1:n,0:(n-1)] = Pi
        Pi = pi0*A + (1-pi0)*B + pi0*C + (1-pi0)*D
        Pi[1:n-1] = 0.5*Pi[1:n-1]
        
        assert(np.all(np.abs(np.sum(Pi,axis=1)-1)<1e-5 ))
    
    return Pi


@njit(fastmath=True)
def mc_simulate(statein,Piin,shocks):
    """This simulates transition one period ahead for a Markov chain
    
    Args: 
        Statein: scalar giving the initial state
        Piin: transition matrix n*n [post,initial]
        shocks: scalar in [0,1]
    
    """ 
    return  np.sum(np.cumsum(Piin[:,statein])<shocks)

@njit(fastmath=True)
def rescale_matrix(Π,tol=1e-4):
    """Rescale transition matrix by eliminating close to zero elements
    
    Args:
        Π: n*n square tranistion matrix
        tol: elements of Π below tol are set to 0
    
    """
    
    zeros = Π < tol
    for i in range(len(Π)):
        
        Π[zeros[:,i],i] = 0.0
        Π[:,i] = Π[:,i] / np.sum(Π[:,i])
        
    return Π 

def var_discretization(rho1, rho2, sigma1, sigma2, rho12, n1, n2):
    
    """
     Discretize bivariate correlated AR(1) processes using VAR approach.
     
     Creates independent grids for each process using Rouwenhorst spacing, then computes 
     joint transition probabilities using multivariate normal densities for the innovations.
     Correlation is captured through the joint distribution of innovations rather than 
     through grid construction.
     
     Parameters:
     -----------
     rho1 : float
         Persistence parameter for first AR(1) process, |rho1| < 1
     rho2 : float  
         Persistence parameter for second AR(1) process, |rho2| < 1
     sigma1 : float
         Standard deviation of innovations for first process, sigma1 > 0
     sigma2 : float
         Standard deviation of innovations for second process, sigma2 > 0
     rho12 : float
         Correlation between innovations, -1 < rho12 < 1
     n1 : int
         Number of grid points for first process
     n2 : int
         Number of grid points for second process
         
     Returns:
     --------
     x1_grid : ndarray, shape (n1,)
         Grid points for first process, spanning ±σ₁/(1-ρ₁²)^0.5 approximately
     x2_grid : ndarray, shape (n2,)  
         Grid points for second process, spanning ±σ₂/(1-ρ₂²)^0.5 approximately
     P : ndarray, shape (n1*n2, n1*n2)
         Transition probability matrix. P[i,j] = probability of transitioning from 
         state i to state j, where states are ordered as (i1*n2 + i2) for 
         grid indices (i1, i2).
     """
    
    # Innovation covariance matrix
    Sigma_eps = np.array([[sigma1**2, sigma1*sigma2*rho12],
                          [sigma1*sigma2*rho12, sigma2**2]])
    
   
    ###############################################################
    #Step 1: Create Individual Grids Using Rouwenhorst Spacing
    #############################################################
    sigma1_unc = sigma1 / np.sqrt(1 - rho1**2)
    sigma2_unc = sigma2 / np.sqrt(1 - rho2**2)
    
    # Create grids
    x1_grid = sigma1_unc * np.sqrt(n1 - 1) * np.linspace(-1, 1, n1)
    x2_grid = sigma2_unc * np.sqrt(n2 - 1) * np.linspace(-1, 1, n2)
    

    ##############################################################################
    #Step 2: Compute Joint Transition Probabilities acccounting for correlation
    #
    # What this does:
    #
    #    -Loops over all possible transitions from current state to next state
    #    -Each "state" is a point (x₁, x₂) in the 2D grid
    #    
    ###########################################################################
    
    n_states = n1 * n2
    P = np.zeros((n_states, n_states))
    
    # For each current state (i, j)
    for i in range(n1):
        for j in range(n2):
            current_state = i * n2 + j
            current_x1 = x1_grid[i]
            current_x2 = x2_grid[j]
            
            # Required innovations for this specific transition
            for i_next in range(n1):
                for j_next in range(n2):
                    next_state = i_next * n2 + j_next
                    next_x1 = x1_grid[i_next]
                    next_x2 = x2_grid[j_next]
                    
                    # Expected innovations
                    eps1_expected = next_x1 - rho1 * current_x1
                    eps2_expected = next_x2 - rho2 * current_x2
                    
                    # What's the probability of getting exactly these innovations?
                    eps_vector = np.array([eps1_expected, eps2_expected])


                    try:
                        prob_density = multivariate_normal.pdf(eps_vector, mean=[0, 0], cov=Sigma_eps)
                        """
                        Explain the line above:
                            
                        Given where we are (x₁, x₂) and where we want to go (x₁', x₂')
                        We can calculate exactly what innovations we'd need:

                            ε₁ = x₁' - ρ₁x₁ (innovation needed for process 1)
                            ε₂ = x₂' - ρ₂x₂ (innovation needed for process 2)
                               
                        The probability of this transition = probability of getting exactly those innovations
                        Since innovations are bivariate normal with covariance Σ, we use multivariate_normal.pdf()
                        """
                    except:
                        prob_density = 1e-10
                    
                    P[current_state, next_state] = prob_density
    
    # Normalize rows (this is approximate but necessary)
    row_sums = P.sum(axis=1)
    for i in range(n_states):
        if row_sums[i] > 1e-10:
            P[i, :] /= row_sums[i]
        else:
            # If all probabilities are essentially zero, make it uniform
            P[i, :] = 1.0 / n_states
    
    return x1_grid, x2_grid, P.T

##########################
# Other routines below
##########################

@njit
def deriv(x,f,ϵ=1e-8):
    """
    Create derivative for array f defined on the x array x
    """
    
    forward,backward=np.empty((2,len(x)))
    linear_interp.interp_1d_vec(x,f,x+ϵ,forward)
    linear_interp.interp_1d_vec(x,f,x-ϵ,backward)
    
    return (forward-backward)/(2*ϵ)
    

def grid_fat_tails(gmin,gmax,gridpoints):
    """Create a grid with fat tail, centered and symmetric around gmin+gmax
    
    Args: 
        gmin (float): min of grid
        gmax(float): max of grids
        gridpoints(int): number of gridpoints (odd number)
    
    """ 
    odd_num = np.mod(gridpoints,2)
    mid=(gmax+gmin)/2.0
    summ=gmin+gmax
    first_part = nonlinspace(gmin,mid,(gridpoints+odd_num)//2,1.3)
    last_part = np.flip(summ - nonlinspace(gmin,mid,(gridpoints-odd_num)//2 + 1,1.3))[1:]
    return np.append(first_part,last_part)

@njit 
def optimizer(obj,a,b,args=(),tol=1e-6): 
    """ golden section search optimizer 
     
    Args: 
 
        obj (callable): 1d function to optimize over 
        a (double): minimum of starting bracket 
        b (double): maximum of starting bracket 
        args (tuple): additional arguments to the objective function 
        tol (double,optional): tolerance 
 
    Returns: 
 
        float,float: optimization result, function at argmin
     
    """ 
     
    inv_phi = (np.sqrt(5) - 1) / 2 # 1/phi                                                                                                                 
    inv_phi_sq = (3 - np.sqrt(5)) / 2 # 1/phi^2      
         
    # a. distance 
    dist = b - a 
    if dist <= tol:  
        return (a+b)/2,  obj((a+b)/2,*args)
 
    # b. number of iterations 
    n = int(np.ceil(np.log(tol/dist)/np.log(inv_phi))) 
 
    # c. potential new mid-points 
    c = a + inv_phi_sq * dist 
    d = a + inv_phi * dist 
    yc = obj(c,*args) 
    yd = obj(d,*args) 
 
    # d. loop 
    for _ in range(n-1): 
        if yc < yd: 
            b = d 
            d = c 
            yd = yc 
            dist = inv_phi*dist 
            c = a + inv_phi_sq * dist 
            yc = obj(c,*args) 
        else: 
            a = c 
            c = d 
            yc = yd 
            dist = inv_phi*dist 
            d = a + inv_phi * dist 
            yd = obj(d,*args) 
 
    # e. return 
    if yc < yd: 
        return (a+d)/2, obj((a+d)/2,*args)
    else: 
        return (c+b)/2, obj((c+b)/2,*args)
    
####################################################################################
# Upper envelop alogorithm - Adapted from Consav to accomodate for couple_decisions
###################################################################################
def create(ufunc):
    """ create upperenvelope function from the utility function ufunc
    
    Args:

        ufunc (callable): utility function with *args (must be decorated with @njit)

    Returns:

        upperenvelope (callable): upperenvelope called as (grid_a,m_vec,c_vec,inv_w_vec,use_inv_w,grid_m,c_ast_vec,v_ast_vec,*args)
    
    """

    @njit
    def upperenvelope(grid_a,m_vec,c_vec,inv_w_vec_w,inv_w_vec_m,power,grid_m,c_ast_vec,v_ast_vec_w,v_ast_vec_m,v_ast_vec_c,*args):
        """ upperenvelope function
        
        Args:

            grid_a (numpy.ndarray): input, end-of-period asset vector of length Na
            m_vec (numpy.ndarray): input, cash-on-hand vector from egm of length Na
            c_vec (numpy.ndarray): input, consumption vector from egm of length Na
            inv_w_vec (numpy.ndarray): input, post decision value-of-choice vector from egm of length Na
            grid_m (numpy.ndarray): input, common grid for cash-on-hand of length Nm
            c_ast_vec (numpy.ndarray): output, consumption on common grid for cash-on-hand of length Nm
            v_ast_vec (numpy.ndarray): output, value-of-choice on common grid for cash-on-hand of length Nm
            *args: additional arguments to the utility function
                    
        """

        # for given m_vec, c_vec and w_vec (coming from grid_a)
        # find the optimal consumption choices (c_ast_vec) at the common grid (grid_m) 
        # using the upper envelope + also value the implied values-of-choice (v_ast_vec)

        Na = grid_a.size
        Nm = grid_m.size

        c_ast_vec[:] = 0
        v_ast_vec = -np.inf*np.ones(c_ast_vec.shape)

        # constraint
        # the constraint is binding if the common m is smaller
        # than the smallest m implied by EGM step (m_vec[0])

        im = 0 
        while im < Nm and grid_m[im] <= m_vec[0]: 
             
            # a. consume all 
            c_ast_vec[im] = grid_m[im]  
 
            # b. value of choice 
            u_w,u_m = ufunc(c_ast_vec[im:im+1],*args) 
            v_ast_vec_w[im] = u_w[0] + inv_w_vec_w[0] 
            v_ast_vec_m[im] = u_m[0] + inv_w_vec_m[0] 
 
            v_ast_vec[im] = power*v_ast_vec_w[im] + (1.0-power)*v_ast_vec_m[im] 
            v_ast_vec_c[im] = v_ast_vec[im]
            im += 1 
            
        # upper envellope
        # apply the upper envelope algorithm
        
        for ia in range(Na-1):

            # a. a inteval and w slope
            a_low  = grid_a[ia]
            a_high = grid_a[ia+1]
            
            inv_w_low_w  = inv_w_vec_w[ia]
            inv_w_high_w = inv_w_vec_w[ia+1]
            
            inv_w_low_m  = inv_w_vec_m[ia]
            inv_w_high_m = inv_w_vec_m[ia+1]

            if a_low > a_high:
                continue

            inv_w_slope_w = (inv_w_high_w-inv_w_low_w)/(a_high-a_low)
            inv_w_slope_m = (inv_w_high_m-inv_w_low_m)/(a_high-a_low)
            
            # b. m inteval and c slope
            m_low  = m_vec[ia]
            m_high = m_vec[ia+1]

            c_low  = c_vec[ia]
            c_high = c_vec[ia+1]

            c_slope = (c_high-c_low)/(m_high-m_low)

            # c. loop through common grid
            for im in range(Nm):

                # i. current m
                m = grid_m[im]

                # ii. interpolate?
                interp = (m >= m_low) and (m <= m_high)            
                extrap_above = ia == Na-2 and m > m_vec[Na-1]

                # iii. interpolation (or extrapolation)
                if interp or extrap_above:

                    # o. implied guess
                    c_guess = np.array([c_low + c_slope * (m - m_low)])
                    a_guess = m - c_guess[0]

                    # oo. implied post-decision value function
                    inv_w = inv_w_low_w + inv_w_slope_w * (a_guess - a_low)     
                    inv_m = inv_w_low_m + inv_w_slope_m * (a_guess - a_low)       

                    # ooo. value-of-choice
                    u_w,u_m = ufunc(c_guess,*args)   
                    v_guess_w = u_w[0] + inv_w
                    v_guess_m = u_m[0] + inv_m

                    v_guess=power*v_guess_w+(1.0-power)*v_guess_m
                    
                    # oooo. update
                    if v_guess > v_ast_vec[im]:
                        v_ast_vec[im] = v_guess
                        c_ast_vec[im] = c_guess[0]
                        
                        # update utility for the couple
                        v_ast_vec_w[im] = v_guess_w
                        v_ast_vec_m[im] = v_guess_m                      
                        v_ast_vec_c[im]=power*v_ast_vec_w[im]+(1.0-power)*v_ast_vec_m[im]
    
    return upperenvelope