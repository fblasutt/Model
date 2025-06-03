import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import cholesky
from scipy.stats import multivariate_normal




def correct_bivariate_discretization(rho1, rho2, sigma1, sigma2, rho12, n1, n2, method='cholesky'):
    """
    Correct implementation using Cholesky decomposition or VAR(1) approach
    """
    if method == 'cholesky':
        return cholesky_bivariate_discretization(rho1, rho2, sigma1, sigma2, rho12, n1, n2)
    elif method == 'var':
        return var_discretization(rho1, rho2, sigma1, sigma2, rho12, n1, n2)
    else:
        raise ValueError("Method must be 'cholesky', 'cholesky_simple', or 'var'")

def cholesky_bivariate_discretization(rho1, rho2, sigma1, sigma2, rho12, n1, n2):
    """
    Discretize bivariate correlated AR(1) processes using Cholesky decomposition.
    
    Uses Cholesky decomposition to transform the correlated innovation system into 
    independent innovations, then computes transition probabilities by exploiting 
    this independence structure. For each transition, solves for the required 
    independent innovations (η₁, η₂) and uses their independent normal densities
    with appropriate Jacobian transformation.
    
    Mathematical approach:
    - Original: ε ~ N(0, Σ) where Σ has correlation ρ₁₂  
    - Transform: ε = L η where η ~ N(0, I) are independent
    - AR(1) becomes: x₁' = ρ₁x₁ + L₁₁η₁, x₂' = ρ₂x₂ + L₂₁η₁ + L₂₂η₂
    - Transition probability: P(η₁) × P(η₂) / |det(L)|
    
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
        Grid points for first process, covering approximately ±2.5 × σ₁/(1-ρ₁²)^0.5
    x2_grid : ndarray, shape (n2,)  
        Grid points for second process, covering approximately ±2.5 × σ₂/(1-ρ₂²)^0.5
    P : ndarray, shape (n1*n2, n1*n2)
        Transition probability matrix where P[i,j] is the probability of moving from 
        state i to state j. States are indexed as (i1*n2 + i2) for grid positions 
        (i1, i2). Computed using independent normal densities for transformed 
        innovations with Jacobian correction.
        
    """
    # Innovation covariance matrix
    Sigma_eps = np.array([[sigma1**2, sigma1*sigma2*rho12],
                          [sigma1*sigma2*rho12, sigma2**2]])
    
    # Cholesky decomposition: Sigma_eps = L @ L.T
    L = cholesky(Sigma_eps, lower=True)
    
    # In the transformed system, we have:
    # [eps1_t]   [L11  0 ] [eta1_t]
    # [eps2_t] = [L21 L22] [eta2_t]
    # where eta1_t, eta2_t are independent N(0,1)
    
    L11 = L[0, 0]  # = sigma1
    L21 = L[1, 0]  # = sigma1 * sigma2 * rho12 / sigma1 = sigma2 * rho12
    L22 = L[1, 1]  # = sigma2 * sqrt(1 - rho12^2)
    
    # The original AR(1) system:
    # x1_t = rho1 * x1_{t-1} + eps1_t
    # x2_t = rho2 * x2_{t-1} + eps2_t
    
    # Becomes:
    # x1_t = rho1 * x1_{t-1} + L11 * eta1_t
    # x2_t = rho2 * x2_{t-1} + L21 * eta1_t + L22 * eta2_t
    
    # This is a VAR(1) system! Let's discretize it properly.
    
    # Method 1: Discretize the independent eta processes and construct joint transitions
    
    # Unconditional variances in the transformed system
    # For x1: still just sigma1^2 / (1 - rho1^2) since only eta1 affects it
    var1_unc = L11**2 / (1 - rho1**2)
    
    # For x2: more complex because both eta1 and eta2 affect it
    # Var(x2) = Var(rho2 * x2_{t-1} + L21 * eta1 + L22 * eta2)
    # This requires solving the VAR system, but approximately:
    var2_unc = (L21**2 + L22**2) / (1 - rho2**2)
    
    
    # Create grids based on unconditional distributions
    std1_unc = np.sqrt(var1_unc)
    std2_unc = np.sqrt(var2_unc)
    
    m = 2.5  # Coverage
    x1_grid = np.linspace(-m * std1_unc, m * std1_unc, n1)
    x2_grid = np.linspace(-m * std2_unc, m * std2_unc, n2)
    
    # Now build transition matrix using the VAR structure
    n_states = n1 * n2
    P = np.zeros((n_states, n_states))
    
    # Grid spacing for integration
    h1 = x1_grid[1] - x1_grid[0] if n1 > 1 else 1.0
    h2 = x2_grid[1] - x2_grid[0] if n2 > 1 else 1.0
    
    for i in range(n1):
        for j in range(n2):
            current_state = i * n2 + j
            x1_curr = x1_grid[i]
            x2_curr = x2_grid[j]
            
            for i_next in range(n1):
                for j_next in range(n2):
                    next_state = i_next * n2 + j_next
                    x1_next = x1_grid[i_next]
                    x2_next = x2_grid[j_next]
                    
                    # Given current state (x1_curr, x2_curr), what's P(x1_next, x2_next)?
                    # We need: x1_next = rho1 * x1_curr + L11 * eta1
                    #          x2_next = rho2 * x2_curr + L21 * eta1 + L22 * eta2
                    
                    # Solve for required eta1, eta2:
                    # eta1 = (x1_next - rho1 * x1_curr) / L11
                    eta1_required = (x1_next - rho1 * x1_curr) / L11
                    
                    # eta2 = (x2_next - rho2 * x2_curr - L21 * eta1) / L22
                    eta2_required = (x2_next - rho2 * x2_curr - L21 * eta1_required) / L22
                    
                    # Probability density (since eta1, eta2 are independent N(0,1))
                    prob_eta1 = norm.pdf(eta1_required)
                    prob_eta2 = norm.pdf(eta2_required)
                    joint_prob = prob_eta1 * prob_eta2
                    
                    # Jacobian: we're transforming from (eta1, eta2) to (x1, x2)
                    # The Jacobian determinant is |L11 * L22| = L11 * L22
                    jacobian = L11 * L22
                    
                    # Final probability (multiply by grid cell area)
                    P[current_state, next_state] = joint_prob * h1 * h2 / jacobian
    
    # Normalize rows
    row_sums = P.sum(axis=1)
    for i in range(n_states):
        if row_sums[i] > 1e-10:
            P[i, :] /= row_sums[i]
        else:
            P[i, :] = 1.0 / n_states
    
    return x1_grid, x2_grid, P

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
    
    return x1_grid, x2_grid, P

def simulate_bivariate_ar1(x1_grid, x2_grid, P, T=10000):
    """Simulate the discretized process"""
    n1, n2 = len(x1_grid), len(x2_grid)
    n_states = n1 * n2
    
    # Initialize in the middle
    state = n_states // 2
    
    x1_sim = np.zeros(T)
    x2_sim = np.zeros(T)
    
    for t in range(T):
        # Get current values
        i = state // n2
        j = state % n2
        x1_sim[t] = x1_grid[i]
        x2_sim[t] = x2_grid[j]
        
        # Draw next state
        state = np.random.choice(n_states, p=P[state, :])
    
    return x1_sim, x2_sim

def compute_theoretical_moments(rho1, rho2, sigma1, sigma2, rho12):
    """Compute theoretical moments correctly"""
    # Unconditional variances
    var1 = sigma1**2 / (1 - rho1**2)
    var2 = sigma2**2 / (1 - rho2**2)
    
    # Unconditional covariance
    cov12 = (sigma1 * sigma2 * rho12) / (1 - rho1 * rho2)
    
    # Unconditional correlation
    corr12 = cov12 / np.sqrt(var1 * var2)
    
    return {
        'var1': var1,
        'var2': var2,
        'cov12': cov12,
        'corr12': corr12,
        'innovation_corr': rho12
    }

def test_discretization():
    """Test the discretization methods"""
    # Parameters
    rho1, rho2 = 0.0, 0.0
    sigma1, sigma2 = 0.1, 0.1
    rho12 = 0.1
    n1, n2 = 3, 3  # Increase grid size for better accuracy
    T = 150000
    
    print("TESTING CORRECTED BIVARIATE DISCRETIZATION")
    print("=" * 50)
    print(f"Parameters: ρ₁={rho1}, ρ₂={rho2}, σ₁={sigma1}, σ₂={sigma2}, ρ₁₂={rho12}")
    
    # Theoretical moments
    theory = compute_theoretical_moments(rho1, rho2, sigma1, sigma2, rho12)
    print(f"\nTheoretical moments:")
    print(f"Var(x₁) = {theory['var1']:.6f}")
    print(f"Var(x₂) = {theory['var2']:.6f}")
    print(f"Cov(x₁,x₂) = {theory['cov12']:.6f}")
    print(f"Corr(x₁,x₂) = {theory['corr12']:.6f}")
    print(f"Innovation correlation = {theory['innovation_corr']:.6f}")
    
    methods = ['var', 'cholesky']
    results = {}
    
    for method in methods:
        print(f"\n{method.upper()} METHOD:")
        print("-" * 20)
        
        np.random.seed(42)  # For reproducibility
        
        try:
            x1_grid, x2_grid, P = correct_bivariate_discretization(
                rho1, rho2, sigma1, sigma2, rho12, n1, n2, method=method
            )
            
            x1_sim, x2_sim = simulate_bivariate_ar1(x1_grid, x2_grid, P, T)
            
            # Compute simulated moments
            sim_var1 = np.var(x1_sim)
            sim_var2 = np.var(x2_sim)
            sim_cov12 = np.cov(x1_sim, x2_sim)[0, 1]
            sim_corr12 = np.corrcoef(x1_sim, x2_sim)[0, 1]
            
            # Persistence
            sim_rho1 = np.corrcoef(x1_sim[:-1], x1_sim[1:])[0, 1]
            sim_rho2 = np.corrcoef(x2_sim[:-1], x2_sim[1:])[0, 1]
            
            # Innovation correlation (approximate)
            eps1_approx = x1_sim[1:] - sim_rho1 * x1_sim[:-1]
            eps2_approx = x2_sim[1:] - sim_rho2 * x2_sim[:-1]
            sim_innov_corr = np.corrcoef(eps1_approx, eps2_approx)[0, 1]
            
            print(f"Var(x₁): {sim_var1:.6f} (error: {abs(sim_var1 - theory['var1']):.6f})")
            print(f"Var(x₂): {sim_var2:.6f} (error: {abs(sim_var2 - theory['var2']):.6f})")
            print(f"Cov(x₁,x₂): {sim_cov12:.6f} (error: {abs(sim_cov12 - theory['cov12']):.6f})")
            print(f"Corr(x₁,x₂): {sim_corr12:.6f} (error: {abs(sim_corr12 - theory['corr12']):.6f})")
            print(f"ρ₁: {sim_rho1:.6f} (target: {rho1})")
            print(f"ρ₂: {sim_rho2:.6f} (target: {rho2})")
            print(f"Innovation corr: {sim_innov_corr:.6f} (target: {rho12})")
            
            results[method] = {
                'x1_sim': x1_sim, 'x2_sim': x2_sim,
                'corr_error': abs(sim_corr12 - theory['corr12'])
            }
            
        except Exception as e:
            print(f"Error with {method} method: {e}")
    
    # Plot results
    if results:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        for i, (method, data) in enumerate(results.items()):
            if i >= 2:
                break
                
            x1_sim, x2_sim = data['x1_sim'], data['x2_sim']
            
            # Time series
            axes[0, i].plot(x1_sim[:1000], label='Process 1', alpha=0.7)
            axes[0, i].plot(x2_sim[:1000], label='Process 2', alpha=0.7)
            axes[0, i].set_title(f'{method.title()} Method - Time Series')
            axes[0, i].legend()
            axes[0, i].grid(True)
            
            # Scatter plot
            axes[1, i].scatter(x1_sim[::50], x2_sim[::50], alpha=0.5, s=1)
            axes[1, i].set_xlabel('Process 1')
            axes[1, i].set_ylabel('Process 2')
            axes[1, i].set_title(f'{method.title()} Method - Joint Distribution')
            axes[1, i].grid(True)
            
            # Add correlation as text
            sim_corr = np.corrcoef(x1_sim, x2_sim)[0, 1]
            axes[1, i].text(0.05, 0.95, f'Corr = {sim_corr:.3f}', 
                           transform=axes[1, i].transAxes, 
                           bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.show()
    
    return results

# Import norm for Tauchen method
from scipy.stats import norm

if __name__ == "__main__":
    test_discretization()