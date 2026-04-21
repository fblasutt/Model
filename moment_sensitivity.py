# -*- coding: utf-8 -*-
"""
Created on Aug 4, 2025

Moment sensitivity analysis script
Varies each parameter one at a time on a grid and records how moments change.

Notes: the calculation for each parameter value could be parallelized for speed, 
but this is not done here

@author: sara
"""
# %% 
# import os
# os.environ.setdefault("OMP_NUM_THREADS", "1")
# os.environ.setdefault("MKL_NUM_THREADS", "1")
# os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")

# %%

import numpy as np
import Bargaining_numba as brg
import UserFunctions_numba as usr
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import getpass
import time

# Initialize seed
np.random.seed(10)

# Root directory setup
user = getpass.getuser()
if user == "sara":
    root = '/Users/sara/Dropbox/Family Risk Sharing'
elif user == "32489":
    root = '/Users/32489/Dropbox/Family Risk Sharing'
else:
    raise RuntimeError(f"Unknown user: {user}")

# %% Functions to compute moments and perform sensitivity analysis

def compute_moments(M):
    """
    Compute all moments for a given model
    Returns a dictionary with moment values
    """
    try:
        age=(np.cumsum(np.ones((M.par.simN,M.par.T)),axis=1)-1)+20#age of hh  
               
        # Sample used for divorce moments and employment/expenditures moments
        sample_div =  (age>=age_initial[:,None]) & (age<=age_final[:,None]) & (M.sim.couple_lag==1) 
        sample_empl =  (age>=age_initial[:,None]) & (age<=age_final[:,None]) & (M.sim.couple==1) 

        # Sample to be used for pass through from total to public good expenditures
        sample_pass= (age>age_initial[:,None]) & (age<=age_final[:,None]) & (M.sim.couple==1) & (M.sim.couple_lag==1)
        sample_pass_m1= np.roll(sample_pass,-1,axis=1)

        sample_pass_single= (age>age_initial[:,None]) & (age<=age_final[:,None]) & (M.sim.couple==0) & (M.sim.couple_lag==0)
        sample_pass_m1_single= np.roll(sample_pass_single,-1,axis=1)

        # This sample will be used for pass throughs regressions (if sample, BPP persistent will not work)
        sample_reg= (age>=age_initial[:,None]) & (age<=age_final[:,None]) & (M.sim.couple_lag==1) & (M.sim.couple==1) 

        wife_empl = np.mean(M.sim.WLP[sample_empl]>0)        
        divorce_rate=np.mean((M.sim.couple==0)[sample_div])
        divorce_rate_young=np.mean((M.sim.couple==0)[(sample_div) & (age<=40)])
        expenditure_x_share=np.mean((M.sim.dw/M.sim.C_tot)[sample_empl])
        

        # ΔC =np.log(M.sim.C_tot[sample_pass])  -np.log(M.sim.C_tot[sample_pass_m1])
        # Δd=np.log(M.sim.dw[sample_pass])  -np.log(M.sim.dw[sample_pass_m1])     
        # samee=(M.sim.WLP[sample_pass]==M.sim.WLP[sample_pass_m1])
        # βdC=np.cov(ΔC[samee],Δd[samee])[0,1]/np.var(ΔC[samee])

        C =np.log(M.sim.C_tot[sample_pass])
        d=np.log(M.sim.dw[sample_pass])     
        βdC=np.cov(C,d)[0,1]/np.var(C)

        # ΔC_single =np.log(M.sim.C_tot[sample_pass_single])  -np.log(M.sim.C_tot[sample_pass_m1_single])
        # Δd_single=np.log(M.sim.dw[sample_pass_single])  -np.log(M.sim.dw[sample_pass_m1_single])     
        # samee_single=(M.sim.WLP[sample_pass_single]==M.sim.WLP[sample_pass_m1_single])
        # βdC_single=np.cov(ΔC_single[samee_single],Δd_single[samee_single])[0,1]/np.var(ΔC_single[samee_single])

        # C_single =np.log(M.sim.C_tot[sample_pass_single])
        # d_single=np.log(M.sim.dw[sample_pass_single])     
        # βdC_single=np.cov(C_single,d_single)[0,1]/np.var(C_single)


        gender_gap_earnings=(M.sim.incwg[sample_empl][M.sim.WLP[sample_empl]>0]).mean()/M.sim.incmg[sample_empl].mean()
        share_full_time=(M.sim.WLP[sample_empl][(M.sim.WLP[sample_empl]>0) & (M.sim.couple[sample_empl]==1)]==(M.par.num_wlp-1)).mean()
 
        moments = {
            'wife_empl': wife_empl,
            'divorce_rate': divorce_rate,
            'divorce_rate_young': divorce_rate_young,
            'expenditure_x_share': expenditure_x_share,
            'βdC': βdC,
            # 'βdC_single': βdC_single,
            'gender_gap_earnings': gender_gap_earnings,
            'share_full_time': share_full_time
        }

        return moments

    except Exception as e:
        print(f"Error computing moments: {e}")
        return {
            'wife_empl': np.nan,
            'divorce_rate': np.nan,
            'divorce_rate_young': np.nan,
            'expenditure_x_share': np.nan,
            'βdC': np.nan,
            # 'βdC_single': np.nan,
            'gender_gap_earnings': np.nan,
            'share_full_time': np.nan
        }


def sensitivity_analysis(param_idx, xl, xu, n_points=11):
    """
    Perform sensitivity analysis for a single parameter
    
    Parameters:
    param_idx: index of parameter to vary (0-4)
    n_points: number of grid points to evaluate
    xl: lower bounds for parameters
    xu: upper bounds for parameters

    Returns:
    dict with parameter values and corresponding moments
    """
    
    param_name = param_names[param_idx]
    print(f"\nRunning sensitivity analysis for parameter {param_name} (index {param_idx})")
    
    # Create grid for the parameter using user-defined bounds
    param_min = xl[param_idx]
    param_max = xu[param_idx]
    param_grid = np.linspace(param_min, param_max, n_points)
    
    # Initialize results storage
    results = {
        'param_values': param_grid,
        'param_name': param_name,
        'moments': {
            'wife_empl': [],
            'divorce_rate': [],
            'divorce_rate_young': [],
            'expenditure_x_share': [],
            'βdC': [],
            'gender_gap_earnings': [],
            'share_full_time': []
        }
    }
    
    # Loop through parameter values
    for i, param_val in enumerate(tqdm(param_grid, desc=f"Varying {param_name}")):
        
        #try:
        # Create a copy of the baseline parameters
        params_current = xc.copy()
        params_current[param_idx] = param_val
            
        # Set up the model with current parameters
        M = model.copy(name=f'sensitivity_{param_name}_{i}')
        M.par.ω = params_current[0]
        M.par.grid_love, M.par.Πl, M.par.Πl0 = usr.addaco_nonst(M.par.T, params_current[1], params_current[1], M.par.num_love)
        M.par.α = params_current[2]
        M.par.χ = params_current[3]
        M.par.wedge = params_current[4]
            
        # Solve and simulate the model
        M.solve()
        M.simulate()
            
        # Compute moments
        moments = compute_moments(M)
            
        # Store results
        for moment_name, moment_value in moments.items():
            results['moments'][moment_name].append(moment_value)
                
        print(f"  {param_name} = {param_val:.4f}: wife_empl = {moments['wife_empl']:.4f}, divorce_rate = {moments['divorce_rate']:.4f}")
            
        # except Exception as e:
        #     print(f"  Error at {param_name} = {param_val:.4f}: {e}")
        #     # Fill with NaN for failed computations
        #     for moment_name in results['moments'].keys():
        #         results['moments'][moment_name].append(np.nan)
    
    return results


def plot_sensitivity_results(results_list, save_path=None):
    """
    Plot sensitivity analysis results
    
    Parameters:
    results_list: list of results dictionaries from sensitivity_analysis
    save_path: path to save plots (optional)
    """
    
    moment_names = ['wife_empl', 'divorce_rate', 'divorce_rate_young', 
                   'expenditure_x_share', 'βdC'] #, 'gender_gap_earnings', 'share_full_time'
    
    # Data targets for reference lines
    targets = {
        'wife_empl': 0.567,
        'divorce_rate': 0.0101,
        'divorce_rate_young': 0.0115,
        'expenditure_x_share': 0.782,
        'βdC': 1.04,
        'gender_gap_earnings': 0.52,  # External moment
        'share_full_time': 0.356      # External moment
    }
    
    # Create subplots
    fig, axes = plt.subplots(len(moment_names), len(results_list), figsize=(4*len(results_list), 3*len(moment_names)))
    
    if len(results_list) == 1:
        axes = axes.reshape(-1, 1)
    
    # First pass: compute y-axis limits for each moment across all parameters
    y_limits = {}
    for i, moment_name in enumerate(moment_names):
        all_values = []
        for results in results_list:
            moment_values = results['moments'][moment_name]
            all_values.extend([v for v in moment_values if not np.isnan(v)])
        
        if all_values:
            y_min = np.min(all_values)
            y_max = np.max(all_values)
            # Add 5% padding
            y_range = y_max - y_min
            y_limits[moment_name] = (y_min - 0.05 * y_range, y_max + 0.05 * y_range)
        else:
            y_limits[moment_name] = (0, 1)
    
    # Second pass: plot with consistent y-axis limits
    for j, results in enumerate(results_list):
        param_name = results['param_name']
        param_values = results['param_values']
        
        for i, moment_name in enumerate(moment_names):
            ax = axes[i, j]
            moment_values = results['moments'][moment_name]
            
            # Plot moment sensitivity
            ax.plot(param_values, moment_values, 'b-', linewidth=2, label='Model')
            
            # Add target line if available
            if moment_name in targets:
                ax.axhline(y=targets[moment_name], color='r', linestyle='--', alpha=0.7, label='Target')
            
            # Add baseline parameter value
            baseline_idx = param_names.index(param_name)
            ax.axvline(x=xc[baseline_idx], color='g', linestyle=':', alpha=0.7, label='Baseline')
            
            # Set consistent y-axis limits for this moment
            ax.set_ylim(y_limits[moment_name])
            
            # Mark NaN values with an 'x'
            nan_indices = np.isnan(moment_values)
            ax.plot(param_values[nan_indices], np.full(np.sum(nan_indices), y_limits[moment_name][0]), 'rx', markersize=10, label='NaN')
            
            ax.set_xlabel(f'{param_name}')
            ax.set_ylabel(moment_name.replace('_', ' ').title())
            ax.grid(True, alpha=0.3)
            
            if i == 0:  # Only show legend for first row
                ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def save_results_to_csv(results_list, save_path):
    """
    Save sensitivity analysis results to CSV files
    
    Parameters:
    results_list: list of results dictionaries
    save_path: base path for saving CSV files
    """
    
    for results in results_list:
        param_name = results['param_name']
        
        # Create DataFrame
        df_data = {'param_value': results['param_values']}
        df_data.update(results['moments'])
        
        df = pd.DataFrame(df_data)
        
        # Save to CSV
        filename = f"{save_path}/sensitivity_{param_name}.csv"
        df.to_csv(filename, index=False)
        print(f"Results for {param_name} saved to {filename}")

# %% Data import and model setup

# Create sample with replacement
N = 10_000  # sample size

# Import information for the sample, then store the relevant variables
baseline_sample = np.array(pd.read_excel(root + '/Output files/data_sample.csv'))

pr = np.ones(baseline_sample.shape[0]) / baseline_sample.shape[0]
indexes = np.array(np.random.choice(baseline_sample[:, 0], size=N, p=pr, replace=True), dtype=np.int64) - 1
final_sample = baseline_sample[:, 1:][indexes]

age_initial = final_sample[:, 0]
age_final = final_sample[:, 1]
cw_cons_share = final_sample[:, 2]
h_income = final_sample[:, 3]
w_income = final_sample[:, 4]
age_marriage = final_sample[:, 5]

# Baseline parameters: [ω,σL,α,χ,wedge]
xc = np.array([0.35870364, 0.00596541, 0.89223739, 2.5, 2.0]) #0.96875457
param_names = ['ω', 'σL', 'α', 'χ', 'wedge']

# Parameter bounds
xl = np.array([0.02, 0.001, 0.1, 0.8, 0.5])
xu = np.array([0.8, 0.2, 0.99, 2.0, 1.2])


# Parametrize the baseline model
par = {'simN': N, 'ω': xc[0], 'σL': xc[1], 'α': xc[2], 'χ': xc[3], 'wedge': xc[4]}
model = brg.HouseholdModelClass(par=par)

# Set the initial conditions for the couples based on baseline sample
model.par.sample_init = age_initial - 20

# Given the parameters, set the initial pareto weight for couples
param = (cw_cons_share / (1.0 - cw_cons_share)) ** model.par.ρ
model.sim.init_power = param / (1.0 + param)

# Set the initial income gridpoints for income, the closest to our value
izm = np.array([np.argmin(np.abs(np.log(model.par.grid_zm)[int(model.par.sample_init[i]), :, 0] - h_income[i])) for i in range(model.par.simN)], dtype=np.int64)
izm[np.isnan(h_income)] = (model.par.num_pm * model.par.num_ϵm) // 2
izw = np.array([np.argmin(np.abs(np.log(model.par.grid_zw)[int(model.par.sample_init[i]), :, 0] - w_income[i])) for i in range(model.par.simN)], dtype=np.int64)
izw[np.isnan(w_income)] = (model.par.num_pw * model.par.num_ϵw) // 2
model.sim.init_z = izm * model.par.num_zm + izw



M = model.copy(name='baseline')
start_time = time.time()
M.solve()
print("--- %s seconds ---" % (time.time() - start_time))
start_time = time.time()
M.simulate()
print("--- %s seconds ---" % (time.time() - start_time))
baseline_moments = compute_moments(M)
print("Baseline moments:")
for moment_name, moment_value in baseline_moments.items():
    print(f"  {moment_name}: {moment_value:.4f}")   




# %% Main execution: run sensitivity analysis for all parameters

if __name__ == '__main__':
    
    # Settings for sensitivity analysis
    n_points = 5  # Number of grid points for each parameter
    
    # Run sensitivity analysis for all parameters
    print("Starting moment sensitivity analysis...")
    
    all_results = []
    
    for param_idx in range(len(param_names)):
        results = sensitivity_analysis(param_idx, xl, xu, n_points=n_points)
        all_results.append(results)
    
    # Create results directory if it doesn't exist
    results_dir = f"{root}/Model/results/sensitivity"
    import os
    os.makedirs(results_dir, exist_ok=True)
    
    # Save results to CSV
    save_results_to_csv(all_results, results_dir)
    
    # Plot results
    plot_path = f"{results_dir}/moment_sensitivity_plots.png"
    plot_sensitivity_results(all_results, save_path=plot_path)
    
    # Print summary
    print("\n" + "="*60)
    print("SENSITIVITY ANALYSIS SUMMARY")
    print("="*60)
    
    for results in all_results:
        param_name = results['param_name']
        print(f"\n{param_name}:")
        print(f"  Range: [{xl[param_names.index(param_name)]:.4f}, {xu[param_names.index(param_name)]:.4f}]")
        print(f"  Baseline: {xc[param_names.index(param_name)]:.4f}")
        
        # Show range of moment values across parameter grid
        for moment_name in ['wife_empl', 'divorce_rate', 'expenditure_x_share', 'βdC', 'gender_gap_earnings', 'share_full_time']:
            moment_vals = np.array(results['moments'][moment_name])
            if not np.all(np.isnan(moment_vals)):
                print(f"  {moment_name}: [{np.nanmin(moment_vals):.4f}, {np.nanmax(moment_vals):.4f}]")
    
    print(f"\nResults saved to: {results_dir}")
    print("Analysis complete!")



# %%
# for key in all_results[5]['moments']:
#     print(key)
#     all_results[5]['moments'][key] = [all_results[5]['moments'][key][i] for i in [0, 2, 4]]
# %%
