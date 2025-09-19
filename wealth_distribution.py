# %% Imports

import numpy as np
from matplotlib import pyplot as plt
import getpass


# Root directory setup
user = getpass.getuser()
if user == "sara":
    root = '/Users/sara/Dropbox/Family Risk Sharing'
elif user == "32489":
    root = '/Users/32489/Dropbox/Family Risk Sharing'
else:
    raise RuntimeError(f"Unknown user: {user}")


# %% 

# Average household income
household_income = M.sim.incw + M.sim.incm
household_income_nonretired = household_income[:, 0:M.par.Tr]
couple_nonretired = M.sim.couple[:,0:M.par.Tr]
male_income_nonretired = M.sim.incm[:,0:M.par.Tr]
mean_Y_husbands = np.nanmean(male_income_nonretired, where=couple_nonretired)
mean_Y_couples = np.nanmean(household_income_nonretired, where=couple_nonretired)

# Wealth distribution over all ages, couples only
couple_assets = M.sim.A[M.sim.couple]/mean_Y_husbands

# Wealth distribution excluding retired individuals
A_nonretired = M.sim.A[:,0:M.par.Tr]
couple_nonretired_assets = A_nonretired[couple_nonretired]/mean_Y_husbands

# Percentiles table for couple_nonretired_assets
percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
percentile_values = np.percentile(couple_nonretired_assets, percentiles)


# Data values from empirical distribution
data_values = [-4.25,  -1.57,  -.67, 0, 1.12, 4.05, 8.55, 12.67, 25.70]

print("Percentiles of Assets/Mean Income (Couples, Non-Retired):")
print("=" * 75)
print(f"{'Percentile':<10} {'Model':<15} {'Data':<15} {'Difference':<15}")
print("=" * 75)
for p, model_val, data_val in zip(percentiles, percentile_values, data_values):
    diff = model_val - data_val
    print(f"{p:2}%{'':<8} {model_val:<15.6f} {data_val:<15.6f} {diff:<15.6f}")
print("=" * 75)

# Additional statistics
print(f"Mean (Model): {np.mean(couple_nonretired_assets):10.6f}    Mean (Data):  2.967")
print(f"Std Dev (Model): {np.std(couple_nonretired_assets):6.6f}    Std Dev (Data): 6.482")
print()


# Average household income over the life cycle
mean_Y_age_couples = np.nanmean(household_income, axis=0, where=M.sim.couple)
plt.figure(figsize=(10, 6))
plt.plot(np.arange(68), mean_Y_age_couples, label='Mean Household Income (Couples)', color='blue')
plt.title("Average Household Income Over the Life Cycle (Couples)")
plt.xlabel("Age")
plt.ylabel("Mean Income")
plt.grid()
plt.legend()
plt.show()  

# Assets over the life cycle
mean_A_age_couples = np.mean(M.sim.A, axis=0, where=M.sim.couple)/mean_Y_husbands
plt.plot(np.arange(68), mean_A_age_couples)
plt.title("Mean Assets Over the Life Cycle (Couples)")
plt.xlabel("Age")
plt.ylabel("Mean Assets Over Mean Income")
plt.grid()
plt.show()

# Wealth distribution
plt.figure(figsize=(12, 8))
plt.hist(couple_assets, bins=50, alpha=0.6, edgecolor='black', label='All Ages', color='blue')
plt.hist(couple_nonretired_assets, bins=50, alpha=0.6, edgecolor='black', label='Non-Retired Only', color='red')
plt.title("Distribution of Assets (Couples) - All Ages vs Non-Retired")
plt.xlabel("Assets over Mean Income")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()






# %%
