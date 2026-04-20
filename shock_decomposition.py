# -*- coding: utf-8 -*-
"""
Variance decomposition of individual consumption volatility by shock source
via fixed-policy re-simulation (à la Huggett, Ventura & Yaron 2011 AER,
"Sources of Lifetime Inequality").

Idea
----
For each shock x ∈ {ζ^w, ζ^m, ε^w, ε^m, ψ^w, ψ^m} (four income shocks + two
match-quality "love" shocks), recompute Var(Δlog TARGET) on a counterfactual
simulation in which x's innovations are zeroed out, while holding the solved
policy functions `m.sol.*` fixed (no re-solve). The gap vs. baseline variance
measures the direct contribution of x to consumption volatility, abstracting
from general-equilibrium responses of policy functions themselves.

TARGET is selectable and can be any of:
    'C_tot'  — total household consumption (C_w + C_m + d)
    'C_priv' — within-couple private consumption (C_w + C_m)
    'Cw'     — wife's private consumption
    'Cm'     — husband's private consumption
    'sw'     — wife's private-consumption share  C_w/(C_w+C_m)
    'sm'     — husband's private-consumption share  C_m/(C_w+C_m)

Reported as a marginal decomposition: 1 baseline simulation + 1 counterfactual
per shock (7 simulations total for 6 shocks). In a nonlinear model the
contributions do not sum exactly to V_baseline; the residual is reported
as 'interaction'.

Human-capital depreciation shocks are not included because depreciation is
shut down in the current calibration.

Integration with this codebase
------------------------------
The income process is a Markov chain on `m.par.Π[t]` (couples) and
`m.par.Πs[t]` (singles). The state index `iz` decomposes as

        iz = pw * (num_ϵw * num_pm * num_ϵm)
             + ϵw * (num_pm * num_ϵm)
             + pm * num_ϵm
             + ϵm

so we can zero a specific income shock by filtering the transition matrix:
    ζ^w zeroed  ⇒  keep only next states with pw' = pw
    ζ^m zeroed  ⇒  keep only next states with pm' = pm
    ε^w zeroed  ⇒  keep only next states with ϵw' = num_ϵw//2
    ε^m zeroed  ⇒  keep only next states with ϵm' = num_ϵm//2
then renormalize each column.

Match-quality (love) shocks live on a separate Markov chain
`m.par.Πl[t] = kron(m.par.Πlw[t], m.par.Πlm[t])`. We zero them by plugging
identity matrices into the relevant factor:
    ψ^w zeroed  ⇒  Πl[t] = kron(I_{num_lovew}, Πlm[t])
    ψ^m zeroed  ⇒  Πl[t] = kron(Πlw[t], I_{num_lovem})
    both        ⇒  Πl[t] = I_{num_love}

The policy functions in `m.sol` are unchanged, so agents continue to behave
as if shocks were drawn from the baseline distributions; only the *realized*
shock paths differ. This is exactly the fixed-policy counterfactual.
"""

import copy

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
INCOME_SHOCKS = ('zeta_w', 'zeta_m', 'eps_w', 'eps_m')
LOVE_SHOCKS   = ('psi_w', 'psi_m')
SHOCKS        = INCOME_SHOCKS + LOVE_SHOCKS      # full set used by decomposition


# ---------------------------------------------------------------------------
# iz decomposition
# ---------------------------------------------------------------------------
def _iz_components(iz, par):
    """Return (pw, ϵw, pm, ϵm) decomposition of compound state iz."""
    ϵm = iz % par.num_ϵm
    pm = (iz // par.num_ϵm) % par.num_pm
    ϵw = (iz // (par.num_ϵm * par.num_pm)) % par.num_ϵw
    pw = iz // (par.num_ϵm * par.num_pm * par.num_ϵw)
    return pw, ϵw, pm, ϵm


def _component_arrays(par):
    """Vectorized (pw, ϵw, pm, ϵm) arrays indexed by iz — built once, reused."""
    n_total = par.num_pw * par.num_ϵw * par.num_pm * par.num_ϵm
    iz_arr = np.arange(n_total)
    ϵm = iz_arr % par.num_ϵm
    pm = (iz_arr // par.num_ϵm) % par.num_pm
    ϵw = (iz_arr // (par.num_ϵm * par.num_pm)) % par.num_ϵw
    pw = iz_arr // (par.num_ϵm * par.num_pm * par.num_ϵw)
    return pw, ϵw, pm, ϵm


# ---------------------------------------------------------------------------
# Build counterfactual transition matrices
# ---------------------------------------------------------------------------
def _filter_transition(Pi, par, zero_set):
    """
    Zero out transitions that violate the `zero_set` constraints and
    renormalize each column. Input Pi has shape (n_total, n_total).
    Convention: Pi[to, from] is Prob(state_next = to | state_now = from).
    """
    pw, ϵw, pm, ϵm = _component_arrays(par)   # each shape (n_total,)
    n = len(pw)
    mid_ϵw = par.num_ϵw // 2
    mid_ϵm = par.num_ϵm // 2

    Pi_new = Pi.copy()

    for j in range(n):  # iterate over source states
        # Mask of allowed destination states given current (pw[j], ϵw[j], pm[j], ϵm[j])
        mask = np.ones(n, dtype=bool)
        if 'zeta_w' in zero_set: mask &= (pw == pw[j])
        if 'zeta_m' in zero_set: mask &= (pm == pm[j])
        if 'eps_w'  in zero_set: mask &= (ϵw == mid_ϵw)
        if 'eps_m'  in zero_set: mask &= (ϵm == mid_ϵm)

        col = Pi_new[:, j]
        col_filtered = np.where(mask, col, 0.0)
        s = col_filtered.sum()
        if s > 0:
            Pi_new[:, j] = col_filtered / s
        else:
            # No allowed destination (numerical edge case): self-transition.
            Pi_new[:, j] = 0.0
            Pi_new[j, j] = 1.0

    return Pi_new


def build_counterfactual_Pi(par, zero_set):
    """Return a list of counterfactual Π[t] (couples' income transitions) with
    income shocks in zero_set zeroed. Ignores non-income entries of zero_set."""
    income_zero = zero_set & set(INCOME_SHOCKS)
    if not income_zero:
        return [Pi.copy() for Pi in par.Π]
    return [_filter_transition(par.Π[t], par, income_zero) for t in range(par.T - 1)]


# ---------------------------------------------------------------------------
# Love-shock transitions: par.Πl[t] = kron(par.Πlw[t], par.Πlm[t])
# State indexing: love = ilw * num_lovem + ilm
# ---------------------------------------------------------------------------
def build_counterfactual_Pil(par, zero_set):
    """
    Return a list of counterfactual love-transition matrices par.Πl[t] with
    love-shock innovations in zero_set zeroed. Implemented by substituting
    the identity for the relevant spouse's Πl[wm] component in the kron
    product. Ignores non-love entries of zero_set.
    """
    love_zero = zero_set & set(LOVE_SHOCKS)
    if not love_zero:
        return [Pi.copy() for Pi in par.Πl]

    Iw = np.eye(par.num_lovew)
    Im = np.eye(par.num_lovem)
    out = []
    for t in range(par.T - 1):
        Aw = Iw if 'psi_w' in love_zero else par.Πlw[t]
        Am = Im if 'psi_m' in love_zero else par.Πlm[t]
        out.append(np.kron(Aw, Am))
    return out


# ---------------------------------------------------------------------------
# Run a counterfactual simulation
# ---------------------------------------------------------------------------
def simulate_counterfactual(m_baseline, zero_set):
    """
    Deep-copy the baseline model, replace the relevant transition matrices
    (par.Π for income shocks, par.Πl for love shocks) with their
    counterfactual versions, and run simulate(). The solved policy functions
    m.sol.* are preserved.

    Returns the modified model instance; baseline is untouched.
    """
    zero_set = set(zero_set)
    m_cf = copy.deepcopy(m_baseline)

    # Couples' income transitions
    if zero_set & set(INCOME_SHOCKS):
        Π_cf = build_counterfactual_Pi(m_cf.par, zero_set)
        for t in range(m_cf.par.T - 1):
            m_cf.par.Π[t] = Π_cf[t]

    # Couples' love (match-quality) transitions
    if zero_set & set(LOVE_SHOCKS):
        Πl_cf = build_counterfactual_Pil(m_cf.par, zero_set)
        for t in range(m_cf.par.T - 1):
            m_cf.par.Πl[t] = Πl_cf[t]

    # NOT TOUCHED:
    #  * m_cf.par.Πs — single-spell income transitions (post-divorce paths).
    #  * m_cf.par.Πh — human-capital transitions (depreciation is shut down
    #    in this calibration, so Πh is already effectively inert).
    #  * m_cf.par.Πl0 — initial-period love distribution (we're decomposing
    #    innovations, not initial conditions).

    m_cf.simulate()
    return m_cf


# ---------------------------------------------------------------------------
# Extract Var(Δlog TARGET) on the same sample convention used in insurance()
# ---------------------------------------------------------------------------
TARGETS = ('C_tot', 'C_priv', 'Cw', 'Cm', 'sw', 'sm')


def _series(m, target):
    """Return the time series (simN × T) for the requested target."""
    if   target == 'C_tot':  return m.sim.C_tot
    elif target == 'C_priv': return m.sim.Cw + m.sim.Cm
    elif target == 'Cw':     return m.sim.Cw
    elif target == 'Cm':     return m.sim.Cm
    elif target == 'sw':     return m.sim.Cw / (m.sim.Cw + m.sim.Cm)
    elif target == 'sm':     return m.sim.Cm / (m.sim.Cw + m.sim.Cm)
    raise ValueError(f"Unknown target: {target!r}. Pick one of {TARGETS}.")


def var_growth(m, sample, target):
    """
    Sample variance of Δlog(target)_t for couples present in both t and t+1.
    target ∈ TARGETS.
    """
    sample1 = np.roll(sample, 1, axis=1)
    sm = (m.sim.couple[sample1] == 1) & (m.sim.couple[sample] == 1)
    x = _series(m, target)
    dx = np.log(x[sample1] / x[sample])[sm]
    return dx.var(ddof=1)


# ---------------------------------------------------------------------------
# Shock decomposition (marginal contributions)
# ---------------------------------------------------------------------------
def shock_decomposition(m_baseline, sample, target='Cw', verbose=False):
    """
    Decompose Var(Δlog TARGET) by shock source via fixed-policy counterfactuals.

    Runs 1 baseline + 6 counterfactual simulations (one per shock). In a
    nonlinear model, contributions do not sum to V_baseline; the residual is
    returned as 'interaction'.

    Parameters
    ----------
    m_baseline : solved & simulated model (baseline).
    sample     : boolean index array defining the analysis sample (same
                 convention as insurance()).
    target     : one of TARGETS = ('C_tot','C_priv','Cw','Cm','sw','sm').
    verbose    : print progress if True.

    Returns
    -------
    dict with entries:
        {shock_name : V_baseline - V_without_shock   for each shock in SHOCKS},
        'interaction' : V_baseline - sum of the above,
        'V_total'     : V_baseline,
        'target'      : target string (for reporting).
    """
    if target not in TARGETS:
        raise ValueError(f"Unknown target: {target!r}. Pick one of {TARGETS}.")

    V_base = var_growth(m_baseline, sample, target)
    out = {'target': target, 'V_total': V_base}
    for x in SHOCKS:
        if verbose:
            print(f"  counterfactual (target={target}): zero {x}")
        m_cf = simulate_counterfactual(m_baseline, zero_set={x})
        V_cf = var_growth(m_cf, sample, target)
        out[x] = V_base - V_cf
    out['interaction'] = V_base - sum(out[x] for x in SHOCKS)
    return out


# ---------------------------------------------------------------------------
# Convenience: shares of total variance
# ---------------------------------------------------------------------------
def shares_of_total(dec):
    """Return {shock: contribution / V_total} for reporting in the paper."""
    V = dec['V_total']
    return {x: dec[x] / V for x in SHOCKS if x in dec}


# ---------------------------------------------------------------------------
# Example usage (mirrors variance_decomposition.py)
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import Bargaining_numba as brg
    import pandas as pd
    import getpass

    np.random.seed(10)

    user = getpass.getuser()
    if user == 'sara':
        root = '/Users/sara/Dropbox/Family Risk Sharing'
    elif user == '32489':
        root = '/Users/32489/Dropbox/Family Risk Sharing'
    else:
        raise RuntimeError(f'Unknown user: {user}')

    # --- Build the baseline sample exactly as in variance_decomposition.py ---
    N = 10_000
    baseline_sample = np.array(pd.read_excel(root + '/Output files/data_sample.csv'))
    baseline_sample = baseline_sample[~np.isnan(baseline_sample).any(axis=1)]
    baseline_sample[:, 0] = np.arange(len(baseline_sample))
    pr = np.ones(baseline_sample.shape[0]) / baseline_sample.shape[0]
    indexes = np.array(np.random.choice(baseline_sample[:, 0], size=N, p=pr, replace=True),
                       dtype=np.int32) - 1
    final_sample = baseline_sample[:, 1:][indexes]
    age_initial  = final_sample[:, 0]
    age_final    = final_sample[:, 1]
    cw_share     = final_sample[:, 2]
    h_income     = final_sample[:, 3]
    w_income     = final_sample[:, 4]
    age_marriage = final_sample[:, 5]
    year         = final_sample[:, 6]
    assets       = final_sample[:, 7] * np.mean(np.exp(h_income))

    xc = np.array([0.53315238, 0.1, 0.80884379, 1.15375904, 0.915, 1.0])
    par = {'simN': N, 'ν': xc[0], 'σL': xc[1], 'α': xc[2], 'ρ': xc[3],
           'wedge': xc[4],
           'sample_init': np.array(age_marriage - 20, dtype=np.int_)}
    model = brg.HouseholdModelClass(par=par)

    param = (cw_share / (1.0 - cw_share)) ** model.par.ρ
    model.sim.init_power = param / (1.0 + param)
    gridzw = model.par.grid_zw[:, :, np.linspace(0, model.par.num_z - 1, model.par.num_zm, dtype=np.int_)]
    gridzm = model.par.grid_zm[:, :, :model.par.num_zw]
    izm = np.array([np.argmin(np.abs(np.log(gridzm)[int(model.par.sample_init[i]), 0, :, 0] - h_income[i]))
                    for i in range(model.par.simN)], dtype=np.int32)
    izm[np.isnan(h_income)] = (model.par.num_pm * model.par.num_ϵm) // 2
    izw = np.array([np.argmin(np.abs(np.log(gridzw)[int(model.par.sample_init[i]), 0, :, 0] - w_income[i]))
                    for i in range(model.par.simN)], dtype=np.int32)
    izw[np.isnan(w_income)] = (model.par.num_pw * model.par.num_ϵw) // 2
    model.sim.init_z = izm * model.par.num_zm + izw
    model.sim.init_A = assets

    age = (np.cumsum(np.ones((model.par.simN, model.par.T)), axis=1) - 1) + 20

    # --- Solve & simulate baseline (limited commitment) ---
    print("Solving baseline LC model...")
    m_LC = model.copy(name='numba_new_copy')
    m_LC.solve()
    m_LC.simulate()
    sample_LC = ((age > age_initial[:, None]) &
                 (age <= age_final[:, None]) &
                 (m_LC.sim.couple_lag == 1))

    # --- Run the decomposition for each target of interest ---
    for target in ('C_tot', 'C_priv', 'Cw', 'Cm', 'sw', 'sm'):
        print(f"\n=== target = {target} ===")
        dec = shock_decomposition(m_LC, sample_LC, target=target, verbose=True)
        print({k: f'{v:.5f}' if isinstance(v, float) else v
               for k, v in dec.items()})
        print('shares of V_total:',
              {k: f'{v:.1%}' for k, v in shares_of_total(dec).items()})
