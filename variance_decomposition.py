# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 15:31:56 2024

@author: 32489

This script:
  1. Solves and simulates the limited-commitment (LC) and full-commitment (FC)
     models on the JPSC sample.
  2. Builds the channel variance decomposition (Var(Δlog c^g) into aggregate +
     rebargaining + covariance), via the existing insurance() routine, and
     writes its LaTeX table to vardec.tex.
  3. Runs a fixed-policy shock decomposition (à la Huggett, Ventura & Yaron 2011):
     for each structural shock x ∈ {ζ^w, ζ^m, ε^w, ε^m, ψ^w, ψ^m}, re-simulates
     the model with x's innovations zeroed out (policy functions unchanged),
     measures the drop in Var(Δlog y) for two targets, and writes one LaTeX
     table per target:
        * shockdec_Cpriv.tex — total within-couple private consumption (cw+cm),
        * shockdec_sw.tex    — wife's private consumption share cw/(cw+cm).
     Each table has two rows: limited and full commitment.

The channel decomposition (Step 2) is reported separately for wife and husband.
"""

import copy
import getpass

import numpy as np
import pandas as pd

import Bargaining_numba as brg
from reg_cons_insurance import insurance


# Initialize seed
np.random.seed(10)


# ---------------------------------------------------------------------------
# Root path
# ---------------------------------------------------------------------------
user = getpass.getuser()
if user == "sara":
    root = '/Users/sara/Dropbox/Family Risk Sharing'
elif user == "32489":
    root = '/Users/32489/Dropbox/Family Risk Sharing'
else:
    raise RuntimeError(f"Unknown user: {user}")


# ---------------------------------------------------------------------------
# Sample
# ---------------------------------------------------------------------------
N = 10_000  # sample size

baseline_sample = np.array(pd.read_excel(root + '/Output files/data_sample.csv'))
baseline_sample = baseline_sample[~np.isnan(baseline_sample).any(axis=1)]
baseline_sample[:, 0] = np.arange(len(baseline_sample))

pr = np.ones(baseline_sample.shape[0]) / baseline_sample.shape[0]
indexes = np.array(np.random.choice(baseline_sample[:, 0], size=N, p=pr, replace=True),
                   dtype=np.int32) - 1
final_sample = baseline_sample[:, 1:][indexes]

age_initial   = final_sample[:, 0]
age_final     = final_sample[:, 1]
cw_cons_share = final_sample[:, 2]
h_income      = final_sample[:, 3]
w_income      = final_sample[:, 4]
age_marriage  = final_sample[:, 5]
year          = final_sample[:, 6]
assets        = final_sample[:, 7] * np.mean(np.exp(h_income))


# ---------------------------------------------------------------------------
# Parameterize the model
# ---------------------------------------------------------------------------
# Internal parameters: [ω, σL, α, ρ, wedge, β]
xc=np.array([0.55, 0.1       , 0.85, 1.2, 0.929     ,1.        ])

par = {'simN':N,'ω': xc[0],'σL':xc[1],'α':xc[2],'ρ':xc[3],'wedge':xc[4],'β':xc[5],'sample_init':np.array(age_marriage-20,dtype=np.int_)}
model = brg.HouseholdModelClass(par=par)


# ---------------------------------------------------------------------------
# Initial conditions (from data sample)
# ---------------------------------------------------------------------------
param = (cw_cons_share / (1.0 - cw_cons_share)) ** model.par.ρ
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
calendar_year = age - age_initial[:, None] + year[:, None]
policy = np.maximum(calendar_year[:, 0], 2007)
age_policy = np.array(np.where(policy[:, None] == calendar_year)[1], dtype=np.int32)


# ---------------------------------------------------------------------------
# Solve & simulate baseline models
# ---------------------------------------------------------------------------
print("Solving LC model...")
m_LC = model.copy(name='numba_new_copy')
m_LC.solve()
m_LC.simulate()
sample_LC = (age > age_initial[:, None]) & (age <= age_final[:, None]) & (m_LC.sim.couple_lag == 1)

# Extract LC's per-agent initial-love grid index. We feed these into the FC
# baseline simulation so both regimes start from the same per-agent love
# state. (LC's PC check is stricter than FC's at sample-init, so without this
# step FC ends up with a wider initial-love distribution, which contaminates
# any cross-regime variance comparison.)
init_love_lc = np.array(
    [m_LC.sim.love[i, int(m_LC.par.sample_init[i])] for i in range(m_LC.par.simN)],
    dtype=np.int_,
)

print("Solving FC model...")
m_FC = model.copy(name='numba_new_copy')
m_FC.par.full = True
m_FC.solve()
m_FC.sim.force_init_love[:] = init_love_lc      # transplant LC's draws
m_FC.simulate()
sample_FC = (age > age_initial[:, None]) & (age <= age_final[:, None]) & (m_FC.sim.couple_lag == 1)

sample_LC = (age > age_initial[:, None]) & (age <= age_final[:, None]) & (m_LC.sim.couple_lag == 1) 
sample_FC = (age > age_initial[:, None]) & (age <= age_final[:, None]) & (m_FC.sim.couple_lag == 1) & (m_LC.sim.couple_lag == 1)


###############################################################################
#                                                                             #
# Channel variance decomposition: Var(Δlog c^g) = Var(Δlog C) + Var(Δlog s^g) #
#                                                + 2 Cov(...)                 #
#                                                                             #
###############################################################################

def vardec_row(B, spouse, label):
    """
    Two LaTeX rows for one (regime × spouse) case from a results dict B
    returned by insurance(): variance values (×100) on row 1, share of total
    in scriptsize on row 2 with negative \\\\[-0.5ex] separator.
    Empty cells (" & & ") sit in narrow operator columns of the header.
    """
    d = B['vardec_w'] if spouse == 'w' else B['vardec_m']
    def v(x):  return '%.2f' % (100.0*x) if abs(x) > 1e-6 else '0.00'
    def p(x):  return '%.0f' % (100.0*x) if abs(x) > 1e-3 else '0'
    pct = lambda x: r'{\scriptsize (' + p(x) + r'\%)}'
    row1 = (label + ' & ' +
            v(d['V_total']) + ' & & ' +
            v(d['V_agg'])   + ' & & ' +
            v(d['V_reb'])   + ' & & ' +
            v(d['2Cov']))
    row2 = (' & ' + pct(1.0)         + ' & & ' +
                    pct(d['sh_agg']) + ' & & ' +
                    pct(d['sh_reb']) + ' & & ' +
                    pct(d['sh_cov']))
    return row1 + r' \\[-0.5ex]' + '\n' + row2


B_LC = insurance(m_LC, sample_LC, ...)
B_FC = insurance(m_FC, sample_FC, ...)

table = '\n'.join([
    r'\textit{A. Wife} & & & & & & & \\',
    r'\addlinespace',
    vardec_row(B_LC, 'w', 'Limited commitment') + r' \\',
    vardec_row(B_FC, 'w', 'Full commitment')    + r' \\',
    r'\addlinespace',
    r'\textit{B. Husband} & & & & & & & \\',
    r'\addlinespace',
    vardec_row(B_LC, 'm', 'Limited commitment') + r' \\',
    vardec_row(B_FC, 'm', 'Full commitment'),     # no trailing \\
])
with open(root + '/Output files/model/vardec.tex', 'w') as f:
    f.write(table)


###############################################################################
#                                                                             #
# Shock decomposition: marginal contribution of each structural shock to      #
# Var(Δlog c^g), via fixed-policy re-simulation (Huggett, Ventura & Yaron     #
# 2011 "Sources of Lifetime Inequality").                                     #
#                                                                             #
# For each shock x, we re-simulate the estimated model with x's innovations   #
# zeroed out, holding the solved policy functions m.sol.* fixed (no re-       #
# solve). The drop in Var(Δlog c^g) is the marginal contribution of x.        #
# In a nonlinear model contributions don't sum to the baseline variance;      #
# the residual is reported as 'interaction'.                                  #
#                                                                             #
###############################################################################

INCOME_SHOCKS = ('zeta_w', 'zeta_m', 'eps_w', 'eps_m')
LOVE_SHOCKS   = ('psi_w',  'psi_m')
SHOCKS        = INCOME_SHOCKS + LOVE_SHOCKS


# ---------------------------------------------------------------------------
# iz decomposition: iz = pw*(num_ϵw·num_pm·num_ϵm) + ϵw*(num_pm·num_ϵm)
#                       + pm*num_ϵm + ϵm
# ---------------------------------------------------------------------------
def _component_arrays(par):
    """Vectorized (pw, ϵw, pm, ϵm) arrays indexed by iz."""
    n_total = par.num_pw * par.num_ϵw * par.num_pm * par.num_ϵm
    iz_arr = np.arange(n_total)
    ϵm = iz_arr % par.num_ϵm
    pm = (iz_arr // par.num_ϵm) % par.num_pm
    ϵw = (iz_arr // (par.num_ϵm * par.num_pm)) % par.num_ϵw
    pw = iz_arr // (par.num_ϵm * par.num_pm * par.num_ϵw)
    return pw, ϵw, pm, ϵm


def _filter_transition(Pi, par, zero_set):
    """
    Filter income transition matrix Pi by zeroing transitions that violate
    the constraints in zero_set, then renormalize each column.
        ζ^w zeroed  ⇒  pw' = pw
        ζ^m zeroed  ⇒  pm' = pm
        ε^w zeroed  ⇒  ϵw' = num_ϵw // 2  (mid-grid = zero transitory)
        ε^m zeroed  ⇒  ϵm' = num_ϵm // 2
    """
    pw, ϵw, pm, ϵm = _component_arrays(par)
    n = len(pw)
    mid_ϵw = par.num_ϵw // 2
    mid_ϵm = par.num_ϵm // 2

    Pi_new = Pi.copy()
    for j in range(n):  # source state
        mask = np.ones(n, dtype=bool)
        if 'zeta_w' in zero_set: mask &= (pw == pw[j])
        if 'zeta_m' in zero_set: mask &= (pm == pm[j])
        if 'eps_w'  in zero_set: mask &= (ϵw == mid_ϵw)
        if 'eps_m'  in zero_set: mask &= (ϵm == mid_ϵm)
        col_filtered = np.where(mask, Pi_new[:, j], 0.0)
        s = col_filtered.sum()
        if s > 0:
            Pi_new[:, j] = col_filtered / s
        else:
            # No allowed destination (numerical edge): self-transition.
            Pi_new[:, j] = 0.0
            Pi_new[j, j] = 1.0
    return Pi_new


def build_counterfactual_Pi(par, zero_set):
    """List of counterfactual Π[t] (couples' income transitions)."""
    income_zero = zero_set & set(INCOME_SHOCKS)
    if not income_zero:
        return [Pi.copy() for Pi in par.Π]
    return [_filter_transition(par.Π[t], par, income_zero) for t in range(par.T - 1)]


def build_counterfactual_Pil(par, zero_set):
    """
    List of counterfactual love transitions par.Πl[t]. Since
    par.Πl[t] = kron(par.Πlw[t], par.Πlm[t]), zeroing ψ^w replaces Πlw with I,
    zeroing ψ^m replaces Πlm with I, zeroing both gives Πl = I.
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


def simulate_counterfactual(m_baseline, zero_set):
    """
    Deep-copy m_baseline, replace par.Π / par.Πl with their counterfactual
    versions, and run simulate(). Solved policy functions m.sol.* are
    preserved. Returns the modified model.
    """
    zero_set = set(zero_set)
    m_cf = copy.deepcopy(m_baseline)

    if zero_set & set(INCOME_SHOCKS):
        Π_cf = build_counterfactual_Pi(m_cf.par, zero_set)
        for t in range(m_cf.par.T - 1):
            m_cf.par.Π[t] = Π_cf[t]

    if zero_set & set(LOVE_SHOCKS):
        Πl_cf = build_counterfactual_Pil(m_cf.par, zero_set)
        for t in range(m_cf.par.T - 1):
            m_cf.par.Πl[t] = Πl_cf[t]

    # NOT TOUCHED:
    #  par.Πs  — single-spell income transitions (post-divorce paths).
    #  par.Πh  — human-capital transitions (depreciation is shut down here).
    #  par.Πl0 — initial-period love distribution (innovations only).

    m_cf.simulate()
    return m_cf


# ---------------------------------------------------------------------------
# Targets and Var(Δlog target) extractor
# ---------------------------------------------------------------------------
TARGETS = ('C_tot', 'C_priv', 'Cw', 'Cm', 'sw', 'sm')


def _series(m, target):
    """Time series (simN × T) for the requested target."""
    if   target == 'C_tot':  return m.sim.C_tot
    elif target == 'C_priv': return m.sim.Cw + m.sim.Cm
    elif target == 'Cw':     return m.sim.Cw
    elif target == 'Cm':     return m.sim.Cm
    elif target == 'sw':     return m.sim.Cw / (m.sim.Cw + m.sim.Cm)
    elif target == 'sm':     return m.sim.Cm / (m.sim.Cw + m.sim.Cm)
    raise ValueError(f"Unknown target: {target!r}. Pick one of {TARGETS}.")


def var_growth(m, sample, target):
    """Sample variance of Δlog(target)_t for couples present in t and t+1."""
    sample1 = np.roll(sample, 1, axis=1)
    sm = (m.sim.couple[sample1] == 1) & (m.sim.couple[sample] == 1)
    x = _series(m, target)
    dx = np.log(x[sample1] / x[sample])[sm]
    return dx.var(ddof=1)


# ---------------------------------------------------------------------------
# Marginal shock decomposition
# ---------------------------------------------------------------------------
def run_counterfactuals(m_baseline, verbose=False):
    """
    Run the 6 fixed-policy counterfactual simulations once and cache them.
    Returns dict keyed by shock name; the baseline model itself is the
    'baseline' key. Reuse to compute Var(Δlog target) for any target without
    re-simulating.
    """
    cfs = {'baseline': m_baseline}
    for x in SHOCKS:
        if verbose:
            print(f"  zero {x} ...")
        cfs[x] = simulate_counterfactual(m_baseline, zero_set={x})
    return cfs


def shock_decomposition(cfs, sample, target):
    """
    Marginal contribution of each shock to Var(Δlog target), from the cached
    counterfactual simulations in `cfs` (output of run_counterfactuals).

    Returns a dict with one entry per shock (= V_baseline - V_without_shock),
    plus 'V_total' and 'interaction' (= V_total - sum of contributions).
    """
    V_base = var_growth(cfs['baseline'], sample, target)
    out = {'target': target, 'V_total': V_base}
    for x in SHOCKS:
        out[x] = V_base - var_growth(cfs[x], sample, target)
    out['interaction'] = V_base - sum(out[x] for x in SHOCKS)
    return out


# ---------------------------------------------------------------------------
# LaTeX row builder for the shock-decomposition table
# ---------------------------------------------------------------------------
def shockdec_row(dec, label):
    """
    Two LaTeX rows for one (regime × target) case, mirroring vardec_row.
        Row 1: V_total, then ΔV_x for each shock, then interaction (×100).
        Row 2: share of V_total in scriptsize, in parentheses.
    If V_total ≈ 0 (e.g., consumption share under full commitment, where
    Δlog s^g ≡ 0 for intact couples), the percentage row is suppressed
    because the shares are all 0/0; only the value row is returned.
    No operator columns — the table header lists shock names directly.
    """
    def v(x):  return '%.2f' % (100.0*x) if abs(x) > 1e-6 else '0.00'
    def p(x):  return '%.0f' % (100.0*x) if abs(x) > 1e-3 else '0'
    pct = lambda x: r'{\scriptsize (' + p(x) + r'\%)}'

    V = dec['V_total']
    keys = list(SHOCKS) + ['interaction']
    cells_v = [v(V)] + [v(dec[k]) for k in keys]
    row1 = label + ' & ' + ' & '.join(cells_v)

    if abs(V) < 1e-8:
        return row1                         # V_total ≈ 0 → drop the % row

    cells_p = [pct(1.0)] + [pct(dec[k] / V) for k in keys]
    row2 = ' & ' + ' & '.join(cells_p)
    return row1 + r' \\[-0.5ex]' + '\n' + row2


# ---------------------------------------------------------------------------
# Run shock decomposition for both regimes and emit one table per target.
#
# Two output tables, each with two rows (LC, FC):
#   * shockdec_Cpriv.tex — target = C_priv (within-couple private cw + cm),
#     measuring the role of each shock for total private consumption volatility.
#   * shockdec_sw.tex    — target = sw (wife's private share cw/(cw+cm)),
#     measuring the role of each shock for the consumption share.
# ---------------------------------------------------------------------------
print("Running fixed-policy counterfactual simulations (LC)...")
cfs_LC = run_counterfactuals(m_LC, verbose=True)

print("Running fixed-policy counterfactual simulations (FC)...")
cfs_FC = run_counterfactuals(m_FC, verbose=True)


def write_shockdec_table(target, filename):
    """Build a 2-row LaTeX table (LC + FC) for the given target and save it."""
    dec_LC = shock_decomposition(cfs_LC, sample_LC, target=target)
    dec_FC = shock_decomposition(cfs_FC, sample_FC, target=target)
    body = '\n'.join([
        shockdec_row(dec_LC, 'Limited commitment') + r' \\',
        shockdec_row(dec_FC, 'Full commitment'),     # no trailing \\
    ])
    with open(root + '/Output files/model/' + filename, 'w') as f:
        f.write(body)


write_shockdec_table(target='C_priv', filename='shockdec_Cpriv.tex')
write_shockdec_table(target='sw',     filename='shockdec_sw.tex')

print("Done. Wrote vardec.tex, shockdec_Cpriv.tex, shockdec_sw.tex.")
