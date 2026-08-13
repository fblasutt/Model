# -*- coding: utf-8 -*-
"""
Shapley shock decomposition — limited AND full commitment.

Exact Shapley decomposition of Var(Δlog y) across the 6 structural shocks of
the current model:

    ζ^w, ζ^m : persistent (random-walk) income innovations
    ε^w, ε^m : transitory income shocks
    ψ^w, ψ^m : individual-specific random-walk love shocks (initial value 0)

For every subset S of active shocks (2^6 = 64 configurations) each regime is
re-simulated with the complement zeroed out, holding the solved policy
functions fixed (Huggett–Ventura–Yaron fixed-policy counterfactual).
Zeroing semantics — ALL shut-down shocks are frozen at their t=0 entry value:

    ζ    frozen at current value (random walk: kill innovations, keep levels)
    ε    frozen at the t=0 entry draw (transitions restricted to ϵ' = ϵ)
    ψ    frozen at current value (random walk; entry value is 0 for everyone)

The Shapley value of shock x is the weighted average of its marginal
contribution V(S ∪ {x}) − V(S) over all orderings, and satisfies exact
adding-up:  Σ_x φ_x = V(all shocks) − V(no shocks).  The leave-one-out
(Kaplan-style) numbers are also echoed from the same cache for comparison.

Outputs (root + '/Output files/model/'), one per regime tag ∈ {LC, FC}:
    shapley_Cpriv_<tag>.tex — Shapley by shock, target C_priv = cw + cm
    shapley_sw_<tag>.tex    — Shapley by shock, target sw = cw/(cw+cm)
    events_sw_<tag>.tex     — decomposition of E[(Δlog s_w)^2] by renegotiation
                              direction (toward wife / toward husband / none)
plus one combined channel table (BASELINE identity, no shock split):
    shapley_channels.tex    — Var(Δlog c^g) = Var(Δlog C) + Var(Δlog s^g)
                              + 2Cov, wife & husband panels, LC & FC rows
"""

import copy
import getpass
from math import factorial
from itertools import combinations

import numpy as np
import pandas as pd

import Bargaining_numba as brg
import init_conditions as ic


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
# Sample (identical to variance_decomposition.py)
# ---------------------------------------------------------------------------
N = 10_000  # sample size

baseline_sample = np.array(pd.read_excel(root + '/Output files/data_sample.csv'))
baseline_sample = baseline_sample[~np.isnan(baseline_sample).any(axis=1)]
baseline_sample[:, 0] = np.arange(len(baseline_sample))

pr = np.ones(baseline_sample.shape[0]) / baseline_sample.shape[0]
indexes = np.array(np.random.choice(baseline_sample[:, 0], size=N, p=pr, replace=True),
                   dtype=np.int32) - 1
final_sample = baseline_sample[:, 1:][indexes]

age_initial   = final_sample[:, 0]*0+25   # forced to 25, as in calibration.py (t=0 = age 25)
age_final     = final_sample[:, 1]
cw_cons_share = final_sample[:, 2]
h_income      = final_sample[:, 3]
w_income      = final_sample[:, 4]
age_marriage  = final_sample[:, 5]*0+25   # forced to 25, as in calibration.py
year          = final_sample[:, 6]
assets        = final_sample[:, 7] * np.mean(np.exp(h_income))

# Pre-drawn uniforms for the posterior-draw initial income split (drawn AFTER
# the sample so sample selection is unchanged; FIXED across evaluations so
# SMM objectives stay deterministic)
u_init_w=np.random.rand(N);u_init_m=np.random.rand(N)
σME2_init=0.0  # measurement-error variance in observed entry income (0 = off)


# ---------------------------------------------------------------------------
# Parameterize the model
# ---------------------------------------------------------------------------
# Current estimates [η, σL, α, ρ, Ω, β] from the shared module (sync with calibration.py)
from estimated_params import xc, par_dict

par = par_dict(N, np.array(age_marriage-25,dtype=np.int_))
model = brg.HouseholdModelClass(par=par)


# ---------------------------------------------------------------------------
# Initial conditions (from data sample)
# ---------------------------------------------------------------------------
param = (cw_cons_share / (1.0 - cw_cons_share)) ** model.par.ρ
model.sim.init_power = param / (1.0 + param)

gridzw = model.par.grid_zw[:, :, np.linspace(0, model.par.num_z - 1, model.par.num_zm, dtype=np.int_)]
gridzm = model.par.grid_zm[:, :, :model.par.num_zw]

izm=ic.draw_init_iz(h_income,model.par.sample_init,gridzm,model.par.grid_pm,model.par.grid_ϵm,u_init_m,σME2=σME2_init)
izm[np.isnan(h_income)] = (model.par.num_pm * model.par.num_ϵm) // 2
izw=ic.draw_init_iz(w_income,model.par.sample_init,gridzw,model.par.grid_pw,model.par.grid_ϵw,u_init_w,σME2=σME2_init)
izw[np.isnan(w_income)] = (model.par.num_pw * model.par.num_ϵw) // 2
model.sim.init_z = izm * model.par.num_zm + izw
model.sim.init_A = assets

age = (np.cumsum(np.ones((model.par.simN, model.par.T)), axis=1) - 1) + 25


# ---------------------------------------------------------------------------
# Solve & simulate the two baselines (LC and FC)
# ---------------------------------------------------------------------------
print("Solving LC model...")
m_LC = model.copy(name='numba_new_copy')
m_LC.solve()
m_LC.simulate()

# Extract LC's per-agent initial-love grid index and feed it into the FC
# baseline so both regimes start from the same per-agent love state (LC's PC
# check is stricter than FC's at sample-init; without this step FC ends up
# with a wider initial-love distribution).
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

sample_LC = (age > age_initial[:, None]) & (age <= age_final[:, None]) & (m_LC.sim.couple_lag == 1)
sample_FC = (age > age_initial[:, None]) & (age <= age_final[:, None]) & (m_FC.sim.couple_lag == 1) & (m_LC.sim.couple_lag == 1)


###############################################################################
# Counterfactual machinery (identical semantics to variance_decomposition.py, #
# EXCEPT ε: frozen at the t=0 entry draw instead of forced to the mid-grid)   #
###############################################################################

INCOME_SHOCKS = ('zeta_w', 'zeta_m', 'eps_w', 'eps_m')
LOVE_SHOCKS   = ('psi_w', 'psi_m')
SHOCKS        = INCOME_SHOCKS + LOVE_SHOCKS
n_shocks      = len(SHOCKS)


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
    the constraints in zero_set, then renormalize each column. Every shut
    shock is FROZEN at its current value, so it keeps the t=0 entry draw
    (from init_z) for the whole simulation:
        ζ^w zeroed  ⇒  pw' = pw
        ζ^m zeroed  ⇒  pm' = pm
        ε^w zeroed  ⇒  ϵw' = ϵw
        ε^m zeroed  ⇒  ϵm' = ϵm
    """
    pw, ϵw, pm, ϵm = _component_arrays(par)
    n = len(pw)

    Pi_new = Pi.copy()
    for j in range(n):  # source state
        mask = np.ones(n, dtype=bool)
        if 'zeta_w' in zero_set: mask &= (pw == pw[j])
        if 'zeta_m' in zero_set: mask &= (pm == pm[j])
        if 'eps_w'  in zero_set: mask &= (ϵw == ϵw[j])
        if 'eps_m'  in zero_set: mask &= (ϵm == ϵm[j])
        col_filtered = np.where(mask, Pi_new[:, j], 0.0)
        s = col_filtered.sum()
        if s > 0:
            Pi_new[:, j] = col_filtered / s
        else:
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
    Counterfactual love transitions: par.Πl[t] = kron(Πlw_[t], Πlm_[t]) with
    joint index iL = iψw*num_lovem + iψm.
        'psi_w' zeroed ⇒ wife's RW factor replaced by the identity
        'psi_m' zeroed ⇒ husband's RW factor replaced by the identity
    A frozen RW keeps its current level; since both spouses enter at ψ = 0,
    freezing kills the shock entirely. Initial draws (Πl0) are untouched.
    """
    love_zero = zero_set & set(LOVE_SHOCKS)
    if not love_zero:
        return [Pi.copy() for Pi in par.Πl]
    I_w = np.eye(par.num_lovew)
    I_m = np.eye(par.num_lovem)
    out = []
    for t in range(par.T - 1):
        A = I_w if 'psi_w' in love_zero else par.Πlw_[t]
        B = I_m if 'psi_m' in love_zero else par.Πlm_[t]
        out.append(np.kron(A, B))
    return out


def simulate_counterfactual(m_baseline, zero_set):
    """
    Deep-copy m_baseline, replace par.Π / par.Πl with their counterfactual
    versions, and run simulate(). Solved policies m.sol.* are preserved.
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

    m_cf.simulate()
    return m_cf


# ---------------------------------------------------------------------------
# Targets and Var(Δlog target) extractor
# ---------------------------------------------------------------------------
TARGETS = ('C_priv', 'Cw', 'Cm', 'sw', 'sm')


def _series(m, target):
    if   target == 'C_priv': return m.sim.Cw + m.sim.Cm
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


###############################################################################
# All 2^7 subset counterfactuals, per regime                                  #
###############################################################################
# V[frozenset(active shocks)] = {target: Var(Δlog target)}. The subset with
# ALL shocks active is the baseline itself (no re-simulation needed). Models
# are discarded after their variances are extracted to bound memory.

def run_all_subsets(m_base, sample, tag):
    print(f"Running {2**n_shocks - 1} fixed-policy counterfactual simulations ({tag})...")
    V = {}
    count = 0
    for k in range(n_shocks + 1):
        for active in combinations(SHOCKS, k):
            active = frozenset(active)
            zero_set = set(SHOCKS) - active
            if not zero_set:
                m_cf = m_base                     # baseline: everything on
            else:
                m_cf = simulate_counterfactual(m_base, zero_set)
            V[active] = {tgt: var_growth(m_cf, sample, tgt) for tgt in TARGETS}
            if zero_set:
                del m_cf
            count += 1
            if count % 16 == 0:
                print(f"  {count}/{2**n_shocks} subsets done")
    return V


ALL  = frozenset(SHOCKS)
NONE = frozenset()

REGIMES = (('LC', m_LC, sample_LC), ('FC', m_FC, sample_FC))
Vcache = {tag: run_all_subsets(m, sample, tag) for tag, m, sample in REGIMES}


# ---------------------------------------------------------------------------
# Shapley values (+ leave-one-out from the same cache)
# ---------------------------------------------------------------------------
# w(a) = a! (n-1-a)! / n!  — weight of a coalition of size a not containing x
W = {a: factorial(a) * factorial(n_shocks - 1 - a) / factorial(n_shocks)
     for a in range(n_shocks)}


def shapley(V, target):
    phi = {}
    for x in SHOCKS:
        others = [s for s in SHOCKS if s != x]
        val = 0.0
        for k in range(len(others) + 1):
            for S in combinations(others, k):
                S = frozenset(S)
                val += W[k] * (V[S | {x}][target] - V[S][target])
        phi[x] = val
    return phi


def leave_one_out(V, target):
    return {x: V[ALL][target] - V[ALL - {x}][target] for x in SHOCKS}


results = {}
for tag, _, _ in REGIMES:
    V = Vcache[tag]
    results[tag] = {}
    for tgt in TARGETS:
        phi = shapley(V, tgt)
        loo = leave_one_out(V, tgt)
        V_all, V_none = V[ALL][tgt], V[NONE][tgt]
        results[tag][tgt] = {'phi': phi, 'loo': loo, 'V_all': V_all, 'V_none': V_none}

        print(f"\n=== Shapley decomposition of Var(Δlog {tgt}) — {tag} ===")
        print(f"  V(all shocks) = {100*V_all:.3f}   V(no shocks) = {100*V_none:.3f}   (x100)")
        print(f"  {'shock':10s} {'Shapley':>10s} {'(share)':>8s} {'leave-1-out':>12s}")
        for x in SHOCKS:
            share = phi[x]/V_all if abs(V_all) > 1e-12 else 0.0
            print(f"  {x:10s} {100*phi[x]:10.3f} {share:8.1%} {100*loo[x]:12.3f}")
        ssum = sum(phi.values())
        print(f"  SUM(Shapley) = {100*ssum:.3f}  vs  V(all)-V(none) = {100*(V_all - V_none):.3f}"
              f"   [adding-up gap {100*abs(ssum - (V_all - V_none)):.2e}]")


# ---------------------------------------------------------------------------
# LaTeX tables: one row of Shapley values (x100), one row of shares, plus
# the no-shock floor V(∅) as its own column so the rows add up exactly:
# V_total = Σ φ_x + V(no shocks).
# ---------------------------------------------------------------------------
def shapley_rows(tag, target, label):
    r = results[tag][target]
    def v(x):  return '%.2f' % (100.0*x) if abs(x) > 1e-6 else '0.00'
    def p(x):  return '%.0f' % (100.0*x) if abs(x) > 1e-3 else '0'
    pct = lambda x: r'{\scriptsize (' + p(x) + r'\%)}'
    Vt = r['V_all']
    cells_v = [v(Vt)] + [v(r['phi'][x]) for x in SHOCKS] + [v(r['V_none'])]
    row1 = label + ' & ' + ' & '.join(cells_v)
    if abs(Vt) < 1e-8:
        return row1                         # V_total ≈ 0 → drop the % row
    cells_p = [pct(1.0)] + [pct(r['phi'][x] / Vt) for x in SHOCKS] + [pct(r['V_none'] / Vt)]
    row2 = ' & ' + ' & '.join(cells_p)
    return row1 + r' \\[-0.5ex]' + '\n' + row2


for tag, _, _ in REGIMES:
    label = 'Limited commitment' if tag == 'LC' else 'Full commitment'
    with open(root + f'/Output files/model/shapley_Cpriv_{tag}.tex', 'w') as f:
        f.write(shapley_rows(tag, 'C_priv', label))
    with open(root + f'/Output files/model/shapley_sw_{tag}.tex', 'w') as f:
        f.write(shapley_rows(tag, 'sw', label))


###############################################################################
# Channel table (BASELINE identity only, no shock split):                     #
#   Var(Δlog c^g) = Var(Δlog C) + Var(Δlog s^g) + 2 Cov(Δlog C, Δlog s^g)     #
###############################################################################
# log c^g = log C_priv + log s^g holds cell by cell, so 2Cov is the exact
# residual V_total − V_agg − V_share on the same growth cells. Layout mirrors
# vardec.tex: value row + scriptsize share-of-total row, with empty operator
# spacer columns (=, +, +) between the four variance columns.

def channel_row(tag, g, label):
    cg, sg = ('Cw', 'sw') if g == 'w' else ('Cm', 'sm')
    r = results[tag]
    V_tot, V_agg, V_sh = r[cg]['V_all'], r['C_priv']['V_all'], r[sg]['V_all']
    V_cov = V_tot - V_agg - V_sh
    def v(x):  return '%.2f' % (100.0*x) if abs(x) > 1e-6 else '0.00'
    def p(x):  return '%.0f' % (100.0*x) if abs(x) > 1e-3 else '0'
    pct = lambda x: r'{\scriptsize (' + p(x) + r'\%)}'
    row1 = (label + ' & ' + v(V_tot) + ' & & ' + v(V_agg) + ' & & ' +
            v(V_sh) + ' & & ' + v(V_cov))
    if abs(V_tot) < 1e-8:
        return row1
    row2 = (' & ' + pct(1.0) + ' & & ' + pct(V_agg/V_tot) + ' & & ' +
            pct(V_sh/V_tot) + ' & & ' + pct(V_cov/V_tot))
    return row1 + r' \\[-0.5ex]' + '\n' + row2


channels_table = '\n'.join([
    r'\textit{A. Wife} & & & & & & & \\',
    r'\addlinespace',
    channel_row('LC', 'w', 'Limited commitment') + r' \\',
    channel_row('FC', 'w', 'Full commitment')    + r' \\',
    r'\addlinespace',
    r'\textit{B. Husband} & & & & & & & \\',
    r'\addlinespace',
    channel_row('LC', 'm', 'Limited commitment') + r' \\',
    channel_row('FC', 'm', 'Full commitment'),      # no trailing \\
])
with open(root + '/Output files/model/shapley_channels.tex', 'w') as f:
    f.write(channels_table)

for tag, g in (('LC', 'w'), ('LC', 'm'), ('FC', 'w'), ('FC', 'm')):
    cg, sg = ('Cw', 'sw') if g == 'w' else ('Cm', 'sm')
    r = results[tag]
    V_tot, V_agg, V_sh = r[cg]['V_all'], r['C_priv']['V_all'], r[sg]['V_all']
    print(f"channels {tag} {'wife' if g=='w' else 'husband':8s}: "
          f"V(c)={100*V_tot:7.3f} = V(C)={100*V_agg:7.3f} + V(s)={100*V_sh:7.3f} "
          f"+ 2Cov={100*(V_tot-V_agg-V_sh):7.3f}   (x100)")


###############################################################################
# Event-direction decomposition of E[(Δlog s_w)^2] — per-regime baseline      #
###############################################################################
# Every Δlog s_w observation is classified by what the bargaining weight did
# in the SECOND period of the growth cell: renegotiation toward the wife
# (power up), toward the husband (power down), or no renegotiation. The three
# groups' contributions to E[(Δlog s_w)^2] add up exactly. (Under FC the
# weight never moves, so everything lands in 'No renegotiation'.)

def event_decomposition(m, sample, tag):
    sample1 = np.roll(sample, 1, axis=1)
    sm = (m.sim.couple[sample1] == 1) & (m.sim.couple[sample] == 1)

    sw_ser = _series(m, 'sw')
    dsw = np.log(sw_ser[sample1] / sw_ser[sample])[sm]
    pw_now = m.sim.power[sample1][sm]
    pw_lag = m.sim.power_lag[sample1][sm]

    tol = 1e-12
    up   = pw_now > pw_lag + tol      # renegotiation toward wife
    down = pw_now < pw_lag - tol      # renegotiation toward husband
    none = ~(up | down)

    E_tot = np.mean(dsw**2)
    groups = [('Toward wife', up), ('Toward husband', down), ('No renegotiation', none)]

    print(f"\n=== Event decomposition of E[(dlog s_w)^2] — {tag} baseline ===")
    print(f"  E[(dlog s_w)^2] = {100*E_tot:.3f}  (x100)")
    rows = []
    for name, g in groups:
        contrib = np.sum(dsw[g]**2) / len(dsw)          # additive contribution
        freq = g.mean()
        share = contrib/E_tot if E_tot > 0 else 0.0
        print(f"  {name:18s} freq {freq:6.1%}   contribution {100*contrib:8.3f}  ({share:6.1%})")
        rows.append((name, freq, contrib))

    def v(x):  return '%.2f' % (100.0*x)
    def p(x):  return '%.0f' % (100.0*x)
    lines = []
    for name, freq, contrib in rows:
        share = contrib / E_tot if E_tot > 0 else 0.0
        lines.append(f'{name} & {p(freq)}\\% & {v(contrib)} & {p(share)}\\%')
    table = (' \\\\\n'.join(lines) + ' \\\\\n\\midrule\n' +
             f'Total & 100\\% & {v(E_tot)} & 100\\%')
    with open(root + f'/Output files/model/events_sw_{tag}.tex', 'w') as f:
        f.write(table)


for tag, m, sample in REGIMES:
    event_decomposition(m, sample, tag)

print("\nDone. Wrote shapley_Cpriv_{LC,FC}.tex, shapley_sw_{LC,FC}.tex, "
      "shapley_channels.tex, events_sw_{LC,FC}.tex.")
