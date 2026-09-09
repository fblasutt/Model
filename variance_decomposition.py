# -*- coding: utf-8 -*-
"""
Variance decomposition — limited AND full commitment (single consolidated
script; supersedes the former shapley_decomposition.py).

Kaplan–Violante-style BLOCK decomposition of Var(Δlog y): starting from the
baseline, whole blocks of structural shocks are shut down one at a time,
holding the solved policy functions fixed (fixed-policy counterfactual à la
Huggett–Ventura–Yaron). The blocks are

    persistent : ζ^w + ζ^m   (random-walk income innovations, both spouses)
    transitory : ε^w + ε^m   (transitory income shocks, both spouses)
    love       : ψ^w + ψ^m   (individual-specific random-walk love shocks)

The wife's human-capital DEPRECIATION shock (δ^w) is NOT part of any block
and stays ACTIVE in every counterfactual (floor included); its own one-at-a-
time contribution is reported in a separate table.

Freezing semantics (a frozen block keeps its t=0 entry values):
    ζ    frozen at current VALUE. The persistent grids are nonstationary
         (width ~ sqrt(t)), so freezing the INDEX would let off-median
         agents' income drift as the grid widens. The frozen component maps
         to the two t+1 gridpoints BRACKETING its current value with
         interpolation weights: E[value'] = value — the conditional mean of
         the true RW kernel, minus its innovation variance.
    ε    frozen at the t=0 entry draw (transitions restricted to ϵ' = ϵ).
         The ε grids are time-constant so index = value; NOTE num_ϵ = 2
         (even) means there is no zero midpoint — do not switch to
         mid-forcing unless num_ϵ is made odd.
    ψ    frozen at current VALUE (same bracketing kernel on the widening
         love grids; matters because entry love is off-median for many
         couples under the conditional draw with σL0 > 0).
    δ    (dep table only) frozen at current h: Πh replaced by the identity.

Var(Δlog y) is computed AFTER residualizing Δlog y on age dummies, so the
deterministic life-cycle profile of growth drops out of every counterfactual
— the pooled variance would put it in the floor, unattributable to shocks.

The contribution of block B is V(all) − V(all∖B): one-at-a-time shutdowns,
no exact adding-up (interactions are not allocated). The FLOOR world has all
three blocks frozen simultaneously (δ^w still active): whatever variance
remains there comes from depreciation risk plus deterministic couple-specific
dynamics (trend-driven renegotiations, participation-switch timing).

Simulations per regime: baseline + 3 block shutdowns + floor + δ^w = 6.

Outputs (root + '/Output files/model/'):
    vardec.tex           — channel identity Var(Δlog c^g) = Var(Δlog C)
                           + Var(Δlog s^g) + 2Cov, wife & husband panels,
                           LC & FC rows (via reg_cons_insurance.insurance())
    shockdec_Cpriv.tex   — block decomposition, target C_priv = cw + cm;
                           columns: V_total | persistent | transitory | love;
                           rows LC, FC (the floor is console-only)
    shockdec_sw.tex      — same, target sw = cw/(cw+cm)
    shockdec_sm.tex      — same, target sm = cm/(cw+cm)
    shockdec_dep.tex     — the DEPRECIATION table: δ^w's one-at-a-time
                           contribution for C_priv and sw, LC & FC rows
    events_sw_<tag>.tex  — decomposition of E[(Δlog s_w)^2] by renegotiation
                           direction (toward wife / toward husband / none)

After a run, M_NOSHOCK['LC'] / M_NOSHOCK['FC'] hold the simulated FLOOR
models for interactive inspection.
"""

import copy
import getpass

import numpy as np
import pandas as pd

import Bargaining_numba as brg
import init_conditions as ic
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
# Current estimates from the shared module (sync with calibration.py)
from estimated_params import xc, par_dict, apply_fc_params

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
model.sim.init_z = izw * model.par.num_zm + izm   # FIXED gender swap: wife is the SLOW joint-index component
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
apply_fc_params(m_FC)   # FC-specific [η, σL, β] (estimated_params.xc_full)
m_FC.solve()
m_FC.sim.force_init_love[:] = init_love_lc      # transplant LC's draws
m_FC.simulate()

sample_LC = (age > age_initial[:, None]) & (age <= age_final[:, None])# & (m_LC.sim.couple_lag == 1)  & (m_LC.sim.couple == 1)
sample_FC = (age > age_initial[:, None]) & (age <= age_final[:, None])# & (m_FC.sim.couple_lag == 1)  & (m_FC.sim.couple == 1)


###############################################################################
# Counterfactual machinery                                                    #
###############################################################################

INCOME_SHOCKS = ('zeta_w', 'zeta_m', 'eps_w', 'eps_m')
LOVE_SHOCKS   = ('psi_w', 'psi_m')
DEP_SHOCKS    = ('dep_w',)              # wife's human-capital depreciation

# BLOCKS shut down together (depreciation deliberately NOT a block: it stays
# active in every block counterfactual and in the floor; its own contribution
# is reported separately in shockdec_dep.tex)
BLOCKS = {
    'persistent': ('zeta_w', 'zeta_m'),
    'transitory': ('eps_w',  'eps_m'),
    'love':       ('psi_w',  'psi_m'),
}
BLOCK_KEYS = tuple(BLOCKS)


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
    the constraints in zero_set, then renormalize each column — this pins
    every frozen component at its current INDEX:
        ζ^w zeroed  ⇒  pw' = pw
        ζ^m zeroed  ⇒  pm' = pm
        ε^w zeroed  ⇒  ϵw' = ϵw   (grid time-constant: index = value)
        ε^m zeroed  ⇒  ϵm' = ϵm
    For ζ the index-freeze is then corrected to a VALUE-freeze by
    _value_freeze_redistribute (the persistent grids widen over time, so a
    frozen index would drift in value).
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


def _freeze_kernel(g_now, g_next):
    """
    Value-preserving no-innovation kernel for a nonstationary (widening)
    grid, column-stochastic F[post, initial]: the current value v = g_now[i]
    maps to the two g_next points bracketing v with linear interpolation
    weights, so E[value'] = v exactly (v is always interior since the grid
    widens around the same center). Median 0 maps to 0 with weight 1.
    """
    n = len(g_now)
    F = np.zeros((n, n))
    for i, v in enumerate(g_now):
        j = int(np.clip(np.searchsorted(g_next, v) - 1, 0, n - 2))
        w = float(np.clip((g_next[j+1] - v)/(g_next[j+1] - g_next[j]), 0.0, 1.0))
        F[j, i] += w
        F[j+1, i] += 1.0 - w
    return F


def _value_freeze_redistribute(Pi, par, t, which):
    """
    Correct the ζ INDEX-freeze produced by _filter_transition into a
    VALUE-freeze: the probability block sitting at persistent index i0
    (whose t-value is g_now[i0]) is moved to the two t+1 gridpoints
    bracketing that value, with interpolation weights. Leaves the other
    components of each destination state untouched (block shift by the
    component's stride), so columns stay stochastic.
    """
    if which == 'w':
        nP, stride = par.num_pw, par.num_ϵw*par.num_pm*par.num_ϵm
        g_now  = par.grid_pw[t,   0, ::par.num_ϵw, 0]
        g_next = par.grid_pw[t+1, 0, ::par.num_ϵw, 0]
        comp = _component_arrays(par)[0]
    else:
        nP, stride = par.num_pm, par.num_ϵm
        g_now  = par.grid_pm[t,   0, ::par.num_ϵm, 0]
        g_next = par.grid_pm[t+1, 0, ::par.num_ϵm, 0]
        comp = _component_arrays(par)[2]
    assert len(g_now) == nP and np.all(np.diff(g_now) > 0), "unexpected persistent-grid layout"

    out = np.zeros_like(Pi)
    for i0 in range(nP):
        v = g_now[i0]
        j = int(np.clip(np.searchsorted(g_next, v) - 1, 0, nP - 2))
        w = float(np.clip((g_next[j+1] - v)/(g_next[j+1] - g_next[j]), 0.0, 1.0))
        rows = np.where(comp == i0)[0]
        out[rows + (j   - i0)*stride, :] += w        * Pi[rows, :]
        out[rows + (j+1 - i0)*stride, :] += (1.0-w)  * Pi[rows, :]
    return out


def build_counterfactual_Pi(par, zero_set):
    """List of counterfactual Π[t] (couples' income transitions)."""
    income_zero = zero_set & set(INCOME_SHOCKS)
    if not income_zero:
        return [Pi.copy() for Pi in par.Π]
    out = []
    for t in range(par.T - 1):
        Pi = _filter_transition(par.Π[t], par, income_zero)
        if 'zeta_w' in income_zero: Pi = _value_freeze_redistribute(Pi, par, t, 'w')
        if 'zeta_m' in income_zero: Pi = _value_freeze_redistribute(Pi, par, t, 'm')
        out.append(Pi)
    return out


def build_counterfactual_Pil(par, zero_set):
    """
    Counterfactual love transitions: par.Πl[t] = kron(Πlw_[t], Πlm_[t]) with
    joint index iL = iψw*num_lovem + iψm.
        'psi_w' zeroed ⇒ wife's RW factor replaced by the value-freeze kernel
        'psi_m' zeroed ⇒ husband's RW factor replaced likewise
    Initial draws (Πl0) are untouched.
    """
    love_zero = zero_set & set(LOVE_SHOCKS)
    if not love_zero:
        return [Pi.copy() for Pi in par.Πl]
    out = []
    for t in range(par.T - 1):
        # VALUE-preserving freeze (not the identity): the love grids widen
        # over time, so an index-freeze would let off-median entry love
        # (conditional draw with σL0 > 0) drift deterministically.
        A = (_freeze_kernel(par.grid_lovew_[t], par.grid_lovew_[t+1])
             if 'psi_w' in love_zero else par.Πlw_[t])
        B = (_freeze_kernel(par.grid_lovem_[t], par.grid_lovem_[t+1])
             if 'psi_m' in love_zero else par.Πlm_[t])
        out.append(np.kron(A, B))
    return out


def simulate_counterfactual(m_baseline, zero_set):
    """
    Deep-copy m_baseline, replace par.Π / par.Πl / par.Πh with their
    counterfactual versions, and run simulate(). Solved policies m.sol.*
    are preserved.
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

    if 'dep_w' in zero_set:
        # Freeze the wife's human capital at its current level: Πh replaced
        # by the identity for every wlp at every t (post-retirement periods
        # already use exactly this identity_block in setup).
        identity_block = np.tile(np.eye(m_cf.par.num_h), (m_cf.par.num_wlp, 1, 1))
        for t in range(m_cf.par.T):
            m_cf.par.Πh[t] = identity_block

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
    """
    Sample variance of Δlog(target)_t for couples present in t and t+1,
    residualized on AGE (period) dummies. The raw pooled variance would also
    count the deterministic life-cycle profile of growth (between-age
    variation of mean growth), which is common to every counterfactual and
    would otherwise sit in the floor unattributable to any shock.
    """
    sample1 = np.roll(sample, 1, axis=1)
    sm = (m.sim.couple[sample1] == 1) & (m.sim.couple[sample] == 1)
    x = _series(m, target)
    dx = np.log(x[sample1] / x[sample])[sm]
    tt = np.tile(np.arange(m.par.T), (m.par.simN, 1))
    t_cell = tt[sample1][sm]                 # period of the growth cell
    for tv in np.unique(t_cell):
        g = t_cell == tv
        dx[g] -= dx[g].mean()                # age-dummy residualization
    return dx.var(ddof=1)


###############################################################################
# The counterfactual RUNS (KV block shutdowns), per regime                    #
###############################################################################
# 'baseline'  : all shocks on (the baseline model itself, no re-simulation)
# one per BLOCK: that block frozen, everything else (incl. δ^w) on
# 'floor'     : ALL three blocks frozen simultaneously (δ^w still ACTIVE)
# 'no_dep'    : only δ^w frozen (for the separate depreciation table)

RUNS = {'baseline': set()}
RUNS.update({b: set(BLOCKS[b]) for b in BLOCK_KEYS})
RUNS['floor']  = set().union(*(set(v) for v in BLOCKS.values()))
RUNS['no_dep'] = set(DEP_SHOCKS)


def floor_diagnostics(m, sample, tag):
    """
    What still moves in the FLOOR world? Frozen shocks are not frozen
    FUNDAMENTALS: income still follows its deterministic life-cycle trend and
    assets evolve along couple-specific paths, so outside options drift and
    participation constraints can be crossed deterministically — each couple
    at its own date (which is why age dummies cannot absorb it). On top of
    that, δ^w is still ACTIVE here by design. Split the floor variance by
    event: renegotiation cells, participation-switch cells (no reneg), and
    quiet cells. sw moves ONLY with power, so its floor should load
    ~entirely on the renegotiation cells (and be ~0 under FC).
    """
    sample1 = np.roll(sample, 1, axis=1)
    sm = (m.sim.couple[sample1] == 1) & (m.sim.couple[sample] == 1)
    reneg = m.sim.power[sample1][sm] != m.sim.power_lag[sample1][sm]
    wlpch = m.sim.WLP[sample1][sm] != m.sim.WLP[sample][sm]
    groups = (('renegotiation', reneg),
              ('wlp switch (no reneg)', wlpch & ~reneg),
              ('neither', ~(reneg | wlpch)))
    print(f"  [floor diagnostics {tag}] blocks frozen (dep active): "
          f"reneg freq {reneg.mean():.2%}, wlp-switch freq {wlpch.mean():.2%}")
    for tgt in ('sw', 'C_priv'):
        x = _series(m, tgt)
        dx = np.log(x[sample1] / x[sample])[sm]
        dx = dx - dx.mean()
        vtot = np.mean(dx**2)
        line = f"    E[(dlog {tgt})^2] = {100*vtot:.3f} (x100):"
        for name, g in groups:
            c = np.mean(dx**2 * g)
            share = c/vtot if vtot > 0 else 0.0
            line += f"   {name} {100*c:.3f} ({share:.0%})"
        print(line)


# The simulated FLOOR model of each regime is KEPT here for inspection after
# the run: e.g. M_NOSHOCK['LC'].sim.power, .sim.WLP, .sim.Cw, ...
# (fixed baseline policies; persistent+transitory+love frozen, dep active).
M_NOSHOCK = {}


def run_counterfactuals(m_base, sample, tag):
    """One V[run][target] = Var(Δlog target) per run in RUNS."""
    print(f"Running {len(RUNS)-1} fixed-policy counterfactual simulations ({tag})...")
    V = {}
    for name, zero_set in RUNS.items():
        if not zero_set:
            m_cf = m_base                    # baseline: everything on
        else:
            print(f"  freezing {sorted(zero_set)} ...")
            m_cf = simulate_counterfactual(m_base, zero_set)
        V[name] = {tgt: var_growth(m_cf, sample, tgt) for tgt in TARGETS}
        if name == 'floor':
            floor_diagnostics(m_cf, sample, tag)
            M_NOSHOCK[tag] = m_cf            # keep for interactive inspection
        elif zero_set:
            del m_cf
    return V


REGIMES = (('LC', m_LC, sample_LC), ('FC', m_FC, sample_FC))
Vcache = {tag: run_counterfactuals(m, sample, tag) for tag, m, sample in REGIMES}


# ---------------------------------------------------------------------------
# KV block contributions: c_B = V(all) − V(all∖B); + floor and dep numbers
# ---------------------------------------------------------------------------
results = {}
for tag, _, _ in REGIMES:
    V = Vcache[tag]
    results[tag] = {}
    for tgt in TARGETS:
        V_all = V['baseline'][tgt]
        con   = {b: V_all - V[b][tgt] for b in BLOCK_KEYS}
        c_dep = V_all - V['no_dep'][tgt]
        V_flr = V['floor'][tgt]
        results[tag][tgt] = {'V_all': V_all, 'contrib': con,
                             'c_dep': c_dep, 'V_floor': V_flr}

        print(f"\n=== KV block decomposition of Var(Δlog {tgt}) — {tag} ===")
        print(f"  V(all shocks) = {100*V_all:.3f}   V(floor: blocks frozen, dep on) = {100*V_flr:.3f}   (x100)")
        print(f"  {'block':12s} {'V(all)-V(all-B)':>16s} {'(share)':>8s}")
        for b in BLOCK_KEYS:
            share = con[b]/V_all if abs(V_all) > 1e-12 else 0.0
            print(f"  {b:12s} {100*con[b]:16.3f} {share:8.1%}")
        share = c_dep/V_all if abs(V_all) > 1e-12 else 0.0
        print(f"  {'dep_w (sep.)':12s} {100*c_dep:16.3f} {share:8.1%}")
        print(f"  SUM(blocks) = {100*sum(con.values()):.3f}  vs  "
              f"V(all)-V(floor) = {100*(V_all - V_flr):.3f}   [KV: no exact adding-up]")


# ---------------------------------------------------------------------------
# LaTeX tables: one row of contributions (x100), one scriptsize row of shares.
# shockdec_<target>.tex columns: V_total | persistent | transitory | love.
# Rows: LC, FC. (The floor is computed and printed to console but not
# tabulated.)
# ---------------------------------------------------------------------------
def _v(x):  return '%.2f' % (100.0*x) if abs(x) > 1e-6 else '0.00'
def _p(x):  return '%.0f' % (100.0*x) if abs(x) > 1e-3 else '0'
_pct = lambda x: r'{\scriptsize (' + _p(x) + r'\%)}'


def block_rows(tag, target, label):
    r = results[tag][target]
    Vt = r['V_all']
    cells_v = [_v(Vt)] + [_v(r['contrib'][b]) for b in BLOCK_KEYS]
    row1 = label + ' & ' + ' & '.join(cells_v)
    if abs(Vt) < 1e-8:
        return row1                          # V_total ≈ 0 → drop the % row
    cells_p = [_pct(1.0)] + [_pct(r['contrib'][b]/Vt) for b in BLOCK_KEYS]
    row2 = ' & ' + ' & '.join(cells_p)
    return row1 + r' \\[-0.5ex]' + '\n' + row2


for tgt, fname in (('C_priv', 'shockdec_Cpriv.tex'), ('sw', 'shockdec_sw.tex'),
                   ('sm', 'shockdec_sm.tex')):
    body = '\n'.join([
        block_rows('LC', tgt, '\hspace{8pt}Limited commitment') + r' \\',
        block_rows('FC', tgt, '\hspace{8pt}Full commitment'),     # no trailing \\
    ])
    with open(root + '/Output files/model/' + fname, 'w') as f:
        f.write(body)


# --- the DEPRECIATION table: δ^w's one-at-a-time contribution --------------
# Columns: C_priv: V_total, ΔV_dep | sw: V_total, ΔV_dep. Rows: LC, FC.
def dep_row(tag, label):
    rc, rs = results[tag]['C_priv'], results[tag]['sw']
    row1 = (label + ' & ' + _v(rc['V_all']) + ' & ' + _v(rc['c_dep']) +
            ' & ' + _v(rs['V_all']) + ' & ' + _v(rs['c_dep']))
    shc = rc['c_dep']/rc['V_all'] if abs(rc['V_all']) > 1e-8 else 0.0
    shs = rs['c_dep']/rs['V_all'] if abs(rs['V_all']) > 1e-8 else 0.0
    row2 = (' & ' + _pct(1.0) + ' & ' + _pct(shc) +
            ' & ' + (_pct(1.0) if abs(rs['V_all']) > 1e-8 else '') +
            ' & ' + (_pct(shs) if abs(rs['V_all']) > 1e-8 else ''))
    return row1 + r' \\[-0.5ex]' + '\n' + row2


with open(root + '/Output files/model/shockdec_dep.tex', 'w') as f:
    f.write('\n'.join([
        dep_row('LC', 'Limited commitment') + r' \\',
        dep_row('FC', 'Full commitment'),
    ]))


###############################################################################
# Channel table vardec.tex (BASELINE identity only, no shock split):          #
#   Var(Δlog c^g) = Var(Δlog C) + Var(Δlog s^g) + 2 Cov(Δlog C, Δlog s^g)     #
# via the insurance() routine (which also writes its pass-through side files) #
###############################################################################

def vardec_row(B, spouse, label):
    """
    Two LaTeX rows for one (regime × spouse) case from a results dict B
    returned by insurance(): variance values (×100) on row 1, share of total
    in scriptsize on row 2. Empty cells (" & & ") sit in narrow operator
    columns of the header.
    """
    d = B['vardec_w'] if spouse == 'w' else B['vardec_m']
    row1 = (label + ' & ' +
            _v(d['V_total']) + ' & & ' +
            _v(d['V_agg'])   + ' & & ' +
            _v(d['V_reb'])   + ' & & ' +
            _v(d['2Cov']))
    row2 = (' & ' + _pct(1.0)         + ' & & ' +
                    _pct(d['sh_agg']) + ' & & ' +
                    _pct(d['sh_reb']) + ' & & ' +
                    _pct(d['sh_cov']))
    return row1 + r' \\[-0.5ex]' + '\n' + row2


B_LC = insurance(m_LC, sample_LC, shock_type='permanent', shock_gender='Male',
                 consumption_gender='Male', name_file='VardecLC', name_line='Limited commitment')
B_FC = insurance(m_FC, sample_FC, shock_type='permanent', shock_gender='Male',
                 consumption_gender='Male', name_file='VardecFC', name_line='Full commitment')

table = '\n'.join([
    r'\textit{A. Wife} & & & & & & & \\',
    r'\addlinespace',
    vardec_row(B_LC, 'w', '\hspace{8pt}Limited comm.') + r' \\',
    vardec_row(B_FC, 'w', '\hspace{8pt}Full comm.')    + r' \\',
    r'\addlinespace',
    r'\textit{B. Husband} & & & & & & & \\',
    r'\addlinespace',
    vardec_row(B_LC, 'm', '\hspace{8pt}Limited comm.') + r' \\',
    vardec_row(B_FC, 'm', '\hspace{8pt}Full comm.'),          # no trailing \\
])
with open(root + '/Output files/model/vardec.tex', 'w') as f:
    f.write(table)


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



print("\nDone. Wrote vardec.tex, shockdec_Cpriv.tex, shockdec_sw.tex, "
      "shockdec_sm.tex, shockdec_dep.tex, events_sw_{LC,FC}.tex.")
