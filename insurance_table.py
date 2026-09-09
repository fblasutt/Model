# -*- coding: utf-8 -*-
"""
Sources-of-individual-consumption-insurance table.

For wife and husband, against own permanent and own transitory income shocks,
under limited and full commitment, this script:

  1. Solves and simulates the LC and FC models on the JPSC sample (same
     setup as variance_decomposition.py).
  2. Calls insurance() from reg_cons_insurance.py for each combination
     and extracts the K1-K6 channel decomposition stored in ins_dec.
  3. Writes a single LaTeX table (insurance_decomposition.tex) with the
     column structure of the existing policy-experiment tables (e.g.,
     Table table:insurance_decomposition_alimony in the paper):

       Passive | Active | Tax | Savings | Private Exp. share | Reneg. | Total

     organized as two panels (A. Limited commitment, B. Full commitment),
     with four rows per panel (wife/permanent, wife/transitory,
     husband/permanent, husband/transitory).
"""

import getpass

import numpy as np
import pandas as pd

import Bargaining_numba as brg
import init_conditions as ic
from reg_cons_insurance import insurance


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
N = 10_000
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
# Current estimates [η,σL,α,ρ,Ω,β] from the shared module (sync with calibration.py;
# NB: position 4 is Ω, the match-quality disagreement shock — not the old 'wedge')
from estimated_params import xc, par_dict, apply_fc_params
par = par_dict(N, np.array(age_marriage-25,dtype=np.int_))
model = brg.HouseholdModelClass(par=par)


# ---------------------------------------------------------------------------
# Initial conditions
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
# Solve & simulate the two regimes
# ---------------------------------------------------------------------------
print("Solving LC model...")
m_LC = model.copy(name='numba_new_copy')
m_LC.solve()
m_LC.simulate()
sample_LC = (age > age_initial[:, None]) & (age <= age_final[:, None]) & (m_LC.sim.couple_lag == 1)

# Equalize the initial-love draw across regimes by transplanting LC's per-agent
# initial-love grid indices into the FC baseline simulation. (Without this,
# FC's laxer mutual-consent PC check at sample-init would yield a wider
# initial-love distribution than LC's, contaminating any cross-regime
# comparison of consumption variances.)
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
sample_FC = (age > age_initial[:, None]) & (age <= age_final[:, None]) & (m_FC.sim.couple_lag == 1)


sample_LC = (age > age_initial[:, None]) & (age <= age_final[:, None]) & (m_LC.sim.couple_lag == 1)  
sample_FC = (age > age_initial[:, None]) & (age <= age_final[:, None]) & (m_FC.sim.couple_lag == 1) & (m_LC.sim.couple_lag == 1)


# ---------------------------------------------------------------------------
# Build one LaTeX row per (regime, consumption-spouse, shock-spouse, shock-type)
# combination. Keyword-only arguments to make the call to insurance() unambiguous.
# ---------------------------------------------------------------------------
def insurance_row(*, m, sample, consumption_gender, shock_gender,
                  shock_type, label, name_file):
    """
    Run insurance() and return one LaTeX row in the alimony/GWG-table format.
    The shock-gender and consumption-gender are independent so we can report
    both own-shock insurance (shock_gender == consumption_gender) and cross-
    spouse-shock insurance (shock_gender != consumption_gender).
    Numbers are in percentage points (×100) with one decimal.
    """
    res = insurance(m=m,
                    sample=sample,
                    shock_type=shock_type,
                    shock_gender=shock_gender,
                    consumption_gender=consumption_gender,
                    name_file=name_file,
                    name_line=label)
    d = res['ins_dec']
    p31 = lambda x: '%3.1f' % (100.0 * x)
    return (label
            + ' & ' + p31(d['Passive_insurance'])
            + ' & ' + p31(d['Active_insuranc'])    # note: existing typo in dict key
            + ' & ' + p31(d['Taxes'])
            + ' & ' + p31(d['Self_insurance'])
            + ' & ' + p31(d['private_shift'])
            + ' & ' + p31(d['bargaining_shift'])
            + ' & ' + p31(d['ind_con_ins']))


def make_rows(m, sample, regime_tag):
    """
    Eight rows for one regime: for each spouse (wife / husband), the four
    insurance contributions against {own, partner} × {persistent, transitory}
    income shocks.
    """
    common = dict(m=m, sample=sample)
    return [
        # ---- Wife's consumption (consumption_gender='Female') ----
        insurance_row(**common, consumption_gender='Female', shock_gender='Female',
                      shock_type='permanent',
                      label=r'$\zeta^w$ (own, persistent)',
                      name_file=f'ins_{regime_tag}_w_zw'),
        insurance_row(**common, consumption_gender='Female', shock_gender='Female',
                      shock_type='transitory',
                      label=r'$\varepsilon^w$ (own, transitory)',
                      name_file=f'ins_{regime_tag}_w_ew'),
        insurance_row(**common, consumption_gender='Female', shock_gender='Male',
                      shock_type='permanent',
                      label=r'$\zeta^m$ (partner, persistent)',
                      name_file=f'ins_{regime_tag}_w_zm'),
        insurance_row(**common, consumption_gender='Female', shock_gender='Male',
                      shock_type='transitory',
                      label=r'$\varepsilon^m$ (partner, transitory)',
                      name_file=f'ins_{regime_tag}_w_em'),
        # ---- Husband's consumption (consumption_gender='Male') ----
        insurance_row(**common, consumption_gender='Male', shock_gender='Male',
                      shock_type='permanent',
                      label=r'$\zeta^m$ (own, persistent)',
                      name_file=f'ins_{regime_tag}_h_zm'),
        insurance_row(**common, consumption_gender='Male', shock_gender='Male',
                      shock_type='transitory',
                      label=r'$\varepsilon^m$ (own, transitory)',
                      name_file=f'ins_{regime_tag}_h_em'),
        insurance_row(**common, consumption_gender='Male', shock_gender='Female',
                      shock_type='permanent',
                      label=r'$\zeta^w$ (partner, persistent)',
                      name_file=f'ins_{regime_tag}_h_zw'),
        insurance_row(**common, consumption_gender='Male', shock_gender='Female',
                      shock_type='transitory',
                      label=r'$\varepsilon^w$ (partner, transitory)',
                      name_file=f'ins_{regime_tag}_h_ew'),
    ]


rows_LC = make_rows(m_LC, sample_LC, 'LC')
rows_FC = make_rows(m_FC, sample_FC, 'FC')


# ---------------------------------------------------------------------------
# Assemble the table body. 8 columns (label + 7 numeric). Two top-level
# panels (LC, FC); inside each, two sub-panels (wife / husband consumption)
# of four rows each.
# ---------------------------------------------------------------------------
EMPTY_PANEL_TAIL = r' & & & & & & & \\'

def render_panel(rows, last_terminator):
    """Render an 8-row panel (4 wife rows then 4 husband rows) with sub-headers."""
    lines = []
    # Sub-panel: wife
    lines.append(r'\quad\textit{Wife consumption}' + EMPTY_PANEL_TAIL)
    for r in rows[:4]:
        lines.append(r'\quad\quad ' + r + r' \\')
    # Sub-panel: husband
    lines.append(r'\quad\textit{Husband consumption}' + EMPTY_PANEL_TAIL)
    for r in rows[4:7]:
        lines.append(r'\quad\quad ' + r + r' \\')
    # Last row carries the panel-end terminator
    lines.append(r'\quad\quad ' + rows[7] + ' ' + last_terminator)
    return '\n'.join(lines)


table = '\n'.join([
    r'\multicolumn{8}{l}{\textit{A. Limited commitment}}\\[1.5ex]',
    render_panel(rows_LC, r'\\[3.5ex]'),
    r'\multicolumn{8}{l}{\textit{B. Full commitment}}\\[1.5ex]',
    render_panel(rows_FC, ''),     # last panel: no trailing \\
])
with open(root + '/Output files/model/insurance_decomposition.tex', 'w') as f:
    f.write(table)

print("Done. Wrote insurance_decomposition.tex.")
