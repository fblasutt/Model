# -*- coding: utf-8 -*-
"""
Manual FULL-COMMITMENT runner / inspector.

Edit xf below (or build a new vector in the console) and call
    M, B, fit = q(xf)
as many times as you like — the data sample is loaded once at import, and
each call solves/simulates a fresh FC model at that parametrization and
returns:

  M   — the solved and simulated model, for agent-level inspection:
            M.sim.couple[i,:]   marriage path of agent i
            M.sim.WLP[i,:]      her participation choices
            M.sim.incwg[i,:]    her gross earnings
            M.sim.power[i,:]    Pareto weight path
            M.sim.Cw[i,:]       her private consumption
  B   — the full output dict of reg_cons_insurance.insurance()
        (pass-throughs, BPP moments, vardec, shockdec, vol_ladder, ...);
        its LaTeX side files are written under name_file (default
        'ManualFull...', so baseline outputs are not overwritten)
  fit — sum of squared relative deviations over the calibration moments
        (the pension-reform moment is skipped: it would need a second,
        post-reform solve)

Running the file itself executes M, B, fit = q(xf) once at the xf below.
"""

import numpy as np
import pandas as pd
import getpass

import Bargaining_numba as brg
import init_conditions as ic
from reg_cons_insurance import insurance

# Initialize seed
np.random.seed(10)

# Root
user = getpass.getuser()
if user == "sara":
    root = '/Users/sara/Dropbox/Family Risk Sharing'
elif user == "32489":
    root = '/Users/32489/Dropbox/Family Risk Sharing'
elif user == 'blasutto':
    root = '/home/users/b/l/blasutto/Family-Risk-Sharing'
else:
    raise RuntimeError(f"Unknown user: {user}")


###############################################################################
# USER PARAMETRIZATION — edit freely. Layout:
#            [η,          σL,         α,          ρ,          wedge_w,    wedge_m,    β,          ι0w]   (σL0 fixed in Bargaining_numba.setup())
###############################################################################
xf=np.array([ 2.17742503,  0.03991867,  0.95089659,  1.80807117,  0.63913244,
       -0.23640997,  0.99523097, -0.90290523])

# xf=np.array([ 1.8,  0.13991867,  0.95089659,  1.80807117,  0.63913244,
#        -0.23640997,  0.998, -0.90290523])


# Commitment regime of the run: True = full commitment (LC twin solved first
# for the initial love distribution), False = limited commitment.
FULL = True

N = 10_000  # sample size


###############################################################################
# Sample (identical to calibration.py) — loaded ONCE at import
###############################################################################
baseline_sample = np.array(pd.read_excel(root + '/Output files/data_sample.csv'))
baseline_sample = baseline_sample[~np.isnan(baseline_sample).any(axis=1)]
baseline_sample[:, 0] = np.arange(len(baseline_sample))

pr = np.ones(baseline_sample.shape[0]) / baseline_sample.shape[0]
indexes = np.array(np.random.choice(baseline_sample[:, 0], size=N, p=pr, replace=True),
                   dtype=np.int32) - 1
final_sample = baseline_sample[:, 1:][indexes]

age_initial   = final_sample[:, 0]*0+25
age_final     = final_sample[:, 1]
cw_cons_share = final_sample[:, 2]
h_income      = final_sample[:, 3]
w_income      = final_sample[:, 4]
age_marriage  = final_sample[:, 5]*0+25
year          = final_sample[:, 6]
assets        = final_sample[:, 7] * np.mean(np.exp(h_income))

# Pre-drawn uniforms for the posterior-draw initial income split (drawn AFTER
# the sample so sample selection is unchanged; FIXED across evaluations so
# SMM objectives stay deterministic)
u_init_w=np.random.rand(N);u_init_m=np.random.rand(N)
σME2_init=0.0  # measurement-error variance in observed entry income (0 = off)

age = (np.cumsum(np.ones((N, 65)), axis=1)-1)+25   # rebuilt per-run inside q() with the model's T


###############################################################################
# One full run at a chosen parametrization
###############################################################################
def q(xf, extra_par=None, name_file='ManualFull', full=False, light=False, init_love=None):
    """
    Build, solve and simulate the FULL-COMMITMENT model at parametrization
    xf = [η, σL, α, ρ, wedge_w, wedge_m, β, ι0w]; print the fit against the
    calibration targets; run insurance(). extra_par is an optional dict of
    additional par overrides (applied at construction, BEFORE grids are
    built, so grid-relevant parameters like σϵw, μ, num_pw ... are fair game).

    Returns (M, B, fit).
    """
    global age  # refreshed with the model's T so inspection helpers match

    # --- Build the FC model ------------------------------------------------
    par = {'simN': N, 'η': xf[0], 'σL': xf[1], 'α': xf[2], 'ρ': xf[3],
           'wedge_w': xf[4], 'wedge_m': xf[5], 'β': xf[6], 'ι0w': xf[7],
           'full': full,
           'sample_init': np.array(age_marriage-25, dtype=np.int_)}
    if extra_par: par.update(extra_par)
    M = brg.HouseholdModelClass(par=par)

    # Initial conditions (from the data sample; uses the CHOSEN ρ and the
    # grids built under the CHOSEN ι0w, so everything is internally consistent)
    param = (cw_cons_share/(1.0-cw_cons_share))**M.par.ρ
    M.sim.init_power = param/(1.0+param)

    gridzw = M.par.grid_zw[:, :, np.linspace(0, M.par.num_z-1, M.par.num_zm, dtype=np.int_)]
    gridzm = M.par.grid_zm[:, :, :M.par.num_zw]

    izm=ic.draw_init_iz(h_income,M.par.sample_init,gridzm,M.par.grid_pm,M.par.grid_ϵm,u_init_m,σME2=σME2_init)
    izm[np.isnan(h_income)] = (M.par.num_pm*M.par.num_ϵm)//2
    izw=ic.draw_init_iz(w_income,M.par.sample_init,gridzw,M.par.grid_pw,M.par.grid_ϵw,u_init_w,σME2=σME2_init)
    izw[np.isnan(w_income)] = (M.par.num_pw*M.par.num_ϵw)//2
    M.sim.init_z = izw*M.par.num_zm+izm   # FIXED gender swap: wife is the SLOW joint-index component
    M.sim.init_A = assets   # as in calibration.py (use np.maximum(assets,0.0) to clamp debt)

    age = (np.cumsum(np.ones((M.par.simN, M.par.T)), axis=1)-1)+25

    print("Solving the model at:")
    print("  [η, σL, α, ρ, wedge_w, wedge_m, β, ι0w] =", np.round(np.asarray(xf), 5))
    if extra_par: print("  extra overrides:", extra_par)

    if init_love is not None:
        # externally supplied initial love (e.g. the LC baseline during the
        # FC estimation loop): transplant and skip the conditional draw
        M.sim.force_init_love[:] = init_love
        M._init_love_rationalized = True
    elif M.par.full:
        # FULL COMMITMENT: the initial love distribution comes from the LC
        # BASELINE at the ORIGINAL LC estimates (estimated_params.xc) — same
        # convention as the experiment scripts. It is computed ONCE and cached,
        # so trying new FC parametrizations does not re-run the LC model.
        M.sim.force_init_love[:] = lc_baseline_init_love()
        M._init_love_rationalized = True

    M.solve()
    M.simulate()

    # --- Fit: calibration moments vs data targets --------------------------
    # (reform moment skipped: FC only, single solve)
    sample_div  = (age >= age_initial[:, None]-1) & (age <= age_final[:, None]) & (M.sim.couple_lag == 1)
    sample_empl = (age >= age_initial[:, None])   & (age <= age_final[:, None]) & (M.sim.couple == 1)
    sample_pass = (age > age_initial[:, None]) & (age <= age_final[:, None]) & (M.sim.couple == 1) & (M.sim.couple_lag == 1)
    sample_reg  = (age > age_initial[:, None]) & (age <= age_final[:, None]) & (M.sim.couple_lag == 1)

    wife_empl       = np.mean(M.sim.WLP[sample_empl] > 0)
    divorce_rate    = np.mean((M.sim.couple == 0)[sample_div])
    divorce_rate_young = np.mean((M.sim.couple == 0)[(sample_div) & (age <= 40)])
    expenditure_x_share = np.mean((M.sim.dw/M.sim.C_tot)[sample_empl])
    wife_cons_share = np.mean((M.sim.Cw/(M.sim.Cw+M.sim.Cm))[sample_empl])
    wife_ratio      = np.mean((M.sim.incwg/(M.sim.incmg+M.sim.incwg))[sample_empl][M.sim.WLP[sample_empl] > 0])
    gender_gap_earnings = (M.sim.incwg[sample_empl][M.sim.WLP[sample_empl] > 0]).mean()/M.sim.incmg[sample_empl].mean()
    couple_assets   = M.sim.A[sample_empl].mean()/M.sim.incmg[sample_empl].mean()

    ΔC = np.log(M.sim.C_tot[sample_pass])
    Δd = np.log(M.sim.dw[sample_pass])
    βCp = np.cov(ΔC, Δd)[0, 1]/np.var(ΔC)   # same moment as calibration's βCp

    moments = [
        # (label,                              data,      model)   targets synced with calibration.py
        ('Employment rate married women',            0.5879,    wife_empl),
        ('Annual divorce rate',                      0.0107513, divorce_rate),
        ('[diag] annual divorce rate, young (<=40)', 0.0118,    divorce_rate_young),
        ('Expenditure share home goods',             0.812,     expenditure_x_share),
        ('Private-exp. elasticity to total cons.',   1.044,     βCp),
        ('Wealth / husband earnings',                2.44,      couple_assets),
        ("Wife's share of private consumption",      0.322,     wife_cons_share),
        ('Gender gap earnings (workers/husbands)',   0.3767,    gender_gap_earnings),
        ('[diag] wife share of HH earnings, workers', np.nan,   wife_ratio),
        ('[skipped] reform effect on wife share',    0.0139,    np.nan),
    ]

    print("\n=== FIT (full commitment, single solve) ===")
    print(f"{'moment':45s} {'data':>9s} {'model':>9s} {'rel.dev':>9s}")
    fit = 0.0
    for name, dat, mod in moments:
        rel = (mod-dat)/dat if np.isfinite(dat) and np.isfinite(mod) and dat != 0 else np.nan
        if name[0] != '[' and np.isfinite(rel): fit += rel**2
        print(f"{name:45s} {dat:9.4f} {mod:9.4f} {rel:9.3f}" if np.isfinite(mod)
              else f"{name:45s} {dat:9.4f} {'---':>9s} {'---':>9s}")
    print(f"\nfit (sum of squared rel. deviations, reform moment excluded) = {fit:.4f}")

    # store the moments on the model for programmatic use (estimation loop)
    M._moments = {'wife_empl': wife_empl, 'divorce_rate': divorce_rate,
                  'divorce_rate_young': divorce_rate_young,
                  'expenditure_x_share': expenditure_x_share, 'βCp': βCp,
                  'couple_assets': couple_assets, 'wife_cons_share': wife_cons_share,
                  'gender_gap_earnings': gender_gap_earnings}

    if light:
        return M, None, fit   # estimation mode: skip the insurance machinery

    # --- Insurance / decomposition outputs ---------------------------------
    B = insurance(M, sample_reg,
                  shock_type='permanent', shock_gender='Male', consumption_gender='Male',
                  name_file=name_file, name_line='Manual FC')

    print("\n=== reg_cons_insurance highlights ===")
    print("pass-throughs (perm. male shock): own cons %.3f | wife cons %.3f | total C %.3f"
          % (B['indc']['per_m_m'], B['indc']['per_m_w'], B['totc']['per_m']))
    print("vardec husband: V_total %.4f = V_agg %.4f + V_reb %.4f + 2Cov %.4f"
          % (B['vardec_m']['V_total'], B['vardec_m']['V_agg'], B['vardec_m']['V_reb'], B['vardec_m']['2Cov']))
    print("vol ladder: V_Y %.4f -> taxes %.4f -> savings %.4f -> V_C %.4f"
          % (B['vol_ladder']['V_Y'], B['vol_ladder']['taxes'], B['vol_ladder']['savings'], B['vol_ladder']['V_C']))

    return M, B, fit


###############################################################################
# Initial love distribution for FULL-COMMITMENT runs: from the LC BASELINE at
# the ORIGINAL LC estimates (estimated_params.xc), computed ONCE and cached —
# new FC parametrizations never re-run the LC model.
###############################################################################
_init_love_cache = {'v': None}

def lc_baseline_init_love():
    if _init_love_cache['v'] is None:
        from estimated_params import xc as xc_lc
        print("(solving the LC baseline at the LC estimates ONCE, for the initial love distribution)")
        M0, _, _ = q(np.asarray(xc_lc), full=False, light=True)
        _init_love_cache['v'] = np.array(
            [M0.sim.love[k, int(M0.par.sample_init[k])] for k in range(M0.par.simN)],
            dtype=np.int_)
    return _init_love_cache['v']


###############################################################################
# ESTIMATE = True: re-calibrate the FULL-COMMITMENT model — [η, σL, β] chosen
# to match the employment rate of married women (0.5879), the annual divorce
# rate (0.0107513) and wealth/husband's earnings (2.44) — holding all other
# parameters at the LC estimates (xf). The FC models use the LC BASELINE's
# initial love distribution (at the ORIGINAL estimates, cached).
# Paste the result into estimated_params.xc_full so the experiments use it.
###############################################################################
ESTIMATE = True

if ESTIMATE:
    import dfols

    # LC baseline at the LC estimates -> the initial love distribution (cached)
    init_love_base = lc_baseline_init_love()

    def q3(pt3):
        """[η, σL, β] -> residuals on (employment, divorce, wealth) for the FC model."""
        xf_ = xf.copy(); xf_[0] = pt3[0]; xf_[1] = pt3[1]; xf_[6] = pt3[2]
        Mx, _, _ = q(xf_, full=True, light=True, init_love=init_love_base)
        mo = Mx._moments
        return [(mo['wife_empl']-0.5879)/0.5879,
                (mo['divorce_rate']-0.0107513)/0.0107513,
                (mo['couple_assets']-2.44)/2.44]

    x0  = np.array([xf[0], xf[1], xf[6]])
    lb  = np.array([0.2,  0.1, 0.99])
    ub  = np.array([3.2, 0.6,  1.005])
    res = dfols.solve(q3, x0, rhobeg=0.1, rhoend=1e-5, maxfun=100, bounds=(lb, ub),
                      npt=len(x0)+5, scaling_within_bounds=True,
                      user_params={'tr_radius.gamma_dec': 0.98, 'tr_radius.gamma_inc': 1.0,
                                   'tr_radius.alpha1': 0.9, 'tr_radius.alpha2': 0.95},
                      objfun_has_noise=False, print_progress=True)
    print('FC estimates [η, σL, β] =', res.x)
    print('LC values were          =', x0)
    print('-> paste into estimated_params.xc_full')
    # full diagnostics at the FC optimum
    xf_opt = xf.copy(); xf_opt[0], xf_opt[1], xf_opt[6] = res.x
    M, B, fit = q(xf_opt, full=True, init_love=init_love_base, name_file='ManualFullFC')

else:
    # Run once at the xf above (regime set by FULL at the top); re-run from
    # the console with e.g.
    #   xf2 = xf.copy(); xf2[0] = 4.5
    #   M, B, fit = q(xf2, full=FULL)
    M, B, fit = q(xf, full=FULL)
