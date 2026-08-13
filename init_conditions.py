# -*- coding: utf-8 -*-
"""
Posterior-draw initial income conditions.

Observed entry income is y0 = trend + z0 + ϵ0 (+ measurement error). The old
nearest-gridpoint assignment handed (nearly) the whole deviation from trend to
the persistent component, overstating initial persistent heterogeneity and
understating early-life mean reversion. draw_init_iz() instead:

  1. computes the deviation d = log y0 − trend;
  2. draws ϵ0 from its POSTERIOR given d (discrete Bayes over the gender's
     transitory gridpoints; likelihood z0 ~ N(0, σ0² + σME²)) — the draw is
     CORRELATED with d, which is what makes Var(z0) come out right: an
     independent draw would inflate it by 2σϵ² instead of removing σϵ²;
  3. backs out z0 = shrink·(d − ϵ0), shrink = σ0²/(σ0²+σME²) (the measurement
     error component, if any, is discarded rather than loaded into the state);
  4. returns the nearest combined gridpoint with that (z0, ϵ0) pair, so the
     observed income level is preserved up to grid snapping.

The uniforms u are passed IN and must be pre-drawn ONCE per gender at module
level (fixed across q() evaluations: DFO-LS needs a deterministic objective).

Grid layout: the per-gender combined index is p-major over (p, ϵ), which is
the layout of both the subsampled total-earnings grids used by the callers
(gridzw = grid_zw[:,:,linspace(0,num_z-1,num_zm)], gridzm = grid_zm[:,:,:num_zw])
and of par.grid_p*/par.grid_ϵ* — verified against the joint-index decoding.
"""
import numpy as np


def draw_init_iz(y_obs, t0, grid_tot, grid_p, grid_eps, u, σME2=0.0):
    """
    Posterior-draw initial combined income index for one gender.

    y_obs    : observed log income at entry (NaN -> middle gridpoint, as before)
    t0       : per-agent initial period (par.sample_init)
    grid_tot : the gender's total-earnings grid [t, iD, c, ih] (LEVELS) —
               pass the same subsampled gridzw/gridzm object the old code used
    grid_p   : par.grid_pw / par.grid_pm  (persistent component, same c index)
    grid_eps : par.grid_ϵw / par.grid_ϵm  (transitory component, same c index)
    u        : pre-drawn uniforms, one per agent (FIXED across evaluations)
    σME2     : measurement-error variance in observed entry income (0 = off)

    The prior dispersion of the initial persistent component is NOT taken from
    par.σ0 (that parameter only sizes the grid): it is estimated from the DATA
    by deconvolution,  Var(z0) = Var(d) - Var(ϵ) - σME2,  with d the observed
    deviations from trend and Var(ϵ) read off the transitory grid. This makes
    the split self-consistent with the dispersion the initialization carries.

    Returns the combined index c (int32), layout-compatible with the old
    nearest-total assignment.
    """
    n = len(y_obs)
    out = np.zeros(n, dtype=np.int32)
    nC = grid_tot.shape[2]
    default = nC//2                          # same NaN fallback as the old code

    # --- deviations from trend, per agent (trend recovered from the grids) ---
    d_all = np.full(n, np.nan)
    for i in range(n):
        if not np.isfinite(y_obs[i]): continue
        t = int(t0[i])
        gv0 = np.log(grid_tot[t, 0, 0, 0])
        trend = gv0-grid_p[t, 0, 0, 0]-grid_eps[t, 0, 0, 0]
        d_all[i] = y_obs[i]-trend

    # --- prior variance of z0 from the data (deconvolution) ---
    t_ref = int(t0[np.isfinite(y_obs)][0]) if np.isfinite(y_obs).any() else 0
    ev_ref = grid_eps[t_ref, 0, :, 0]
    e_var  = float(np.mean(ev_ref**2)-np.mean(ev_ref)**2)   # Var(ϵ) on the grid
    σz2 = max(float(np.nanvar(d_all))-e_var-σME2, 1e-4)
    s2 = σz2+σME2                            # var of d given ϵ0
    shrink = σz2/s2

    for i in range(n):
        if not np.isfinite(d_all[i]):
            out[i] = default
            continue
        t = int(t0[i])
        pv = grid_p[t, 0, :, 0]              # persistent component per c
        ev = grid_eps[t, 0, :, 0]            # transitory component per c
        d = d_all[i]

        # posterior over the transitory gridpoints (grid multiplicity = prior)
        uniq = np.unique(ev)
        w = np.array([(np.isclose(ev, e)).sum()*np.exp(-0.5*((d-e)**2)/s2) for e in uniq])
        cdf = np.cumsum(w/w.sum())
        e = uniq[min(np.searchsorted(cdf, u[i]), len(uniq)-1)]

        # persistent component: shrunk residual deviation, nearest p-gridpoint
        # among combined states carrying the drawn ϵ0
        zhat = shrink*(d-e)
        cand = np.where(np.isclose(ev, e))[0]
        out[i] = cand[np.argmin(np.abs(pv[cand]-zhat))]

    return out
