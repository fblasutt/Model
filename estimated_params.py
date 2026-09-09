# -*- coding: utf-8 -*-
"""
Single source of truth for the estimated parametrization used by ALL
counterfactual experiments (alimony, gender wage gap, taxes) and the
variance decomposition. Keep xc in sync with the current estimate in
calibration.py ([η, σL, α, ρ, wedge_w, wedge_m, β] — 7 parameters).

NOTE the parameter layout of the CURRENT model: two individual-specific
random-walk love shocks (ψw, ψm) with common innovation sd σL and initial
value zero, plus a GENDER-SPECIFIC single-couple utility wedge in positions
4 (wife) and 5 (husband); β moved to position 6.

On import this module compares xc against the LAST active `xc = np.array([...])`
assignment in calibration.py and prints a loud warning if they differ, so a
re-estimation that is not propagated here cannot go unnoticed.
"""
import os
import io
import re
import numpy as np

# Current SMM estimates [η, σL, α, ρ, wedge_w, wedge_m, β, ι0w] — sync with
# calibration.py. σL0 (initial match-quality dispersion) is FIXED in
# Bargaining_numba.setup() (no longer estimated; young-divorce target dropped).
xc=np.array([ 2.17742503,  0.03991867,  0.95089659,  1.80807117,  0.63913244,
       -0.23640997,  0.99523097, -0.90290523])



def par_dict(N, sample_init):
    """Common par dict for HouseholdModelClass under the current estimates."""
    return {'simN': N, 'η': xc[0], 'σL': xc[1], 'α': xc[2], 'ρ': xc[3],
            'wedge_w': xc[4], 'wedge_m': xc[5], 'β': xc[6], 'ι0w': xc[7],
            'sample_init': sample_init}


# ---------------------------------------------------------------------------
# FULL-COMMITMENT re-calibrated parameters [η, σL, β] — estimated in
# run_full.py (ESTIMATE=True), targeting the employment rate of married women,
# the annual divorce rate and wealth/husband's earnings, holding everything
# else at the LC estimates above. PLACEHOLDER = LC values until estimated.
# ---------------------------------------------------------------------------
xc_full = np.array([1.8394,0.23844,0.99635 ])


def apply_fc_params(M):
    """
    Apply the FC re-calibrated [η, σL, β] to a model (call AFTER copying it
    and setting par.full=True, BEFORE solve). σL differs from the LC value,
    so the love grids are rebuilt (σL0 unchanged). Initial-love TRANSPLANTS
    from an LC model remain valid: they are grid INDICES (relative positions),
    whose values are read off the FC grids.
    """
    import UserFunctions_numba as usr
    p = M.par
    p.η = xc_full[0]; p.σL = xc_full[1]; p.β = xc_full[2]
    p.grid_lovew_, p.Πlw_, p.Πlw0_ = usr.rouw_nonst(p.T, p.σL, p.σL0, p.num_lovew)
    p.grid_lovem_, p.Πlm_, p.Πlm0_ = usr.rouw_nonst(p.T, p.σL, p.σL0, p.num_lovem)
    p.grid_lovew = [np.repeat(p.grid_lovew_[t], p.num_lovem) for t in range(p.T)]
    p.grid_lovem = [np.tile(p.grid_lovem_[t], p.num_lovew) for t in range(p.T)]
    p.Πl  = [np.kron(p.Πlw_[t],  p.Πlm_[t])  for t in range(p.T-1)]
    p.Πl0 = [np.kron(p.Πlw0_[t], p.Πlm0_[t]) for t in range(p.T-1)]
    return M


def _warn_if_out_of_sync():
    """Compare xc with the last (= active) xc assignment in calibration.py."""
    try:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'calibration.py')
        src = io.open(path, encoding='utf-8').read()
        hits = re.findall(r'^xc\s*=\s*np\.array\(\[([^\]]+)\]\)', src, re.M)
        if not hits:
            return
        cal = np.array([float(v) for v in hits[-1].replace('\n', ' ').split(',') if v.strip()])
        if len(cal) != len(xc) or not np.allclose(cal, xc, rtol=1e-6, atol=0):
            print('!'*74)
            print('WARNING [estimated_params.py]: xc is OUT OF SYNC with calibration.py')
            print(f'  experiments use: {xc}')
            print(f'  calibration.py : {cal}')
            print('  -> update xc in estimated_params.py after re-estimating.')
            print('!'*74)
    except Exception:
        pass  # never let the check break an experiment run


_warn_if_out_of_sync()
