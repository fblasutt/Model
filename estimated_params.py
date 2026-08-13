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

# Current SMM estimates [η, σL, α, ρ, wedge_w, wedge_m, β, ι0w] — sync with calibration.py
# PLACEHOLDER pending re-estimation with ι0w internally estimated (targets the
# mean wife-to-husband earnings ratio among working wives, 0.46): last 7-param
# estimate with the previous fixed ι0w = -0.47 appended.
xc=np.array([4.52531362,  0.01574232,  0.92576666,  1.67436856,  2.65417813, -0.11096475,0.98380537, -0.56896309])
xc=np.array([4.85457701,  0.0207449,   0.93030974,  1.700733,    2.84106809, -0.08532483,0.98470179, -0.62552763])
xc=np.array([6.21926302,  0.02625619,  0.96,        2.05119648,  3.45904837, -0.11929322,0.98213518, -0.66810006])

def par_dict(N, sample_init):
    """Common par dict for HouseholdModelClass under the current estimates."""
    return {'simN': N, 'η': xc[0], 'σL': xc[1], 'α': xc[2], 'ρ': xc[3],
            'wedge_w': xc[4], 'wedge_m': xc[5], 'β': xc[6], 'ι0w': xc[7],
            'sample_init': sample_init}


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
