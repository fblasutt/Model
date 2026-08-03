# -*- coding: utf-8 -*-
"""
Single source of truth for the estimated parametrization used by ALL
counterfactual experiments (alimony, gender wage gap, taxes) and the
variance decomposition. Keep xc in sync with the current estimate in
calibration.py ([η, σL, α, ρ, Ω, β]).

NOTE the parameter layout of the NEW model: position 4 is Ω (the ±Ω
transitory match-quality disagreement shock), NOT the old utility 'wedge'.
Passing it under the wrong key would silently turn OFF the disagreement
shocks and turn ON a large single-couple utility wedge.

On import this module compares xc against the LAST active `xc = np.array([...])`
assignment in calibration.py and prints a loud warning if they differ, so a
re-estimation that is not propagated here cannot go unnoticed.
"""
import os
import io
import re
import numpy as np

# Current SMM estimates [η, σL, α, ρ, Ω, β] — sync with calibration.py
xc=np.array([9.59993703, 0.03871525, 0.93100763, 1.70843712, 1.2026561,  0.973201])


def par_dict(N, sample_init):
    """Common par dict for HouseholdModelClass under the current estimates."""
    return {'simN': N, 'η': xc[0], 'σL': xc[1], 'α': xc[2], 'ρ': xc[3],
            'Ω': xc[4], 'β': xc[5], 'sample_init': sample_init}


def _warn_if_out_of_sync():
    """Compare xc with the last (= active) xc assignment in calibration.py."""
    try:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'calibration.py')
        src = io.open(path, encoding='utf-8').read()
        hits = re.findall(r'^xc\s*=\s*np\.array\(\[([^\]]+)\]\)', src, re.M)
        if not hits:
            return
        cal = np.array([float(v) for v in hits[-1].replace('\n', ' ').split(',') if v.strip()])
        if len(cal) == len(xc) and not np.allclose(cal, xc, rtol=1e-6, atol=0):
            print('!'*74)
            print('WARNING [estimated_params.py]: xc is OUT OF SYNC with calibration.py')
            print(f'  experiments use: {xc}')
            print(f'  calibration.py : {cal}')
            print('  -> update xc in estimated_params.py after re-estimating.')
            print('!'*74)
    except Exception:
        pass  # never let the check break an experiment run


_warn_if_out_of_sync()
