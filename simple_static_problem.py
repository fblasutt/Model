# A script to solve the simple static consumption allocation problem in couples

# %% Imports
import numpy as np
from typing import Dict, Optional, Sequence
from scipy.optimize import root
from matplotlib import pyplot as plt
import argparse

# %% FOCs system

def F(x: np.ndarray, p: Dict) -> np.ndarray:
    cW, cH, d, Q, lam = x
    theta   = p["theta"]
    alpha   = p["alpha"]
    sigma   = p["sigma"]
    nu      = p["nu"]
    chi     = p["chi"]
    wH      = p["wH"]
    wF      = p["wF"]
    phi     = p["phi"] 

    # 1) theta*(1-alpha)*(cW)^(-sigma) - lambda = 0
    f1 = theta * (1 - alpha) * (cW ** (-sigma)) - lam

    # 2) (1-theta)*(1-alpha)*(cH)^(-sigma) - lambda = 0
    f2 = (1 - theta) * (1 - alpha) * (cH ** (-sigma)) - lam

    # 3) alpha*(d^nu*(2*phi)^(1-nu))^-chi * nu*d^(nu-1)*(2*phi)^(1-nu) - lambda = 0
    term = (d**nu) * ((2 * phi) ** (1 - nu))
    f3 = alpha * (term ** (-chi)) * nu * (d ** (nu - 1)) * ((2 * phi) ** (1 - nu)) - lam

    # 4) Q
    f4 = (d**nu)*((2 * phi)**(1-nu))-Q

    # 5) cW + cH + d = wH + wF
    f5 = cW + cH + d - (wH + wF)

    return np.array([f1, f2, f3, f4, f5], dtype=float)

def solve_static(params: Dict, x0: np.ndarray, method: str = "hybr", tol: float = 1e-12):
    """Solve the static FOC system for given params and initial guess.

    Returns
    -------
    success: bool
        Whether the root finder succeeded.
    x: np.ndarray
        Solution vector [cW, cH, d, Q, lambda] (NaNs if failed).
    msg: str
        Solver message.
    """
    try:
        sol = root(F, x0, args=(params), method=method, tol=tol)
        if sol.success and np.all(sol.x[:3] > 0):
            return True, sol.x, sol.message
        else:
            # Mark failure if any of consumption/public good is non-positive
            return False, np.array([np.nan, np.nan, np.nan, np.nan, np.nan]), sol.message
    except Exception as e:
        return False, np.array([np.nan, np.nan, np.nan, np.nan, np.nan]), str(e)


def sweep_parameter(param_name: str,
                    grid: np.ndarray,
                    base_params: Dict,
                    x0: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """Sweep a parameter value over a grid and solve for [cW, cH, d].

    Parameters
    ----------
    param_name : str
        Name of the parameter key in params to vary (e.g., 'theta').
    grid : np.ndarray
        Array of parameter values to evaluate.
    base_params : Dict
        Baseline parameter dictionary to copy and modify.
    x0 : np.ndarray, optional
        Initial guess for the solver. If None, uses [1.0, 1.0, 0.5, 1.0].

    Returns
    -------
    results : Dict[str, np.ndarray]
        Dictionary with keys: 'param', 'cW', 'cH', 'd', 'lambda', 'success'.
    """
    if x0 is None:
        x0 = np.array([1.0, 1.0, 0.5, 1.0, 1.0])

    cW_lst, cH_lst, d_lst, Q_lst, lam_lst, suc_lst = [], [], [], [], [], []
    prev_x = x0.copy()

    for val in grid:
        p = dict(base_params)  # shallow copy
        p[param_name] = float(val)

        success, x_sol, msg = solve_static(p, prev_x)
        suc_lst.append(success)
        if success:
            cW_lst.append(x_sol[0])
            cH_lst.append(x_sol[1])
            d_lst.append(x_sol[2])
            Q_lst.append(x_sol[3])
            lam_lst.append(x_sol[4])
            prev_x = x_sol  # warm start next run
        else:
            cW_lst.append(np.nan)
            cH_lst.append(np.nan)
            d_lst.append(np.nan)
            Q_lst.append(np.nan)
            lam_lst.append(np.nan)

    return {
        "param": np.asarray(grid, float),
        "cW": np.asarray(cW_lst, float),
        "cH": np.asarray(cH_lst, float),
        "d": np.asarray(d_lst, float),
        "Q": np.asarray(Q_lst, float),
        "lambda": np.asarray(lam_lst, float),
        "success": np.asarray(suc_lst, bool),
    }


def plot_sweep(results: Dict[str, np.ndarray], param_label: Optional[str] = None):
    """Plot cW, cH, d against the swept parameter.

    NaN values are ignored by matplotlib (gaps in lines where solver failed).
    """
    x = results["param"]
    cW = results["cW"]
    cH = results["cH"]
    d = results["d"]
    Q = results["Q"]

    if param_label is None:
        param_label = "parameter"

    plt.figure(figsize=(10, 6))
    plt.plot(x, cW, label="cW (wife private)", lw=2)
    plt.plot(x, cH, label="cH (husband private)", lw=2)
    plt.plot(x, d, label="d (public good input)", lw=2)
    plt.plot(x, Q, label="Q (public good)", lw=2)
    plt.xlabel(param_label)
    plt.ylabel("levels")
    plt.title(f"Static solution vs {param_label}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def elasticity_1pct(param_name: str,
                    base_params: Dict,
                    x0: Optional[np.ndarray] = None,
                    change: float = 0.01,
                    method: str = "central") -> Dict[str, float]:
    """Compute percent change in cW, cH, d, Q for a 1% change in a parameter.

    Uses log-difference elasticity: d ln y / d ln p ≈ (ln y_up - ln y_dn) / (ln p_up - ln p_dn).
    Returned numbers can be read as: "percent change in y for a 1% change in p".

    Parameters
    ----------
    param_name : str
        Parameter to perturb (e.g., 'theta', 'alpha', 'sigma', 'nu', 'chi', 'wH', 'wF', 'phi').
    base_params : Dict
        Baseline parameter dictionary.
    x0 : np.ndarray, optional
        Initial guess. If None, defaults to [1.0, 1.0, 0.5, 1.0].
    change : float
        Fractional total change used for the approximation (default 0.01 for 1%).
    method : str
        'central' (default) uses ±change/2 around baseline; 'forward' uses +change only.

    Returns
    -------
    Dict[str, float]
        Elasticities for keys 'cW', 'cH', 'd', 'Q'. If a computation fails, value is np.nan.
    """
    if x0 is None:
        x0 = np.array([1.0, 1.0, 0.5, 1.0, 1.0])

    if param_name not in base_params:
        raise KeyError(f"Parameter '{param_name}' not found in base_params")

    # Solve at baseline (also provides a good warm start)
    ok0, x_baseline, _ = solve_static(base_params, x0)
    if not ok0:
        return {k: np.nan for k in ["cW", "cH", "d", "Q"]}

    y0 = {
        "cW": x_baseline[0],
        "cH": x_baseline[1],
        "d": x_baseline[2],
        "Q": x_baseline[3],
    }

    p0 = float(base_params[param_name])

    def _safe_log(val):
        return np.log(val) if (val is not None and np.isfinite(val) and val > 0) else np.nan

    if method == "central":
        # Symmetric ±change/2
        p_up = p0 * (1.0 + change / 2.0)
        p_dn = p0 * (1.0 - change / 2.0)

        # Up solve
        p_up_dict = dict(base_params)
        p_up_dict[param_name] = p_up
        ok_up, x_up, _ = solve_static(p_up_dict, x_baseline)

        # Down solve
        p_dn_dict = dict(base_params)
        p_dn_dict[param_name] = p_dn
        ok_dn, x_dn, _ = solve_static(p_dn_dict, x_baseline)

        if not (ok_up and ok_dn):
            return {k: np.nan for k in ["cW", "cH", "d", "Q"]}

        y_up = {"cW": x_up[0], "cH": x_up[1], "d": x_up[2], "Q": x_up[3]}
        y_dn = {"cW": x_dn[0], "cH": x_dn[1], "d": x_dn[2], "Q": x_dn[3]}

        dlnp = _safe_log(p_up) - _safe_log(p_dn)
        if not np.isfinite(dlnp) or dlnp == 0:
            return {k: np.nan for k in ["cW", "cH", "d", "Q"]}

        out = {}
        for key in ["cW", "cH", "d", "Q"]:
            dlny = _safe_log(y_up[key]) - _safe_log(y_dn[key])
            out[key] = dlny / dlnp if np.isfinite(dlny) else np.nan
        return out

    elif method == "forward":
        p_up = p0 * (1.0 + change)
        p_up_dict = dict(base_params)
        p_up_dict[param_name] = p_up
        ok_up, x_up, _ = solve_static(p_up_dict, x_baseline)
        if not ok_up:
            return {k: np.nan for k in ["cW", "cH", "d", "Q"]}

        y_up = {"cW": x_up[0], "cH": x_up[1], "d": x_up[2], "Q": x_up[3]}
        dlnp = _safe_log(p_up) - _safe_log(p0)
        if not np.isfinite(dlnp) or dlnp == 0:
            return {k: np.nan for k in ["cW", "cH", "d", "Q"]}

        out = {}
        for key in ["cW", "cH", "d", "Q"]:
            dlny = _safe_log(y_up[key]) - _safe_log(y0[key])
            out[key] = dlny / dlnp if np.isfinite(dlny) else np.nan
        return out

    else:
        raise ValueError("method must be 'central' or 'forward'")



# %% Main script

# Parameters
params = dict(
        theta=0.5, alpha=0.89, sigma=1.5, nu=0.36, chi=0.5,
        wH=1.0, wF=1.0, phi=0.32
    )

# Initial guess 
x0 = np.array([1.0, 1.0, 0.5, 1.0, 1.0])  # [cW, cH, d, lambda, Q]

# Solver
sol = solve_static(params, x0)
print("message:", sol[2])
print("[cW, cH, d, Q, lambda] =", sol[1])


# %% Sweep parameter on a grid and plot

sweep_param = 'wH'

# grid = np.linspace(0.5, 2, 100)
# sweep_res =  sweep_parameter(param_name=sweep_param,
#                     grid=grid,
#                     base_params=params,
#                     x0=x0)
# plot_sweep(sweep_res, param_label=sweep_param)

# Print elasticities for a 1% change in the same parameter
el = elasticity_1pct(sweep_param, params, x0=x0, change=0.01, method="central")
print(f"\nElasticities (percent change in outcome for a 1% change in '{sweep_param}'):")
print(f"  cW: {el['cW']:.6f}")
print(f"  cH: {el['cH']:.6f}")
print(f"  d : {el['d']:.6f}")
print(f"  Q : {el['Q']:.6f}")


# %%
