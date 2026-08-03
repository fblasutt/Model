# -*- coding: utf-8 -*-
"""
Run ALL counterfactual experiments and the variance decomposition under the
current parametrization (estimated_params.py), sequentially:

    1. alimony experiment.py       (post-divorce transfers)
    2. WLP_experiment.py           (gender wage gap, LC vs FC)
    3. tax_experiment.py           (tax progressivity, budget-balanced)
    4. variance_decomposition.py   (consumption-volatility decomposition)

Each experiment remains fully runnable on its own (e.g. in Spyder) from its
own file; this runner launches each as a SEPARATE subprocess so that
  - numba/global state is fresh for every experiment,
  - a crash in one experiment does not kill the others,
  - matplotlib runs headless (MPLBACKEND=Agg): plt.show() does not block and
    all figures/tables are still written to the usual Output files folders.

Usage:
    python run_experiments.py            # run everything
    python run_experiments.py tax        # run only scripts whose name contains 'tax'
"""
import os
import sys
import time
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))

EXPERIMENTS = [
    "alimony experiment.py",
    "variance_decomposition.py",
    "insurance_table.py",
    "WLP_experiment.py",
    "tax_experiment.py",
]


def main(argv):
    # optional name filter: run only matching experiments
    todo = [s for s in EXPERIMENTS if (len(argv) == 0 or any(a.lower() in s.lower() for a in argv))]
    if not todo:
        print(f"nothing matches {argv}; available: {EXPERIMENTS}")
        return 1

    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"          # headless: plt.show() never blocks
    env["PYTHONIOENCODING"] = "utf-8"

    results = []
    for script in todo:
        print(f"\n{'='*72}\n  RUNNING: {script}\n{'='*72}", flush=True)
        t0 = time.time()
        r = subprocess.run([sys.executable, os.path.join(HERE, script)],
                           env=env, cwd=HERE)
        results.append((script, r.returncode, time.time() - t0))

    print(f"\n{'='*72}\n  SUMMARY\n{'='*72}")
    for script, code, secs in results:
        status = "OK" if code == 0 else f"FAILED (exit {code})"
        print(f"  {script:32s} {status:20s} {secs/60:6.1f} min")

    return 0 if all(code == 0 for _, code, _ in results) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
