# Global sensitivity using Latin Hypercube Sampling and PRCC
# (Marino et al. 2008 style). Output is peak prevalence and attack rate.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.stats import qmc

from helpers import load_england, RES, FIG, ensure_dirs


def prcc(X, y):
    # rank transform then partial Spearman via residuals
    Rx = np.argsort(np.argsort(X, axis=0), axis=0).astype(float)
    Ry = np.argsort(np.argsort(y)).astype(float)
    n = X.shape[1]
    vals = np.zeros(n)
    for i in range(n):
        others = np.delete(Rx, i, axis=1)
        A = np.column_stack([others, np.ones(len(y))])
        bi, *_ = np.linalg.lstsq(A, Rx[:, i], rcond=None)
        by, *_ = np.linalg.lstsq(A, Ry, rcond=None)
        ei = Rx[:, i] - A @ bi
        ey = Ry - A @ by
        if ei.std() == 0 or ey.std() == 0:
            vals[i] = 0.0
        else:
            vals[i] = np.corrcoef(ei, ey)[0, 1]
    return vals


def run_sir(beta, gamma, N, I0, R0, t_end=200):
    S0 = max(N - I0 - R0, 1.0)

    def f(t, y):
        S, I, R = y
        inf = beta * S * I / N
        return [-inf, inf - gamma * I, gamma * I]

    t = np.arange(0, t_end + 1)
    sol = solve_ivp(f, [0, t_end], [S0, I0, R0], t_eval=t, rtol=1e-6, atol=1e-6, max_step=1.0)
    if not sol.success:
        return np.nan, np.nan
    S, I, R = sol.y
    peak = I.max() / N
    attack = 1.0 - S[-1] / N
    return peak, attack


def main():
    ensure_dirs()
    df = load_england()
    N = float(df["population"].iloc[0])
    I0 = max(float(df["cases_7d"].iloc[0]) * 6.0, 1000.0)
    R0 = 350000.0

    p = pd.read_csv(RES / "level1_params.csv")
    row = p[p["model"] == "SIR"].iloc[0]
    beta_hat = float(row["beta"])
    gamma_hat = float(row["gamma"])

    # sample around the fitted values
    n = 200
    sampler = qmc.LatinHypercube(d=3, seed=7)
    u = sampler.random(n)
    lo = np.array([beta_hat * 0.5, gamma_hat * 0.5, I0 * 0.3])
    hi = np.array([beta_hat * 1.8, gamma_hat * 1.8, I0 * 3.0])
    X = qmc.scale(u, lo, hi)

    names = ["beta", "gamma", "I0"]
    peak = np.zeros(n)
    attack = np.zeros(n)
    for i in range(n):
        peak[i], attack[i] = run_sir(X[i, 0], X[i, 1], N, X[i, 2], R0)
        if i % 50 == 0:
            print("LHS", i)

    samples = pd.DataFrame(X, columns=names)
    samples["peak_prev"] = peak
    samples["attack_rate"] = attack
    samples.to_csv(RES / "lhs_samples.csv", index=False)

    prcc_peak = prcc(X, peak)
    prcc_att = prcc(X, attack)
    out = pd.DataFrame({"param": names, "PRCC_peak_prevalence": prcc_peak, "PRCC_attack_rate": prcc_att})
    out.to_csv(RES / "prcc.csv", index=False)
    print(out)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ypos = np.arange(len(names))
    ax.barh(ypos - 0.15, prcc_peak, height=0.3, label="peak prevalence")
    ax.barh(ypos + 0.15, prcc_att, height=0.3, label="attack rate")
    ax.set_yticks(ypos)
    ax.set_yticklabels(names)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlim(-1, 1)
    ax.set_xlabel("PRCC")
    ax.legend(frameon=False)
    ax.set_title("Sensitivity of SIR outcomes")
    fig.tight_layout()
    fig.savefig(FIG / "fig11_prcc.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("saved fig11_prcc.png")


if __name__ == "__main__":
    main()
