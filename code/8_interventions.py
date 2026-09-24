# Intervention scenarios on the national SIR with mobility-scaled beta.
# 1) vaccination coverage  2) extra mobility cuts

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.integrate import solve_ivp

from helpers import load_england, FIG, RES, ensure_dirs, TRAIN_END


def simulate(beta0, gamma, t, y0, N, m, vax_flow):
    def f(tt, y):
        S, I, R, V = y
        mt = np.interp(tt, t, m)
        nu = np.interp(tt, t, vax_flow)
        nu = min(nu, max(S, 0.0))
        inf = beta0 * mt * max(S, 0.0) * max(I, 0.0) / N
        return [-inf - nu, inf - gamma * I, gamma * I, nu]

    sol = solve_ivp(f, [t[0], t[-1]], y0, t_eval=t, method="RK45",
                    rtol=1e-6, atol=1e-6, max_step=1.0)
    S, I, R, V = sol.y
    mt = np.array([np.interp(tt, t, m) for tt in t])
    inc = beta0 * mt * S * I / N
    return inc, S, I, V


def main():
    ensure_dirs()
    df = load_england()
    N = float(df["population"].iloc[0])
    t = np.arange(len(df), dtype=float)
    m_obs = df["workplace_m"].to_numpy()
    vax_obs = df["vax_first_dose"].to_numpy()

    # parameters from SIR + mobility if present, else SIR
    p = pd.read_csv(RES / "level1_params.csv")
    row = p[p["model"] == "SIR"].iloc[0]
    beta0 = float(row["beta"])
    gamma = float(row["gamma"])
    I0 = float(row["I0"])
    R0 = 350000.0
    V0 = 0.0
    S0 = max(N - I0 - R0 - V0, 1.0)
    y0 = [S0, I0, R0, V0]

    scenarios = {}

    # vaccination
    scenarios["vax observed"] = simulate(beta0, gamma, t, y0, N, m_obs, vax_obs)
    scenarios["vax none"] = simulate(beta0, gamma, t, y0, N, m_obs, np.zeros_like(vax_obs))
    scenarios["vax +20%"] = simulate(beta0, gamma, t, y0, N, m_obs, vax_obs * 1.2)

    # mobility (keep observed vax)
    m_base = np.ones_like(m_obs)
    m_cut = np.clip(m_obs * 0.80, 0.05, 1.5)
    scenarios["mobility observed"] = scenarios["vax observed"]
    scenarios["mobility baseline"] = simulate(beta0, gamma, t, y0, N, m_base, vax_obs)
    scenarios["mobility extra 20% cut"] = simulate(beta0, gamma, t, y0, N, m_cut, vax_obs)

    summary = []
    series = df[["date"]].copy()
    series["obs_cases_7d"] = df["cases_7d"]
    for name, (inc, S, I, V) in scenarios.items():
        series[name] = inc
        summary.append({
            "scenario": name,
            "peak_incidence": float(np.max(inc)),
            "total_incidence": float(np.sum(inc)),
            "final_V": float(V[-1]),
            "attack_from_S": float(1.0 - S[-1] / N),
        })
    pd.DataFrame(summary).to_csv(RES / "intervention_summary.csv", index=False)
    series.to_csv(RES / "intervention_series.csv", index=False)
    print(pd.DataFrame(summary).to_string(index=False))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["date"], df["cases_7d"], color="0.6", lw=1, label="observed 7d")
    for name in ["vax none", "vax observed", "vax +20%"]:
        ax.plot(df["date"], series[name], lw=1.3, label=name)
    ax.axvline(pd.Timestamp(TRAIN_END), color="0.4", ls="--", lw=1)
    ax.legend(frameon=False)
    ax.set_ylabel("New cases (model)")
    ax.set_title("Vaccination scenarios (SIR)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.tight_layout()
    fig.savefig(FIG / "fig12_vax_scenarios.png", dpi=200, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["date"], df["cases_7d"], color="0.6", lw=1, label="observed 7d")
    for name in ["mobility baseline", "mobility observed", "mobility extra 20% cut"]:
        ax.plot(df["date"], series[name], lw=1.3, label=name)
    ax.axvline(pd.Timestamp(TRAIN_END), color="0.4", ls="--", lw=1)
    ax.legend(frameon=False)
    ax.set_ylabel("New cases (model)")
    ax.set_title("Mobility scenarios (SIR, beta scaled by workplace_m)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.tight_layout()
    fig.savefig(FIG / "fig13_mobility_scenarios.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("saved intervention figures")


if __name__ == "__main__":
    main()
