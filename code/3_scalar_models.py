# Level 1 - scalar ODE models fitted with non-linear least squares.
# I fit to the 7-day mean of new cases so weekend reporting does not dominate.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from helpers import (
    load_england, split_train_test, days_since, fit_metrics,
    FIG, RES, ensure_dirs, shade_test, TRAIN_END, NLS_START, NLS_END,
)


def sir_ode(t, y, beta, gamma, N):
    S, I, R = y
    inf = beta * S * I / N
    return [-inf, inf - gamma * I, gamma * I]


def seir_ode(t, y, beta, sigma, gamma, N):
    S, E, I, R = y
    inf = beta * S * I / N
    return [-inf, inf - sigma * E, sigma * E - gamma * I, gamma * I]


def seirs_ode(t, y, beta, sigma, gamma, omega, N):
    S, E, I, R = y
    inf = beta * S * I / N
    return [-inf + omega * R, inf - sigma * E, sigma * E - gamma * I, gamma * I - omega * R]


def simulate(model, theta, t, y0, N, m=None):
    t = np.asarray(t, dtype=float)
    tspan = [t[0], t[-1]]

    gamma = 1.0 / 7.0
    sigma = 1.0 / 5.2
    if model == "sir":
        beta = theta[0]

        def f(tt, y):
            return sir_ode(tt, y, beta, gamma, N)

        n_comp = 3
    elif model == "sir_m":
        beta0 = theta[0]

        def f(tt, y):
            mt = np.interp(tt, t, m)
            return sir_ode(tt, y, beta0 * mt, gamma, N)

        n_comp = 3
    elif model == "seir":
        beta = theta[0]

        def f(tt, y):
            return seir_ode(tt, y, beta, sigma, gamma, N)

        n_comp = 4
    else:
        beta, omega = theta[0], theta[1]

        def f(tt, y):
            return seirs_ode(tt, y, beta, sigma, gamma, omega, N)

        n_comp = 4

    sol = solve_ivp(f, tspan, y0, t_eval=t, method="RK45", rtol=1e-6, atol=1e-6, max_step=1.0)
    if not sol.success or sol.y.shape[1] != len(t):
        return np.full(len(t), 1e12), None

    if n_comp == 3:
        S, I, R = sol.y
        E = None
    else:
        S, E, I, R = sol.y

    if model == "sir_m":
        beta_t = theta[0] * m
        inc = beta_t * S * I / N
    else:
        inc = theta[0] * S * I / N
    return inc, {"S": S, "E": E, "I": I, "R": R}


def fit_model(model, t, y, N, I0, R0, m=None):
    S0 = max(N - I0 - R0, 1.0)
    # infectious period fixed at 7 days, incubation 5.2 days
    # (beta and gamma are not separately identifiable from cases alone)
    gamma = 1.0 / 7.0
    sigma = 1.0 / 5.2
    if model in ("sir", "sir_m"):
        y0 = [S0, I0, R0]
        x0 = np.array([0.40])
        lo = np.array([0.08])
        hi = np.array([1.20])
    elif model == "seir":
        E0 = 0.3 * I0
        y0 = [max(N - E0 - I0 - R0, 1.0), E0, I0, R0]
        x0 = np.array([0.28])
        lo = np.array([0.08])
        hi = np.array([0.80])
    else:
        E0 = 0.3 * I0
        y0 = [max(N - E0 - I0 - R0, 1.0), E0, I0, R0]
        x0 = np.array([0.45, 0.004])
        lo = np.array([0.08, 0.0008])
        hi = np.array([1.40, 0.02])

    def sse(theta):
        pred, _ = simulate(model, theta, t, y0, N, m=m)
        if not np.isfinite(pred).all():
            return 1e30
        return np.sum((pred - y) ** 2)

    # 1-parameter models: grid is more reliable than least_squares here
    if len(x0) == 1:
        grid = np.linspace(lo[0], hi[0], 30)
        scores = [sse(np.array([b])) for b in grid]
        b_best = grid[int(np.argmin(scores))]
        # small refine
        refine = np.linspace(max(lo[0], b_best - 0.03), min(hi[0], b_best + 0.03), 15)
        scores2 = [sse(np.array([b])) for b in refine]
        theta = np.array([refine[int(np.argmin(scores2))]])
        cost = min(scores2)
    else:
        fit = least_squares(lambda th: simulate(model, th, t, y0, N, m=m)[0] - y,
                            x0, bounds=(lo, hi), max_nfev=300)
        theta = fit.x
        cost = fit.cost
    pred, states = simulate(model, theta, t, y0, N, m=m)
    return theta, pred, states, y0, cost


def extend_forecast(model, theta, t_all, y0, N, m_all, n_train):
    # run over the whole window using params fitted on train
    pred, states = simulate(model, theta, t_all, y0, N, m=m_all)
    return pred, states


def main():
    ensure_dirs()
    df = load_england()
    train, test = split_train_test(df)
    nls = df[(df["date"] >= NLS_START) & (df["date"] <= NLS_END)].copy()
    N = float(df["population"].iloc[0])

    # starting recovered - roughly confirmed cases before Sep 2020
    R0 = 1500000.0
    I0 = max(float(nls["cases_7d"].iloc[0]) * 8.0, 1000.0)

    t_all = days_since(df["date"])
    nls_idx = df.index[(df["date"] >= NLS_START) & (df["date"] <= NLS_END)].to_numpy()
    t_nls = t_all[nls_idx]
    # integrate from the first nls day, so shift to start at 0 for the ODE
    t_nls0 = t_nls - t_nls[0]
    y_nls = nls["cases_7d"].to_numpy()
    y_all = df["cases_7d"].to_numpy()
    m_all = df["workplace_m"].to_numpy()
    m_nls = m_all[nls_idx]

    models = ["sir", "seir", "seirs"]
    labels = {
        "sir": "SIR",
        "seir": "SEIR",
        "seirs": "SEIRS",
    }

    rows = []
    preds = {}
    params_out = []

    for model in models:
        print("fitting", model, NLS_START, "to", NLS_END)
        theta, pred_nls, states_nls, y0, cost = fit_model(model, t_nls0, y_nls, N, I0, R0, m=None)
        # replay on the same winter window then leave NaN outside for the csv
        pred_all = np.full(len(df), np.nan)
        # continue from winter start to the end of the study (shows later waves are missed)
        t_from = t_all[nls_idx[0]:] - t_all[nls_idx[0]]
        pred_tail, states_all = simulate(model, theta, t_from, y0, N, m=None)
        pred_all[nls_idx[0]:] = pred_tail
        preds[model] = pred_all

        met_nls = fit_metrics(y_nls, pred_all[nls_idx])
        tr_mask = df["date"] <= TRAIN_END
        te_mask = df["date"] > TRAIN_END
        # train/test metrics only where the model has a prediction
        pred_tr = pred_all[tr_mask.to_numpy()]
        obs_tr = df.loc[tr_mask, "cases_7d"].to_numpy()
        ok = np.isfinite(pred_tr)
        met_tr = fit_metrics(obs_tr[ok], pred_tr[ok])
        met_te = fit_metrics(df.loc[te_mask, "cases_7d"].to_numpy(), pred_all[te_mask.to_numpy()])
        print("  wave RMSE %.1f  train RMSE %.1f  test RMSE %.1f  test R2 %.3f" % (
            met_nls["rmse"], met_tr["rmse"], met_te["rmse"], met_te["r2"]))

        rows.append({"model": labels[model], "split": "nls_wave", **met_nls})
        rows.append({"model": labels[model], "split": "train", **met_tr})
        rows.append({"model": labels[model], "split": "test", **met_te})

        gamma = 1.0 / 7.0
        sigma = 1.0 / 5.2
        if model == "sir":
            beta = theta[0]
            params_out.append({"model": "SIR", "beta": beta, "gamma": gamma, "sigma": np.nan,
                               "omega": np.nan, "R0": beta / gamma, "I0": I0, "cost": cost})
        elif model == "seir":
            beta = theta[0]
            params_out.append({"model": "SEIR", "beta": beta, "gamma": gamma, "sigma": sigma,
                               "omega": np.nan, "R0": beta / gamma, "I0": I0, "cost": cost})
        else:
            beta, omega = theta[0], theta[1]
            params_out.append({"model": "SEIRS", "beta": beta, "gamma": gamma, "sigma": sigma,
                               "omega": omega, "R0": beta / gamma, "I0": I0, "cost": cost})

        # keep last train state for interventions later
        if states_all is not None:
            np.savez(RES / ("states_%s.npz" % model),
                     t=t_all, S=states_all["S"], I=states_all["I"],
                     R=states_all["R"], pred=pred_all, theta=theta,
                     y0=np.array(y0), N=N)

    pd.DataFrame(rows).to_csv(RES / "level1_metrics.csv", index=False)
    pd.DataFrame(params_out).to_csv(RES / "level1_params.csv", index=False)
    sim = df[["date", "cases", "cases_7d"]].copy()
    for model in models:
        sim[model] = preds[model]
    sim.to_csv(RES / "level1_fitted.csv", index=False)

    # plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["date"], df["cases"], color="0.75", lw=0.5, label="daily cases")
    ax.plot(df["date"], df["cases_7d"], color="k", lw=1.3, label="7-day mean")
    colours = {"sir": "C0", "seir": "C1", "seirs": "C2", "sir_m": "C3"}
    for model in models:
        ax.plot(df["date"], preds[model], lw=1.4, color=colours[model], label=labels[model])
    shade_test(ax)
    ax.legend(frameon=False, ncol=2)
    ax.set_ylabel("New cases")
    ax.axvline(pd.Timestamp(NLS_START), color="C1", ls=":", lw=1)
    ax.axvline(pd.Timestamp(NLS_END), color="C1", ls=":", lw=1)
    ax.set_title("Level 1 NLS (fitted to Dec 2020-Feb 2021 peak)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.tight_layout()
    fig.savefig(FIG / "fig3_level1_fits.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("saved fig3_level1_fits.png")

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for ax, model in zip(axes, ["sir", "seir"]):
        resid = df["cases_7d"].to_numpy() - preds[model]
        ax.plot(df["date"], resid, lw=1)
        ax.axhline(0, color="k", lw=0.8)
        shade_test(ax)
        ax.set_ylabel("residual")
        ax.set_title(labels[model])
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.tight_layout()
    fig.savefig(FIG / "fig4_level1_residuals.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("saved fig4_level1_residuals.png")


if __name__ == "__main__":
    main()
