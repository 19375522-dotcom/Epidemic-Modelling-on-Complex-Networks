# Level 3 - SIRD-V with an extended Kalman filter.
# I first use NLS for gamma, then let the EKF track beta through time.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from helpers import (
    load_england, split_train_test, fit_metrics,
    FIG, RES, ensure_dirs, shade_test,
)


def jacobian(fun, x, eps=1e-5):
    n = len(x)
    fx = fun(x)
    J = np.zeros((len(fx), n))
    for i in range(n):
        xp = x.copy()
        step = eps * max(abs(x[i]), 1.0)
        xp[i] += step
        J[:, i] = (fun(xp) - fx) / step
    return J


def main():
    ensure_dirs()
    df = load_england()
    train, test = split_train_test(df)
    N = float(df["population"].iloc[0])

    gamma = 1.0 / 7.0
    pfile = RES / "level1_params.csv"
    if pfile.exists():
        p = pd.read_csv(pfile)
        if (p["model"] == "SIR").any():
            gamma = float(p.loc[p["model"] == "SIR", "gamma"].iloc[0])

    # infection fatality among the I compartment - rough
    mu = float(train["deaths"].mean() / max(train["cases_7d"].mean(), 1.0) * gamma)
    mu = min(max(mu, 1e-4), 0.02)
    print("gamma", gamma, "mu", mu)

    cases = df["cases_7d"].to_numpy()
    deaths = df["deaths"].to_numpy()
    vax = df["vax_first_dose"].to_numpy()
    T = len(df)

    I0 = max(cases[0] * 6.0, 1000.0)
    D0 = 40000.0
    R0 = 350000.0
    V0 = 0.0
    S0 = max(N - I0 - R0 - D0 - V0, 1.0)
    beta0 = 0.30

    # state: S, I, R, D, V, beta
    x = np.array([S0, I0, R0, D0, V0, beta0], dtype=float)
    P = np.diag([1e10, 1e8, 1e10, 1e8, 1e8, 0.05 ** 2])
    # process noise - allow beta to move
    Q = np.diag([1e8, 1e6, 1e8, 1e4, 1e6, 0.04 ** 2])
    # observation noise for [incidence, deaths]
    R = np.diag([3e7, 4e3])

    xs = np.zeros((T, 6))
    yhat = np.zeros((T, 2))
    openloop = np.zeros(T)  # forecast with frozen beta after train
    n_train = len(train)

    def f_proc(x, nu):
        S, I, R, D, V, beta = x
        S = max(S, 0.0)
        I = max(I, 0.0)
        inf = beta * S * I / N
        nu = min(nu, S)
        Sn = S - inf - nu
        In = I + inf - gamma * I - mu * I
        Rn = R + gamma * I
        Dn = D + mu * I
        Vn = V + nu
        return np.array([Sn, In, Rn, Dn, Vn, beta])

    def h_obs(x):
        S, I, R, D, V, beta = x
        inf = beta * max(S, 0.0) * max(I, 0.0) / N
        return np.array([inf, mu * max(I, 0.0)])

    x_end_train = None
    for k in range(T):
        nu = vax[k]
        # predict
        x_pred = f_proc(x, nu)
        F = jacobian(lambda z: f_proc(z, nu), x)
        P_pred = F @ P @ F.T + Q

        if k == n_train:
            x_end_train = x_pred.copy()

        # update using today's cases and deaths
        y = np.array([cases[k], deaths[k]])
        y_pred = h_obs(x_pred)
        H = jacobian(h_obs, x_pred)
        S_y = H @ P_pred @ H.T + R
        try:
            K = P_pred @ H.T @ np.linalg.inv(S_y)
        except np.linalg.LinAlgError:
            K = np.zeros((6, 2))
        x = x_pred + K @ (y - y_pred)
        P = (np.eye(6) - K @ H) @ P_pred

        # keep compartments non-negative and beta in a sensible range
        x[0:5] = np.clip(x[0:5], 0.0, N)
        x[5] = np.clip(x[5], 0.02, 2.5)
        xs[k] = x
        yhat[k] = h_obs(x)

    # open-loop from end of train (no more measurement updates)
    if x_end_train is None:
        x_end_train = xs[n_train - 1]
    xol = x_end_train.copy()
    for k in range(T):
        if k < n_train:
            openloop[k] = yhat[k, 0]
        else:
            xol = f_proc(xol, vax[k])
            xol[0:5] = np.clip(xol[0:5], 0.0, N)
            openloop[k] = h_obs(xol)[0]

    met_tr_f = fit_metrics(cases[:n_train], yhat[:n_train, 0])
    met_te_f = fit_metrics(cases[n_train:], yhat[n_train:, 0])
    met_te_ol = fit_metrics(cases[n_train:], openloop[n_train:])
    print("EKF filter train RMSE", met_tr_f["rmse"], "test", met_te_f["rmse"])
    print("EKF open-loop test RMSE", met_te_ol["rmse"])

    pd.DataFrame([
        {"model": "EKF (filter)", "split": "train", **met_tr_f},
        {"model": "EKF (filter)", "split": "test", **met_te_f},
        {"model": "EKF (open-loop)", "split": "test", **met_te_ol},
    ]).to_csv(RES / "level3_metrics.csv", index=False)

    out = df[["date", "cases_7d", "deaths", "stringency", "workplace_m"]].copy()
    out["inc_filter"] = yhat[:, 0]
    out["deaths_filter"] = yhat[:, 1]
    out["inc_openloop"] = openloop
    out["S"] = xs[:, 0]
    out["I"] = xs[:, 1]
    out["R"] = xs[:, 2]
    out["D"] = xs[:, 3]
    out["V"] = xs[:, 4]
    out["beta"] = xs[:, 5]
    out["Reff"] = xs[:, 5] / gamma * (xs[:, 0] / N)
    out.to_csv(RES / "level3_ekf.csv", index=False)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(df["date"], cases, color="k", lw=1, label="cases 7d")
    axes[0].plot(df["date"], yhat[:, 0], color="C0", lw=1.3, label="EKF filter")
    axes[0].plot(df["date"], openloop, color="C1", lw=1.2, ls="--", label="open-loop after train")
    shade_test(axes[0])
    axes[0].legend(frameon=False)
    axes[0].set_ylabel("New cases")
    axes[0].set_title("SIRD-V EKF")

    axes[1].plot(df["date"], xs[:, 5], color="C3")
    shade_test(axes[1])
    axes[1].set_ylabel("beta")
    axes[1].set_title("Filtered transmission rate")

    axes[2].plot(df["date"], out["Reff"], color="C4")
    axes[2].axhline(1.0, color="k", ls=":")
    shade_test(axes[2])
    axes[2].set_ylabel("Reff")
    axes[2].set_title("Reff = (beta/gamma) * S/N")
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.tight_layout()
    fig.savefig(FIG / "fig8_ekf.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("saved fig8_ekf.png")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["date"], deaths, color="k", lw=1, label="deaths")
    ax.plot(df["date"], yhat[:, 1], color="C3", lw=1.2, label="EKF")
    shade_test(ax)
    ax.legend(frameon=False)
    ax.set_ylabel("Deaths")
    ax.set_title("EKF deaths")
    fig.tight_layout()
    fig.savefig(FIG / "fig9_ekf_deaths.png", dpi=200, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
