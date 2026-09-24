# Level 2 - 9 English regions.
# Each region is a SIR with beta_i(t) = b_i * workplace_m_i(t).
# I also put a small amount of coupling along neighbouring regions.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from scipy.integrate import solve_ivp
from helpers import (
    load_england, load_regions, split_train_test, days_since, fit_metrics,
    FIG, RES, ensure_dirs, shade_test, NLS_START, NLS_END,
)

# neighbouring English regions (undirected)
EDGES = [
    ("North East", "North West"),
    ("North East", "Yorkshire and The Humber"),
    ("North West", "Yorkshire and The Humber"),
    ("North West", "West Midlands"),
    ("Yorkshire and The Humber", "East Midlands"),
    ("East Midlands", "West Midlands"),
    ("East Midlands", "East of England"),
    ("East Midlands", "South East"),
    ("West Midlands", "South East"),
    ("West Midlands", "South West"),
    ("East of England", "London"),
    ("East of England", "South East"),
    ("London", "South East"),
    ("South East", "South West"),
]


def build_graph(names):
    G = nx.Graph()
    G.add_nodes_from(names)
    for a, b in EDGES:
        if a in names and b in names:
            G.add_edge(a, b)
    A = nx.to_numpy_array(G, nodelist=names)
    # row-normalise so coupling is a weighted average of neighbours
    deg = A.sum(axis=1, keepdims=True)
    deg[deg == 0] = 1.0
    A_hat = A / deg
    return G, A, A_hat


def fit_one_region(t, y, m, N, I0, R0, gamma):
    # only b is free, gamma taken from the national SIR
    S0 = max(N - I0 - R0, 1.0)
    y0 = [S0, I0, R0]

    def simulate(b):
        def f(tt, z):
            S, I, R = z
            beta = b * np.interp(tt, t, m)
            inf = beta * S * I / N
            return [-inf, inf - gamma * I, gamma * I]
        sol = solve_ivp(f, [t[0], t[-1]], y0, t_eval=t, method="RK45",
                        rtol=1e-6, atol=1e-6, max_step=1.0)
        if not sol.success or sol.y.shape[1] != len(t):
            return np.full(len(t), 1e12)
        S, I, R = sol.y
        beta_t = b * m
        return beta_t * S * I / N

    grid = np.linspace(0.12, 0.55, 20)
    best_sse = np.inf
    best_b = grid[0]
    for b in grid:
        pred = simulate(b)
        sse = np.sum((pred - y) ** 2)
        if np.isfinite(sse) and sse < best_sse:
            best_sse = sse
            best_b = b
    pred = simulate(best_b)
    return best_b, pred


def coupled_sim(b_vec, gamma, eps, t, m, N, I0, R0, A_hat):
    # state = [S1..Sk, I1..Ik, R1..Rk]
    k = len(N)
    S0 = np.maximum(N - I0 - R0, 1.0)
    y0 = np.concatenate([S0, I0, R0])

    def f(tt, z):
        S = z[:k]
        I = z[k:2 * k]
        R = z[2 * k:]
        mt = np.empty(k)
        for i in range(k):
            mt[i] = np.interp(tt, t, m[:, i])
        # force of infection: local + a bit from neighbours
        I_force = (1.0 - eps) * I + eps * (A_hat @ I)
        inf = b_vec * mt * S * I_force / N
        dS = -inf
        dI = inf - gamma * I
        dR = gamma * I
        return np.concatenate([dS, dI, dR])

    sol = solve_ivp(f, [t[0], t[-1]], y0, t_eval=t, method="RK45",
                    rtol=1e-6, atol=1e-6, max_step=1.0)
    if not sol.success:
        return None
    S = sol.y[:k]
    I = sol.y[k:2 * k]
    mt = m.T  # (k, T) wait m is (T, k)
    # incidence
    inc = np.zeros((k, len(t)))
    for j, tt in enumerate(t):
        mtj = m[j]
        I_force = (1.0 - eps) * I[:, j] + eps * (A_hat @ I[:, j])
        inc[:, j] = b_vec * mtj * S[:, j] * I_force / N
    return inc, S, I


def main():
    ensure_dirs()
    eng = load_england()
    reg = load_regions()
    names = sorted(reg["area_name"].unique())
    G, A, A_hat = build_graph(names)
    rho = float(np.max(np.abs(np.linalg.eigvals(A))))
    print("nodes", names)
    print("spectral radius of A:", rho)

    # gamma from national SIR if available
    gamma = 1.0 / 7.0
    pfile = RES / "level1_params.csv"
    if pfile.exists():
        p = pd.read_csv(pfile)
        if (p["model"] == "SIR").any():
            gamma = float(p.loc[p["model"] == "SIR", "gamma"].iloc[0])
    print("using gamma =", gamma)

    train_end = pd.Timestamp("2021-08-31")
    nls_end = pd.Timestamp(NLS_END)
    rows = []
    pred_store = []
    b_list = []
    N_list = []
    I0_list = []
    R0_list = []

    dates = None
    m_mat = []
    y_mat = []

    for name in names:
        g = reg[reg["area_name"] == name].reset_index(drop=True)
        if dates is None:
            dates = g["date"]
        tr = g[g["date"] <= train_end]
        nls = g[(g["date"] >= pd.Timestamp(NLS_START)) & (g["date"] <= nls_end)]
        N = float(g["population"].iloc[0])
        t = days_since(g["date"])
        t_tr = days_since(nls["date"])
        y_tr = nls["cases_7d"].to_numpy()
        m_tr = nls["workplace_m"].to_numpy()
        m_all = g["workplace_m"].ffill().fillna(1.0).to_numpy()
        I0 = max(float(nls["cases_7d"].iloc[0]) * 8.0, 200.0)
        R0 = 0.01 * N
        print("fitting", name)
        b, pred_tr = fit_one_region(t_tr, y_tr, m_tr, N, I0, R0, gamma)

        # replay on full window
        def simulate_full(b):
            y0 = [max(N - I0 - R0, 1.0), I0, R0]

            def f(tt, z):
                S, I, R = z
                beta = b * np.interp(tt, t, m_all)
                inf = beta * S * I / N
                return [-inf, inf - gamma * I, gamma * I]
            sol = solve_ivp(f, [t[0], t[-1]], y0, t_eval=t, method="RK45",
                            rtol=1e-6, atol=1e-6, max_step=1.0)
            S, I, R = sol.y
            return (b * m_all) * S * I / N

        pred_all = simulate_full(b)
        y_all = g["cases_7d"].to_numpy()
        ntr = len(tr)
        met_tr = fit_metrics(y_all[:ntr], pred_all[:ntr])
        met_te = fit_metrics(y_all[ntr:], pred_all[ntr:])
        rows.append({"region": name, "b": b, "R0_base": b / gamma, "split": "train", **met_tr})
        rows.append({"region": name, "b": b, "R0_base": b / gamma, "split": "test", **met_te})
        tmp = g[["date", "area_name", "cases_7d"]].copy()
        tmp["pred"] = pred_all
        pred_store.append(tmp)
        b_list.append(b)
        N_list.append(N)
        I0_list.append(I0)
        R0_list.append(R0)
        m_mat.append(m_all)
        y_mat.append(y_all)
        print("  b=%.3f  test RMSE=%.1f" % (b, met_te["rmse"]))

    metrics = pd.DataFrame(rows)
    metrics.to_csv(RES / "level2_region_metrics.csv", index=False)
    pd.concat(pred_store, ignore_index=True).to_csv(RES / "level2_fitted.csv", index=False)

    b_vec = np.array(b_list)
    N_vec = np.array(N_list)
    I0_vec = np.array(I0_list)
    R0_vec = np.array(R0_list)
    m_mat = np.column_stack(m_mat)  # T x k
    y_mat = np.column_stack(y_mat)
    t = days_since(dates)

    # coupled run with a small eps
    eps = 0.08
    coupled = coupled_sim(b_vec, gamma, eps, t, m_mat, N_vec, I0_vec, R0_vec, A_hat)
    if coupled is not None:
        inc, S, I = coupled
        # national sum vs England
        eng_cases = eng.set_index("date")["cases_7d"]
        nat_pred = inc.sum(axis=0)
        nat_obs = y_mat.sum(axis=1)
        ntr = int((dates <= train_end).sum())
        met_tr = fit_metrics(nat_obs[:ntr], nat_pred[:ntr])
        met_te = fit_metrics(nat_obs[ntr:], nat_pred[ntr:])
        print("coupled national train RMSE", met_tr["rmse"], "test", met_te["rmse"])
        uncoupled_nat = pd.concat(pred_store).groupby("date")["pred"].sum().to_numpy()
        pd.DataFrame([
            {"model": "network uncoupled", "split": "train", **fit_metrics(nat_obs[:ntr], uncoupled_nat[:ntr])},
            {"model": "network uncoupled", "split": "test", **fit_metrics(nat_obs[ntr:], uncoupled_nat[ntr:])},
            {"model": "network coupled", "split": "train", **met_tr},
            {"model": "network coupled", "split": "test", **met_te},
        ]).to_csv(RES / "level2_national_metrics.csv", index=False)
        pd.DataFrame({"date": dates, "obs_sum": nat_obs, "uncoupled": uncoupled_nat, "coupled": nat_pred}).to_csv(
            RES / "level2_national_series.csv", index=False
        )
        np.savez(RES / "level2_states.npz", t=t, S=S, I=I, inc=inc, names=np.array(names),
                 b=b_vec, gamma=gamma, eps=eps, A=A, rho=rho, N=N_vec)

    # Reff(t) from mean of b_i * m_i(t) / gamma
    reff = (b_vec * m_mat / gamma).mean(axis=1)
    pd.DataFrame({"date": dates, "Reff_mean": reff, "rho_A": rho}).to_csv(RES / "level2_reff.csv", index=False)

    # plots
    fig, ax = plt.subplots(figsize=(7, 6))
    pos = {
        "North East": (1.0, 3.0),
        "North West": (0.0, 2.2),
        "Yorkshire and The Humber": (1.6, 2.2),
        "East Midlands": (1.6, 1.3),
        "West Midlands": (0.4, 1.3),
        "East of England": (2.4, 0.7),
        "London": (2.0, 0.0),
        "South East": (1.2, 0.15),
        "South West": (0.0, 0.2),
    }
    nx.draw(G, pos=pos, with_labels=True, node_color="#a6cee3", node_size=1800,
            font_size=8, edge_color="0.5", ax=ax)
    ax.set_title("Region graph  (rho(A) = %.2f)" % rho)
    fig.tight_layout()
    fig.savefig(FIG / "fig5_region_graph.png", dpi=200, bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(3, 3, figsize=(11, 8), sharex=True)
    fitted = pd.concat(pred_store, ignore_index=True)
    for ax, name in zip(axes.ravel(), names):
        g = fitted[fitted["area_name"] == name]
        ax.plot(g["date"], g["cases_7d"], color="k", lw=1)
        ax.plot(g["date"], g["pred"], color="C0", lw=1)
        shade_test(ax)
        ax.set_title(name, fontsize=9)
    fig.suptitle("Level 2: mobility-scaled SIR per region")
    fig.tight_layout()
    fig.savefig(FIG / "fig6_level2_regions.png", dpi=200, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(dates, reff, color="C3")
    ax.axhline(1.0, color="k", ls=":")
    shade_test(ax)
    ax.set_ylabel("mean Reff")
    ax.set_title("Mean regional Reff(t) = b_i m_i(t) / gamma")
    fig.tight_layout()
    fig.savefig(FIG / "fig7_level2_reff.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("saved level 2 figures")


if __name__ == "__main__":
    main()
