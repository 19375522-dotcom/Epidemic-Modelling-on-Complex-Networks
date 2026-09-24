# Quick look at the processed data before fitting anything.

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from helpers import load_england, load_regions, FIG, RES, ensure_dirs, shade_test, START, END


def fmt_dates(ax):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")


def main():
    ensure_dirs()
    eng = load_england()
    reg = load_regions()

    # missing values
    miss = eng.isna().mean().rename("frac_missing").to_frame()
    miss["n_missing"] = eng.isna().sum()
    miss.to_csv(RES / "missing_england.csv")
    print(miss[miss["n_missing"] > 0])

    # summary table
    rows = []
    rows.append({
        "series": "England",
        "n_days": len(eng),
        "cases_sum": eng["cases"].sum(),
        "deaths_sum": eng["deaths"].sum(),
        "vax1_end": eng["vax_first_cum"].iloc[-1],
        "workplace_m_min": eng["workplace_m"].min(),
        "workplace_m_max": eng["workplace_m"].max(),
    })
    for name, g in reg.groupby("area_name"):
        rows.append({
            "series": name,
            "n_days": len(g),
            "cases_sum": g["cases"].sum(),
            "deaths_sum": g["deaths"].sum(),
            "vax1_end": g["vax_first_cum"].iloc[-1],
            "workplace_m_min": g["workplace_m"].min(),
            "workplace_m_max": g["workplace_m"].max(),
        })
    pd.DataFrame(rows).to_csv(RES / "data_summary.csv", index=False)

    # figure 1 - England overview
    fig, axes = plt.subplots(3, 2, figsize=(11, 9), sharex=True)
    axes[0, 0].plot(eng["date"], eng["cases"], color="0.7", lw=0.6)
    axes[0, 0].plot(eng["date"], eng["cases_7d"], color="C0", lw=1.4)
    axes[0, 0].set_ylabel("New cases")
    axes[0, 0].set_title("Cases (grey = daily, blue = 7-day mean)")
    shade_test(axes[0, 0])

    axes[0, 1].plot(eng["date"], eng["deaths"], color="C3", lw=1)
    axes[0, 1].set_ylabel("Deaths")
    axes[0, 1].set_title("Deaths within 28 days")
    shade_test(axes[0, 1])

    axes[1, 0].plot(eng["date"], eng["admissions"], color="C1", lw=1)
    axes[1, 0].set_ylabel("Admissions")
    axes[1, 0].set_title("Hospital admissions")
    shade_test(axes[1, 0])

    axes[1, 1].plot(eng["date"], eng["vax_first_cum"] / 1e6, label="1st dose")
    axes[1, 1].plot(eng["date"], eng["vax_second_cum"] / 1e6, label="2nd dose")
    axes[1, 1].set_ylabel("People (millions)")
    axes[1, 1].set_title("Cumulative vaccinations")
    axes[1, 1].legend(frameon=False)
    shade_test(axes[1, 1])

    axes[2, 0].plot(eng["date"], eng["workplace_m"], color="C2", lw=1)
    axes[2, 0].axhline(1.0, color="0.5", ls=":")
    axes[2, 0].set_ylabel("workplace_m")
    axes[2, 0].set_title("Google workplace mobility (1 = baseline)")
    shade_test(axes[2, 0])

    axes[2, 1].plot(eng["date"], eng["stringency"], color="C4", lw=1)
    axes[2, 1].set_ylabel("Index")
    axes[2, 1].set_title("OxCGRT stringency (UK)")
    shade_test(axes[2, 1])

    for ax in axes.ravel():
        fmt_dates(ax)
    fig.suptitle("England, %s to %s. Shaded = hold-out" % (START, END), y=1.01)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_england_overview.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("saved fig1_england_overview.png")

    # figure 2 - regions
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, g in reg.groupby("area_name"):
        ax.plot(g["date"], g["cases_7d"], lw=1.1, label=name)
    shade_test(ax)
    fmt_dates(ax)
    ax.set_ylabel("New cases (7-day mean)")
    ax.set_title("Regional cases")
    ax.legend(ncol=2, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_region_cases.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("saved fig2_region_cases.png")

    # check region sum vs England
    reg_sum = reg.groupby("date")["cases"].sum()
    merged = eng.set_index("date")["cases"].to_frame("england").join(reg_sum.rename("regions"))
    merged["diff"] = merged["england"] - merged["regions"]
    print("mean daily gap England - sum(regions):", merged["diff"].mean())
    merged.to_csv(RES / "england_vs_region_cases.csv")


if __name__ == "__main__":
    main()
