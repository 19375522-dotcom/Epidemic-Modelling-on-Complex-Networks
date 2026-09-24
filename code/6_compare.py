# Put the error metrics from the three levels next to each other.

import pandas as pd
import matplotlib.pyplot as plt

from helpers import RES, FIG, ensure_dirs


def main():
    ensure_dirs()
    bits = []
    for name in ["level1_metrics.csv", "level2_national_metrics.csv", "level3_metrics.csv"]:
        p = RES / name
        if p.exists():
            d = pd.read_csv(p)
            if "model" not in d.columns:
                continue
            bits.append(d[["model", "split", "rmse", "mape", "r2"]])
    tab = pd.concat(bits, ignore_index=True)
    tab.to_csv(RES / "comparison_metrics.csv", index=False)
    print(tab.to_string(index=False))

    test = tab[tab["split"].isin(["test", "nls_wave"])].copy()
    # keep EKF test + NLS winter-wave in-sample so the chart is readable
    keep = []
    for _, r in test.iterrows():
        if str(r["model"]).startswith("EKF") and r["split"] == "test":
            keep.append(True)
        elif r["split"] == "nls_wave":
            keep.append(True)
        elif r["split"] == "test" and "EKF" not in str(r["model"]) and "network" not in str(r["model"]):
            keep.append(False)
        elif r["split"] == "test" and "network" in str(r["model"]):
            keep.append(True)
        else:
            keep.append(False)
    test = test[keep].copy()
    test["label"] = test["model"] + " (" + test["split"] + ")"
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(range(len(test)), test["rmse"].values, color="C0")
    ax.set_xticks(range(len(test)))
    ax.set_xticklabels(test["label"].values, rotation=25, ha="right")
    ax.set_ylabel("RMSE (new cases, 7-day mean)")
    ax.set_title("Hold-out RMSE")
    fig.tight_layout()
    fig.savefig(FIG / "fig10_rmse_compare.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("saved fig10_rmse_compare.png")


if __name__ == "__main__":
    main()
