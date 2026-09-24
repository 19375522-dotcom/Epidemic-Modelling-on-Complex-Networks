# shared bits used by the other scripts
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
FIG = ROOT / "figures"
RES = ROOT / "results"

START = "2020-09-01"
NLS_START = "2020-12-01"  # single winter peak (constant beta cannot fit several waves)
NLS_END = "2021-02-28"
TRAIN_END = "2021-08-31"
END = "2022-03-31"


def ensure_dirs():
    FIG.mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)
    DATA_PROC.mkdir(parents=True, exist_ok=True)


def load_england():
    df = pd.read_csv(DATA_PROC / "england_daily.csv", parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def load_regions():
    df = pd.read_csv(DATA_PROC / "region_daily.csv", parse_dates=["date"])
    return df.sort_values(["area_name", "date"]).reset_index(drop=True)


def split_train_test(df):
    train = df[df["date"] <= TRAIN_END].copy()
    test = df[df["date"] > TRAIN_END].copy()
    return train.reset_index(drop=True), test.reset_index(drop=True)


def days_since(dates):
    dates = pd.to_datetime(dates)
    t0 = dates.iloc[0]
    return (dates - t0).dt.days.to_numpy().astype(float)


def fit_metrics(y, yhat):
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    err = yhat - y
    rmse = np.sqrt(np.mean(err ** 2))
    mape = np.mean(np.abs(err) / np.maximum(np.abs(y), 1.0)) * 100.0
    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {"rmse": rmse, "mape": mape, "r2": r2}


def shade_test(ax, train_end=None):
    if train_end is None:
        train_end = pd.Timestamp(TRAIN_END)
    ax.axvline(train_end, color="0.4", ls="--", lw=1)
    ax.axvspan(train_end, pd.Timestamp(END), color="0.85", alpha=0.5)
