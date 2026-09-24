# Processed dataset

Study window: **2020-09-01 to 2022-03-31** (inclusive).
Geography: **England** (national) and **9 English regions** (ITL1 / former GOR).

## Files to use in the models

| File | Rows | Use |
|---|---|---|
| `england_daily.csv` | 577 | Level 1 scalar models and Level 3 SIRD-V / EKF |
| `region_daily.csv` | 5193 (9 regions) | Level 2 networked / mobility-scaled model |
| `population.csv` | 10 | Fixed mid-2020 denominators |

England mid-2020 population: **56,550,138**.
England cases in window (sum of daily new cases): **17,609,247**.
England rows with missing workplace mobility: **0**.

## Column notes

- `cases` — new lab-confirmed cases by **specimen date** (not publish date).
- `deaths` — deaths within 28 days of a positive test, by **death date**.
- `admissions` / `hospital_occupancy` — England only (national NHS series).
- `vax_*_dose` — new people vaccinated that day; `vax_first_cum` is the running total.
- `workplace_pct_*` — Google percent change from Jan–Feb 2020 baseline.
- `workplace_m` — `1 + workplace_pct/100` (1.0 = baseline; 0.6 = 40% below baseline).
- `stringency` — OxCGRT UK StringencyIndex_Average (national; copied onto England).

## Important modelling caveats

1. Google mobility is **not** origin–destination passenger flow. Level 2 should use `workplace_m` as a time-varying scalar on transmission, not as a flow matrix.
2. Regional mobility is the unweighted mean of Google `sub_region_1` areas mapped into each ONS region (county / combined-authority rows only, to avoid double-counting districts).
3. OxCGRT is **UK** policy, applied to England. That is standard and should be stated in Chapter 3.
4. Fit on `cases` or `cases_7d`; report which. Weekend reporting noise is why `cases_7d` exists.
5. Train: 2020-09-01 to 2021-08-31. Hold-out: 2021-09-01 to 2022-03-31.

## Sources

- UKHSA COVID-19 archive (decommissioned GOV.UK dashboard): cases, deaths, healthcare, vaccinations.
- Google COVID-19 Community Mobility Reports (last update 2022-10-15).
- Oxford COVID-19 Government Response Tracker, compact national v1.
- ONS mid-2020 population estimates (2021 geography), England and 9 regions.
