# Build the daily csv files from the raw downloads.
# England (national) and the 9 English regions, 1 Sep 2020 - 31 Mar 2022.

import hashlib
import zipfile
from pathlib import Path

import pandas as pd

from helpers import DATA_RAW, DATA_PROC, START, END, ensure_dirs

ENGLAND_CODE = "E92000001"

# Google names -> ONS region. I used the county / combined authority rows
# (sub_region_2 empty) so districts are not counted twice.
MOBILITY_TO_REGION = {
    "Bath and North East Somerset": "South West",
    "Bedford": "East of England",
    "Blackburn with Darwen": "North West",
    "Blackpool": "North West",
    "Borough of Halton": "North West",
    "Bracknell Forest": "South East",
    "Brighton and Hove": "South East",
    "Bristol City": "South West",
    "Buckinghamshire": "South East",
    "Cambridgeshire": "East of England",
    "Central Bedfordshire": "East of England",
    "Cheshire East": "North West",
    "Cheshire West and Chester": "North West",
    "Cornwall": "South West",
    "County Durham": "North East",
    "Cumbria": "North West",
    "Darlington": "North East",
    "Derby": "East Midlands",
    "Derbyshire": "East Midlands",
    "Devon": "South West",
    "Dorset": "South West",
    "East Riding of Yorkshire": "Yorkshire and The Humber",
    "East Sussex": "South East",
    "Essex": "East of England",
    "Gloucestershire": "South West",
    "Greater London": "London",
    "Greater Manchester": "North West",
    "Hampshire": "South East",
    "Hartlepool": "North East",
    "Herefordshire": "West Midlands",
    "Hertfordshire": "East of England",
    "Isle of Wight": "South East",
    "Kent": "South East",
    "Kingston upon Hull": "Yorkshire and The Humber",
    "Lancashire": "North West",
    "Leicester": "East Midlands",
    "Leicestershire": "East Midlands",
    "Lincolnshire": "East Midlands",
    "Luton": "East of England",
    "Medway": "South East",
    "Merseyside": "North West",
    "Middlesbrough": "North East",
    "Milton Keynes": "South East",
    "Norfolk": "East of England",
    "North East Lincolnshire": "Yorkshire and The Humber",
    "North Lincolnshire": "Yorkshire and The Humber",
    "North Somerset": "South West",
    "North Yorkshire": "Yorkshire and The Humber",
    "Northamptonshire": "East Midlands",
    "Northumberland": "North East",
    "Nottingham": "East Midlands",
    "Nottinghamshire": "East Midlands",
    "Oxfordshire": "South East",
    "Peterborough": "East of England",
    "Plymouth": "South West",
    "Portsmouth": "South East",
    "Reading": "South East",
    "Redcar and Cleveland": "North East",
    "Rutland": "East Midlands",
    "Shropshire": "West Midlands",
    "Slough": "South East",
    "Somerset": "South West",
    "South Gloucestershire": "South West",
    "South Yorkshire": "Yorkshire and The Humber",
    "Southampton": "South East",
    "Southend-on-Sea": "East of England",
    "Staffordshire": "West Midlands",
    "Stockton-on-Tees": "North East",
    "Stoke-on-Trent": "West Midlands",
    "Suffolk": "East of England",
    "Surrey": "South East",
    "Swindon": "South West",
    "Thurrock": "East of England",
    "Torbay": "South West",
    "Tyne and Wear": "North East",
    "Warrington": "North West",
    "Warwickshire": "West Midlands",
    "West Berkshire": "South East",
    "West Midlands": "West Midlands",
    "West Sussex": "South East",
    "West Yorkshire": "Yorkshire and The Humber",
    "Wiltshire": "South West",
    "Windsor and Maidenhead": "South East",
    "Wokingham": "South East",
    "Worcestershire": "West Midlands",
    "York": "Yorkshire and The Humber",
}

REGION_CODES = {
    "North East": "E12000001",
    "North West": "E12000002",
    "Yorkshire and The Humber": "E12000003",
    "East Midlands": "E12000004",
    "West Midlands": "E12000005",
    "East of England": "E12000006",
    "London": "E12000007",
    "South East": "E12000008",
    "South West": "E12000009",
}


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_metric(zip_path, member, new_name, area_code=None):
    with zipfile.ZipFile(zip_path) as z:
        df = pd.read_csv(z.open(member), usecols=["date", "area_code", "area_name", "value"])
    df["date"] = pd.to_datetime(df["date"])
    if area_code is not None:
        df = df[df["area_code"] == area_code].copy()
    df = df.rename(columns={"value": new_name})
    return df[["date", "area_code", "area_name", new_name]]


def load_population():
    xls = DATA_RAW / "ukpopestimatesmid2020on2021geography.xls"
    raw = pd.read_excel(xls, sheet_name="MYE2 - Persons", header=None)
    df = raw.iloc[8:, [0, 1, 2, 3]].copy()
    df.columns = ["area_code", "area_name", "geography", "population"]
    df["area_code"] = df["area_code"].astype(str).str.strip()
    df["population"] = pd.to_numeric(df["population"], errors="coerce")
    keep = [ENGLAND_CODE] + list(REGION_CODES.values())
    df = df[df["area_code"].isin(keep)].copy()
    names = {ENGLAND_CODE: "England"}
    names.update({code: name for name, code in REGION_CODES.items()})
    df["area_name"] = df["area_code"].map(names)
    return df[["area_code", "area_name", "geography", "population"]]


def load_oxcgrt():
    df = pd.read_csv(DATA_RAW / "OxCGRT_compact_national_v1.csv")
    uk = df[df["CountryName"] == "United Kingdom"].copy()
    uk["date"] = pd.to_datetime(uk["Date"].astype(str), format="%Y%m%d")
    out = pd.DataFrame({"date": uk["date"]})
    mapping = {
        "StringencyIndex_Average": "stringency",
        "C4M_Restrictions on gatherings": "c4_gatherings",
        "C6M_Stay at home requirements": "c6_stay_home",
        "C7M_Restrictions on internal movement": "c7_movement",
        "H7_Vaccination policy": "h7_vaccine_policy",
    }
    for src, dst in mapping.items():
        if src in uk.columns:
            out[dst] = uk[src].values
    return out.drop_duplicates("date")


def load_mobility():
    frames = []
    zpath = DATA_RAW / "Region_Mobility_Report_CSVs.zip"
    with zipfile.ZipFile(zpath) as z:
        for year in (2020, 2021, 2022):
            frames.append(pd.read_csv(z.open(f"{year}_GB_Region_Mobility_Report.csv")))
    mob = pd.concat(frames, ignore_index=True)
    mob["date"] = pd.to_datetime(mob["date"])
    val_cols = [
        "retail_and_recreation_percent_change_from_baseline",
        "grocery_and_pharmacy_percent_change_from_baseline",
        "transit_stations_percent_change_from_baseline",
        "workplaces_percent_change_from_baseline",
        "residential_percent_change_from_baseline",
    ]

    uk = mob[mob["sub_region_1"].isna() & mob["sub_region_2"].isna()][["date"] + val_cols]
    uk = uk.rename(columns={
        "workplaces_percent_change_from_baseline": "workplace_pct_gb",
        "transit_stations_percent_change_from_baseline": "transit_pct_gb",
        "retail_and_recreation_percent_change_from_baseline": "retail_pct_gb",
        "residential_percent_change_from_baseline": "residential_pct_gb",
        "grocery_and_pharmacy_percent_change_from_baseline": "grocery_pct_gb",
    })

    eng = mob[mob["sub_region_2"].isna() & mob["sub_region_1"].isin(MOBILITY_TO_REGION)].copy()
    eng["region"] = eng["sub_region_1"].map(MOBILITY_TO_REGION)
    regional = eng.groupby(["date", "region"], as_index=False)[val_cols].mean()
    regional = regional.rename(columns={
        "workplaces_percent_change_from_baseline": "workplace_pct",
        "transit_stations_percent_change_from_baseline": "transit_pct",
        "retail_and_recreation_percent_change_from_baseline": "retail_pct",
        "residential_percent_change_from_baseline": "residential_pct",
        "grocery_and_pharmacy_percent_change_from_baseline": "grocery_pct",
    })

    england_mean = regional.groupby("date", as_index=False)[
        ["workplace_pct", "transit_pct", "retail_pct", "residential_pct", "grocery_pct"]
    ].mean()
    england_mean = england_mean.rename(columns={
        "workplace_pct": "workplace_pct_england",
        "transit_pct": "transit_pct_england",
        "retail_pct": "retail_pct_england",
        "residential_pct": "residential_pct_england",
        "grocery_pct": "grocery_pct_england",
    })
    uk = uk.merge(england_mean, on="date", how="outer")
    return uk, regional


def in_window(df):
    return df[(df["date"] >= START) & (df["date"] <= END)].copy()


def main():
    ensure_dirs()
    print("reading population and covariates...")
    pop = load_population()
    pop.to_csv(DATA_PROC / "population.csv", index=False)
    england_pop = int(pop.loc[pop["area_code"] == ENGLAND_CODE, "population"].iloc[0])
    print("England population (mid-2020):", england_pop)

    cases_z = DATA_RAW / "cases.zip"
    deaths_z = DATA_RAW / "deaths.zip"
    health_z = DATA_RAW / "healthcare.zip"
    vax_z = DATA_RAW / "vaccinations.zip"

    print("reading England series from the zips...")
    cases = read_metric(cases_z, "Cases/nation_newCasesBySpecimenDate.csv", "cases", ENGLAND_CODE)
    deaths = read_metric(deaths_z, "Deaths/nation_newDeaths28DaysByDeathDate.csv", "deaths", ENGLAND_CODE)
    adm = read_metric(health_z, "Healthcare/nation_newAdmissions.csv", "admissions", ENGLAND_CODE)
    hosp = read_metric(health_z, "Healthcare/nation_hospitalCases.csv", "hospital_occupancy", ENGLAND_CODE)
    vax1 = read_metric(vax_z, "Vaccinations/nation_newPeopleVaccinatedFirstDoseByVaccinationDate.csv", "vax_first_dose", ENGLAND_CODE)
    vax2 = read_metric(vax_z, "Vaccinations/nation_newPeopleVaccinatedSecondDoseByVaccinationDate.csv", "vax_second_dose", ENGLAND_CODE)
    vax3 = read_metric(vax_z, "Vaccinations/nation_newPeopleVaccinatedThirdInjectionByVaccinationDate.csv", "vax_third_dose", ENGLAND_CODE)

    eng = cases[["date", "cases"]]
    for extra in (deaths, adm, hosp, vax1, vax2, vax3):
        col = extra.columns[-1]
        eng = eng.merge(extra[["date", col]], on="date", how="outer")

    print("reading mobility and OxCGRT...")
    uk_mob, region_mob = load_mobility()
    ox = load_oxcgrt()
    eng = eng.merge(uk_mob, on="date", how="left")
    eng = eng.merge(ox, on="date", how="left")
    eng["population"] = england_pop
    eng["area_name"] = "England"
    eng["area_code"] = ENGLAND_CODE
    for col in ["cases", "deaths", "admissions", "hospital_occupancy",
                "vax_first_dose", "vax_second_dose", "vax_third_dose"]:
        eng[col] = eng[col].fillna(0)
    eng = eng.sort_values("date")
    eng["vax_first_cum"] = eng["vax_first_dose"].cumsum()
    eng["vax_second_cum"] = eng["vax_second_dose"].cumsum()
    eng["cases_7d"] = eng["cases"].rolling(7, min_periods=1).mean()
    eng["workplace_m"] = 1.0 + eng["workplace_pct_england"] / 100.0
    eng["transit_m"] = 1.0 + eng["transit_pct_england"] / 100.0

    eng_w = in_window(eng)
    cols = [
        "date", "area_code", "area_name", "population",
        "cases", "cases_7d", "deaths", "admissions", "hospital_occupancy",
        "vax_first_dose", "vax_second_dose", "vax_third_dose",
        "vax_first_cum", "vax_second_cum",
        "workplace_pct_england", "workplace_m",
        "transit_pct_england", "transit_m",
        "retail_pct_england", "residential_pct_england", "workplace_pct_gb",
        "stringency", "c4_gatherings", "c6_stay_home", "c7_movement", "h7_vaccine_policy",
    ]
    eng_w = eng_w[[c for c in cols if c in eng_w.columns]]
    eng_w.to_csv(DATA_PROC / "england_daily.csv", index=False, date_format="%Y-%m-%d")
    print("wrote england_daily.csv", len(eng_w), "rows")

    print("reading region series...")
    rcases = read_metric(cases_z, "Cases/region_newCasesBySpecimenDate.csv", "cases")
    rdeaths = read_metric(deaths_z, "Deaths/region_newDeaths28DaysByDeathDate.csv", "deaths")
    rvax1 = read_metric(vax_z, "Vaccinations/region_newPeopleVaccinatedFirstDoseByVaccinationDate.csv", "vax_first_dose")
    rvax2 = read_metric(vax_z, "Vaccinations/region_newPeopleVaccinatedSecondDoseByVaccinationDate.csv", "vax_second_dose")

    reg = rcases.merge(rdeaths[["date", "area_code", "deaths"]], on=["date", "area_code"], how="outer")
    reg = reg.merge(rvax1[["date", "area_code", "vax_first_dose"]], on=["date", "area_code"], how="outer")
    reg = reg.merge(rvax2[["date", "area_code", "vax_second_dose"]], on=["date", "area_code"], how="outer")
    pop_r = pop[pop["area_code"] != ENGLAND_CODE][["area_code", "population"]]
    reg = reg.merge(pop_r, on="area_code", how="left")
    code_to_name = {v: k for k, v in REGION_CODES.items()}
    reg["area_name"] = reg["area_code"].map(code_to_name).fillna(reg["area_name"])
    region_mob = region_mob.rename(columns={"region": "area_name"})
    reg = reg.merge(region_mob, on=["date", "area_name"], how="left")
    for col in ["cases", "deaths", "vax_first_dose", "vax_second_dose"]:
        reg[col] = reg[col].fillna(0)
    reg = reg.sort_values(["area_name", "date"])
    reg["vax_first_cum"] = reg.groupby("area_name")["vax_first_dose"].cumsum()
    reg["workplace_m"] = 1.0 + reg["workplace_pct"] / 100.0
    reg["transit_m"] = 1.0 + reg["transit_pct"] / 100.0
    reg["cases_7d"] = reg.groupby("area_name")["cases"].transform(lambda s: s.rolling(7, min_periods=1).mean())
    reg_w = in_window(reg)
    rcols = [
        "date", "area_code", "area_name", "population",
        "cases", "cases_7d", "deaths",
        "vax_first_dose", "vax_second_dose", "vax_first_cum",
        "workplace_pct", "workplace_m", "transit_pct", "transit_m",
        "retail_pct", "residential_pct",
    ]
    reg_w = reg_w[[c for c in rcols if c in reg_w.columns]]
    reg_w.to_csv(DATA_PROC / "region_daily.csv", index=False, date_format="%Y-%m-%d")
    print("wrote region_daily.csv", len(reg_w), "rows")

    # hashes so I can show the files were not changed later
    lines = ["# hashes of the raw files I used", ""]
    for name in [
        "cases.zip", "deaths.zip", "healthcare.zip", "vaccinations.zip",
        "Region_Mobility_Report_CSVs.zip", "OxCGRT_compact_national_v1.csv",
        "ukpopestimatesmid2020on2021geography.xls",
    ]:
        p = DATA_RAW / name
        lines.append(f"{name}  {p.stat().st_size}  {file_hash(p)}")
    (DATA_PROC / "manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("done.")


if __name__ == "__main__":
    main()
