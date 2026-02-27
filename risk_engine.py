from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


NUMERIC_COLUMNS = [
    "resting_hr",
    "hrv",
    "sleep_hours",
    "steps",
    "spo2",
    "weight_kg",
    "systolic_bp",
    "diastolic_bp",
    "glucose_fasting",
    "temperature_f",
]


COLUMN_ALIASES = {
    "date": "date",
    "timestamp": "date",
    "datetime": "date",
    "day": "date",
    "local_date": "date",
    "start": "start",
    "start_time": "start",
    "startdate": "start_date",
    "start_date": "start_date",
    "end": "end",
    "end_time": "end",
    "enddate": "end_date",
    "end_date": "end_date",
    "recorded_at": "date",
    "created_at": "date",
    "creation_date": "date",
    "sample_date": "date",
    "cycle_start_time": "date",
    "cycle_end_time": "end",
    "sleep_onset": "start",
    "wake_onset": "end",
    "restinghr": "resting_hr",
    "resting_hr": "resting_hr",
    "resting_heart_rate": "resting_hr",
    "resting_heart_rate_bpm": "resting_hr",
    "rhr": "resting_hr",
    "hrv": "hrv",
    "hrv_rmssd": "hrv",
    "hrv_rmssd_ms": "hrv",
    "hrv_rmssd_milli_seconds": "hrv",
    "heart_rate_variability": "hrv",
    "heart_rate_variability_ms": "hrv",
    "heart_rate_variability_rmssd": "hrv",
    "heart_rate_variability_rmssd_milli_seconds": "hrv",
    "sleep": "sleep_hours",
    "sleep_hours": "sleep_hours",
    "sleephrs": "sleep_hours",
    "sleep_duration": "sleep_hours",
    "sleep_duration_hours": "sleep_hours",
    "sleep_minutes": "sleep_hours",
    "sleep_duration_minutes": "sleep_hours",
    "sleep_duration_seconds": "sleep_hours",
    "sleep_duration_ms": "sleep_hours",
    "sleep_duration_milliseconds": "sleep_hours",
    "total_sleep_time": "sleep_hours",
    "total_sleep_time_ms": "sleep_hours",
    "total_sleep_duration": "sleep_hours",
    "total_sleep_duration_ms": "sleep_hours",
    "sleep_time": "sleep_hours",
    "asleep_duration_min": "sleep_hours",
    "asleep_duration_mins": "sleep_hours",
    "asleep_duration_minutes": "sleep_hours",
    "steps": "steps",
    "step_count": "steps",
    "stepcount": "steps",
    "steps_count": "steps",
    "spo2": "spo2",
    "spo2_percent": "spo2",
    "spo2_percentage": "spo2",
    "blood_oxygen": "spo2",
    "blood_oxygen_percent": "spo2",
    "blood_oxygen_percentage": "spo2",
    "blood_oxygen_saturation": "spo2",
    "oxygen_saturation": "spo2",
    "blood_oxygen": "spo2",
    "blood_oxygen_percent": "spo2",
    "weight": "weight_kg",
    "weight_kg": "weight_kg",
    "body_mass": "weight_kg",
    "body_weight": "weight_kg",
    "weight_lbs": "weight_kg",
    "weight_lb": "weight_kg",
    "systolic": "systolic_bp",
    "systolic_bp": "systolic_bp",
    "systolic_blood_pressure": "systolic_bp",
    "diastolic": "diastolic_bp",
    "diastolic_bp": "diastolic_bp",
    "diastolic_blood_pressure": "diastolic_bp",
    "glucose": "glucose_fasting",
    "glucose_fasting": "glucose_fasting",
    "fasting_glucose": "glucose_fasting",
    "blood_glucose": "glucose_fasting",
    "fasting_blood_glucose": "glucose_fasting",
    "temp_f": "temperature_f",
    "temperature_f": "temperature_f",
    "temperature": "temperature_f",
    "body_temperature": "temperature_f",
    "skin_temperature": "temperature_f",
    "skin_temp_celsius": "temperature_f",
    "skin_temp": "temperature_f",
    "body_temperature_celsius": "temperature_f",
}


LONG_METRIC_VALUE_ALIASES = {
    "hkquantitytypeidentifierrestingheartrate": "resting_hr",
    "restingheartrate": "resting_hr",
    "resting_heart_rate": "resting_hr",
    "resting heart rate": "resting_hr",
    "hkquantitytypeidentifierheartratevariabilitysdnn": "hrv",
    "heart_rate_variability_sdnn": "hrv",
    "heart rate variability": "hrv",
    "hrv": "hrv",
    "hkquantitytypeidentifierstepcount": "steps",
    "step_count": "steps",
    "steps": "steps",
    "hkquantitytypeidentifieroxygensaturation": "spo2",
    "oxygen_saturation": "spo2",
    "blood_oxygen_saturation": "spo2",
    "spo2": "spo2",
    "hkquantitytypeidentifierbodymass": "weight_kg",
    "body_mass": "weight_kg",
    "weight": "weight_kg",
    "hkquantitytypeidentifierbloodpressuresystolic": "systolic_bp",
    "blood_pressure_systolic": "systolic_bp",
    "systolic_blood_pressure": "systolic_bp",
    "hkquantitytypeidentifierbloodpressurediastolic": "diastolic_bp",
    "blood_pressure_diastolic": "diastolic_bp",
    "diastolic_blood_pressure": "diastolic_bp",
    "hkquantitytypeidentifierbloodglucose": "glucose_fasting",
    "blood_glucose": "glucose_fasting",
    "glucose": "glucose_fasting",
    "hkquantitytypeidentifierbodytemperature": "temperature_f",
    "body_temperature": "temperature_f",
    "skin_temperature": "temperature_f",
    "temperature": "temperature_f",
    "sleep_duration": "sleep_hours",
    "total_sleep_time": "sleep_hours",
    "sleep": "sleep_hours",
}


@dataclass
class RiskItem:
    name: str
    probability: float
    severity: str
    trend: str
    confidence: float
    why: list[str]
    protective_factors: list[str]
    next_steps: list[str]
    raw_score: float


def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return float(max(lo, min(hi, value)))


def _normalize_columns(columns: list[str]) -> list[str]:
    out = []
    for col in columns:
        normalized = col.strip().lower()
        normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
        normalized = re.sub(r"_+", "_", normalized).strip("_")
        out.append(COLUMN_ALIASES.get(normalized, normalized))
    return out


def _find_best_date_column(ts: pd.DataFrame) -> str | None:
    if ts.empty:
        return None
    def has_date_token(col: str) -> bool:
        tokens = set(str(col).split("_"))
        return bool(tokens & {"date", "time", "timestamp", "onset", "start", "end"})

    candidates = [c for c in ts.columns if has_date_token(c)]
    best_col = None
    best_score = -1.0
    for col in candidates:
        parsed = pd.to_datetime(ts[col], errors="coerce")
        score = float(parsed.notna().mean()) if len(parsed) else 0.0
        if score > best_score and score >= 0.5:
            best_col = col
            best_score = score
    return best_col


def _normalize_metric_value(metric: Any) -> str | None:
    if metric is None:
        return None
    raw = str(metric).strip().lower()
    if not raw:
        return None
    compact = raw.replace(" ", "_").replace("-", "_")
    compact_key = compact.replace("_", "")
    if compact in LONG_METRIC_VALUE_ALIASES:
        return LONG_METRIC_VALUE_ALIASES[compact]
    if compact_key in LONG_METRIC_VALUE_ALIASES:
        return LONG_METRIC_VALUE_ALIASES[compact_key]

    # Heuristics for WHOOP/Apple variants and other exports.
    if "resting" in compact and "heart" in compact and "rate" in compact:
        return "resting_hr"
    if "hrv" in compact or ("variability" in compact and "heart" in compact):
        return "hrv"
    if "step" in compact:
        return "steps"
    if "oxygen" in compact or "spo2" in compact:
        return "spo2"
    if ("body" in compact and "mass" in compact) or compact in {"weight", "body_weight"}:
        return "weight_kg"
    if "systolic" in compact:
        return "systolic_bp"
    if "diastolic" in compact:
        return "diastolic_bp"
    if "glucose" in compact:
        return "glucose_fasting"
    if "temperature" in compact or "temp" in compact:
        return "temperature_f"
    if "sleep" in compact and ("duration" in compact or "time" in compact or compact == "sleep"):
        return "sleep_hours"
    return None


def _coerce_units_in_timeseries(ts: pd.DataFrame) -> pd.DataFrame:
    if ts.empty:
        return ts
    out = ts.copy()
    for col in NUMERIC_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    # Sleep duration normalization: minutes/seconds/milliseconds -> hours
    if "sleep_hours" in out.columns:
        s = out["sleep_hours"].dropna()
        if not s.empty:
            median = float(s.median())
            if median > 10000:
                out["sleep_hours"] = out["sleep_hours"] / 3_600_000.0
            elif median > 1440:
                # Values larger than a full day are most likely seconds.
                out["sleep_hours"] = out["sleep_hours"] / 3600.0
            elif median > 24:
                # Typical wearable exports often store sleep as minutes.
                out["sleep_hours"] = out["sleep_hours"] / 60.0

    # SpO2 can be exported as fraction (0.97) instead of percent (97)
    if "spo2" in out.columns:
        s = out["spo2"].dropna()
        if not s.empty and float(s.median()) <= 1.5:
            out["spo2"] = out["spo2"] * 100.0

    # Temperature may be Celsius; convert when values look like body temp C.
    if "temperature_f" in out.columns:
        s = out["temperature_f"].dropna()
        if not s.empty:
            median = float(s.median())
            if 30 <= median <= 45:
                out["temperature_f"] = (out["temperature_f"] * 9.0 / 5.0) + 32.0

    # Glucose may appear in mmol/L.
    if "glucose_fasting" in out.columns:
        s = out["glucose_fasting"].dropna()
        if not s.empty:
            median = float(s.median())
            if 2 <= median <= 25:
                out["glucose_fasting"] = out["glucose_fasting"] * 18.0

    return out


def _aggregate_daily_timeseries(ts: pd.DataFrame) -> pd.DataFrame:
    if ts.empty:
        return ts
    agg_map: dict[str, Any] = {}
    for col in NUMERIC_COLUMNS:
        if col not in ts.columns:
            continue
        if col in {"steps", "sleep_hours"}:
            agg_map[col] = lambda s: s.sum(min_count=1)
        else:
            agg_map[col] = "mean"
    daily = ts.groupby("date", as_index=False).agg(agg_map)
    return daily


def _prepare_long_format_timeseries(ts: pd.DataFrame) -> pd.DataFrame | None:
    cols = set(ts.columns)
    metric_col = next((c for c in ["metric", "type", "record_type", "data_type", "datatype", "name", "identifier"] if c in cols), None)
    value_col = next((c for c in ["value", "numeric_value", "quantity", "measurement", "reading"] if c in cols), None)
    if metric_col is None:
        return None

    date_col = next(
        (c for c in ["date", "start_date", "start", "timestamp", "datetime", "day", "end_date", "end"] if c in cols),
        None,
    )
    if date_col is None:
        return None

    work = ts.copy()
    work["date"] = pd.to_datetime(work[date_col], errors="coerce")
    work["metric_canonical"] = work[metric_col].map(_normalize_metric_value)

    unit_col = next((c for c in ["unit", "units", "measurement_unit"] if c in cols), None)

    # If numeric values are absent (common for sleep intervals), try duration from start/end timestamps.
    if value_col is None and {"start", "end"} <= cols:
        work["start_tmp"] = pd.to_datetime(work["start"], errors="coerce")
        work["end_tmp"] = pd.to_datetime(work["end"], errors="coerce")
        work["value_num"] = (work["end_tmp"] - work["start_tmp"]).dt.total_seconds()
    elif value_col is not None:
        work["value_num"] = pd.to_numeric(work[value_col], errors="coerce")
    else:
        return None

    work = work.dropna(subset=["date", "metric_canonical"])
    if work.empty:
        return None

    if unit_col is not None:
        units = work[unit_col].astype(str).str.lower().fillna("")
        # Strong unit-based conversions before heuristics.
        mmol_mask = (work["metric_canonical"] == "glucose_fasting") & units.str.contains("mmol")
        work.loc[mmol_mask, "value_num"] = work.loc[mmol_mask, "value_num"] * 18.0

        spo2_frac_mask = (work["metric_canonical"] == "spo2") & (
            units.str.contains("fraction") | units.str.contains("ratio")
        )
        work.loc[spo2_frac_mask, "value_num"] = work.loc[spo2_frac_mask, "value_num"] * 100.0

        temp_c_mask = (work["metric_canonical"] == "temperature_f") & units.str.contains("c")
        work.loc[temp_c_mask, "value_num"] = (work.loc[temp_c_mask, "value_num"] * 9.0 / 5.0) + 32.0

        weight_lb_mask = (work["metric_canonical"] == "weight_kg") & (units.str.contains("lb") | units.str.contains("pound"))
        work.loc[weight_lb_mask, "value_num"] = work.loc[weight_lb_mask, "value_num"] / 2.20462

        sleep_ms_mask = (work["metric_canonical"] == "sleep_hours") & (units.str.contains("ms") | units.str.contains("millisecond"))
        work.loc[sleep_ms_mask, "value_num"] = work.loc[sleep_ms_mask, "value_num"] / 3_600_000.0
        sleep_sec_mask = (work["metric_canonical"] == "sleep_hours") & units.str.contains("sec")
        work.loc[sleep_sec_mask, "value_num"] = work.loc[sleep_sec_mask, "value_num"] / 3600.0
        sleep_min_mask = (work["metric_canonical"] == "sleep_hours") & units.str.contains("min")
        work.loc[sleep_min_mask, "value_num"] = work.loc[sleep_min_mask, "value_num"] / 60.0

    work["date"] = work["date"].dt.normalize()
    work = work.dropna(subset=["value_num"])
    if work.empty:
        return None

    frames: list[pd.Series] = []
    for metric in NUMERIC_COLUMNS:
        metric_rows = work[work["metric_canonical"] == metric]
        if metric_rows.empty:
            continue
        if metric in {"steps", "sleep_hours"}:
            agg = metric_rows.groupby("date")["value_num"].sum(min_count=1)
        else:
            agg = metric_rows.groupby("date")["value_num"].mean()
        agg.name = metric
        frames.append(agg)

    if not frames:
        return None

    out = pd.concat(frames, axis=1).reset_index()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    return out


def prepare_timeseries(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", *NUMERIC_COLUMNS])

    ts = df.copy()
    ts.columns = _normalize_columns(list(ts.columns))

    # Try long-format exports first (Apple/WHOOP and other platforms sometimes export `type/value/date` rows).
    long_ts = _prepare_long_format_timeseries(ts)
    if long_ts is not None and not long_ts.empty:
        ts = long_ts

    if "date" not in ts.columns:
        for candidate in ["start_date", "start", "end_date", "end"]:
            if candidate in ts.columns:
                ts = ts.rename(columns={candidate: "date"})
                break

    if "date" not in ts.columns:
        inferred_date_col = _find_best_date_column(ts)
        if inferred_date_col is not None:
            ts = ts.rename(columns={inferred_date_col: "date"})

    if "date" not in ts.columns:
        raise ValueError("CSV needs a date/timestamp column (or a long format with date/type/value columns).")

    ts["date"] = pd.to_datetime(ts["date"], errors="coerce")
    ts["date"] = ts["date"].dt.normalize()
    ts = ts.dropna(subset=["date"]).sort_values("date")

    for col in NUMERIC_COLUMNS:
        if col in ts.columns:
            ts[col] = pd.to_numeric(ts[col], errors="coerce")
        else:
            ts[col] = np.nan

    ts = ts[["date", *NUMERIC_COLUMNS]]
    ts = _coerce_units_in_timeseries(ts)
    ts = _aggregate_daily_timeseries(ts)
    ts = ts.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    return ts


def summarize_timeseries(ts: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "n_days": 0,
        "date_min": None,
        "date_max": None,
        "freshness_days": None,
        "features": {},
    }
    if ts is None or ts.empty:
        return summary

    summary["n_days"] = int(ts["date"].nunique())
    summary["date_min"] = ts["date"].min()
    summary["date_max"] = ts["date"].max()
    summary["freshness_days"] = int((pd.Timestamp.now().normalize() - ts["date"].max().normalize()).days)

    ts = ts.sort_values("date")
    recent7 = ts.tail(7)
    recent14 = ts.tail(14)
    prev7 = recent14.head(max(0, len(recent14) - len(recent7)))

    for col in NUMERIC_COLUMNS:
        series = ts[col].dropna()
        if series.empty:
            continue
        latest_value = float(series.iloc[-1])
        recent7_nonnull = recent7[col].dropna()
        recent14_nonnull = recent14[col].dropna()
        prev7_nonnull = prev7[col].dropna() if not prev7.empty else pd.Series(dtype=float)
        avg7 = float(recent7_nonnull.mean()) if not recent7_nonnull.empty else np.nan
        avg14 = float(recent14_nonnull.mean()) if not recent14_nonnull.empty else np.nan
        prev7_mean = float(prev7_nonnull.mean()) if not prev7_nonnull.empty else np.nan
        trend_delta = avg7 - prev7_mean if not np.isnan(avg7) and not np.isnan(prev7_mean) else np.nan
        summary["features"][col] = {
            "latest": latest_value,
            "avg7": None if np.isnan(avg7) else float(avg7),
            "avg14": None if np.isnan(avg14) else float(avg14),
            "trend_delta": None if np.isnan(trend_delta) else float(trend_delta),
            "count": int(series.shape[0]),
        }
    return summary


def merge_manual_inputs(summary: dict[str, Any], manual: dict[str, Any]) -> dict[str, Any]:
    merged = dict(summary)
    merged_features = {**summary.get("features", {})}
    for key, value in manual.items():
        if value is None:
            continue
        if key not in NUMERIC_COLUMNS:
            continue
        try:
            val = float(value)
        except (TypeError, ValueError):
            continue
        entry = dict(merged_features.get(key, {}))
        entry["latest"] = val
        if "avg7" not in entry or entry["avg7"] is None:
            entry["avg7"] = val
        merged_features[key] = entry
    merged["features"] = merged_features
    return merged


def _metric(features: dict[str, Any], key: str, field: str = "latest") -> float | None:
    value = features.get(key, {}).get(field)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _add_contrib(score: float, delta: float, message: str, store: list[str]) -> float:
    if delta == 0:
        return score
    store.append(message)
    return score + delta


def _trend_label(score: float) -> str:
    if score >= 75:
        return "Rising"
    if score <= 30:
        return "Stable/Low"
    return "Watch"


def _severity_label(prob: float) -> str:
    if prob >= 70:
        return "High"
    if prob >= 40:
        return "Moderate"
    return "Low"


def _probability_from_score(score: float) -> float:
    # Smooth logistic conversion to a pseudo-probability for UI ranking.
    x = (score - 50.0) / 12.0
    return _clamp(100.0 / (1.0 + np.exp(-x)))


def _data_confidence(summary: dict[str, Any], manual_inputs: dict[str, Any]) -> float:
    feature_count = len(summary.get("features", {}))
    n_days = summary.get("n_days", 0) or 0
    freshness_days = summary.get("freshness_days", None)
    manual_count = sum(1 for v in manual_inputs.values() if v not in (None, ""))

    conf = 0.25
    conf += min(feature_count, 8) * 0.05
    conf += min(n_days, 14) / 14.0 * 0.25
    conf += min(manual_count, 6) * 0.02
    if freshness_days is None:
        conf -= 0.05
    elif freshness_days > 14:
        conf -= 0.15
    elif freshness_days > 3:
        conf -= 0.05
    return float(max(0.2, min(0.95, conf)))


def _positive_behaviors(
    features: dict[str, Any],
    profile: dict[str, Any],
    symptoms: list[str],
) -> list[str]:
    notes: list[str] = []
    sleep_avg = _metric(features, "sleep_hours", "avg7")
    steps_avg = _metric(features, "steps", "avg7")
    hrv_delta = _metric(features, "hrv", "trend_delta")
    rhr_delta = _metric(features, "resting_hr", "trend_delta")
    bp_sys = _metric(features, "systolic_bp")
    bp_dia = _metric(features, "diastolic_bp")
    spo2 = _metric(features, "spo2")
    glucose = _metric(features, "glucose_fasting")

    if sleep_avg is not None and sleep_avg >= 7:
        notes.append(f"Sleep average is {sleep_avg:.1f}h, which supports recovery, glucose control, and resilience.")
    if steps_avg is not None and steps_avg >= 7000:
        notes.append(f"Activity is strong at ~{steps_avg:,.0f} steps/day, which is protective for cardiometabolic health.")
    if hrv_delta is not None and hrv_delta > 3:
        notes.append("HRV trend is improving, which can signal better recovery and lower physiological strain.")
    if rhr_delta is not None and rhr_delta < -2:
        notes.append("Resting heart rate trend is falling, which often reflects improved conditioning or recovery.")
    if bp_sys is not None and bp_dia is not None and bp_sys < 130 and bp_dia < 80:
        notes.append("Current blood pressure is in a favorable range, lowering short-term cardiovascular risk.")
    if spo2 is not None and spo2 >= 96:
        notes.append("Oxygen saturation looks stable, which is reassuring for respiratory status.")
    if glucose is not None and glucose < 100:
        notes.append("Fasting glucose is in a healthy range, supporting lower metabolic risk.")
    if not symptoms:
        notes.append("No current symptoms were reported, which improves signal quality and reduces immediate concern.")
    if str(profile.get("smoking_status", "")).lower() == "never":
        notes.append("Non-smoking status is a major protective factor across cardiovascular and respiratory outcomes.")

    return notes[:5]


def _data_gaps(summary: dict[str, Any], conditions: list[str]) -> list[str]:
    features = summary.get("features", {})
    gaps: list[str] = []
    for key, label in [
        ("sleep_hours", "Sleep duration"),
        ("resting_hr", "Resting heart rate"),
        ("hrv", "HRV"),
        ("systolic_bp", "Blood pressure"),
        ("weight_kg", "Weight"),
        ("glucose_fasting", "Fasting glucose"),
    ]:
        if key not in features:
            gaps.append(f"Add {label} tracking to improve predictive confidence.")

    if "Heart failure" in conditions and "weight_kg" not in features:
        gaps.append("Daily weight is especially high-value when monitoring heart failure decompensation risk.")
    if "Diabetes" in conditions and "glucose_fasting" not in features:
        gaps.append("Frequent glucose readings would improve metabolic risk trend detection.")
    return gaps[:6]


def generate_risks(
    summary: dict[str, Any],
    manual_inputs: dict[str, Any],
    profile: dict[str, Any],
    symptoms: list[str],
    conditions: list[str],
) -> tuple[list[RiskItem], list[str], list[str], list[str]]:
    features = summary.get("features", {})
    confidence = _data_confidence(summary, manual_inputs)

    age = int(profile.get("age", 40) or 40)
    bmi = float(profile.get("bmi", 0) or 0)
    smoking = str(profile.get("smoking_status", "Never"))

    rhr = _metric(features, "resting_hr")
    rhr_delta = _metric(features, "resting_hr", "trend_delta")
    hrv = _metric(features, "hrv")
    hrv_delta = _metric(features, "hrv", "trend_delta")
    sleep = _metric(features, "sleep_hours", "avg7") or _metric(features, "sleep_hours")
    sleep_delta = _metric(features, "sleep_hours", "trend_delta")
    steps = _metric(features, "steps", "avg7") or _metric(features, "steps")
    spo2 = _metric(features, "spo2")
    weight = _metric(features, "weight_kg")
    weight_delta = _metric(features, "weight_kg", "trend_delta")
    sys_bp = _metric(features, "systolic_bp")
    dia_bp = _metric(features, "diastolic_bp")
    glucose = _metric(features, "glucose_fasting")
    temp = _metric(features, "temperature_f")

    symptom_set = {s.lower() for s in symptoms}
    condition_set = {c.lower() for c in conditions}

    risks: list[RiskItem] = []

    # 1) Cardiovascular strain / hypertension risk
    score = 30 + (age - 40) * 0.2
    why: list[str] = []
    protective: list[str] = []
    next_steps = ["Track blood pressure at least 3-5 days/week for better trend accuracy."]
    if sys_bp is not None:
        if sys_bp >= 160:
            score = _add_contrib(score, 32, f"Systolic BP {sys_bp:.0f} is very elevated.", why)
            next_steps.append("Consider urgent medical evaluation for sustained severe blood pressure.")
        elif sys_bp >= 140:
            score = _add_contrib(score, 18, f"Systolic BP {sys_bp:.0f} is elevated.", why)
        elif sys_bp < 130:
            protective.append(f"Systolic BP {sys_bp:.0f} is within a favorable range.")
            score -= 6
    if dia_bp is not None:
        if dia_bp >= 100:
            score = _add_contrib(score, 18, f"Diastolic BP {dia_bp:.0f} is significantly elevated.", why)
        elif dia_bp >= 90:
            score = _add_contrib(score, 10, f"Diastolic BP {dia_bp:.0f} is elevated.", why)
        elif dia_bp < 80:
            protective.append(f"Diastolic BP {dia_bp:.0f} is in range.")
            score -= 4
    if rhr is not None and rhr >= 90:
        score = _add_contrib(score, 10, f"Resting HR {rhr:.0f} may indicate increased cardiovascular strain.", why)
    if rhr_delta is not None and rhr_delta > 4:
        score = _add_contrib(score, 8, "Resting HR trend is rising vs prior week.", why)
    if steps is not None and steps < 4000:
        score = _add_contrib(score, 7, f"Low activity (~{steps:,.0f} steps/day) may worsen cardiovascular risk.", why)
    elif steps is not None and steps >= 8000:
        protective.append(f"Activity level (~{steps:,.0f} steps/day) is protective.")
        score -= 6
    if bmi >= 30:
        score = _add_contrib(score, 8, f"BMI {bmi:.1f} increases cardiometabolic load.", why)
    if "hypertension" in condition_set:
        score = _add_contrib(score, 12, "History of hypertension increases baseline risk.", why)
    if smoking in {"Current", "current"}:
        score = _add_contrib(score, 12, "Current smoking elevates cardiovascular risk.", why)
    prob = _probability_from_score(_clamp(score))
    risks.append(
        RiskItem(
            name="Cardiovascular strain / hypertension worsening",
            probability=prob,
            severity=_severity_label(prob),
            trend=_trend_label(score),
            confidence=confidence,
            why=why[:5] or ["Limited cardiovascular signals available; continue collecting BP, HR, and activity data."],
            protective_factors=protective[:3],
            next_steps=list(dict.fromkeys(next_steps))[:4],
            raw_score=_clamp(score),
        )
    )

    # 2) Metabolic dysregulation / diabetes progression
    score = 28 + max(0, age - 45) * 0.15
    why = []
    protective = []
    next_steps = ["Add fasting glucose (or CGM summaries) regularly to strengthen trend prediction."]
    if glucose is not None:
        if glucose >= 180:
            score = _add_contrib(score, 35, f"Fasting glucose {glucose:.0f} is severely elevated.", why)
        elif glucose >= 126:
            score = _add_contrib(score, 24, f"Fasting glucose {glucose:.0f} is in a diabetes range.", why)
        elif glucose >= 100:
            score = _add_contrib(score, 10, f"Fasting glucose {glucose:.0f} is in a prediabetes range.", why)
        else:
            protective.append(f"Fasting glucose {glucose:.0f} is in a healthy range.")
            score -= 8
    if bmi >= 30:
        score = _add_contrib(score, 14, "Higher BMI raises insulin resistance risk.", why)
    elif 18.5 <= bmi < 25:
        protective.append("BMI is in a generally favorable range.")
        score -= 4
    if steps is not None and steps < 5000:
        score = _add_contrib(score, 8, "Low daily activity can worsen glucose control.", why)
    elif steps is not None and steps >= 7500:
        protective.append("Consistent movement likely supports insulin sensitivity.")
        score -= 5
    if sleep is not None and sleep < 6:
        score = _add_contrib(score, 8, f"Low sleep ({sleep:.1f}h) can impair glucose regulation.", why)
    if "diabetes" in condition_set or "prediabetes" in condition_set:
        score = _add_contrib(score, 15, "Existing metabolic history increases progression risk.", why)
    prob = _probability_from_score(_clamp(score))
    risks.append(
        RiskItem(
            name="Metabolic dysregulation / diabetes progression",
            probability=prob,
            severity=_severity_label(prob),
            trend=_trend_label(score),
            confidence=confidence,
            why=why[:5] or ["No glucose data yet; upload labs or device trends for better metabolic predictions."],
            protective_factors=protective[:3],
            next_steps=next_steps[:3],
            raw_score=_clamp(score),
        )
    )

    # 3) Recovery failure / sleep-stress overload
    score = 25
    why = []
    protective = []
    next_steps = ["Capture daily sleep, resting HR, and HRV to detect stress and recovery changes earlier."]
    if sleep is not None:
        if sleep < 5.5:
            score = _add_contrib(score, 24, f"Sleep average {sleep:.1f}h is very low.", why)
        elif sleep < 6.5:
            score = _add_contrib(score, 12, f"Sleep average {sleep:.1f}h is below target.", why)
        elif sleep >= 7.5:
            protective.append(f"Sleep average {sleep:.1f}h supports recovery.")
            score -= 7
    if sleep_delta is not None and sleep_delta < -0.75:
        score = _add_contrib(score, 10, "Sleep duration has dropped vs prior week.", why)
    if hrv is not None and hrv < 30:
        score = _add_contrib(score, 12, f"HRV {hrv:.0f} may reflect elevated physiological stress.", why)
    if hrv_delta is not None and hrv_delta < -5:
        score = _add_contrib(score, 10, "HRV trend is worsening vs prior week.", why)
    if rhr_delta is not None and rhr_delta > 5:
        score = _add_contrib(score, 8, "Resting HR increase suggests reduced recovery.", why)
    if steps is not None and steps > 16000 and sleep is not None and sleep < 6.5:
        score = _add_contrib(score, 7, "High workload plus low sleep can increase burnout/overtraining risk.", why)
    if "fatigue" in symptom_set or "poor sleep" in symptom_set:
        score = _add_contrib(score, 10, "Reported fatigue/poor sleep aligns with recovery risk.", why)
    prob = _probability_from_score(_clamp(score))
    risks.append(
        RiskItem(
            name="Recovery failure / sleep-stress overload",
            probability=prob,
            severity=_severity_label(prob),
            trend=_trend_label(score),
            confidence=confidence,
            why=why[:5] or ["No strong sleep/recovery signals yet; wearable trend uploads will improve this prediction."],
            protective_factors=protective[:3],
            next_steps=next_steps[:3],
            raw_score=_clamp(score),
        )
    )

    # 4) Respiratory / infectious stress
    score = 18
    why = []
    protective = []
    next_steps = ["Record temperature, resting HR, and SpO2 when feeling unwell to improve early detection."]
    if temp is not None:
        if temp >= 101.0:
            score = _add_contrib(score, 30, f"Temperature {temp:.1f}F suggests fever.", why)
        elif temp >= 99.5:
            score = _add_contrib(score, 14, f"Temperature {temp:.1f}F is mildly elevated.", why)
    if spo2 is not None:
        if spo2 < 92:
            score = _add_contrib(score, 35, f"SpO2 {spo2:.0f}% is low.", why)
            next_steps.append("Seek urgent medical evaluation for low oxygen saturation.")
        elif spo2 < 95:
            score = _add_contrib(score, 15, f"SpO2 {spo2:.0f}% is below baseline for many adults.", why)
        elif spo2 >= 97:
            protective.append(f"SpO2 {spo2:.0f}% is reassuring.")
            score -= 5
    if rhr_delta is not None and rhr_delta > 6:
        score = _add_contrib(score, 8, "Resting HR rose sharply, which can precede illness.", why)
    if hrv_delta is not None and hrv_delta < -8:
        score = _add_contrib(score, 7, "HRV drop can be an early illness/stress signal.", why)
    if {"cough", "shortness of breath", "fever"} & symptom_set:
        score = _add_contrib(score, 20, "Reported respiratory/infection symptoms increase short-term risk.", why)
    prob = _probability_from_score(_clamp(score))
    risks.append(
        RiskItem(
            name="Respiratory / infectious stress event",
            probability=prob,
            severity=_severity_label(prob),
            trend=_trend_label(score),
            confidence=confidence,
            why=why[:5] or ["No infection-related signals detected from current data."],
            protective_factors=protective[:3],
            next_steps=list(dict.fromkeys(next_steps))[:4],
            raw_score=_clamp(score),
        )
    )

    # 5) Fluid retention / heart failure decompensation (generalized if no HF history)
    score = 15
    why = []
    protective = []
    next_steps = ["Daily weight and symptom logging is high leverage for catching decompensation early."]
    if "heart failure" in condition_set:
        score = _add_contrib(score, 20, "Heart failure history raises baseline decompensation risk.", why)
    if weight_delta is not None and weight_delta > 1.5:
        score = _add_contrib(score, 18, f"Weight trend increased by {weight_delta:.1f} kg vs prior week.", why)
    if {"leg swelling", "shortness of breath"} & symptom_set:
        score = _add_contrib(score, 18, "Reported swelling/breathlessness can reflect fluid overload.", why)
    if rhr is not None and rhr > 95:
        score = _add_contrib(score, 6, "Elevated resting HR may accompany cardiac strain.", why)
    if steps is not None and steps < 3000:
        score = _add_contrib(score, 6, "Sharp reduction in activity can accompany worsening symptoms.", why)
    if spo2 is not None and spo2 < 94:
        score = _add_contrib(score, 10, "Lower oxygen saturation increases concern for cardiopulmonary issues.", why)
    if weight_delta is not None and weight_delta < 0.5:
        protective.append("Weight appears relatively stable across the recent trend window.")
        score -= 4
    prob = _probability_from_score(_clamp(score))
    label = (
        "Heart failure decompensation / fluid overload"
        if "heart failure" in condition_set
        else "Fluid retention / cardiopulmonary worsening"
    )
    risks.append(
        RiskItem(
            name=label,
            probability=prob,
            severity=_severity_label(prob),
            trend=_trend_label(score),
            confidence=confidence,
            why=why[:5] or ["Insufficient weight/symptom data for stronger fluid overload prediction."],
            protective_factors=protective[:3],
            next_steps=next_steps[:3],
            raw_score=_clamp(score),
        )
    )

    risks.sort(key=lambda r: r.probability, reverse=True)
    positives = _positive_behaviors(features, profile, symptoms)
    gaps = _data_gaps(summary, conditions)
    alerts = urgent_alerts(summary, symptoms)
    return risks, positives, gaps, alerts


def urgent_alerts(summary: dict[str, Any], symptoms: list[str]) -> list[str]:
    features = summary.get("features", {})
    alerts: list[str] = []
    sys_bp = _metric(features, "systolic_bp")
    dia_bp = _metric(features, "diastolic_bp")
    spo2 = _metric(features, "spo2")
    temp = _metric(features, "temperature_f")
    glucose = _metric(features, "glucose_fasting")
    symptom_set = {s.lower() for s in symptoms}

    if sys_bp is not None and sys_bp >= 180:
        alerts.append("Systolic BP is extremely high (>=180). Seek urgent medical care if this is real/repeated.")
    if dia_bp is not None and dia_bp >= 120:
        alerts.append("Diastolic BP is extremely high (>=120). This can be an emergency.")
    if spo2 is not None and spo2 < 90:
        alerts.append("SpO2 is <90%, which can indicate a serious breathing problem.")
    if temp is not None and temp >= 103:
        alerts.append("High fever (>=103F) detected. Consider urgent evaluation, especially with other symptoms.")
    if glucose is not None and glucose >= 250:
        alerts.append("Very high glucose detected (>=250). This may require urgent clinical guidance.")
    if "chest pain" in symptom_set:
        alerts.append("Chest pain reported. Seek urgent medical evaluation immediately.")
    if "confusion" in symptom_set:
        alerts.append("Confusion reported. This can indicate a medical emergency.")
    return alerts


def build_summary_text(
    profile: dict[str, Any],
    symptoms: list[str],
    conditions: list[str],
    risks: list[RiskItem],
    positives: list[str],
    gaps: list[str],
    alerts: list[str],
) -> str:
    lines: list[str] = []
    lines.append("CareNav Predictive Health Summary (Educational Prototype)")
    lines.append("=" * 60)
    age = profile.get("age")
    sex = profile.get("sex")
    bmi = profile.get("bmi")
    smoking = profile.get("smoking_status")
    age_text = str(age) if age not in (None, "") else "unknown"
    sex_text = str(sex) if sex not in (None, "") else "unspecified"
    smoking_text = str(smoking) if smoking not in (None, "") else "unknown"
    bmi_text = f"{float(bmi):.1f}" if bmi not in (None, "") else "unknown"
    lines.append(
        f"Patient profile: age {age_text}, sex {sex_text}, BMI {bmi_text}, smoking {smoking_text}."
    )
    if conditions:
        lines.append("Known conditions: " + ", ".join(conditions))
    if symptoms:
        lines.append("Current symptoms: " + ", ".join(symptoms))
    if alerts:
        lines.append("")
        lines.append("Urgent flags:")
        for a in alerts:
            lines.append(f"- {a}")

    lines.append("")
    lines.append("Predicted watchlist (next risk domains to monitor):")
    for r in risks:
        lines.append(
            f"- {r.name}: {r.probability:.0f}% ({r.severity}, confidence {r.confidence:.0%}). Top reasons: "
            + "; ".join(r.why[:3])
        )
        if r.protective_factors:
            lines.append("  Protective factors: " + "; ".join(r.protective_factors[:2]))

    if positives:
        lines.append("")
        lines.append("Positive behaviors to reinforce:")
        for p in positives:
            lines.append(f"- {p}")

    if gaps:
        lines.append("")
        lines.append("High-value data gaps:")
        for g in gaps:
            lines.append(f"- {g}")

    lines.append("")
    lines.append("Note: This tool is not a medical diagnosis system and should be used to support, not replace, clinician judgement.")
    return "\n".join(lines)


def simulate_what_if(
    summary: dict[str, Any],
    manual_inputs: dict[str, Any],
    profile: dict[str, Any],
    symptoms: list[str],
    conditions: list[str],
    adjustments: dict[str, float],
) -> list[RiskItem]:
    # Clone the summary and adjust selected "latest" / "avg7" values to estimate directionality.
    sim_summary = {
        **summary,
        "features": {k: dict(v) for k, v in summary.get("features", {}).items()},
    }
    for key, delta in adjustments.items():
        if key not in NUMERIC_COLUMNS:
            continue
        entry = dict(sim_summary["features"].get(key, {}))
        if "latest" in entry and entry["latest"] is not None:
            entry["latest"] = float(entry["latest"]) + float(delta)
        if "avg7" in entry and entry["avg7"] is not None:
            entry["avg7"] = float(entry["avg7"]) + float(delta)
        sim_summary["features"][key] = entry
    sim_risks, _, _, _ = generate_risks(sim_summary, manual_inputs, profile, symptoms, conditions)
    return sim_risks
