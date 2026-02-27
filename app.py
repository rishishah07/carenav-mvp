from __future__ import annotations

import json
import os
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from risk_engine import (
    NUMERIC_COLUMNS,
    build_summary_text,
    generate_risks,
    merge_manual_inputs,
    prepare_timeseries,
    summarize_timeseries,
)

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency at runtime
    OpenAI = None


st.set_page_config(page_title="CareNav", page_icon="🩺", layout="wide")


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root { color-scheme: dark; }
        html, body, .stApp {
            background: #060b16;
            color: #e5e7eb;
        }
        [data-testid="stHeader"] {
            background: transparent;
        }
        [data-testid="stSidebar"] {
            background: #0b1324;
            border-right: 1px solid rgba(148, 163, 184, 0.16);
        }
        [data-testid="stSidebar"] * {
            color: #dbe3f0;
        }
        .main {
            background:
                radial-gradient(circle at 8% 4%, rgba(59, 130, 246, 0.22), transparent 44%),
                radial-gradient(circle at 92% 8%, rgba(14, 165, 233, 0.20), transparent 42%),
                radial-gradient(circle at 75% 85%, rgba(16, 185, 129, 0.14), transparent 35%),
                linear-gradient(#060b16, #0a1120);
        }
        .carenav-hero {
            background:
                linear-gradient(180deg, rgba(15,23,42,0.86), rgba(15,23,42,0.74));
            backdrop-filter: blur(8px);
            -webkit-backdrop-filter: blur(8px);
            color: #e2e8f0;
            border-radius: 22px;
            padding: 20px 22px;
            margin-bottom: 8px;
            border: 1px solid rgba(148, 163, 184, 0.20);
            box-shadow: 0 10px 36px rgba(2, 6, 23, 0.45);
            position: relative;
            overflow: hidden;
        }
        .carenav-hero::before {
            content: "";
            position: absolute;
            inset: 0 0 auto 0;
            height: 4px;
            background: linear-gradient(90deg, #2563eb, #0ea5e9, #10b981);
        }
        .carenav-hero h1 {
            margin: 0;
            font-size: 2.0rem;
            font-weight: 680;
            letter-spacing: -0.02em;
        }
        .carenav-hero p {
            margin: 7px 0 0 0;
            color: #cbd5e1;
            font-size: 0.96rem;
            max-width: 54rem;
        }
        .mini-card {
            background: rgba(15,23,42,0.78);
            border: 1px solid rgba(148, 163, 184, 0.16);
            border-radius: 16px;
            padding: 13px 14px;
            box-shadow: 0 6px 20px rgba(2, 6, 23, 0.38);
            position: relative;
            overflow: hidden;
        }
        .mini-card::before {
            content: "";
            position: absolute;
            left: 0; top: 0; right: 0;
            height: 3px;
            background: linear-gradient(90deg, #cbd5e1, #e2e8f0);
        }
        .mini-card.blue::before {
            background: linear-gradient(90deg, #2563eb, #38bdf8);
        }
        .mini-card.red::before {
            background: linear-gradient(90deg, #ef4444, #fb7185);
        }
        .mini-card.green::before {
            background: linear-gradient(90deg, #10b981, #34d399);
        }
        .mini-card.purple::before {
            background: linear-gradient(90deg, #4f46e5, #60a5fa);
        }
        .mini-label {
            color: #94a3b8;
            font-size: 0.8rem;
            margin-bottom: 2px;
        }
        .mini-value {
            color: #e2e8f0;
            font-size: 1.25rem;
            font-weight: 700;
            line-height: 1.2;
        }
        .mini-sub {
            color: #9aa6bb;
            font-size: 0.78rem;
            margin-top: 3px;
        }
        .pill {
            display: inline-block;
            border-radius: 999px;
            padding: 4px 10px;
            font-size: 0.78rem;
            font-weight: 600;
            margin-right: 6px;
            margin-bottom: 6px;
            border: 1px solid transparent;
        }
        .pill-high { background:#fee2e2; color:#991b1b; border-color:#fecaca; }
        .pill-mod { background:#fef3c7; color:#92400e; border-color:#fde68a; }
        .pill-low { background:#dcfce7; color:#166534; border-color:#bbf7d0; }
        .section-title {
            margin-top: 0.25rem;
            margin-bottom: 0.35rem;
            font-weight: 650;
            color: #e5e7eb;
        }
        .subtle-note {
            color: #94a3b8;
            font-size: 0.82rem;
        }
        .csv-preview-card {
            background: rgba(15,23,42,0.70);
            border: 1px solid rgba(59, 130, 246, 0.22);
            border-radius: 16px;
            padding: 10px 12px;
            margin-top: 6px;
            margin-bottom: 10px;
            box-shadow: 0 6px 18px rgba(2, 6, 23, 0.34);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _sample_csv() -> pd.DataFrame:
    rows = [
        ["2026-02-12", 58, 48, 7.8, 8200, 98, 72.2, 122, 78, 94, 98.2],
        ["2026-02-13", 59, 46, 7.1, 7600, 98, 72.3, 124, 79, 96, 98.4],
        ["2026-02-14", 60, 44, 6.6, 6300, 97, 72.3, 126, 80, 98, 98.7],
        ["2026-02-15", 62, 42, 6.0, 5400, 97, 72.6, 130, 83, 102, 99.1],
        ["2026-02-16", 64, 39, 5.8, 4100, 96, 73.0, 134, 86, 108, 99.4],
        ["2026-02-17", 66, 35, 5.4, 3000, 95, 73.5, 138, 88, 116, 99.8],
        ["2026-02-18", 69, 31, 5.2, 2200, 94, 74.2, 142, 92, 124, 100.6],
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "date",
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
        ],
    )


def _clean_text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _parse_optional_float(value: str) -> float | None:
    text = _clean_text(value)
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _parse_optional_int(value: str) -> int | None:
    text = _clean_text(value)
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _build_ai_payload(
    profile: dict[str, Any],
    manual_inputs: dict[str, Any],
    summary: dict[str, Any],
    risks: list[Any],
    positives: list[str],
    gaps: list[str],
    alerts: list[str],
    symptoms: list[str],
    conditions: list[str],
    main_concern: str,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for key in [
        "resting_hr",
        "hrv",
        "sleep_hours",
        "steps",
        "spo2",
        "systolic_bp",
        "diastolic_bp",
        "glucose_fasting",
        "weight_kg",
        "temperature_f",
    ]:
        item = summary.get("features", {}).get(key, {})
        if item:
            metrics[key] = {
                "latest": item.get("latest"),
                "avg7": item.get("avg7"),
                "trend_delta": item.get("trend_delta"),
                "count": item.get("count"),
            }
        elif manual_inputs.get(key) is not None:
            metrics[key] = {"latest": manual_inputs.get(key)}

    return {
        "app": "CareNav",
        "main_concern": main_concern or None,
        "profile": {
            "age": profile.get("age"),
            "sex": profile.get("sex"),
            "bmi": round(profile["bmi"], 1) if profile.get("bmi") is not None else None,
            "smoking_status": profile.get("smoking_status"),
        },
        "known_conditions": conditions,
        "current_symptoms": symptoms,
        "trend_summary": {
            "n_days": summary.get("n_days"),
            "freshness_days": summary.get("freshness_days"),
            "metrics": metrics,
        },
        "predicted_risks": [
            {
                "risk_domain": r.name,
                "probability_percent": round(r.probability, 1),
                "severity": r.severity,
                "confidence": round(r.confidence, 3),
                "trend": r.trend,
                "drivers": r.why[:4],
                "protective_factors": r.protective_factors[:3],
                "next_steps": r.next_steps[:3],
            }
            for r in risks
        ],
        "positive_behaviors": positives[:4],
        "data_gaps": gaps[:5],
        "urgent_alerts": alerts[:4],
    }


def _extract_response_text(response: object) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()
    try:
        chunks: list[str] = []
        for item in getattr(response, "output", []) or []:
            for content in getattr(item, "content", []) or []:
                text = getattr(content, "text", None)
                if isinstance(text, str) and text.strip():
                    chunks.append(text.strip())
        if chunks:
            return "\n\n".join(chunks)
    except Exception:
        pass
    return ""


def _generate_ai_narrative(
    api_key: str,
    model: str,
    payload: dict[str, Any],
) -> str:
    if not api_key:
        raise ValueError("Missing OpenAI API key.")
    if OpenAI is None:
        raise RuntimeError("The `openai` package is not installed. Install dependencies from requirements.txt.")

    client = OpenAI(api_key=api_key)
    system_prompt = (
        "You are CareNav's healthcare education assistant. "
        "Write a short plain-language explanation of only the top predicted risk from the provided CareNav results. "
        "Keep it very brief and easy for non-clinical users. "
        "Format exactly: 1 short paragraph, then 2 bullet points. "
        "Bullet 1: why this risk may be elevated for this user. "
        "Bullet 2: one practical next measurement/action. "
        "If urgent alerts exist, add one urgent warning line first. "
        "Do not diagnose. Do not list all risks. Do not invent data."
    )
    user_prompt = (
        "Use the structured risk output as the source of truth. Do not invent measurements.\n\n"
        + json.dumps(payload, indent=2)
    )

    try:
        if hasattr(client, "responses"):
            response = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            text = _extract_response_text(response)
            if text:
                return text

        chat = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = chat.choices[0].message.content if chat.choices else ""
        return (content or "").strip()
    except Exception as exc:  # pragma: no cover - network/runtime path
        raise RuntimeError(str(exc)) from exc


def _fallback_analysis(risks: list[Any], positives: list[str], gaps: list[str], alerts: list[str]) -> str:
    if not risks:
        return "Not enough data yet to generate meaningful risk analysis."
    top = risks[0]
    lines: list[str] = []
    if alerts:
        lines.append(f"**Urgent:** {alerts[0]}")
    lines.append(
        f"Top current risk is **{top.name} ({top.probability:.0f}%)**. "
        "This is a directional estimate based on your latest available trends."
    )
    driver = top.why[0] if top.why else "Limited data is lowering confidence."
    lines.append(f"- Why this may be elevated: {driver}")
    if top.next_steps:
        lines.append(f"- Next best action: {top.next_steps[0]}")
    elif gaps:
        lines.append(f"- Next best action: {gaps[0]}")
    elif positives:
        lines.append(f"- Keep doing: {positives[0]}")
    return "\n".join(lines)


def _extract_actions_from_text(text: str) -> list[str]:
    if not text:
        return []
    actions: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("- "):
            line = line[2:].strip()
        elif len(line) > 2 and line[0].isdigit() and line[1] in {".", ")"}:
            line = line[2:].strip()
        if not line:
            continue
        actions.append(line)
    deduped: list[str] = []
    for item in actions:
        if item not in deduped:
            deduped.append(item)
    return deduped[:5]


def _generate_ai_mitigation_actions(
    api_key: str,
    model: str,
    payload: dict[str, Any],
) -> list[str]:
    if not api_key:
        raise ValueError("Missing OpenAI API key.")
    if OpenAI is None:
        raise RuntimeError("The `openai` package is not installed. Install dependencies from requirements.txt.")

    client = OpenAI(api_key=api_key)
    system_prompt = (
        "You are a healthcare education assistant. "
        "Generate a short list of practical mitigation actions for the top predicted risk only. "
        "Use only the provided data. "
        "Output 3 to 5 bullets, no intro text. "
        "Prioritize behavior and monitoring actions that reduce risk impact. "
        "Avoid generic advice and avoid asking for extra data collection unless essential. "
        "Do not diagnose."
    )
    user_prompt = "Return mitigation actions for the top risk from this structured payload:\n\n" + json.dumps(payload, indent=2)

    try:
        if hasattr(client, "responses"):
            response = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            text = _extract_response_text(response)
            actions = _extract_actions_from_text(text)
            if actions:
                return actions

        chat = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = chat.choices[0].message.content if chat.choices else ""
        actions = _extract_actions_from_text(content or "")
        if actions:
            return actions
    except Exception as exc:  # pragma: no cover - network/runtime path
        raise RuntimeError(str(exc)) from exc
    return []


def _normalize_no_selection(values: list[str], none_label: str) -> list[str]:
    if not values:
        return []
    if none_label in values:
        return []
    return [v for v in values if v != none_label]


def _build_trend_view(
    ts_df: pd.DataFrame,
    selected_cols: list[str],
    view_mode: str,
) -> pd.DataFrame:
    if ts_df is None or ts_df.empty or not selected_cols:
        return pd.DataFrame()
    work = ts_df[["date", *selected_cols]].copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work = work.dropna(subset=["date"]).sort_values("date").set_index("date")
    if work.empty:
        return pd.DataFrame()

    if view_mode == "Day":
        return work.tail(90)
    if view_mode == "Week":
        return work.resample("W-MON").mean().dropna(how="all")
    if view_mode == "Month":
        return work.resample("MS").mean().dropna(how="all")
    return work


def _build_priority_actions(risks: list[Any], alerts: list[str]) -> list[str]:
    actions: list[str] = []
    if alerts:
        actions.append("Urgent symptom or reading detected: seek immediate medical evaluation if the alert is accurate/current.")
    if risks:
        top = risks[0]
        top_name = top.name.lower()
        if "sleep" in top_name or "recovery" in top_name:
            actions.extend(
                [
                    "Set a fixed sleep-wake window for the next 7 days, including weekends.",
                    "Reduce late caffeine/alcohol and heavy evening meals to improve overnight recovery.",
                    "Use a low-intensity recovery day after nights with poor sleep.",
                ]
            )
        elif "metabolic" in top_name or "diabetes" in top_name or "glucose" in top_name:
            actions.extend(
                [
                    "Prioritize lower-glycemic meals and pair carbs with protein/fiber.",
                    "Add a 10-20 minute walk after major meals.",
                    "Aim for consistent sleep and meal timing to improve glucose stability.",
                ]
            )
        elif "respiratory" in top_name or "infectious" in top_name:
            actions.extend(
                [
                    "Scale down intensity and prioritize rest/hydration until vitals normalize.",
                    "Track symptom progression daily and escalate care quickly if breathing worsens.",
                    "Avoid high-exertion sessions while symptoms are active.",
                ]
            )
        else:
            actions.extend(
                [
                    "Keep moderate daily movement and avoid sudden spikes in exertion.",
                    "Reduce sodium-heavy meals and alcohol for the next few days.",
                    "Prioritize sleep consistency and stress management to lower physiologic strain.",
                ]
            )
        for step in top.next_steps:
            if step not in actions:
                actions.append(step)
            if len(actions) >= 5:
                break

    deduped: list[str] = []
    for item in actions:
        if item not in deduped:
            deduped.append(item)
        if len(deduped) >= 5:
            break
    return deduped[:5]


def _severity_class(severity: str) -> str:
    s = (severity or "").lower()
    if s == "high":
        return "pill-high"
    if s == "moderate":
        return "pill-mod"
    return "pill-low"


def _card(title: str, value: str, subtitle: str = "", tone: str = "") -> None:
    tone_class = f" {tone}" if tone else ""
    st.markdown(
        f"""
        <div class="mini-card{tone_class}">
          <div class="mini-label">{title}</div>
          <div class="mini-value">{value}</div>
          <div class="mini-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _get_api_key() -> str:
    def _clean(value: Any) -> str:
        if value is None:
            return ""
        text = str(value).strip().strip('"').strip("'")
        return text

    # 1) Environment variables
    for env_name in ["OPENAI_API_KEY", "OPENAI_KEY"]:
        env_val = _clean(os.getenv(env_name, ""))
        if env_val:
            return env_val

    # 2) Streamlit secrets (supports multiple common layouts)
    try:
        # Flat keys
        for key in ["OPENAI_API_KEY", "openai_api_key", "OPENAI_KEY", "openai_key"]:
            val = _clean(st.secrets.get(key, ""))
            if val:
                return val

        # Nested blocks, e.g. [openai] api_key="..."
        for section in ["openai", "OPENAI"]:
            block = st.secrets.get(section, {})
            if hasattr(block, "get"):
                for key in ["api_key", "API_KEY", "openai_api_key", "OPENAI_API_KEY"]:
                    val = _clean(block.get(key, ""))
                    if val:
                        return val
    except Exception:
        return ""

    return ""


def _format_prefill_value(metric: str, value: float | None) -> str:
    if value is None:
        return ""
    if metric in {"steps", "systolic_bp", "diastolic_bp", "resting_hr", "hrv", "glucose_fasting", "spo2"}:
        return str(int(round(value)))
    return f"{float(value):.1f}"


def _preview_upload(uploaded_file: Any) -> tuple[pd.DataFrame, dict[str, Any] | None, str | None]:
    if uploaded_file is None:
        return pd.DataFrame(), None, None
    try:
        uploaded_file.seek(0)
        raw = pd.read_csv(uploaded_file)
        ts = prepare_timeseries(raw)
        summary = summarize_timeseries(ts)
        return ts, summary, None
    except Exception as exc:
        return pd.DataFrame(), None, str(exc)


def _prefill_form_from_summary(summary: dict[str, Any] | None) -> list[str]:
    if not summary:
        return []
    features = summary.get("features", {})
    widget_map = {
        "weight_kg": "weight_text",
        "systolic_bp": "systolic_bp_text",
        "diastolic_bp": "diastolic_bp_text",
        "resting_hr": "resting_hr_text",
        "hrv": "hrv_text",
        "spo2": "spo2_text",
        "sleep_hours": "sleep_hours_text",
        "steps": "steps_text",
        "glucose_fasting": "glucose_fasting_text",
        "temperature_f": "temp_f_text",
    }
    filled: list[str] = []
    for metric, widget_key in widget_map.items():
        latest = features.get(metric, {}).get("latest")
        if latest is None:
            continue
        current = st.session_state.get(widget_key, "")
        if str(current).strip():
            continue
        st.session_state[widget_key] = _format_prefill_value(metric, float(latest))
        filled.append(metric)
    return filled


def _recommended_check_in(risks: list[Any], confidence: float, alerts: list[str]) -> str:
    if alerts:
        return "Now / urgent care"
    top_prob = risks[0].probability if risks else 0
    if top_prob >= 70:
        return "Re-check in 24h"
    if top_prob >= 45 or confidence < 0.45:
        return "Re-check in 2-3 days"
    return "Weekly check-in"


def _data_coverage(summary: dict[str, Any]) -> tuple[int, int]:
    features = summary.get("features", {})
    present = sum(1 for c in NUMERIC_COLUMNS if c in features and features[c].get("latest") is not None)
    return present, len(NUMERIC_COLUMNS)


def _run_analysis(
    profile: dict[str, Any],
    manual_inputs: dict[str, Any],
    symptoms: list[str],
    conditions: list[str],
    uploaded_file: Any,
    ai_enabled: bool,
    ai_model: str,
    main_concern: str,
) -> dict[str, Any]:
    ts_df = pd.DataFrame()
    upload_error = None
    if uploaded_file is not None:
        try:
            uploaded_file.seek(0)
            raw = pd.read_csv(uploaded_file)
            ts_df = prepare_timeseries(raw)
        except Exception as exc:
            upload_error = str(exc)

    summary = summarize_timeseries(ts_df)
    summary = merge_manual_inputs(summary, manual_inputs)
    risks, positives, gaps, alerts = generate_risks(summary, manual_inputs, profile, symptoms, conditions)
    confidence = risks[0].confidence if risks else 0.0

    ai_text = None
    ai_error = None
    ai_actions: list[str] = []
    ai_actions_error = None
    payload: dict[str, Any] | None = None
    if ai_enabled:
        api_key = _get_api_key()
        if api_key:
            payload = _build_ai_payload(
                profile,
                manual_inputs,
                summary,
                risks,
                positives,
                gaps,
                alerts,
                symptoms,
                conditions,
                main_concern,
            )
            try:
                ai_text = _generate_ai_narrative(api_key=api_key, model=ai_model, payload=payload)
            except Exception as exc:
                ai_error = str(exc)
            try:
                ai_actions = _generate_ai_mitigation_actions(api_key=api_key, model=ai_model, payload=payload)
            except Exception as exc:
                ai_actions_error = str(exc)
        else:
            ai_error = "No OpenAI API key found. Set OPENAI_API_KEY or .streamlit/secrets.toml."

    if not ai_text:
        ai_text = _fallback_analysis(risks, positives, gaps, alerts)

    summary_text = build_summary_text(profile, symptoms, conditions, risks, positives, gaps, alerts)
    return {
        "ts_df": ts_df,
        "upload_error": upload_error,
        "summary": summary,
        "risks": risks,
        "positives": positives,
        "gaps": gaps,
        "alerts": alerts,
        "confidence": confidence,
        "ai_text": ai_text,
        "ai_error": ai_error,
        "ai_actions": ai_actions,
        "ai_actions_error": ai_actions_error,
        "summary_text": summary_text,
        "main_concern": main_concern,
        "ai_model": ai_model,
    }


def main() -> None:
    _inject_styles()

    st.markdown(
        """
        <div class="carenav-hero">
          <h1>CareNav</h1>
          <p>CareNav helps make care more predictive by spotting early warning signals from current readings and wearable trends, then translating them into clear risks, likely drivers, and next steps before issues escalate.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.subheader("AI Analysis")
        ai_enabled = st.toggle("Use ChatGPT analysis", value=True)
        model_preset = st.selectbox(
            "Model",
            ["gpt-5", "gpt-5-mini", "gpt-4.1-mini", "gpt-4.1"],
            index=0,
            help="Pick a model your account can access.",
        )
        custom_model = st.text_input("Custom model", placeholder="e.g. gpt-5")
        ai_model = _clean_text(custom_model) or model_preset
        detected_key = _get_api_key()
        if detected_key:
            st.caption("OpenAI key detected.")
        else:
            st.caption("Set `OPENAI_API_KEY` or `.streamlit/secrets.toml` to enable ChatGPT analysis.")
            with st.expander("API key setup example"):
                st.code('OPENAI_API_KEY = "sk-..."', language="toml")

        st.divider()
        st.subheader("Wearable / Device Data")
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        st.download_button(
            "Sample CSV template",
            data=_sample_csv().to_csv(index=False).encode("utf-8"),
            file_name="carenav_sample_wearable_vitals.csv",
            mime="text/csv",
        )
        with st.expander("Supported columns"):
            st.code(", ".join(["date", *NUMERIC_COLUMNS]), language="text")
            st.caption("CareNav also auto-maps many Apple Health / WHOOP-style column names and long-format `date/type/value` exports.")

    upload_preview_df, upload_preview_summary, upload_preview_error = _preview_upload(uploaded_file)
    prefilled_metrics = _prefill_form_from_summary(upload_preview_summary)

    st.markdown("### Inputs")
    st.caption("All fields are optional. Leave blanks for anything you do not know.")
    if upload_preview_error:
        st.warning(f"CSV preview could not be parsed yet: {upload_preview_error}")
    elif upload_preview_summary and prefilled_metrics:
        pretty = ", ".join(m.replace("_", " ") for m in prefilled_metrics[:6])
        extra = "" if len(prefilled_metrics) <= 6 else f" +{len(prefilled_metrics) - 6} more"
        st.markdown(
            f'<div class="subtle-note">CSV parsed and prefilled blank fields from latest readings: {pretty}{extra}.</div>',
            unsafe_allow_html=True,
        )
    if upload_preview_summary and upload_preview_summary.get("features"):
        preview_rows = []
        feature_order = ["resting_hr", "hrv", "sleep_hours", "steps", "spo2", "systolic_bp", "diastolic_bp", "weight_kg", "glucose_fasting", "temperature_f"]
        for metric in feature_order:
            item = upload_preview_summary.get("features", {}).get(metric)
            if not item or item.get("latest") is None:
                continue
            preview_rows.append(
                {
                    "Metric": metric.replace("_", " "),
                    "Latest": round(float(item["latest"]), 2),
                    "7d avg": None if item.get("avg7") is None else round(float(item["avg7"]), 2),
                    "Records": item.get("count"),
                }
            )
        if preview_rows:
            st.markdown('<div class="csv-preview-card">', unsafe_allow_html=True)
            st.markdown("**Detected from CSV**")
            st.dataframe(pd.DataFrame(preview_rows), use_container_width=True, hide_index=True, height=min(300, 45 + len(preview_rows) * 35))
            st.markdown("</div>", unsafe_allow_html=True)

    with st.form("carenav_form", clear_on_submit=False):
        p1, p2, p3 = st.columns(3)
        with p1:
            age_text = st.text_input("Age", placeholder="e.g. 42", key="age_text")
            height_text = st.text_input("Height cm", placeholder="e.g. 170", key="height_text")
            sex = st.selectbox("Sex", ["Female", "Male", "Other / Prefer not to say"], index=None, key="sex_select")
        with p2:
            weight_text = st.text_input("Weight kg", placeholder="e.g. 72.5", key="weight_text")
            smoking_status = st.selectbox("Smoking status", ["Never", "Former", "Current"], index=None, key="smoking_status_select")
            main_concern_choices = st.multiselect(
                "Main concerns today",
                [
                    "No concerns",
                    "High blood pressure",
                    "Poor sleep / recovery",
                    "Fatigue / low energy",
                    "Stress / burnout",
                    "High resting heart rate",
                    "Low HRV",
                    "Blood sugar concerns",
                    "Breathing / oxygen concerns",
                    "Weight gain / fluid retention",
                    "Palpitations",
                ],
                default=[],
                placeholder="Choose any that apply",
                key="main_concern_choices",
            )
            main_concern_other = st.text_input("Other concern", placeholder="If not listed", key="main_concern_other")
        with p3:
            conditions = st.multiselect(
                "Known conditions",
                [
                    "Hypertension",
                    "Prediabetes",
                    "Diabetes",
                    "Heart failure",
                    "Coronary artery disease",
                    "Asthma",
                    "COPD",
                    "Kidney disease",
                    "Sleep apnea",
                ],
                default=[],
                placeholder="Select any that apply",
                key="conditions_select",
            )
            symptoms = st.multiselect(
                "Current symptoms",
                [
                    "No symptoms",
                    "Chest pain",
                    "Shortness of breath",
                    "Cough",
                    "Fever",
                    "Fatigue",
                    "Poor sleep",
                    "Palpitations",
                    "Leg swelling",
                    "Dizziness",
                    "Confusion",
                ],
                default=[],
                placeholder="Select any that apply",
                key="symptoms_select",
            )

        st.markdown("#### Current Measurements")
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            systolic_bp_text = st.text_input("Systolic BP", placeholder="e.g. 124", key="systolic_bp_text")
            diastolic_bp_text = st.text_input("Diastolic BP", placeholder="e.g. 79", key="diastolic_bp_text")
            glucose_fasting_text = st.text_input("Fasting glucose (mg/dL)", placeholder="e.g. 95", key="glucose_fasting_text")
        with m2:
            resting_hr_text = st.text_input("Resting HR", placeholder="e.g. 62", key="resting_hr_text")
            hrv_text = st.text_input("HRV (ms)", placeholder="e.g. 40", key="hrv_text")
            spo2_text = st.text_input("SpO2 (%)", placeholder="e.g. 98", key="spo2_text")
        with m3:
            sleep_hours_text = st.text_input("Sleep hours", placeholder="e.g. 7.2", key="sleep_hours_text")
            steps_text = st.text_input("Steps", placeholder="e.g. 6500", key="steps_text")
            temp_f_text = st.text_input("Temperature F", placeholder="e.g. 98.6", key="temp_f_text")
        with m4:
            st.info(
                "Fastest value: add BP, resting HR, sleep, and steps. CSV uploads can fill blanks automatically from wearable data."
            )

        submitted = st.form_submit_button("Run Analysis", type="primary", use_container_width=True)

    if submitted:
        age = _parse_optional_int(age_text)
        height_cm = _parse_optional_float(height_text)
        weight_kg = _parse_optional_float(weight_text)
        bmi = None
        if height_cm and weight_kg and height_cm > 0:
            bmi = weight_kg / ((height_cm / 100) ** 2)

        normalized_concerns = _normalize_no_selection(main_concern_choices, "No concerns")
        normalized_symptoms = _normalize_no_selection(symptoms, "No symptoms")

        concern_parts = [c for c in normalized_concerns if c]
        if _clean_text(main_concern_other) and "No concerns" not in main_concern_choices:
            concern_parts.append(_clean_text(main_concern_other))
        main_concern = ", ".join(concern_parts)

        profile = {
            "age": age,
            "sex": sex,
            "height_cm": height_cm,
            "weight_kg": weight_kg,
            "bmi": bmi,
            "smoking_status": smoking_status or "Unknown",
        }
        manual_inputs = {
            "systolic_bp": _parse_optional_float(systolic_bp_text),
            "diastolic_bp": _parse_optional_float(diastolic_bp_text),
            "resting_hr": _parse_optional_float(resting_hr_text),
            "hrv": _parse_optional_float(hrv_text),
            "spo2": _parse_optional_float(spo2_text),
            "sleep_hours": _parse_optional_float(sleep_hours_text),
            "steps": _parse_optional_float(steps_text),
            "glucose_fasting": _parse_optional_float(glucose_fasting_text),
            "temperature_f": _parse_optional_float(temp_f_text),
            "weight_kg": weight_kg,
        }
        st.session_state["carenav_result"] = _run_analysis(
            profile=profile,
            manual_inputs=manual_inputs,
            symptoms=normalized_symptoms,
            conditions=conditions,
            uploaded_file=uploaded_file,
            ai_enabled=ai_enabled,
            ai_model=ai_model,
            main_concern=main_concern,
        )

    result = st.session_state.get("carenav_result")
    if not result:
        st.info("Add any values you know above and click `Run Analysis` to generate a prediction summary.")
        return

    summary = result["summary"]
    risks = result["risks"]
    alerts = result["alerts"]
    positives = result["positives"]
    gaps = result["gaps"]
    ts_df = result["ts_df"]
    upload_error = result["upload_error"]
    confidence = float(result["confidence"] or 0)
    ai_text = result["ai_text"]
    ai_error = result["ai_error"]
    ai_actions = result.get("ai_actions", [])
    ai_actions_error = result.get("ai_actions_error")
    summary_text = result["summary_text"]

    if upload_error:
        st.warning(f"CSV upload could not be parsed: {upload_error}")

    if alerts:
        st.error("Urgent warning: " + " | ".join(alerts[:2]))

    top_risk = risks[0] if risks else None
    top_prob = f"{top_risk.probability:.0f}%" if top_risk else "N/A"
    check_in = _recommended_check_in(risks, confidence, alerts)
    present_metrics, total_metrics = _data_coverage(summary)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _card("Top risk", top_risk.name if top_risk else "No result", top_risk.severity if top_risk else "", tone="red")
    with c2:
        _card("Top probability", top_prob, "Rule-based estimate", tone="blue")
    with c3:
        _card("Confidence", f"{confidence:.0%}", f"Data coverage {present_metrics}/{total_metrics}", tone="purple")
    with c4:
        _card("Next check-in", check_in, "Suggested cadence", tone="green")

    st.markdown("### Top Risk Explanation")
    if ai_error and _get_api_key():
        st.warning(f"ChatGPT analysis failed, showing local summary instead. Error: {ai_error}")
    elif ai_error and "No OpenAI API key" in ai_error:
        st.info("ChatGPT key not configured. Showing local summary.")
    st.markdown(ai_text)

    st.markdown("### Next Best Actions")
    if ai_actions_error and _get_api_key():
        st.caption("AI mitigation actions unavailable right now. Showing local fallback actions.")
    actions = ai_actions if ai_actions else _build_priority_actions(risks, alerts)
    if actions:
        for i, action in enumerate(actions, start=1):
            st.write(f"{i}. {action}")
    else:
        st.write("1. Keep healthy routines consistent and rerun after new readings are available.")

    b1, b2 = st.columns([1.1, 1.0])
    with b1:
        st.markdown('<div class="section-title">Top Risks</div>', unsafe_allow_html=True)
        for r in risks[:3]:
            st.markdown(
                f'<span class="pill {_severity_class(r.severity)}">{r.name}: {r.probability:.0f}%</span>',
                unsafe_allow_html=True,
            )
    with b2:
        st.markdown('<div class="section-title">Positive Signal</div>', unsafe_allow_html=True)
        if positives:
            st.success(positives[0])
        else:
            st.caption("No clear protective signal detected yet. Add more data over time.")

    st.markdown("### Positive Behaviors")
    if positives:
        for p in positives[:4]:
            st.success(p)
    else:
        st.caption("No positive behaviors detected yet from current inputs.")

    with st.expander("Detailed Risk Breakdown"):
        for r in risks[:3]:
            st.markdown(f"#### {r.name}")
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Probability", f"{r.probability:.0f}%")
            d2.metric("Severity", r.severity)
            d3.metric("Trend", r.trend)
            d4.metric("Confidence", f"{r.confidence:.0%}")
            if r.why:
                st.write("Why")
                for line in r.why:
                    st.write(f"- {line}")
            if r.protective_factors:
                st.write("Protective factors")
                for line in r.protective_factors:
                    st.write(f"- {line}")
            if r.next_steps:
                st.write("Next monitoring/actions")
                for line in r.next_steps[:3]:
                    st.write(f"- {line}")
            st.divider()

    with st.expander("Trend Data (Uploaded CSV)"):
        if ts_df is None or ts_df.empty:
            st.caption("No CSV uploaded. Analysis used manual inputs only.")
        else:
            st.write(
                f"Loaded {len(ts_df)} rows from {summary.get('date_min').date()} to {summary.get('date_max').date()}."
            )
            present_cols = [c for c in NUMERIC_COLUMNS if c in ts_df.columns and ts_df[c].notna().any()]
            trend_granularity = st.radio(
                "View",
                ["Day", "Week", "Month"],
                horizontal=True,
                key="trend_granularity",
            )
            selected_cols = st.multiselect(
                "Metrics to chart",
                present_cols,
                default=present_cols[:3],
                placeholder="Choose metrics",
                key="trend_chart_metrics",
            )
            if selected_cols:
                chart_df = _build_trend_view(ts_df, selected_cols, trend_granularity)
                if chart_df.empty:
                    st.caption("No chartable points for this view.")
                else:
                    chart_long = (
                        chart_df.reset_index()
                        .melt(id_vars="date", var_name="Metric", value_name="Value")
                        .dropna(subset=["Value"])
                    )
                    line = (
                        alt.Chart(chart_long)
                        .mark_line()
                        .encode(
                            x=alt.X("date:T", title="Date"),
                            y=alt.Y("Value:Q", title="Value"),
                            color=alt.Color("Metric:N", legend=alt.Legend(orient="top-right", title="Metric")),
                            tooltip=["date:T", "Metric:N", "Value:Q"],
                        )
                        .properties(height=320)
                    )
                    st.altair_chart(line, use_container_width=True)
                    if trend_granularity == "Day":
                        st.caption("Daily values (last 90 days).")
                    elif trend_granularity == "Week":
                        st.caption("Weekly averages.")
                    else:
                        st.caption("Monthly averages.")
            st.dataframe(ts_df.tail(14), use_container_width=True, hide_index=True)

    with st.expander("Care Team Summary / Export"):
        st.text_area("Summary text", value=summary_text, height=320)
        st.download_button(
            "Download summary (.txt)",
            data=summary_text.encode("utf-8"),
            file_name="carenav_summary.txt",
            mime="text/plain",
        )
        if ai_text:
            st.download_button(
                "Download AI analysis (.md)",
                data=ai_text.encode("utf-8"),
                file_name="carenav_ai_analysis.md",
                mime="text/markdown",
            )


if __name__ == "__main__":
    main()
