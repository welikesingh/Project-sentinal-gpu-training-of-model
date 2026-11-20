"""SENTINEL dashboard: LoRA vs QLoRA vs XGBoost — stats, benchmark, live + batch tests, Go / No-Go decision.

    streamlit run ui/app.py
Endpoints come from env and can be edited in the sidebar:
  LORA_URL  / LORA_API_KEY    (falls back to LLM_URL / LLM_API_KEY)   default http://localhost:8000
  QLORA_URL / QLORA_API_KEY   (key falls back to LLM_API_KEY)         default http://localhost:8002
  XGB_URL   / XGB_API_KEY                                             default http://localhost:8001
XGBoost is the current production model (baseline). LoRA and QLoRA are the candidates.
"""

from __future__ import annotations

import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from client import Endpoint  # noqa: E402
from style import (CSS, DECISION_COLOR, GRAY, LORA_COLOR, QLORA_COLOR, RED, XGB_COLOR,  # noqa: E402
                   decision_pill, dot, esc, kpi, pill, plotly_layout, prob_bar)

st.set_page_config(page_title="SENTINEL", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")
st.markdown(CSS, unsafe_allow_html=True)

BASELINE = "XGBoost"
CANDIDATES = ["LoRA", "QLoRA"]
NAMES = CANDIDATES + [BASELINE]
COLOR = {"LoRA": LORA_COLOR, "QLoRA": QLORA_COLOR, "XGBoost": XGB_COLOR}
FALLBACK_SCENARIOS = [
    {"name": "Everyday grocery (UPI)", "expect": "ALLOW", "transaction": {
        "transaction_id": "SCN_A", "amount": 640, "mcc": "grocery", "channel": "UPI", "hour": 19, "weekday": "Tue",
        "customer_age": 41, "account_age_days": 1450, "avg_90d": 720, "txns_30d": 14, "fails_24h": 0,
        "new_device": False, "city": "Pune", "merchant_city": "Pune", "new_city": False, "dist_km": "<5",
        "amt_ratio": 0.9, "txns_1h": 0, "amount_24h": 300, "device_users_7d": 1, "history": []}},
    {"name": "3 AM electronics, new device, failed attempts", "expect": "BLOCK", "transaction": {
        "transaction_id": "SCN_B", "amount": 74999, "mcc": "electronics", "channel": "card", "hour": 3, "weekday": "Sun",
        "customer_age": 24, "account_age_days": 420, "avg_90d": 1800, "txns_30d": 6, "fails_24h": 4,
        "new_device": True, "city": "Patna", "merchant_city": "Mumbai", "new_city": True, "dist_km": "1000+",
        "amt_ratio": 41.7, "txns_1h": 3, "amount_24h": 12000, "device_users_7d": 2, "history": []}},
]
MCC = ["grocery", "quick_commerce", "food", "electronics", "gaming", "gift_cards", "healthcare", "travel", "education", "other"]
CHANNELS = ["UPI", "card", "wallet", "net_banking"]
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
DIST = ["<5", "50-200", "200-500", "500-1000", "1000+", "unk"]

ss = st.session_state
ss.setdefault("live_log", [])
ss.setdefault("batch", None)


def env(*names, default=""):
    for n in names:
        if os.getenv(n):
            return os.getenv(n)
    return default


# ------------------------------------------------------------------ sidebar: connections + targets
with st.sidebar:
    st.markdown('<p class="large-title" style="font-size:24px">Connections</p>', unsafe_allow_html=True)
    st.caption("All three models run on the Hyperstack GPU VM · XGBoost = current production baseline")
    cfg = {
        "LoRA": (env("LORA_URL", "LLM_URL", default="http://localhost:8000"), env("LORA_API_KEY", "LLM_API_KEY")),
        "QLoRA": (env("QLORA_URL", default="http://localhost:8002"), env("QLORA_API_KEY", "LLM_API_KEY")),
        "XGBoost": (env("XGB_URL", default="http://localhost:8001"), env("XGB_API_KEY")),
    }
    EP = {}
    for name in NAMES:
        url = st.text_input(f"{name} URL", cfg[name][0])
        key = st.text_input(f"{name} API key", cfg[name][1], type="password")
        EP[name] = Endpoint(name, url or "", key or "")
    if st.button("Refresh", use_container_width=True):
        st.cache_data.clear()
    with st.expander("Acceptance targets"):
        TARGETS = {
            "recall": st.number_input("Min recall at REVIEW", 0.0, 1.0, 0.85, 0.01),
            "precision_block": st.number_input("Min precision at BLOCK", 0.0, 1.0, 0.70, 0.01),
            "pr_auc": st.number_input("Min PR-AUC (holdout)", 0.0, 1.0, 0.80, 0.01),
            "uplift": st.number_input("Min PR-AUC uplift over XGBoost", -1.0, 1.0, 0.03, 0.01),
            "fpr_tol": st.number_input("Max extra false-positive rate vs XGBoost", 0.0, 1.0, 0.005, 0.001, format="%.3f"),
            "p99_ms": st.number_input("Max latency p99 (ms)", 1, 10_000, 250, 10),
        }


@st.cache_data(ttl=20, show_spinner=False)
def cached_health(url, key, name):
    return Endpoint(name, url, key).health()


@st.cache_data(ttl=120, show_spinner=False)
def cached_stats(url, key, name):
    return Endpoint(name, url, key).stats()


@st.cache_data(ttl=300, show_spinner=False)
def cached_scenarios(url, key):
    ok, data, _ = Endpoint("XGBoost", url, key).scenarios()
    return data if ok and isinstance(data, list) and data else FALLBACK_SCENARIOS


health, stats = {}, {}
for name, ep in EP.items():
    health[name] = cached_health(ep.url, ep.key, name)
    ok_s, data, _ = cached_stats(ep.url, ep.key, name) if health[name][0] else (False, health[name][1], 0)
    stats[name] = data if ok_s else None
    if health[name][0] and not ok_s:
        health[name] = (False, data, 0)
ONLINE = [n for n in NAMES if health[n][0]]

with st.sidebar:
    st.markdown('<div class="section">Status</div>', unsafe_allow_html=True)
    for name in NAMES:
        ok, data, ms = health[name]
        if ok:
            st.markdown(f'<div class="card">{dot(COLOR[name])}<b>{name}</b> · online · {ms:.0f} ms'
                        f'<div class="muted">{esc(data.get("run", data.get("kind", "")))} {esc(data.get("device", ""))}</div></div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="card">{dot(RED)}<b>{name}</b> · offline'
                        f'<div class="muted">{esc(str(data)[:160])}</div></div>', unsafe_allow_html=True)


# ------------------------------------------------------------------ helpers
def m_of(name, split="holdout"):
    try:
        return stats[name]["metrics"][split]["calibrated"]
    except (TypeError, KeyError):
        return None


def fmt(v, digits=3):
    return "—" if v is None else f"{v:.{digits}f}"


def pct(v):
    return "—" if v is None else f"{100 * v:.1f}%"


def status_chip(ok):
    if ok is None:
        return pill("n/a", "p-gray")
    return pill("PASS", "p-allow") if ok else pill("FAIL", "p-block")


def offline_note(name):
    st.info(f"{name} is not reachable. Check its URL / API key in the sidebar.")


def live_latencies(name):
    return [r[f"{name}_ms"] for r in ss.live_log if r.get(f"{name}_ms") is not None]


def checks_for(name):
    """Acceptance checks for one candidate: list of (id, objective, target, measured, pass|None)."""
    m, b = m_of(name), m_of(BASELINE)
    lat = live_latencies(name)
    p99 = float(np.percentile(lat, 99)) if len(lat) >= 5 else None
    uplift = (m["pr_auc"] - b["pr_auc"]) if m and b and m["pr_auc"] is not None and b["pr_auc"] is not None else None
    fpr, bfpr = (m and m["at_review"]["false_positive_rate"]), (b and b["at_review"]["false_positive_rate"])
    t = TARGETS
    return [
        ("OBJ-01", "Recall on fraud at REVIEW (holdout)", f"≥ {pct(t['recall'])}", pct(m and m["at_review"]["recall"]),
         None if not m else m["at_review"]["recall"] >= t["recall"]),
        ("OBJ-02", "Precision at BLOCK threshold", f"≥ {pct(t['precision_block'])}", pct(m and m["at_block"]["precision"]),
         None if not m else m["at_block"]["precision"] >= t["precision_block"]),
        ("OBJ-03", "PR-AUC on holdout", f"≥ {t['pr_auc']:.2f}", fmt(m and m["pr_auc"]),
         None if not m or m["pr_auc"] is None else m["pr_auc"] >= t["pr_auc"]),
        ("OBJ-04", "PR-AUC uplift over XGBoost", f"≥ {t['uplift']:+.2f}", "—" if uplift is None else f"{uplift:+.3f}",
         None if uplift is None else uplift >= t["uplift"]),
        ("OBJ-05", "False-positive rate at REVIEW vs XGBoost", f"≤ XGB + {100 * t['fpr_tol']:.1f} pt",
         "—" if fpr is None or bfpr is None else f"{pct(fpr)} vs {pct(bfpr)}",
         None if fpr is None or bfpr is None else fpr <= bfpr + t["fpr_tol"]),
        ("OBJ-10", "Scoring latency p99 (live tests, round trip)", f"≤ {t['p99_ms']} ms",
         "—" if p99 is None else f"{p99:.0f} ms ({len(lat)} calls)", None if p99 is None else p99 <= t["p99_ms"]),
    ]


def verdict_for(name):
    """('GO' | 'NO-GO' | 'INCOMPLETE', explanation) for one candidate."""
    if not stats.get(name):
        return "INCOMPLETE", f"{name} is offline."
    if not stats.get(BASELINE):
        return "INCOMPLETE", "XGBoost baseline is offline - nothing to compare against."
    cs = checks_for(name)
    failed = [f"{c[0]} {c[1]} ({c[3]}, target {c[2]})" for c in cs if c[4] is False]
    missing = [c[0] for c in cs if c[4] is None]
    if failed:
        return "NO-GO", "Failed: " + "; ".join(failed)
    if missing:
        return "INCOMPLETE", "All measured checks pass; still missing " + ", ".join(missing) + \
            (" (run ≥ 5 Live Test calls)" if "OBJ-10" in missing else "")
    return "GO", "Meets every acceptance target and beats XGBoost."


def checks_table(cols):
    """HTML table: one row per objective, one measured+status column per candidate in cols."""
    per = {n: checks_for(n) for n in cols}
    head = "".join(f"<td>{dot(COLOR[n])}{n}</td><td></td>" for n in cols)
    body = ""
    for i, c in enumerate(per[cols[0]]):
        cells = "".join(f"<td><b>{esc(per[n][i][3])}</b></td><td>{status_chip(per[n][i][4])}</td>" for n in cols)
        body += f"<tr><td><b>{c[0]}</b></td><td>{esc(c[1])}</td><td class='muted'>{esc(c[2])}</td>{cells}</tr>"
    return (f'<div class="card"><table style="width:100%;border-collapse:collapse;font-size:14px">'
            f'<tr class="muted"><td>ID</td><td>Objective</td><td>Target</td>{head}</tr>{body}</table>'
            f'<div class="muted" style="margin-top:8px">Latency fills in as you run Live Test calls (≥ 5 needed). '
            f'Targets are editable in the sidebar.</div></div>')


st.markdown('<p class="large-title">SENTINEL</p><p class="subtitle">Fraud LLM (LoRA · QLoRA) vs XGBoost · '
            'benchmark · live testing · deployment decision</p>', unsafe_allow_html=True)
t_over, t_ft, t_bench, t_live, t_batch, t_dec = st.tabs(
    ["Overview", "LoRA vs QLoRA", "Benchmark", "Live Test", "Batch Test", "Decision"])


# ------------------------------------------------------------------ Overview
with t_over:
    cols = st.columns(3)
    for col, name in zip(cols, NAMES):
        with col:
            s, m = stats.get(name), m_of(name)
            tag = "baseline (production)" if name == BASELINE else "candidate"
            st.markdown(f'<div class="card"><h4>{dot(COLOR[name])}{name} <span class="muted">· {tag}</span></h4>'
                        f'<div class="muted">{esc(s["model"]) if s else "offline"}</div></div>', unsafe_allow_html=True)
            k = st.columns(2)
            k[0].markdown(kpi("PR-AUC", fmt(m and m["pr_auc"]), "holdout"), unsafe_allow_html=True)
            k[1].markdown(kpi("ROC-AUC", fmt(m and m["roc_auc"]), "holdout"), unsafe_allow_html=True)
            k = st.columns(2)
            k[0].markdown(kpi("Recall", pct(m and m["at_review"]["recall"]), "at REVIEW"), unsafe_allow_html=True)
            k[1].markdown(kpi("Precision", pct(m and m["at_block"]["precision"]), "at BLOCK"), unsafe_allow_html=True)
    st.markdown('<div class="section">Acceptance targets (candidates, holdout)</div>', unsafe_allow_html=True)
    st.markdown(checks_table(CANDIDATES), unsafe_allow_html=True)


# ------------------------------------------------------------------ LoRA vs QLoRA
with t_ft:
    ft = [n for n in CANDIDATES if stats.get(n)]
    if not ft:
        offline_note("Neither LoRA nor QLoRA")
    else:
        rows = {}
        for n in ft:
            info, m, cal = stats[n].get("info", {}), m_of(n) or {}, stats[n].get("calibration", {})
            rows[n] = {"Run": stats[n].get("run"), "Method": str(info.get("method", "")).upper(),
                       "Base model": str(info.get("base_model", "—")).split("/")[-1], "Device": str(info.get("device", "—")),
                       "Train rows": info.get("train_rows"), "Train time (min)": round((info.get("train_runtime_s") or 0) / 60, 1),
                       "Final train loss": info.get("train_loss"), "Holdout PR-AUC": m.get("pr_auc"),
                       "Holdout ROC-AUC": m.get("roc_auc"), "ECE (↓)": m.get("ece"),
                       "Recall@REVIEW": (m.get("at_review") or {}).get("recall"),
                       "Precision@BLOCK": (m.get("at_block") or {}).get("precision"),
                       "REVIEW threshold": cal.get("review_threshold"), "BLOCK threshold": cal.get("block_threshold")}
        st.dataframe(pd.DataFrame(rows).astype(str), use_container_width=True)

        left, right = st.columns(2)
        with left:
            fig = go.Figure()
            for n in ft:
                lc = stats[n].get("loss_curve") or []
                if lc:
                    df = pd.DataFrame(lc)
                    fig.add_trace(go.Scatter(x=df["step"], y=df["loss"].rolling(10, min_periods=1).mean(), mode="lines",
                                             name=n, line=dict(color=COLOR[n], width=2.5)))
            st.plotly_chart(plotly_layout(fig, title="Training loss (smoothed)", xaxis_title="step"), use_container_width=True)
        with right:
            fig = go.Figure()
            for n in ft:
                try:
                    per = stats[n]["metrics"]["holdout"]["reasons"]["per_code_f1"]
                except (KeyError, TypeError):
                    continue
                fig.add_trace(go.Bar(name=n, x=list(per.values()), orientation="h", marker_color=COLOR[n],
                                     y=[k.split("_", 1)[1].replace("_", " ").title() for k in per]))
            st.plotly_chart(plotly_layout(fig, title="Reason-code F1 (holdout)", barmode="group", xaxis_range=[0, 1]),
                            use_container_width=True)


# ------------------------------------------------------------------ Benchmark
with t_bench:
    split = st.radio("Split", ["holdout", "validation"], horizontal=True, label_visibility="collapsed") or "holdout"
    ms_ = {n: m_of(n, split) for n in NAMES}
    if not any(ms_.values()):
        offline_note("No model")
    else:
        def row(m):
            if not m:
                return [None] * 8
            return [m["pr_auc"], m["roc_auc"], m["ece"], m["at_review"]["recall"], m["at_review"]["precision"],
                    m["at_block"]["recall"], m["at_block"]["precision"], m["at_review"]["false_positive_rate"]]
        metric_names = ["PR-AUC", "ROC-AUC", "ECE (↓)", "Recall@REVIEW", "Precision@REVIEW", "Recall@BLOCK",
                        "Precision@BLOCK", "FPR@REVIEW (↓)"]
        df = pd.DataFrame({n: row(ms_[n]) for n in NAMES}, index=metric_names, dtype=float)
        for n in CANDIDATES:
            df[f"Δ {n} − XGB"] = df[n] - df[BASELINE]
        k = st.columns(5)
        for col, n in zip(k[:3], NAMES):
            col.markdown(kpi(f"{n} PR-AUC", fmt(ms_[n] and ms_[n]["pr_auc"]), split), unsafe_allow_html=True)
        for col, n in zip(k[3:], CANDIDATES):
            d = df.loc["PR-AUC", f"Δ {n} − XGB"]
            col.markdown(kpi(f"{n} uplift", "—" if pd.isna(d) else f"{d:+.3f}", f"target ≥ {TARGETS['uplift']:+.2f}"),
                         unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            for n in NAMES:
                cv = (ms_[n] or {}).get("curves", {}).get("pr")
                if cv:
                    fig.add_trace(go.Scatter(x=cv["recall"], y=cv["precision"], mode="lines", name=n,
                                             line=dict(color=COLOR[n], width=2.5)))
            fig.add_vline(x=TARGETS["recall"], line_dash="dot", line_color=GRAY)
            fig.add_hline(y=TARGETS["precision_block"], line_dash="dot", line_color=GRAY)
            st.plotly_chart(plotly_layout(fig, title="Precision–Recall", xaxis_title="Recall", yaxis_title="Precision",
                                          xaxis_range=[0, 1], yaxis_range=[0, 1.02]), use_container_width=True)
        with c2:
            fig = go.Figure()
            for n in NAMES:
                cv = (ms_[n] or {}).get("curves", {}).get("roc")
                if cv:
                    fig.add_trace(go.Scatter(x=cv["fpr"], y=cv["tpr"], mode="lines", name=n, line=dict(color=COLOR[n], width=2.5)))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=GRAY, dash="dot"), showlegend=False))
            st.plotly_chart(plotly_layout(fig, title="ROC", xaxis_title="False positive rate", yaxis_title="True positive rate"),
                            use_container_width=True)

        st.dataframe(df.round(4), use_container_width=True)
        fig = go.Figure()
        keep = [0, 1, 3, 4, 5, 6]
        for n in NAMES:
            fig.add_trace(go.Bar(name=n, x=[metric_names[i] for i in keep], y=df[n].iloc[keep], marker_color=COLOR[n]))
        st.plotly_chart(plotly_layout(fig, title="Side by side", barmode="group", yaxis_range=[0, 1]), use_container_width=True)

        fi = (stats.get(BASELINE) or {}).get("feature_importance") or {}
        if fi:
            top = list(fi.items())[:15][::-1]
            fig = go.Figure(go.Bar(x=[v for _, v in top], y=[k for k, _ in top], orientation="h", marker_color=XGB_COLOR))
            st.plotly_chart(plotly_layout(fig, height=420, title="XGBoost feature importance (gain share)"),
                            use_container_width=True)


# ------------------------------------------------------------------ Live Test
def result_card(name, ok, res, ms, server_ms):
    if not ok:
        return f'<div class="card"><h4>{dot(RED)}{name}</h4><div class="muted">{esc(str(res)[:300])}</div></div>'
    p = res["fraud_probability"]
    reasons = "".join(pill(r.split("_", 1)[1].replace("_", " ").lower(), "p-blue") for r in res["reason_codes"]) \
        or '<span class="muted">no reason codes</span>'
    srv = f" · server {server_ms:.0f} ms" if server_ms is not None else ""
    return (f'<div class="card"><h4>{dot(COLOR[name])}{name} &nbsp;{decision_pill(res["decision"])}</h4>'
            f'<div class="kpi-value">{100 * p:.2f}%</div><div class="kpi-sub">fraud probability · confidence '
            f'{100 * res["confidence"]:.1f}%</div>{prob_bar(p, DECISION_COLOR[res["decision"]])}'
            f'<div style="margin:8px 0">{reasons}</div><div>{esc(res["explanation"])}</div>'
            f'<div class="muted" style="margin-top:8px">round trip {ms:.0f} ms{srv}</div></div>')


with t_live:
    scen = cached_scenarios(EP[BASELINE].url, EP[BASELINE].key)
    snames = [s["name"] for s in scen]
    top = st.columns([3, 1])
    pick = top[0].selectbox("Scenario", snames, index=min(2, len(snames) - 1) if len(snames) > 2 else 0)
    mode = top[1].radio("Input", ["Form", "Raw prompt"], horizontal=True)
    sc = next(s for s in scen if s["name"] == pick)
    t0 = dict(sc["transaction"])
    if "expect" in sc:
        st.markdown(f'<span class="muted">Scenario expectation:</span> {decision_pill(sc["expect"])}', unsafe_allow_html=True)

    item = None
    if mode == "Form":
        with st.form(f"live_{pick}"):
            st.markdown('<div class="section">Transaction</div>', unsafe_allow_html=True)
            a = st.columns(5)
            amount = a[0].number_input("Amount (₹)", 0.0, 2_000_000.0, float(t0.get("amount", 1000)), step=100.0)
            mcc = a[1].selectbox("MCC", MCC, index=MCC.index(t0.get("mcc", "grocery")) if t0.get("mcc") in MCC else 0)
            channel = a[2].selectbox("Channel", CHANNELS, index=CHANNELS.index(t0.get("channel", "UPI")) if t0.get("channel") in CHANNELS else 0)
            hour = a[3].slider("Hour", 0, 23, int(t0.get("hour", 12)))
            weekday = a[4].selectbox("Weekday", DAYS, index=DAYS.index(t0.get("weekday", "Mon")) if t0.get("weekday") in DAYS else 0)
            st.markdown('<div class="section">Customer</div>', unsafe_allow_html=True)
            b = st.columns(5)
            age = b[0].number_input("Age", 16, 100, int(t0.get("customer_age", 35)))
            acct = b[1].number_input("Account age (days)", 0, 10000, int(t0.get("account_age_days", 365)))
            avg = b[2].number_input("Avg 90d (₹, 0 = unknown)", 0.0, 2_000_000.0, float(t0.get("avg_90d") or 0))
            t30 = b[3].number_input("Txns last 30d", 0, 500, int(t0.get("txns_30d", 0)))
            fails = b[4].number_input("Failed attempts 24h", 0, 20, int(t0.get("fails_24h", 0)))
            st.markdown('<div class="section">Context</div>', unsafe_allow_html=True)
            c = st.columns(5)
            new_dev = c[0].toggle("New device", bool(t0.get("new_device")))
            new_city = c[1].toggle("New city", bool(t0.get("new_city")))
            city = c[2].text_input("Customer city", t0.get("city", "Pune"))
            mcity = c[3].text_input("Merchant city", t0.get("merchant_city", "Pune"))
            dist = c[4].selectbox("Distance (km)", DIST, index=DIST.index(t0.get("dist_km", "<5")) if t0.get("dist_km") in DIST else 0)
            d = st.columns(4)
            ratio = d[0].number_input("Amount ratio vs avg (0 = unknown)", 0.0, 1000.0, float(t0.get("amt_ratio") or 0))
            t1h = d[1].number_input("Txns last hour", 0, 100, int(t0.get("txns_1h", 0)))
            a24 = d[2].number_input("Amount last 24h (₹)", 0.0, 5_000_000.0, float(t0.get("amount_24h", 0)))
            dusers = d[3].number_input("Device users 7d", 0, 20, int(t0.get("device_users_7d", 1)))
            st.markdown('<div class="section">History (most recent first)</div>', unsafe_allow_html=True)
            hist_df = pd.DataFrame(t0.get("history") or [], columns=["days_ago", "channel", "amount", "mcc", "city"])
            hist = st.data_editor(hist_df, num_rows="dynamic", use_container_width=True, key=f"hist_{pick}")
            go_btn = st.form_submit_button("Score with all models")
        if go_btn:
            item = {"transaction_id": t0.get("transaction_id", "LIVE"), "amount": amount, "mcc": mcc, "channel": channel,
                    "hour": hour, "weekday": weekday, "customer_age": age, "account_age_days": acct,
                    "avg_90d": avg or None, "txns_30d": t30, "fails_24h": fails, "new_device": new_dev,
                    "city": city, "merchant_city": mcity, "new_city": new_city, "dist_km": dist,
                    "amt_ratio": ratio or None, "txns_1h": t1h, "amount_24h": a24, "device_users_7d": dusers,
                    "history": [h for h in hist.dropna(subset=["amount"]).to_dict("records")]}
    else:
        raw = st.text_area("fmt-v1 prompt", height=180, value=(
            "[TXN] id=TX_RAW amt=74999.0 mcc=electronics channel=card time=03h Sun\n"
            "[CUST] age=24 acct_age_days=420 avg_90d=1800.00 txns_30d=6 fails_24h=4\n"
            "[CTX] new_device=yes city=Patna merchant_city=Mumbai new_city=yes dist_km=1000+ amt_ratio=41.7 "
            "txns_1h=3 amount_24h=12000.00 device_users_7d=2\n[HIST] none observed\n"
            "[TASK] Assess fraud risk using only the observed context."))
        if st.button("Score prompt with all models"):
            item = {"transaction_id": "TX_RAW", "prompt": raw}

    if item is not None:
        if not ONLINE:
            st.error("No model is online - check the sidebar.")
        cols = st.columns(3)
        entry = {"time": time.strftime("%H:%M:%S"), "scenario": pick if mode == "Form" else "raw prompt"}
        decisions = {}
        for col, name in zip(cols, NAMES):
            if name not in ONLINE:
                col.markdown(result_card(name, False, "offline - skipped", 0, None), unsafe_allow_html=True)
                continue
            ok, res, ms, srv = EP[name].score_one(item)
            col.markdown(result_card(name, ok, res, ms, srv), unsafe_allow_html=True)
            entry[f"{name}_decision"] = res["decision"] if ok else "error"
            entry[f"{name}_p"] = res["fraud_probability"] if ok else None
            entry[f"{name}_ms"] = ms if ok else None
            if ok:
                decisions[name] = res["decision"]
        if len(decisions) >= 2:
            agree = len(set(decisions.values())) == 1
            st.markdown(f'<div class="card">{pill("All models agree", "p-allow") if agree else pill("Models disagree", "p-review")} '
                        + " · ".join(f"{n} {decision_pill(d)}" for n, d in decisions.items()) + "</div>",
                        unsafe_allow_html=True)
        ss.live_log.append(entry)

    if ss.live_log:
        st.markdown('<div class="section">Session log</div>', unsafe_allow_html=True)
        log = pd.DataFrame(ss.live_log)
        st.dataframe(log.iloc[::-1], use_container_width=True, hide_index=True)
        k = st.columns(4)
        for col, name in zip(k, NAMES):
            v = pd.Series(live_latencies(name), dtype=float)
            if len(v):
                col.markdown(kpi(f"{name} latency p50 / p99", f"{v.quantile(.5):.0f} / {v.quantile(.99):.0f} ms",
                                 f"{len(v)} calls, round trip"), unsafe_allow_html=True)
        if k[3].button("Clear log"):
            ss.live_log = []
            st.rerun()


# ------------------------------------------------------------------ Batch Test
SHORT = {"LoRA": "lora", "QLoRA": "qlora", "XGBoost": "xgb"}


def sample_metrics(y, p, decisions):
    from sklearn.metrics import average_precision_score, roc_auc_score

    y, p = np.asarray(y), np.asarray(p, dtype=float)
    flagged = np.isin(decisions, ["REVIEW", "BLOCK"])
    blocked = np.asarray(decisions) == "BLOCK"
    tp = int((flagged & (y == 1)).sum())
    both = len(np.unique(y)) == 2
    return {"PR-AUC": average_precision_score(y, p) if both else None,
            "ROC-AUC": roc_auc_score(y, p) if both else None,
            "Recall (REVIEW+BLOCK)": tp / max(1, int((y == 1).sum())),
            "Precision (REVIEW+BLOCK)": tp / max(1, int(flagged.sum())),
            "Precision@BLOCK": int((blocked & (y == 1)).sum()) / max(1, int(blocked.sum())),
            "Flagged": int(flagged.sum())}


with t_batch:
    st.markdown('<div class="card"><h4>Live benchmark on labelled samples</h4><div class="muted">Pulls labelled rows '
                'from the XGBoost service, scores the identical prompts on every online model, and computes '
                'metrics on that sample. Or upload your own JSONL / CSV.</div></div>', unsafe_allow_html=True)
    src = st.radio("Source", ["Samples from XGBoost service", "Upload file"], horizontal=True) or "Samples"
    rows = None
    if src.startswith("Samples"):
        c = st.columns(5)
        bsplit = c[0].selectbox("Split", ["holdout", "validation", "train"]) or "holdout"
        n = c[1].number_input("Rows", 10, 2000, 200, step=10)
        fr = c[2].slider("Fraud share", 0.0, 1.0, 0.3, 0.05)
        seed = c[3].number_input("Seed", 0, 10_000, 1)
        c[4].markdown("<br>", unsafe_allow_html=True)
        if c[4].button("Fetch & score", use_container_width=True):
            ok, data, _ = EP[BASELINE].samples(bsplit, int(n), float(fr), int(seed))
            if not ok:
                st.error(f"Could not fetch samples: {data}")
            else:
                rows = data["rows"]
    else:
        up = st.file_uploader("JSONL (one transaction per line) or CSV; optional column fraud_label", type=["jsonl", "json", "csv"])
        if up is not None and st.button("Score file"):
            txt = up.getvalue().decode("utf-8")
            if up.name.endswith(".csv"):
                rows = pd.read_csv(io.StringIO(txt)).to_dict("records")
            elif txt.strip().startswith("["):
                rows = json.loads(txt)
            else:
                rows = [json.loads(line) for line in txt.splitlines() if line.strip()]
            for i, r in enumerate(rows):
                r.setdefault("transaction_id", f"UP_{i:05d}")
                if isinstance(r.get("history"), str):
                    r["history"] = json.loads(r["history"])

    if rows is not None:
        items = [{k: v for k, v in r.items() if k not in ("fraud_label", "reasons")} for r in rows]
        res = {}
        df = pd.DataFrame({"transaction_id": [r.get("transaction_id") for r in rows]})
        if all("fraud_label" in r for r in rows):
            df["fraud_label"] = [int(r["fraud_label"]) for r in rows]
        for name in ONLINE:
            with st.spinner(f"Scoring {len(items)} rows on {name}…"):
                ok, data, ms = EP[name].score(items)
            s = SHORT[name]
            if ok:
                df[f"{s}_p"] = [d["fraud_probability"] for d in data]
                df[f"{s}_decision"] = [d["decision"] for d in data]
                df[f"{s}_reasons"] = [", ".join(d["reason_codes"]) for d in data]
            else:
                st.error(f"{name}: {data}")
            res[name] = (ok, ms)
        ss.batch = {"df": df, "res": res, "n": len(rows), "prompts": {r.get("transaction_id"): r.get("prompt") for r in rows}}

    if ss.batch is not None:
        df, res, n = ss.batch["df"], ss.batch["res"], ss.batch["n"]
        scored = [nm for nm in NAMES if f"{SHORT[nm]}_p" in df]
        k = st.columns(4)
        k[0].markdown(kpi("Rows", str(n), f"{int(df['fraud_label'].sum()) if 'fraud_label' in df else '—'} fraud"), unsafe_allow_html=True)
        for col, name in zip(k[1:], NAMES):
            if name in res:
                ok, ms = res[name]
                col.markdown(kpi(f"{name} time", f"{ms / 1000:.1f} s" if ok else "error",
                                 f"{ms / max(1, n):.1f} ms / txn (batched)" if ok else ""), unsafe_allow_html=True)
        if BASELINE in scored:
            ag = st.columns(2)
            for col, name in zip(ag, [c for c in CANDIDATES if c in scored]):
                agree = (df[f"{SHORT[name]}_decision"] == df["xgb_decision"]).mean()
                col.markdown(kpi("Decision agreement", pct(agree), f"{name} vs XGBoost"), unsafe_allow_html=True)

        if "fraud_label" in df and scored:
            mt = {nm: sample_metrics(df["fraud_label"], df[f"{SHORT[nm]}_p"], df[f"{SHORT[nm]}_decision"].values)
                  for nm in scored}
            st.markdown('<div class="section">Metrics on this sample</div>', unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(mt).round(4), use_container_width=True)
            st.caption("Samples are enriched with fraud, so precision here is higher than at the real 0.6 % base rate. "
                       "Use the Benchmark tab (full holdout) for acceptance numbers.")
        if BASELINE in scored and any(c in scored for c in CANDIDATES):
            fig = go.Figure()
            for name in [c for c in CANDIDATES if c in scored]:
                fig.add_trace(go.Scatter(x=df["xgb_p"], y=df[f"{SHORT[name]}_p"], mode="markers", name=name,
                                         text=df["transaction_id"], marker=dict(
                                             color=COLOR[name], size=8, opacity=.75,
                                             symbol=df["fraud_label"].map({1: "x", 0: "circle"}) if "fraud_label" in df else "circle")))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=GRAY, dash="dot"), showlegend=False))
            st.plotly_chart(plotly_layout(fig, height=380, title="Candidate vs XGBoost probability (× = fraud)",
                                          xaxis_title="XGBoost p", yaxis_title="Candidate p"), use_container_width=True)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button("Download results CSV", df.to_csv(index=False), "sentinel_batch_results.csv", "text/csv")
        with st.expander("Inspect a prompt"):
            tid = st.selectbox("Transaction", df["transaction_id"].tolist())
            p = ss.batch["prompts"].get(tid)
            st.code(p or "(structured input, no prompt stored)", language="text")


# ------------------------------------------------------------------ Decision
with t_dec:
    verdicts = {n: verdict_for(n) for n in CANDIDATES}
    go_models = [n for n in CANDIDATES if verdicts[n][0] == "GO"]
    if go_models:
        best = max(go_models, key=lambda n: (m_of(n) or {}).get("pr_auc") or 0)
        cls, head = "v-go", f"GO — deploy {best}"
        sub = f"{best} meets every acceptance target and beats the XGBoost baseline on holdout."
    elif any(v[0] == "NO-GO" for v in verdicts.values()) and not any(v[0] == "INCOMPLETE" for v in verdicts.values()):
        cls, head = "v-nogo", "NO-GO — keep XGBoost in production"
        sub = "No candidate meets all acceptance targets."
    elif any(v[0] == "NO-GO" for v in verdicts.values()):
        cls, head = "v-nogo", "NO-GO so far — keep XGBoost in production"
        sub = "No candidate passes yet; finish the missing checks for the incomplete one before closing the decision."
    else:
        cls, head = "v-wait", "Decision pending"
        sub = "Some checks are not measured yet (bring the models online and run ≥ 5 Live Test calls)."
    st.markdown(f'<div class="verdict {cls}"><h2>{esc(head)}</h2><div>{esc(sub)}</div></div>', unsafe_allow_html=True)

    cols = st.columns(2)
    for col, n in zip(cols, CANDIDATES):
        v, why = verdicts[n]
        chip = {"GO": pill("GO", "p-allow"), "NO-GO": pill("NO-GO", "p-block")}.get(v, pill("INCOMPLETE", "p-gray"))
        col.markdown(f'<div class="card"><h4>{dot(COLOR[n])}{n} &nbsp;{chip}</h4><div>{esc(why)}</div></div>',
                     unsafe_allow_html=True)
    st.markdown('<div class="section">Checklist (candidate vs targets and vs XGBoost)</div>', unsafe_allow_html=True)
    st.markdown(checks_table(CANDIDATES), unsafe_allow_html=True)

    lines = [f"# SENTINEL deployment decision — {time.strftime('%Y-%m-%d %H:%M')}", "", f"**{head}**", "", sub, "",
             "| Model | Run | Holdout PR-AUC | Verdict |", "|---|---|---|---|"]
    for n in NAMES:
        m = m_of(n)
        lines.append(f"| {n} | {(stats.get(n) or {}).get('run', 'offline')} | {fmt(m and m['pr_auc'])} | "
                     f"{'baseline' if n == BASELINE else verdicts[n][0]} |")
    for n in CANDIDATES:
        lines += ["", f"## {n}: {verdicts[n][0]}", verdicts[n][1], "", "| ID | Objective | Target | Measured | Result |",
                  "|---|---|---|---|---|"]
        lines += [f"| {c[0]} | {c[1]} | {c[2]} | {c[3]} | {'n/a' if c[4] is None else 'PASS' if c[4] else 'FAIL'} |"
                  for c in checks_for(n)]
    st.download_button("Download decision report (Markdown)", "\n".join(lines), "sentinel_decision.md", "text/markdown")
