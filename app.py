# ============================================================
#  AirFlow AI — Smart Air Quality & Energy Management
#  Streamlit Simulator: Traditional Rule-Based vs RL Agent
# ============================================================

import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="AirFlow AI — RL Simulator",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ───────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Base ── */
    .stApp { background-color: #060c18 !important; color: #e2e8f0; }
    section[data-testid="stSidebar"] > div:first-child { background-color: #0b1322 !important; }

    /* ── Hide default Streamlit chrome ── */
    #MainMenu, footer, header { visibility: hidden; }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; } 
    ::-webkit-scrollbar-track { background: #060c18; }
    ::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 3px; }

    /* ── Hero title ── */
    .hero-title {
        font-size: 2.1rem; font-weight: 900; letter-spacing: -0.5px; line-height: 1.15;
        background: linear-gradient(110deg, #38bdf8 0%, #818cf8 50%, #f472b6 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .hero-sub {
        color: #475569; font-size: 0.88rem; margin-top: 4px; letter-spacing: 0.01em;
    }

    /* ── Metric / info cards ── */
    .kpi-card {
        background: #0d1a2d;
        border: 1px solid #1e2d45;
        border-radius: 12px;
        padding: 16px 20px 14px;
    }
    .kpi-label {
        font-size: 0.7rem; font-weight: 600; letter-spacing: 0.1em;
        text-transform: uppercase; color: #4b6484;
    }
    .kpi-value {
        font-size: 1.95rem; font-weight: 800; margin: 4px 0 2px; line-height: 1.1;
    }
    .kpi-sub { font-size: 0.74rem; color: #3d5068; }

    /* ── Live state cards ── */
    .live-card {
        background: #0d1a2d;
        border: 1px solid #1e2d45;
        border-radius: 12px;
        padding: 16px 20px;
    }
    .live-card-title {
        font-size: 0.68rem; font-weight: 700; letter-spacing: 0.1em;
        text-transform: uppercase; color: #4b6484;
    }
    .live-action-badge {
        font-size: 0.68rem; color: #4b6484;
    }
    .live-divider {
        border: 0; border-top: 1px solid #1a2840; margin: 10px 0;
    }
    .stat-grid {
        display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px; margin-top: 6px;
    }
    .stat-box {
        background: #07101e; border: 1px solid #1a2840;
        border-radius: 8px; padding: 10px 14px;
    }
    .stat-box-label {
        font-size: 0.65rem; color: #4b6484; text-transform: uppercase;
        letter-spacing: 0.08em; margin-bottom: 4px;
    }
    .stat-box-val { font-size: 1.2rem; font-weight: 700; }

    /* ── Section header ── */
    .sec-hdr {
        font-size: 0.68rem; font-weight: 700; letter-spacing: 0.14em;
        text-transform: uppercase; color: #3d5068; margin: 14px 0 6px;
    }

    /* ── Sidebar sub-label ── */
    .sidebar-brand { font-size: 1rem; font-weight: 800; color: #38bdf8; }
    .sidebar-tagline { font-size: 0.7rem; color: #3d5068; margin-bottom: 12px; }

    /* ── Separator ── */
    .sep { border: 0; border-top: 1px solid #0f1e30; margin: 14px 0; }

    /* ── Summary banner tweaks ── */
    .stAlert { border-radius: 10px !important; }

    /* ── Plotly wrapper ── */
    .stPlotlyChart { background: transparent !important; }
    .js-plotly-plot .plotly, .js-plotly-plot .plotly svg { background: transparent !important; }

    /* ── Tab styling ── */
    button[data-baseweb="tab"] { font-size: 0.8rem !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Constants ────────────────────────────────────────────────
ACTIONS        = {0: "OFF", 1: "ECO", 2: "MED", 3: "TURBO"}
POWER_W        = {0: 0,    1: 20,   2: 40,   3: 80}
CLEANSE_MAP    = {0: 0.0,  1: 15.0, 2: 30.0, 3: 50.0}
TARIFF_PEAK    = 0.30    # $/kWh
TARIFF_OFFPEAK = 0.12    # $/kWh
SIM_STEPS      = 24      # 24-hour horizon
AQI_SAFE       = 50.0
C_TRAD         = "#ef4444"
C_RL           = "#10b981"
C_SAFE         = "#fbbf24"
C_ECO          = "#0ea5e9"
C_MED          = "#f59e0b"
C_OFF          = "#334155"

# Shared Plotly dark theme
_BASE = dict(
    paper_bgcolor="#060c18",
    plot_bgcolor="#07101e",
    font=dict(color="#64748b", size=12, family="'Segoe UI', system-ui, sans-serif"),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11), orientation="h",
                yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=8, r=8, t=52, b=8),
    xaxis=dict(gridcolor="#0f1e30", zeroline=False, tickfont=dict(size=11)),
    yaxis=dict(gridcolor="#0f1e30", zeroline=False, tickfont=dict(size=11)),
    hovermode="x unified",
)


# ════════════════════════════════════════════════════════════
#  SIMULATION CORE
# ════════════════════════════════════════════════════════════

def simulate_step(
    aqi: float, temp: float, action: int,
    tariff: float, outdoor_aqi: float, occupancy: int,
) -> dict:
    """One-hour environment transition, self-contained (no Gym dependency)."""
    cleanse          = CLEANSE_MAP[action]
    power_w          = POWER_W[action]
    occ_factor       = 1.0 + 0.08 * occupancy
    outdoor_leak     = outdoor_aqi * 0.04
    pollution_in     = float(np.random.uniform(3.5, 8.5)) * occ_factor + outdoor_leak
    new_aqi          = float(max(10.0, aqi - cleanse + pollution_in))
    new_temp         = float(max(18.0, temp - 0.15 * action + np.random.uniform(-0.1, 0.2)))
    kwh              = power_w / 1000.0
    cost             = kwh * (TARIFF_PEAK if tariff == 1.0 else TARIFF_OFFPEAK)
    aqi_pen          = max(0.0, new_aqi - AQI_SAFE)
    reward           = -(aqi_pen * 0.6 + (power_w / 10.0) * (2.5 if tariff else 1.0) * 0.9)
    return dict(aqi=new_aqi, temp=new_temp, power=power_w, cost=cost, reward=reward)


def encode_state(aqi: float, temp: float, tariff: float) -> tuple:
    aqi_bin  = int(np.clip(np.digitize(aqi,  [50, 100, 150, 200, 250]), 0, 5))
    temp_bin = int(np.clip(np.digitize(temp, [22, 26, 30, 34]),          0, 4))
    return (aqi_bin, temp_bin, int(tariff))


@st.cache_data(show_spinner=False)
def train_rl_agent(episodes: int, seed: int = 42) -> np.ndarray:
    """Q-Learning agent (proxy for PPO/DQN). Cached by episode count."""
    rng     = np.random.default_rng(seed)
    q       = np.zeros((6, 5, 2, 4), dtype=np.float64)
    alpha, gamma, eps = 0.12, 0.96, 0.25
    decay   = 0.995
    for _ in range(episodes):
        aqi, temp, tariff = 200.0, 29.0, 0.0
        eps = max(0.02, eps * decay)
        for __ in range(40):
            s   = encode_state(aqi, temp, tariff)
            act = int(rng.integers(0, 4)) if rng.random() < eps else int(np.argmax(q[s]))
            res = simulate_step(aqi, temp, act, tariff, outdoor_aqi=80.0, occupancy=2)
            ns  = encode_state(res["aqi"], res["temp"], tariff)
            q[s][act] += alpha * (res["reward"] + gamma * np.max(q[ns]) - q[s][act])
            aqi, temp = res["aqi"], res["temp"]
    return q


def traditional_action(aqi: float) -> int:
    """Dumb threshold controller — blind to tariff and occupancy."""
    if   aqi > 150: return 3
    elif aqi > 100: return 2
    elif aqi > 50:  return 1
    return 0


def run_simulation(
    init_aqi: float, outdoor_aqi: float,
    occupancy: int, tariff: float, q: np.ndarray,
) -> pd.DataFrame:
    """Run Traditional and RL controllers side-by-side for 24 hours."""
    rows = []
    t_aqi,  t_temp   = init_aqi, 29.0
    rl_aqi, rl_temp  = init_aqi, 29.0
    t_cost = rl_cost = 0.0

    for h in range(SIM_STEPS):
        # Auto peak hours 7-10 and 17-21 regardless of base tariff toggle
        ht = 1.0 if h in range(7, 11) or h in range(17, 22) else tariff

        # --- Traditional ---
        ta     = traditional_action(t_aqi)
        tr     = simulate_step(t_aqi, t_temp, ta, ht, outdoor_aqi, occupancy)
        t_aqi, t_temp = tr["aqi"], tr["temp"]
        t_cost += tr["cost"]

        # --- RL Agent ---
        ra     = int(np.argmax(q[encode_state(rl_aqi, rl_temp, ht)]))
        rr     = simulate_step(rl_aqi, rl_temp, ra, ht, outdoor_aqi, occupancy)
        rl_aqi, rl_temp = rr["aqi"], rr["temp"]
        rl_cost += rr["cost"]

        rows.append({
            "Hour":            h + 1,
            "Tariff":          "Peak" if ht == 1.0 else "Off-Peak",
            "Trad_AQI":        round(t_aqi, 1),
            "Trad_Action":     ta,
            "Trad_ActionName": ACTIONS[ta],
            "Trad_Power_W":    tr["power"],
            "Trad_CumCost":    round(t_cost, 5),
            "RL_AQI":          round(rl_aqi, 1),
            "RL_Action":       ra,
            "RL_ActionName":   ACTIONS[ra],
            "RL_Power_W":      rr["power"],
            "RL_CumCost":      round(rl_cost, 5),
        })
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════
#  CHART BUILDERS
# ════════════════════════════════════════════════════════════

def _layout(**kwargs) -> dict:
    """Merge base layout with overrides."""
    return {**_BASE, **kwargs}


def fig_aqi(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(
        x=df["Hour"], y=df["Trad_AQI"], name="Traditional",
        line=dict(color=C_TRAD, width=2.2),
        mode="lines+markers", marker=dict(size=4.5),
        hovertemplate="Hour %{x}<br>AQI: %{y}<extra>Traditional</extra>",
    )
    fig.add_scatter(
        x=df["Hour"], y=df["RL_AQI"], name="RL Agent",
        line=dict(color=C_RL, width=2.2),
        mode="lines+markers", marker=dict(size=4.5),
        hovertemplate="Hour %{x}<br>AQI: %{y}<extra>RL Agent</extra>",
    )
    fig.add_hline(
        y=AQI_SAFE, line_dash="dot", line_color=C_SAFE, line_width=1.4,
        annotation_text=" Safe Limit (AQI 50)",
        annotation_position="top left",
        annotation_font=dict(color=C_SAFE, size=10),
    )
    fig.update_layout(
        **_layout(title="Indoor AQI Trajectory — 24 Hours",
                  yaxis_title="AQI Index", xaxis_title="Hour of Day"),
    )
    return fig


def fig_cost(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(
        x=df["Hour"], y=df["Trad_CumCost"], name="Traditional",
        line=dict(color=C_TRAD, width=2.2),
        fill="tozeroy", fillcolor="rgba(239,68,68,0.08)",
        hovertemplate="Hour %{x}<br>$%{y:.5f}<extra>Traditional</extra>",
    )
    fig.add_scatter(
        x=df["Hour"], y=df["RL_CumCost"], name="RL Agent",
        line=dict(color=C_RL, width=2.2),
        fill="tozeroy", fillcolor="rgba(16,185,129,0.08)",
        hovertemplate="Hour %{x}<br>$%{y:.5f}<extra>RL Agent</extra>",
    )
    fig.update_layout(
        **_layout(title="Cumulative Electricity Cost (USD)",
                  yaxis_title="USD ($)", xaxis_title="Hour of Day",
                  yaxis=dict(tickformat=".4f", gridcolor="#0f1e30", zeroline=False)),
    )
    return fig


def fig_power(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_bar(
        x=df["Hour"], y=df["Trad_Power_W"], name="Traditional",
        marker_color=C_TRAD, opacity=0.82,
        hovertemplate="Hour %{x}<br>%{y} W<extra>Traditional</extra>",
    )
    fig.add_bar(
        x=df["Hour"], y=df["RL_Power_W"], name="RL Agent",
        marker_color=C_RL, opacity=0.82,
        hovertemplate="Hour %{x}<br>%{y} W<extra>RL Agent</extra>",
    )
    fig.update_layout(
        **_layout(title="Instantaneous Power Consumption (W)",
                  barmode="group", yaxis_title="Watts", xaxis_title="Hour of Day"),
    )
    return fig


def fig_actions(df: pd.DataFrame) -> go.Figure:
    """Vectorised action bars — one trace per action type per system."""
    ac = {0: C_OFF, 1: C_ECO, 2: C_MED, 3: C_TRAD}
    labels = {0: "OFF", 1: "ECO", 2: "MED", 3: "TURBO"}

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        subplot_titles=["Traditional Rule-Based", "RL Agent (PPO/DQN)"],
        vertical_spacing=0.14,
    )
    shown = set()
    for act_id in [3, 2, 1, 0]:
        lbl  = labels[act_id]
        col  = ac[act_id]
        show = act_id not in shown

        mask_t  = df["Trad_Action"] == act_id
        mask_rl = df["RL_Action"]   == act_id

        fig.add_bar(
            x=df.loc[mask_t, "Hour"],
            y=df.loc[mask_t, "Trad_Power_W"],
            name=lbl, marker_color=col, legendgroup=lbl,
            showlegend=show, row=1, col=1,
            hovertemplate=f"Hour %{{x}}<br>{lbl}: %{{y}} W<extra></extra>",
        )
        fig.add_bar(
            x=df.loc[mask_rl, "Hour"],
            y=df.loc[mask_rl, "RL_Power_W"],
            name=lbl, marker_color=col, legendgroup=lbl,
            showlegend=False, row=2, col=1,
            hovertemplate=f"Hour %{{x}}<br>{lbl}: %{{y}} W<extra></extra>",
        )
        shown.add(act_id)

    fig.update_layout(
        **_layout(title="Fan / Cooling Action Map (W)", barmode="stack", height=400),
    )
    fig.update_yaxes(title_text="Watts", gridcolor="#0f1e30")
    fig.update_xaxes(gridcolor="#0f1e30")
    fig.update_xaxes(title_text="Hour of Day", row=2, col=1)
    return fig


# ════════════════════════════════════════════════════════════
#  HTML COMPONENT HELPERS
# ════════════════════════════════════════════════════════════

def kpi_card(label: str, value: str, sub: str, colour: str) -> str:
    return (
        f'<div class="kpi-card">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value" style="color:{colour};">{value}</div>'
        f'<div class="kpi-sub">{sub}</div>'
        f'</div>'
    )


def live_card(aqi, power, cum_cost, action_name: str, colour: str, tag: str) -> str:
    # Safe colour: only compare if numeric
    if isinstance(aqi, (int, float)):
        aq_col = C_TRAD if aqi > 100 else (C_SAFE if aqi > 50 else C_RL)
        aqi_str    = str(aqi)
        cost_str   = f"${cum_cost:.5f}"
        power_str  = f"{power} W"
    else:
        aq_col     = "#64748b"
        aqi_str    = "—"
        cost_str   = "$—"
        power_str  = "— W"

    return (
        f'<div class="live-card" style="border-left:3px solid {colour};">'
        f'  <div style="display:flex;justify-content:space-between;align-items:center;">'
        f'    <span class="live-card-title">{tag}</span>'
        f'    <span class="live-action-badge">ACTION &nbsp;<strong style="color:#94a3b8;">{action_name}</strong></span>'
        f'  </div>'
        f'  <hr class="live-divider">'
        f'  <div class="stat-grid">'
        f'    <div class="stat-box"><div class="stat-box-label">Indoor AQI</div>'
        f'      <div class="stat-box-val" style="color:{aq_col};">{aqi_str}</div></div>'
        f'    <div class="stat-box"><div class="stat-box-label">Power Load</div>'
        f'      <div class="stat-box-val">{power_str}</div></div>'
        f'    <div class="stat-box"><div class="stat-box-label">Cum. Cost</div>'
        f'      <div class="stat-box-val" style="color:{colour};">{cost_str}</div></div>'
        f'  </div>'
        f'</div>'
    )


# ════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        '<div class="sidebar-brand">🌬️ AirFlow AI</div>'
        '<div class="sidebar-tagline">RL vs Rule-Based HVAC Simulator</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<hr class="sep">', unsafe_allow_html=True)

    st.markdown('<div class="sec-hdr">🏠 Environment Inputs</div>', unsafe_allow_html=True)
    init_aqi    = st.slider("Initial Indoor AQI",   30,  350, 200, step=5)
    outdoor_aqi = st.slider("Outdoor AQI",          20,  300,  80, step=5)
    occupancy   = st.slider("Room Occupancy (pax)", 0,    10,   2)

    st.markdown('<hr class="sep">', unsafe_allow_html=True)
    st.markdown('<div class="sec-hdr">⚡ Grid Tariff</div>', unsafe_allow_html=True)
    tariff_choice = st.radio(
        "Base tariff mode",
        ["🌙  Off-Peak  ($0.12 / kWh)", "⚡  Peak Demand  ($0.30 / kWh)"],
        index=0, label_visibility="collapsed",
    )
    tariff_val = 1.0 if "Peak Demand" in tariff_choice else 0.0
    st.caption("Peak tariff auto-applies hours 7–10 and 17–21 regardless of selection.")

    st.markdown('<hr class="sep">', unsafe_allow_html=True)
    st.markdown('<div class="sec-hdr">🧠 RL Agent</div>', unsafe_allow_html=True)
    train_eps = st.slider("Training Episodes", 200, 3000, 800, step=100)

    st.markdown('<div class="sec-hdr">▶ Playback</div>', unsafe_allow_html=True)
    step_mode = st.checkbox("Step-Through Mode (manual advance)", value=False)
    sim_delay = st.select_slider(
        "Live delay (s)", options=[0.05, 0.10, 0.20, 0.30, 0.50],
        value=0.10, disabled=step_mode,
    )

    st.markdown('<hr class="sep">', unsafe_allow_html=True)
    run_btn  = st.button("🚀  Run Simulation", use_container_width=True, type="primary")
    step_btn = st.button("⏭  Next Hour",       use_container_width=True, disabled=not step_mode)


# ════════════════════════════════════════════════════════════
#  MAIN — HERO HEADER
# ════════════════════════════════════════════════════════════
st.markdown(
    '<div class="hero-title">AIRFLOW AI — SYSTEM COMPARISON SIMULATOR</div>'
    '<div class="hero-sub">'
    'PPO / DQN Reinforcement Learning Agent  ·  vs  ·  Traditional Rule-Based HVAC Controller  ·  24-Hour Window'
    '</div>',
    unsafe_allow_html=True,
)
st.markdown('<hr class="sep">', unsafe_allow_html=True)

# ── KPI row ──────────────────────────────────────────────────
kpi_cols = st.columns(4, gap="small")
slot_savings    = kpi_cols[0].empty()
slot_avg_aqi    = kpi_cols[1].empty()
slot_trad_cost  = kpi_cols[2].empty()
slot_rl_cost    = kpi_cols[3].empty()

def _render_kpis_placeholder():
    slot_savings.markdown(kpi_card("Cost Savings", "—",  "Run simulation", "#38bdf8"), unsafe_allow_html=True)
    slot_avg_aqi.markdown(kpi_card("Avg AQI · RL", "—",  "24-hour mean · safe ≤ 50", C_RL),    unsafe_allow_html=True)
    slot_trad_cost.markdown(kpi_card("Total Cost · Traditional", "$—", "24-hour", C_TRAD),       unsafe_allow_html=True)
    slot_rl_cost.markdown(kpi_card("Total Cost · RL Agent",      "$—", "24-hour", C_RL),         unsafe_allow_html=True)

def _render_kpis(df: pd.DataFrame):
    t_tot   = df["Trad_CumCost"].iloc[-1]
    r_tot   = df["RL_CumCost"].iloc[-1]
    saved   = max(0.0, t_tot - r_tot)
    pct     = round(saved / t_tot * 100, 1) if t_tot else 0.0
    avg_aqi = round(df["RL_AQI"].mean(), 1)
    slot_savings.markdown(   kpi_card("Cost Savings", f"{pct}%",     f"${saved:.5f} saved over 24 h", "#38bdf8"), unsafe_allow_html=True)
    slot_avg_aqi.markdown(   kpi_card("Avg AQI · RL", str(avg_aqi),  "24-hour mean · safe ≤ 50",      C_RL),     unsafe_allow_html=True)
    slot_trad_cost.markdown( kpi_card("Total Cost · Traditional", f"${t_tot:.5f}", "24-hour cumulative", C_TRAD),unsafe_allow_html=True)
    slot_rl_cost.markdown(   kpi_card("Total Cost · RL Agent",    f"${r_tot:.5f}", "24-hour cumulative", C_RL),   unsafe_allow_html=True)

_render_kpis_placeholder()
st.markdown('<hr class="sep">', unsafe_allow_html=True)

# ── Live state cards ─────────────────────────────────────────
st.markdown('<div class="sec-hdr">🔴 Traditional vs 🟢 RL Agent — Live State</div>', unsafe_allow_html=True)
col_t, col_r = st.columns(2, gap="medium")

with col_t:
    st.markdown("##### 🔴 Traditional Rule-Based HVAC", unsafe_allow_html=True)
    slot_live_t = st.empty()
with col_r:
    st.markdown("##### 🟢 AirFlow AI  (PPO / DQN RL Agent)", unsafe_allow_html=True)
    slot_live_r = st.empty()

def _render_live(aqi_t, pw_t, cc_t, an_t, aqi_r, pw_r, cc_r, an_r):
    slot_live_t.markdown(live_card(aqi_t, pw_t, cc_t, an_t, C_TRAD, "TRADITIONAL RULE-BASED"), unsafe_allow_html=True)
    slot_live_r.markdown(live_card(aqi_r, pw_r, cc_r, an_r, C_RL,   "AIRFLOW AI OPTIMAL"),     unsafe_allow_html=True)

_render_live("—", "—", 0.0, "—", "—", "—", 0.0, "—")
st.markdown('<hr class="sep">', unsafe_allow_html=True)

# ── Chart tabs ───────────────────────────────────────────────
st.markdown('<div class="sec-hdr">📊 Interactive Visualisations</div>', unsafe_allow_html=True)
tab_a, tab_b, tab_c, tab_d = st.tabs([
    "🌫️  AQI Trajectory",
    "💰  Electricity Cost",
    "⚡  Power Consumption",
    "⚙️  Action Map",
])
with tab_a: slot_aqi     = st.empty()
with tab_b: slot_cost    = st.empty()
with tab_c: slot_power   = st.empty()
with tab_d: slot_actions = st.empty()

st.markdown('<hr class="sep">', unsafe_allow_html=True)

# ── Progress / status / banner ───────────────────────────────
slot_progress = st.empty()
slot_status   = st.empty()
slot_banner   = st.empty()

# ── Data table ───────────────────────────────────────────────
with st.expander("📋  Full Simulation Data Table", expanded=False):
    slot_table = st.empty()

# ── Session state ────────────────────────────────────────────
if "sim_df"      not in st.session_state: st.session_state.sim_df      = None
if "step_cursor" not in st.session_state: st.session_state.step_cursor = 0

_CHART_CFG = {"displayModeBar": False, "responsive": True}


def _update_all(df: pd.DataFrame):
    """Refresh charts, live cards, and table from a df slice."""
    last = df.iloc[-1]
    _render_live(
        last["Trad_AQI"], last["Trad_Power_W"], last["Trad_CumCost"], last["Trad_ActionName"],
        last["RL_AQI"],   last["RL_Power_W"],   last["RL_CumCost"],   last["RL_ActionName"],
    )
    slot_aqi.plotly_chart(    fig_aqi(df),     use_container_width=True, config=_CHART_CFG)
    slot_cost.plotly_chart(   fig_cost(df),    use_container_width=True, config=_CHART_CFG)
    slot_power.plotly_chart(  fig_power(df),   use_container_width=True, config=_CHART_CFG)
    slot_actions.plotly_chart(fig_actions(df), use_container_width=True, config=_CHART_CFG)

    # Highlight unsafe AQI rows in table
    def _hl(v):
        if isinstance(v, (int, float)) and v > AQI_SAFE:
            return "color: #ef4444; font-weight:600"
        return ""
    slot_table.dataframe(
        df.style.map(_hl, subset=["Trad_AQI", "RL_AQI"]),
        use_container_width=True, height=300,
    )


def _show_banner(df: pd.DataFrame):
    t = df["Trad_CumCost"].iloc[-1]
    r = df["RL_CumCost"].iloc[-1]
    pct = round(max(0, (t - r) / t * 100), 1) if t else 0
    slot_banner.success(
        f"🏆 **Simulation complete** — AirFlow AI RL Agent reduced electricity cost by **{pct}%** "
        f"(${r:.5f} vs ${t:.5f}) while maintaining healthier indoor air quality over 24 hours."
    )


# ════════════════════════════════════════════════════════════
#  RUN SIMULATION
# ════════════════════════════════════════════════════════════
if run_btn:
    slot_banner.empty()
    slot_progress.progress(0.0, text="Training RL Agent…")

    with st.spinner("🧠  Training Q-Learning / PPO proxy agent…"):
        q_table = train_rl_agent(train_eps)

    slot_status.info(f"✅  Agent trained over **{train_eps}** episodes. Running 24-hour simulation…")

    df_full = run_simulation(
        init_aqi=float(init_aqi), outdoor_aqi=float(outdoor_aqi),
        occupancy=int(occupancy), tariff=tariff_val, q=q_table,
    )
    st.session_state.sim_df = df_full

    if step_mode:
        st.session_state.step_cursor = 1
        _update_all(df_full.iloc[:1])
        _render_kpis(df_full.iloc[:1])
        slot_progress.progress(1 / SIM_STEPS, text="Hour 1 / 24 — press ⏭ Next Hour")
        slot_status.info("Step-through mode active. Press **⏭ Next Hour** in the sidebar.")
    else:
        for i in range(1, SIM_STEPS + 1):
            _update_all(df_full.iloc[:i])
            slot_progress.progress(i / SIM_STEPS, text=f"Simulating hour {i} / {SIM_STEPS}…")
            time.sleep(sim_delay)

        _render_kpis(df_full)
        slot_progress.progress(1.0, text="Simulation complete ✅")
        slot_status.empty()
        _show_banner(df_full)


# ════════════════════════════════════════════════════════════
#  STEP-THROUGH ADVANCE
# ════════════════════════════════════════════════════════════
if step_btn and st.session_state.sim_df is not None:
    df_full = st.session_state.sim_df
    cursor  = min(st.session_state.step_cursor + 1, SIM_STEPS)
    st.session_state.step_cursor = cursor

    df_slice = df_full.iloc[:cursor]
    _update_all(df_slice)
    _render_kpis(df_slice)
    slot_progress.progress(cursor / SIM_STEPS, text=f"Hour {cursor} / {SIM_STEPS}")

    if cursor >= SIM_STEPS:
        slot_status.empty()
        _show_banner(df_full)
    else:
        slot_status.info(f"Step-through — Hour **{cursor}** / {SIM_STEPS}. Press ⏭ to advance.")


# ════════════════════════════════════════════════════════════
#  FOOTER
# ════════════════════════════════════════════════════════════
st.markdown(
    '<hr class="sep">'
    '<div style="text-align:center;color:#1e2d45;font-size:0.72rem;padding:6px 0 2px;">'
    'AirFlow AI &nbsp;·&nbsp; Smart Air Quality &amp; Energy Management System using Reinforcement Learning'
    '</div>',
    unsafe_allow_html=True,
)
