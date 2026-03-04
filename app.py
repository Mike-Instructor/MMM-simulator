import numpy as np
import streamlit as st
import plotly.graph_objects as go
from scipy.optimize import minimize

# ============================================================
# INSTRUCTOR CONTROLS
# ============================================================

# Revenue per unit sold (explicit, easy to change)
REVENUE_PER_UNIT = 25.0  # dollars per unit

# To avoid the trivial "spend $0 => infinite ROAS" solution, require optimizer
# to spend at least this fraction of the budget when maximizing ROAS.
MIN_SPEND_FRACTION_FOR_ROAS_OPT = 0.20  # 20% of budget (tweak as desired)

# Hide only the "Optimize ROAS" button (set False to show it, e.g. for instructor).
HIDE_ROAS_BUTTON = True

# ------------------------------------------------------------
# 5 SCENARIOS (same hill_n across channels within each scenario)
# ------------------------------------------------------------
SCENARIOS = {

    "Mixed World": {
        "Search": {"scale": 820.0, "half_sat": 690.0, "hill_n": 0.45},
        "Paid Social": {"scale": 490.0, "half_sat": 3000.0, "hill_n": 0.95},
        "Display": {"scale": 790.0, "half_sat": 1290.0, "hill_n": 0.9},    
    },

    "Spiky World (very steep early lift)": {
        "Search":      {"scale": 680.0, "half_sat": 280.0, "hill_n": 0.55},
        "Paid Social": {"scale": 660.0, "half_sat": 500.0, "hill_n": 0.55},
        "Display":     {"scale": 640.0, "half_sat": 820.0, "hill_n": 0.55},
    },
    "Punchy World (strong early lift)": {
        "Search":      {"scale": 700.0, "half_sat": 300.0, "hill_n": 0.70},
        "Paid Social": {"scale": 690.0, "half_sat": 540.0, "hill_n": 0.70},
        "Display":     {"scale": 680.0, "half_sat": 880.0, "hill_n": 0.70},
    },
    "Classic Diminishing Returns": {
        "Search":      {"scale": 700.0, "half_sat": 320.0, "hill_n": 0.85},
        "Paid Social": {"scale": 700.0, "half_sat": 580.0, "hill_n": 0.85},
        "Display":     {"scale": 700.0, "half_sat": 950.0, "hill_n": 0.85},
    },
    "Smooth World (gradual response)": {
        "Search":      {"scale": 700.0, "half_sat": 420.0, "hill_n": 0.90},
        "Paid Social": {"scale": 710.0, "half_sat": 780.0, "hill_n": 0.90},
        "Display":     {"scale": 720.0, "half_sat": 1200.0, "hill_n": 0.90},
    },

    "Near-Linear World (very smooth saturation)": {
        "Search":      {"scale": 700.0, "half_sat": 650.0, "hill_n": 1.00},
        "Paid Social": {"scale": 710.0, "half_sat": 1100.0, "hill_n": 1.00},
        "Display":     {"scale": 720.0, "half_sat": 1700.0, "hill_n": 1.00},
    },

    "Channel Collapse (Display completely dominated)": {
        "Search":      {"scale": 800.0, "half_sat": 250.0, "hill_n": 0.80},
        "Paid Social": {"scale": 750.0, "half_sat": 500.0, "hill_n": 0.80},
        "Display":     {"scale": 120.0, "half_sat": 2500.0, "hill_n": 0.80},
    },
}

DEFAULT_SCENARIO = "Classic Diminishing Returns"
CHANNELS = SCENARIOS[DEFAULT_SCENARIO]

APP_BUDGET_MIN = 500
APP_BUDGET_MAX = 5000

# ✅ Only change requested: budget slider increments of $500
APP_BUDGET_STEP = 500

SPEND_STEP = 10


# ============================================================
# MODEL
# ============================================================

def contribution(spend: float, scale: float, half_sat: float, hill_n: float) -> float:
    spend = max(0.0, float(spend))
    scale = max(0.0, float(scale))
    half_sat = max(1e-6, float(half_sat))
    hill_n = max(1e-6, float(hill_n))
    s_n = spend ** hill_n
    h_n = half_sat ** hill_n
    return float(scale * (s_n / (s_n + h_n)))


def marginal_units(ch: str, current_spend: float, delta: float) -> float:
    """Approx marginal units gained by adding delta spend to channel ch."""
    params = CHANNELS[ch]
    return contribution(current_spend + delta, **params) - contribution(current_spend, **params)


def spend_key(ch: str) -> str:
    return f"spend_{ch.replace(' ', '_')}"


def get_spend(ch: str) -> float:
    return float(st.session_state.get(spend_key(ch), 0.0))


def set_spend(ch: str, v: float):
    st.session_state[spend_key(ch)] = int(v)


def total_units_from_spends(spends: np.ndarray, ch_names: list[str]) -> float:
    return float(sum(contribution(spends[i], **CHANNELS[ch_names[i]]) for i in range(len(ch_names))))


def max_sales_allocation(budget: float) -> tuple[np.ndarray, float]:
    """
    Maximize total units subject to:
      spends >= 0
      sum(spends) = budget
    """
    ch_names = list(CHANNELS.keys())
    n = len(ch_names)
    budget = float(max(0.0, budget))
    if budget <= 0:
        return np.zeros(n), 0.0

    def objective(x):
        x = np.maximum(x, 0.0)
        return -total_units_from_spends(x, ch_names)

    cons = [{"type": "eq", "fun": lambda x: float(np.sum(np.maximum(x, 0.0)) - budget)}]
    bounds = [(0.0, budget) for _ in range(n)]
    x0 = np.ones(n) * (budget / n)

    res = minimize(objective, x0=x0, method="SLSQP", bounds=bounds, constraints=cons, options={"maxiter": 400})
    if not res.success:
        # fallback: best single-channel
        best_units = -1.0
        best_x = None
        for j in range(n):
            x = np.zeros(n)
            x[j] = budget
            u = total_units_from_spends(x, ch_names)
            if u > best_units:
                best_units, best_x = u, x
        return best_x, float(best_units)

    x_best = np.maximum(res.x, 0.0)
    u_best = total_units_from_spends(x_best, ch_names)
    return x_best, float(u_best)


def max_roas_allocation(budget: float) -> tuple[np.ndarray, float]:
    """
    Maximize ROAS = revenue/spend = (units*rev_per_unit)/spend
    subject to:
      spends >= 0
      sum(spends) <= budget
      sum(spends) >= min_spend (avoid trivial spend=0 solution)
    """
    ch_names = list(CHANNELS.keys())
    n = len(ch_names)
    budget = float(max(0.0, budget))
    if budget <= 0:
        return np.zeros(n), 0.0

    min_spend = max(float(SPEND_STEP), budget * float(MIN_SPEND_FRACTION_FOR_ROAS_OPT))

    def roas(x):
        x = np.maximum(x, 0.0)
        spend = float(np.sum(x))
        units = total_units_from_spends(x, ch_names)
        revenue = units * REVENUE_PER_UNIT
        return revenue / spend if spend > 0 else 0.0

    def objective(x):
        return -roas(x)

    cons = [
        {"type": "ineq", "fun": lambda x: float(budget - np.sum(np.maximum(x, 0.0)))},    # sum <= budget
        {"type": "ineq", "fun": lambda x: float(np.sum(np.maximum(x, 0.0)) - min_spend)}  # sum >= min_spend
    ]
    bounds = [(0.0, budget) for _ in range(n)]
    x0 = np.ones(n) * (min_spend / n)

    res = minimize(objective, x0=x0, method="SLSQP", bounds=bounds, constraints=cons, options={"maxiter": 500})
    if not res.success:
        # fallback: try min_spend all in best ROAS channel
        best_r = -1.0
        best_x = None
        for j in range(n):
            x = np.zeros(n)
            x[j] = min_spend
            r = roas(x)
            if r > best_r:
                best_r, best_x = r, x
        return best_x, float(best_r)

    x_best = np.maximum(res.x, 0.0)
    r_best = roas(x_best)
    return x_best, float(r_best)


def apply_allocation_vector(x: np.ndarray, force_full_budget: bool = True):
    """
    Apply spend vector to the three spend widgets (rounded to step).
    If force_full_budget=True, redistribute leftover (due to rounding)
    so total spend == budget (within SPEND_STEP granularity).
    """
    ch_names = list(CHANNELS.keys())
    budget = float(st.session_state.budget)

    # 1) Floor to step
    spends = []
    for i, ch in enumerate(ch_names):
        v = float(max(0.0, x[i]))
        v = float(np.floor(v / SPEND_STEP) * SPEND_STEP)
        spends.append(v)

    spends = np.array(spends, dtype=float)

    # 2) If requested, top up leftover budget by allocating step-by-step
    if force_full_budget and budget > 0:
        total = float(spends.sum())
        leftover = float(np.floor((budget - total) / SPEND_STEP) * SPEND_STEP)

        while leftover >= SPEND_STEP - 1e-9:
            best_ch_idx = None
            best_gain = -1e18

            for i, ch in enumerate(ch_names):
                if spends[i] + SPEND_STEP > budget:
                    continue
                gain = marginal_units(ch, spends[i], SPEND_STEP)
                if gain > best_gain:
                    best_gain = gain
                    best_ch_idx = i

            if best_ch_idx is None:
                break

            spends[best_ch_idx] += SPEND_STEP
            leftover -= SPEND_STEP

    # 3) Write to session state
    for i, ch in enumerate(ch_names):
        set_spend(ch, spends[i])


# ============================================================
# STREAMLIT STATE INIT
# ============================================================

st.set_page_config(page_title="Media Mix Allocation", layout="wide")

st.markdown(
    """
    <style>
    /* Reduce top whitespace in the main page container */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# ✅ URL QUERY PARAMS (Instructor flag + optional scenario/budget)
#
# Usage examples:
#   ?Instructor               -> show instructor tools (presence-only)
#   ?Instructor=1             -> show instructor tools (also works)
#   ?scenario=Smooth%20World%20(gradual%20response)&budget=3000&Instructor
# ============================================================

def _is_truthy(v) -> bool:
    if v is None:
        return False
    s = str(v).strip().lower()
    return s in {"1", "true", "yes", "y", "on"}

def apply_query_params_once():
    qp = st.query_params  # Streamlit's query param interface

    if st.session_state.get("_qp_applied", False):
        return

    # Presence-only "Instructor" flag:
    # - If "Instructor" exists in URL at all, enable tools (ignore value).
    # - Also allow common truthy values.
    instructor_present = ("Instructor" in qp) or ("instructor" in qp)
    instructor_truthy = _is_truthy(qp.get("Instructor", None)) or _is_truthy(qp.get("instructor", None))
    st.session_state.show_instructor_tools = bool(instructor_present or instructor_truthy)

    # Optional scenario override via URL
    scenario = qp.get("scenario", None)
    if scenario and scenario in SCENARIOS:
        st.session_state.scenario_name = scenario

    # Optional budget override via URL
    budget = qp.get("budget", None)
    if budget is not None:
        try:
            b = int(float(budget))
            b = max(APP_BUDGET_MIN, min(APP_BUDGET_MAX, b))
            b = int(round(b / APP_BUDGET_STEP) * APP_BUDGET_STEP)  # snap to step
            st.session_state.budget = b
            st.session_state["budget_slider"] = b
        except ValueError:
            pass

    st.session_state["_qp_applied"] = True

apply_query_params_once()


if "budget" not in st.session_state:
    st.session_state.budget = 2500

# Widget keys must be initialized BEFORE widgets, and then widgets should NOT pass value=
if "budget_slider" not in st.session_state:
    st.session_state["budget_slider"] = int(st.session_state.budget)

# Scenario selection state
if "scenario_name" not in st.session_state:
    st.session_state.scenario_name = DEFAULT_SCENARIO

# Apply scenario at startup (ensures CHANNELS matches session selection)
CHANNELS = SCENARIOS[st.session_state.scenario_name]

# Ensure spend keys exist
for ch in CHANNELS:
    k = spend_key(ch)
    if k not in st.session_state:
        st.session_state[k] = 0

# Axis ranges fixed until budget changes
if "axis_budget" not in st.session_state:
    st.session_state.axis_budget = None
if "sales_axis_max" not in st.session_state:
    st.session_state.sales_axis_max = None
if "roas_axis_max" not in st.session_state:
    st.session_state.roas_axis_max = None


def recompute_axes_for_budget():
    b = float(st.session_state.budget)

    # Sales axis max (optimal sales under full budget)
    _, max_units = max_sales_allocation(b)
    st.session_state.sales_axis_max = float(max_units)

    # ROAS axis max (optimal ROAS under budget with min spend rule)
    _, max_roas = max_roas_allocation(b)
    st.session_state.roas_axis_max = float(max_roas)

    st.session_state.axis_budget = float(b)


def on_budget_change():
    st.session_state.budget = int(st.session_state["budget_slider"])

    # Reset allocations to 0 whenever budget changes
    for ch in CHANNELS:
        st.session_state[spend_key(ch)] = 0

    # Recompute axes only when budget changes
    recompute_axes_for_budget()


def on_scenario_change():
    """Apply scenario preset and reset spends + axes (so charts update cleanly)."""
    global CHANNELS
    CHANNELS = SCENARIOS[st.session_state.scenario_name]

    # Make sure spend keys exist
    for ch in CHANNELS:
        k = spend_key(ch)
        if k not in st.session_state:
            st.session_state[k] = 0

    # Reset spends when switching scenario
    for ch in CHANNELS:
        st.session_state[spend_key(ch)] = 0

    # Force axis recalculation because response curves changed
    recompute_axes_for_budget()


def on_spend_change(ch: str):
    # Clamp AFTER release: if sum exceeds budget, snap back changed slider only.
    budget = float(st.session_state.budget)
    total = sum(get_spend(k) for k in CHANNELS)

    if total <= budget:
        return

    others_sum = sum(get_spend(k) for k in CHANNELS if k != ch)
    allowed = max(0.0, budget - others_sum)
    allowed = float(np.floor(allowed / SPEND_STEP) * SPEND_STEP)
    set_spend(ch, allowed)


# Initialize axes on first load
if st.session_state.axis_budget is None or float(st.session_state.axis_budget) != float(st.session_state.budget):
    recompute_axes_for_budget()


# ============================================================
# UI
# ============================================================

st.title("Media Mix Allocation")
st.caption("Allocate budget across Search, Paid Social, and Display. Predict sales volume (units) and ROAS.")

outer_left, outer_right = st.columns([2.4, 1.1], gap="large")

# ---------------- LEFT: sliders + curves ----------------
with outer_left:
    st.subheader("Dashboard (Budget and Channel Allocations)")

    st.slider(
        "Total Budget ($)",
        min_value=APP_BUDGET_MIN,
        max_value=APP_BUDGET_MAX,
        step=APP_BUDGET_STEP,  # ✅ $500 increments
        key="budget_slider",
        on_change=on_budget_change,
    )
    st.session_state.budget = int(st.session_state["budget_slider"])
    slider_max = int(st.session_state.budget)

    # Instructor tools + scenario selector (hidden in same expander)
    with st.expander("", expanded=False):
        # Default: hidden. If URL contains ?Instructor (any value), show tools.
        if st.session_state.get("show_instructor_tools", False):
            st.radio(
                "Curve Scenario",
                options=list(SCENARIOS.keys()),
                key="scenario_name",
                on_change=on_scenario_change,
            )

            if st.button("🛠️"):
                x_best, _ = max_sales_allocation(float(st.session_state.budget))
                apply_allocation_vector(x_best, force_full_budget=True)

            if not HIDE_ROAS_BUTTON:
                if st.button("🧠 Optimize ROAS"):
                    x_best, _ = max_roas_allocation(float(st.session_state.budget))
                    apply_allocation_vector(x_best, force_full_budget=False)
        else:
            st.caption("Student Version. Robert H. Smith School of Business")

    c1, c2, c3 = st.columns(3, gap="small")
    for col, ch in zip([c1, c2, c3], CHANNELS.keys()):
        with col:
            st.slider(
                f"{ch} Spend ($)",
                min_value=0,
                max_value=slider_max,
                step=SPEND_STEP,
                key=spend_key(ch),
                on_change=on_spend_change,
                args=(ch,),
            )

    total_spend = float(sum(get_spend(ch) for ch in CHANNELS))
    remaining = max(0.0, float(st.session_state.budget) - total_spend)

    m1, m2, m3 = st.columns(3, gap="small")
    m1.metric("Allocated", f"${total_spend:,.0f}")
    m2.metric("Remaining", f"${remaining:,.0f}")
    m3.metric("Budget", f"${float(st.session_state.budget):,.0f}")

    st.divider()

    st.subheader("Diminishing Returns Curves (Spend → Sales Units)")
    x_max = max(100.0, float(st.session_state.budget))
    x_grid = np.linspace(0, x_max, 160)

    # ✅ FIXED Y-AXIS ACROSS ALL THREE CURVES (within current scenario)
    global_max_units = max(params["scale"] for params in CHANNELS.values())
    y_axis_max = global_max_units * 1.05

    curve_cols = st.columns(3, gap="small")
    for col, ch in zip(curve_cols, CHANNELS.keys()):
        with col:
            y_grid = [contribution(x, **CHANNELS[ch]) for x in x_grid]
            x_pt = float(get_spend(ch))
            y_pt = float(contribution(x_pt, **CHANNELS[ch]))

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x_grid, y=y_grid, mode="lines"))
            fig.add_trace(go.Scatter(x=[x_pt], y=[y_pt], mode="markers", marker=dict(size=12)))
            fig.update_layout(
                title=ch,
                height=300,
                margin=dict(l=10, r=10, t=45, b=10),
                xaxis_title="Spend ($)",
                yaxis=dict(title="Units Sold", range=[0, y_axis_max]),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

# ---------------- RIGHT: sales bar + roas bar ----------------
with outer_right:
    # Compute current totals
    total_units = float(sum(contribution(get_spend(ch), **CHANNELS[ch]) for ch in CHANNELS))
    total_revenue = total_units * float(REVENUE_PER_UNIT)
    total_spend = float(sum(get_spend(ch) for ch in CHANNELS))
    total_roas = (total_revenue / total_spend) if total_spend > 0 else 0.0

    # Fixed axis max until budget changes
    sales_max = float(st.session_state.sales_axis_max or 0.0)
    roas_max = float(st.session_state.roas_axis_max or 0.0)

    # add tiny headroom so labels don't clip
    sales_axis_max = max(1.0, sales_max * 1.05)
    roas_axis_max = max(0.1, roas_max * 1.05)

    # Two narrow bars side-by-side (UNCHANGED)
    rb1, rb2 = st.columns(2, gap="small")

    with rb1:
        fig_sales = go.Figure()
        fig_sales.add_trace(go.Bar(
            x=[""],
            y=[total_units],
            text=[f"{total_units:,.0f}"],
            textposition="outside",
            cliponaxis=False,
        ))
        fig_sales.update_layout(
            title=dict(text=f"Predicted Sales: {total_units:,.0f}", x=0.0, xanchor="left", y=0.98),
            height=640,
            margin=dict(l=8, r=8, t=60, b=10),
            xaxis=dict(showticklabels=False),
            yaxis=dict(title="Units", range=[0, sales_axis_max]),
            showlegend=False,
        )
        st.plotly_chart(fig_sales, use_container_width=True)

    with rb2:
        fig_roas = go.Figure()
        fig_roas.add_trace(go.Bar(
            x=[""],
            y=[total_roas],
            text=[f"{total_roas:,.2f}"],
            textposition="outside",
            cliponaxis=False,
        ))
        fig_roas.update_layout(
            title=dict(text=f"Total ROAS: {total_roas:,.2f}", x=0.0, xanchor="left", y=0.98),
            height=640,
            margin=dict(l=8, r=8, t=60, b=10),
            xaxis=dict(showticklabels=False),
            yaxis=dict(title="Revenue / Spend", range=[0, roas_axis_max]),
            showlegend=False,
        )
        st.plotly_chart(fig_roas, use_container_width=True)

    st.caption(f"Revenue per unit = ${REVENUE_PER_UNIT:,.2f}")