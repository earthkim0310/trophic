# streamlit_app.py — Ecosystem Balance → Shock → Recovery
import time
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Ecosystem Recovery Simulation", layout="wide")
st.title("Ecosystem → Shock → Recovery (Producers · Primary Consumers · Secondary Consumers)")

st.info(
    "Parameters are adjusted so that all equilibrium values are positive, "
    "and terminology is unified as 'Producers / Primary consumers / Secondary consumers'."
)

# ---------- Fixed parameters (positive interior equilibrium) ----------
r  = 1.0
K  = 500.0
a  = 0.0015   # Producer → Primary consumer feeding rate
b  = 0.0030   # Primary → Secondary consumer feeding rate
e1 = 0.30
e2 = 0.25
m1 = 0.0      # Producer natural death (ignored in this model)
m2 = 0.08
m3 = 0.15
h  = 0.03     # Secondary consumer harvest pressure

# Time
T_total = 30.0
dt = 0.05
t_axis = np.arange(0, T_total + dt, dt)
steps = len(t_axis)

# ---------- Equilibrium ----------
def compute_equilibrium():
    C1_star = (m3 + h) / (e2 * b)          # Primary consumers
    P_star  = K * (1.0 - a * C1_star / r)  # Producers
    C2_star = (e1 * a * P_star - m2) / b   # Secondary consumers
    return P_star, C1_star, C2_star

P_star, C1_star, C2_star = compute_equilibrium()

with st.expander("Show equilibrium values", expanded=False):
    st.write(f"Producers P* = {P_star:.2f}")
    st.write(f"Primary consumers C1* = {C1_star:.2f}")
    st.write(f"Secondary consumers C2* = {C2_star:.2f}")

# ---------- Shock controls ----------
st.sidebar.header("Shock (population change)")
t_shock = st.sidebar.slider("Shock time", 1.0, T_total - 1.0, 5.0, 0.5)
target = st.sidebar.selectbox("Target", ["Producers", "Primary consumers", "Secondary consumers"], index=1)
mode   = st.sidebar.radio("Mode", ["Multiplier (×)", "Increase (+)", "Decrease (−)"], index=0)
amount = st.sidebar.slider("Magnitude", 0.1, 3.0, 1.5, 0.1)
speed  = st.sidebar.slider("Animation delay (sec/frame)", 0.0, 0.2, 0.02, 0.005)
start  = st.sidebar.button("Run simulation")

# ---------- Simulation ----------
def simulate_with_shock():
    P = np.zeros(steps); C1 = np.zeros(steps); C2 = np.zeros(steps)
    P[0], C1[0], C2[0] = P_star, C1_star, C2_star
    shock_index = int(t_shock / dt)

    for t in range(steps - 1):
        # Apply shock at the chosen time step
        if t == shock_index:
            if target == "Producers":
                if mode == "Multiplier (×)":
                    P[t] *= amount
                elif mode == "Increase (+)":
                    P[t] += amount * P_star
                else:  # Decrease (−)
                    P[t] = max(0.0, P[t] - amount * P_star)

            elif target == "Primary consumers":
                if mode == "Multiplier (×)":
                    C1[t] *= amount
                elif mode == "Increase (+)":
                    C1[t] += amount * C1_star
                else:
                    C1[t] = max(0.0, C1[t] - amount * C1_star)

            else:  # Secondary consumers
                if mode == "Multiplier (×)":
                    C2[t] *= amount
                elif mode == "Increase (+)":
                    C2[t] += amount * C2_star
                else:
                    C2[t] = max(0.0, C2[t] - amount * C2_star)

        # Lotka–Volterra with logistic growth for producers
        dP  = r * P[t] * (1 - P[t] / K) - a * P[t] * C1[t]
        dC1 = e1 * a * P[t] * C1[t] - b * C1[t] * C2[t] - m2 * C1[t]
        dC2 = e2 * b * C1[t] * C2[t] - (m3 + h) * C2[t]

        P[t + 1]  = max(P[t]  + dP  * dt, 0.0)
        C1[t + 1] = max(C1[t] + dC1 * dt, 0.0)
        C2[t + 1] = max(C2[t] + dC2 * dt, 0.0)

    return P, C1, C2

P, C1, C2 = simulate_with_shock()

# ---------- Plots (animated) ----------
colA, colB = st.columns([2, 1])
graph_ph = colA.empty()
pyr_ph   = colB.empty()

def draw_frame(k: int):
    # Time-series
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t_axis[:k], P[:k],  label="Producers")
    ax.plot(t_axis[:k], C1[:k], label="Primary consumers", linewidth=2.5)
    ax.plot(t_axis[:k], C2[:k], label="Secondary consumers")
    ax.axvline(t_shock, linestyle="--")
    ax.set_xlabel("Time")
    ax.set_ylabel("Population (relative)")
    ax.legend(loc="best")
    graph_ph.pyplot(fig)

    # Simple population pyramid (horizontal bars, normalized to current max)
    kk = max(0, k - 1)
    p, c1, c2 = P[kk], C1[kk], C2[kk]
    maxw = max(p, c1, c2) if max(p, c1, c2) > 0 else 1.0

    fig2, ax2 = plt.subplots(figsize=(4, 4))
    for y, w, label in zip(
        [1, 2, 3],
        [p / maxw, c1 / maxw, c2 / maxw],
        ["Producers", "Primary consumers", "Secondary consumers"],
    ):
        ax2.barh(y, w, height=0.6)
        ax2.text(w + 0.02, y, label, va="center")
    ax2.set_xlim(0, 1.2)
    ax2.set_ylim(0.5, 3.5)
    ax2.set_yticks([])
    ax2.set_xticks([])
    ax2.set_title(f"t = {t_axis[kk]:.2f}")
    pyr_ph.pyplot(fig2)

# Initial draw (shows pre-shock and a little after)
draw_frame(int(t_shock / dt) + 2)

# Animate if requested
if start:
    for k in range(2, steps + 1):
        draw_frame(k)
        if speed > 0:
            time.sleep(speed)
