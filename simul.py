
# streamlit_app: 생태계 평형 → 충격 → 회복 (생산자/1차/2차 소비자 표기)
import time
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib import font_manager

st.set_page_config(page_title="생태계 평형 회복 시뮬레이션", layout="wide")
st.title("생태계 평형 → 충격 → 회복 (생산자 · 1차 소비자 · 2차 소비자)")

# ---------- Korean font ----------
def set_korean_font():
    candidates = ["AppleGothic", "Malgun Gothic", "NanumGothic", "Noto Sans CJK KR", "Noto Sans KR"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return
    plt.rcParams["axes.unicode_minus"] = False
set_korean_font()

st.info("현재 버전은 **모든 단계의 평형값이 양수**가 되도록 매개변수를 조정했으며, 용어를 ‘생산자/1차 소비자/2차 소비자’로 통일했습니다.")

# ---------- Fixed parameters (positive interior equilibrium) ----------
r  = 1.0
K  = 500.0
a  = 0.0015   # 생산자→1차 소비자 섭식율
b  = 0.0030   # 1차→2차 소비자 섭식율
e1 = 0.30
e2 = 0.25
m1 = 0.0      # 생산자 자연사(모형상 생략)
m2 = 0.08
m3 = 0.15
h  = 0.03     # 2차 소비자 포획 압력

# Time
T_total = 30.0
dt = 0.05
t_axis = np.arange(0, T_total + dt, dt)
steps = len(t_axis)

# ---------- Equilibrium ----------
def compute_equilibrium():
    C1_star = (m3 + h) / (e2 * b)          # 1차 소비자
    P_star  = K * (1.0 - a * C1_star / r)  # 생산자
    C2_star = (e1 * a * P_star - m2) / b   # 2차 소비자
    return P_star, C1_star, C2_star

P_star, C1_star, C2_star = compute_equilibrium()

with st.expander("계산된 평형값 보기", expanded=False):
    st.write(f"생산자 P* = {P_star:.2f}")
    st.write(f"1차 소비자 C1* = {C1_star:.2f}")
    st.write(f"2차 소비자 C2* = {C2_star:.2f}")

# ---------- Shock controls ----------
st.sidebar.header("충격(개체수 변경)")
t_shock = st.sidebar.slider("충격 시점", 1.0, T_total-1.0, 5.0, 0.5)
target = st.sidebar.selectbox("대상", ["생산자", "1차 소비자", "2차 소비자"], index=1)
mode   = st.sidebar.radio("방식", ["배율(×)", "증가량(+)", "감소량(−)"], index=0)
amount = st.sidebar.slider("크기", 0.1, 3.0, 1.5, 0.1)
speed  = st.sidebar.slider("애니메이션 속도(초/프레임)", 0.0, 0.2, 0.02, 0.005)
start  = st.sidebar.button("시뮬레이션 실행")

# ---------- Simulation ----------
def simulate_with_shock():
    P = np.zeros(steps); C1 = np.zeros(steps); C2 = np.zeros(steps)
    P[0], C1[0], C2[0] = P_star, C1_star, C2_star
    shock_index = int(t_shock / dt)
    for t in range(steps-1):
        if t == shock_index:
            if target == "생산자":
                if mode == "배율(×)": P[t] *= amount
                elif mode == "증가량(+)": P[t] += amount * P_star
                else: P[t] = max(0.0, P[t] - amount * P_star)
            elif target == "1차 소비자":
                if mode == "배율(×)": C1[t] *= amount
                elif mode == "증가량(+)": C1[t] += amount * C1_star
                else: C1[t] = max(0.0, C1[t] - amount * C1_star)
            else:  # 2차 소비자
                if mode == "배율(×)": C2[t] *= amount
                elif mode == "증가량(+)": C2[t] += amount * C2_star
                else: C2[t] = max(0.0, C2[t] - amount * C2_star)

        dP  = r*P[t]*(1 - P[t]/K) - a*P[t]*C1[t]
        dC1 = e1*a*P[t]*C1[t] - b*C1[t]*C2[t] - m2*C1[t]
        dC2 = e2*b*C1[t]*C2[t] - (m3 + h)*C2[t]

        P[t+1]  = max(P[t]  + dP*dt, 0.0)
        C1[t+1] = max(C1[t] + dC1*dt, 0.0)
        C2[t+1] = max(C2[t] + dC2*dt, 0.0)
    return P, C1, C2

P, C1, C2 = simulate_with_shock()

# ---------- Plots (animated) ----------
colA, colB = st.columns([2,1])
graph_ph = colA.empty(); pyr_ph = colB.empty()

def draw_frame(k):
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(t_axis[:k], P[:k],  label="생산자")
    ax.plot(t_axis[:k], C1[:k], label="1차 소비자", linewidth=2.5)
    ax.plot(t_axis[:k], C2[:k], label="2차 소비자")
    ax.axvline(t_shock, linestyle="--")
    ax.set_xlabel("시간"); ax.set_ylabel("개체수(상대)")
    ax.legend(loc="best")
    graph_ph.pyplot(fig)

    kk = max(0, k-1)
    p,c1,c2 = P[kk], C1[kk], C2[kk]
    maxw = max(p,c1,c2) if max(p,c1,c2) > 0 else 1.0
    fig2, ax2 = plt.subplots(figsize=(4,4))
    for y,w,label in zip([1,2,3],[p/maxw,c1/maxw,c2/maxw],["생산자","1차 소비자","2차 소비자"]):
        ax2.barh(y, w, height=0.6)
        ax2.text(w+0.02, y, label, va="center")
    ax2.set_xlim(0,1.2); ax2.set_ylim(0.5,3.5)
    ax2.set_yticks([]); ax2.set_xticks([])
    ax2.set_title(f"t = {t_axis[kk]:.2f}")
    pyr_ph.pyplot(fig2)

# initial draw
draw_frame(int(t_shock/dt)+2)

if start:
    for k in range(2, steps+1):
        draw_frame(k)
        if speed>0: time.sleep(speed)
