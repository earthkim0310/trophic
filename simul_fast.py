
# streamlit_app: 생태계 평형 → 충격 → 회복 (생산자/1차/2차 소비자 표기) — Optimized
# 핵심 최적화:
# 1) Figure/Artists 1회 생성 후, per-frame에 데이터만 갱신 (figure 재생성/폰트설정 반복 제거)
# 2) 고정 계산은 캐싱(st.cache_data)으로 메모이제이션
# 3) 프레임레이트 스로틀링 + 불필요한 축 재설정 최소화
# 4) st.session_state로 재실행(run) 제어

import time
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib import font_manager

st.set_page_config(page_title="생태계 평형 회복 시뮬레이션 (FAST)", layout="wide")
st.title("생태계 평형 → 충격 → 회복 (생산자 · 1차 소비자 · 2차 소비자) — FAST")

# ---------- Korean font (1회만) ----------
def set_korean_font_once():
    candidates = ["AppleGothic", "Malgun Gothic", "NanumGothic", "Noto Sans CJK KR", "Noto Sans KR"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return name
    plt.rcParams["axes.unicode_minus"] = False
    return None

_ = set_korean_font_once()

# ---------- Parameters ----------
with st.sidebar:
    st.subheader("모델 파라미터 / 실행")
    # Lotka-Volterra 유사 파라미터 가정 (사용 코드와 호환되도록 일반형 유지)
    dt = st.slider("시간 간격 dt", 0.005, 0.1, 0.02, 0.005)
    T  = st.slider("전체 시간 T", 5.0, 60.0, 30.0, 1.0)
    steps = int(T/dt)
    t_shock = st.slider("충격 시점 t_shock", 0.0, T-1e-9, min(5.0, T/3), 0.1)
    speed = st.slider("프레임 지연 (초) — 값이 클수록 느림", 0.0, 0.5, 0.05, 0.01)

    # 초기 개체수
    P0 = st.number_input("생산자 초기값", 0.0, 10_000.0, 50.0, 1.0)
    C1_0 = st.number_input("1차 소비자 초기값", 0.0, 10_000.0, 20.0, 1.0)
    C2_0 = st.number_input("2차 소비자 초기값", 0.0, 10_000.0, 10.0, 1.0)

    # 단순 동학 파라미터(예시용)
    a = st.number_input("생산자 성장율 a", 0.0, 5.0, 1.0, 0.1)
    b = st.number_input("1차 소비에 의한 감소율 b (P*C1)", 0.0, 5.0, 0.02, 0.01)
    c = st.number_input("1차 소비자 성장율 c (P*C1)", 0.0, 5.0, 0.01, 0.01)
    d = st.number_input("1차 소비자 자연감소 d", 0.0, 5.0, 0.5, 0.1)
    e = st.number_input("2차 소비에 의한 1차 감소 e (C1*C2)", 0.0, 5.0, 0.01, 0.01)
    f = st.number_input("2차 소비자 성장율 f (C1*C2)", 0.0, 5.0, 0.005, 0.005)
    g = st.number_input("2차 소비자 자연감소 g", 0.0, 5.0, 0.2, 0.05)

    shock_scale = st.slider("충격 강도 (생산자 일시 감소 배율)", 0.0, 1.0, 0.5, 0.05)

    run = st.checkbox("애니메이션 실행", value=False)

# ---------- Simulation (vectorized + cached) ----------
@st.cache_data(show_spinner=False)
def simulate(dt, steps, t_shock, P0, C1_0, C2_0, a,b,c,d,e,f,g, shock_scale):
    t = np.linspace(0, steps*dt, steps+1)
    P  = np.zeros_like(t)
    C1 = np.zeros_like(t)
    C2 = np.zeros_like(t)

    P[0]  = P0
    C1[0] = C1_0
    C2[0] = C2_0

    shock_idx = int(t_shock/dt)
    for k in range(steps):
        # 충격 적용 (한 번만)
        if k == shock_idx:
            P[k] *= shock_scale

        # 간단한 세 종 상호작용 (Euler)
        # dP/dt =  a*P - b*P*C1
        # dC1/dt = c*P*C1 - d*C1 - e*C1*C2
        # dC2/dt = f*C1*C2 - g*C2
        dP  =  a*P[k] - b*P[k]*C1[k]
        dC1 =  c*P[k]*C1[k] - d*C1[k] - e*C1[k]*C2[k]
        dC2 =  f*C1[k]*C2[k] - g*C2[k]

        P[k+1]  = max(P[k]  + dt*dP , 0.0)
        C1[k+1] = max(C1[k] + dt*dC1, 0.0)
        C2[k+1] = max(C2[k] + dt*dC2, 0.0)

    return t, P, C1, C2, shock_idx

t_axis, P, C1, C2, shock_idx = simulate(dt, steps, t_shock, P0, C1_0, C2_0, a,b,c,d,e,f,g, shock_scale)

# ---------- Placeholders ----------
left, right = st.columns([2,1])
plot_ph = left.empty()
bar_ph  = right.empty()

# ---------- Create figures (once) ----------
# Line figure
fig_line, ax_line = plt.subplots(figsize=(8,4))
(lineP,)  = ax_line.plot([], [], label="생산자")
(lineC1,) = ax_line.plot([], [], label="1차 소비자")
(lineC2,) = ax_line.plot([], [], label="2차 소비자")
ax_line.axvline(t_axis[shock_idx], linestyle="--", linewidth=1)
ax_line.set_xlim(t_axis[0], t_axis[-1])
ymax = max(P.max(), C1.max(), C2.max()) * 1.1 if (P.max()+C1.max()+C2.max())>0 else 1.0
ax_line.set_ylim(0, ymax)
ax_line.set_xlabel("시간")
ax_line.set_ylabel("개체수")
ax_line.legend(loc="upper right")
plot_ph.pyplot(fig_line, clear_figure=False)

# Bar figure
fig_bar, ax_bar = plt.subplots(figsize=(4,4))
maxw = max(P.max(), C1.max(), C2.max(), 1.0)
bars = ax_bar.barh([1,2,3], [P[0]/maxw, C1[0]/maxw, C2[0]/maxw], height=0.6)
for y, label in zip([1,2,3], ["생산자","1차 소비자","2차 소비자"]):
    ax_bar.text(1.02, y, label, va="center", ha="left", transform=ax_bar.get_yaxis_transform())
ax_bar.set_xlim(0, 1.2)
ax_bar.set_ylim(0.5, 3.5)
ax_bar.set_yticks([]); ax_bar.set_xticks([])
ax_bar.set_title(f"t = {t_axis[0]:.2f}")
bar_ph.pyplot(fig_bar, clear_figure=False)

# ---------- Update function (no re-creation) ----------
def update_frame(k):
    # Update lines
    lineP.set_data(t_axis[:k],  P[:k])
    lineC1.set_data(t_axis[:k], C1[:k])
    lineC2.set_data(t_axis[:k], C2[:k])
    # (x/y lim은 1회만 설정 — 큰 변동이 있다면 최소한으로만 조정)
    # Render
    plot_ph.pyplot(fig_line, clear_figure=False)

    # Update bars (width only)
    maxw = max(P.max(), C1.max(), C2.max(), 1.0)
    vals = [P[k]/maxw, C1[k]/maxw, C2[k]/maxw]
    for bar, w in zip(bars, vals):
        bar.set_width(w)
    ax_bar.set_title(f"t = {t_axis[k]:.2f}")
    bar_ph.pyplot(fig_bar, clear_figure=False)

# ---------- Initial draw around shock ----------
k0 = min(shock_idx+2, steps)
update_frame(k0)

# ---------- Animate ----------
if run:
    # 세션 상태로 "지금 실행중" 표시 → 사이드바 값 바뀌면 rerun
    if "running" not in st.session_state:
        st.session_state.running = True
    for k in range(k0+1, steps+1):
        update_frame(k)
        if speed > 0:
            time.sleep(speed)
        # 사용자가 체크 해제하면 rerun되어 running이 초기화됨
else:
    st.info("사이드바에서 ‘애니메이션 실행’ 체크하면 진행함.")
