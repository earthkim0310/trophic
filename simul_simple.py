
# streamlit_app: 생태계 평형 (간단 버전) — 버튼 시작 + 3개 입력만
import time
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib import font_manager

st.set_page_config(page_title="생태계 평형 (간단 버전)", layout="wide")
st.title("생태계 평형 → 충격 → 회복 (간단 버전)")

# ---------- Korean font (한 번만) ----------
def set_korean_font_once():
    candidates = ["AppleGothic", "Malgun Gothic", "NanumGothic", "Noto Sans CJK KR", "Noto Sans KR"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return
    plt.rcParams["axes.unicode_minus"] = False

set_korean_font_once()

# ---------- 고정 파라미터(내부에서만 사용) ----------
dt = 0.02            # 시간 간격
T = 30.0             # 전체 시간
steps = int(T/dt)
t_shock = T/3        # 충격 시점
shock_scale = 0.5    # 생산자 일시 감소 배율
frame_delay = 0.05   # 프레임 지연 (초): 너무 빠르면 CPU 사용량↑

# 상호작용 계수 (예시용 고정 값)
a = 1.0    # 생산자 성장율
b = 0.02   # 1차 소비로 인한 P 감소
c = 0.01   # 1차 소비자 성장율 (P*C1)
d = 0.5    # 1차 소비자 자연감소
e = 0.01   # 2차 소비로 인한 1차 감소
f = 0.005  # 2차 소비자 성장율 (C1*C2)
g = 0.2    # 2차 소비자 자연감소

# ---------- 사이드바: 입력 3개 + 버튼 ----------
with st.sidebar:
    st.subheader("초기 개체 수 설정")
    P0 = st.number_input("생산자", 0.0, 10000.0, 50.0, 1.0, help="생산자 초기값")
    C1_0 = st.number_input("1차 소비자", 0.0, 10000.0, 20.0, 1.0, help="1차 소비자 초기값")
    C2_0 = st.number_input("2차 소비자", 0.0, 10000.0, 10.0, 1.0, help="2차 소비자 초기값")
    start = st.button("애니메이션 시작", use_container_width=True)

# ---------- 시뮬레이션 (캐시) ----------
@st.cache_data(show_spinner=False)
def simulate(P0, C1_0, C2_0):
    t = np.linspace(0, steps*dt, steps+1)
    P  = np.zeros_like(t)
    C1 = np.zeros_like(t)
    C2 = np.zeros_like(t)

    P[0]  = P0
    C1[0] = C1_0
    C2[0] = C2_0

    shock_idx = int(t_shock/dt)
    for k in range(steps):
        # 충격 (한 번)
        if k == shock_idx:
            P[k] *= shock_scale

        dP  =  a*P[k] - b*P[k]*C1[k]
        dC1 =  c*P[k]*C1[k] - d*C1[k] - e*C1[k]*C2[k]
        dC2 =  f*C1[k]*C2[k] - g*C2[k]

        P[k+1]  = max(P[k]  + dt*dP , 0.0)
        C1[k+1] = max(C1[k] + dt*dC1, 0.0)
        C2[k+1] = max(C2[k] + dt*dC2, 0.0)

    return t, P, C1, C2, shock_idx

t_axis, P, C1, C2, shock_idx = simulate(P0, C1_0, C2_0)

# ---------- 레이아웃 ----------
left, right = st.columns([2,1])
plot_ph = left.empty()
bar_ph  = right.empty()

# ---------- Figure 1회 생성 ----------
# 선 그래프
fig_line, ax_line = plt.subplots(figsize=(8,4))
(lineP,)  = ax_line.plot([], [], label="생산자")
(lineC1,) = ax_line.plot([], [], label="1차 소비자")
(lineC2,) = ax_line.plot([], [], label="2차 소비자")
ax_line.axvline(t_axis[shock_idx], linestyle="--", linewidth=1)
ax_line.set_xlim(t_axis[0], t_axis[-1])
ymax = max(P.max(), C1.max(), C2.max(), 1.0) * 1.1
ax_line.set_ylim(0, ymax)
ax_line.set_xlabel("시간")
ax_line.set_ylabel("개체수")
ax_line.legend(loc="upper right")
plot_ph.pyplot(fig_line, clear_figure=False)

# 막대 그래프
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

# ---------- 업데이트 함수 ----------
def update_frame(k):
    lineP.set_data(t_axis[:k],  P[:k])
    lineC1.set_data(t_axis[:k], C1[:k])
    lineC2.set_data(t_axis[:k], C2[:k])
    plot_ph.pyplot(fig_line, clear_figure=False)

    maxw = max(P.max(), C1.max(), C2.max(), 1.0)
    vals = [P[k]/maxw, C1[k]/maxw, C2[k]/maxw]
    for bar, w in zip(bars, vals):
        bar.set_width(w)
    ax_bar.set_title(f"t = {t_axis[k]:.2f}")
    bar_ph.pyplot(fig_bar, clear_figure=False)

# ---------- 초기 프레임 ----------
k0 = min(int(t_shock/dt)+2, steps)
update_frame(k0)

# ---------- 버튼 눌렀을 때만 실행 ----------
if start:
    for k in range(k0+1, steps+1):
        update_frame(k)
        time.sleep(frame_delay)
    st.success("애니메이션 완료")
else:
    st.info("사이드바에서 초기값을 정하고 ‘애니메이션 시작’ 버튼을 누르세요.")
