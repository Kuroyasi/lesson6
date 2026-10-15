import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)
LAST_YEAR = 2025  # 수업 기준 기간의 끝
MIN_DAYS = 300  # 연간 최소 관측일 수

st.set_page_config(page_title="기온 예측기", page_icon="🌡️", layout="centered")


@st.cache_data(show_spinner="데이터를 불러오는 중...")
def load_yearly() -> pd.DataFrame:
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    df["날짜"] = pd.to_datetime(df["날짜"].astype(str).str.strip(), errors="coerce")
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")
    df = df.dropna(subset=["날짜", "평균기온"])
    df["연도"] = df["날짜"].dt.year
    df = df[df["연도"] <= LAST_YEAR]  # 2025년 이후 데이터 제외

    yearly = (
        df.groupby("연도")["평균기온"]
        .agg(연평균기온="mean", 관측일수="count")
        .reset_index()
    )
    yearly = yearly[yearly["관측일수"] >= MIN_DAYS]  # 관측일 300일 미만 연도 제외
    return yearly.reset_index(drop=True)


st.title("🌡️ 기온 예측기")
st.caption("서울의 연평균기온 추세를 회귀 직선으로 이어서 미래(또는 과거)의 기온을 예상해 봅니다.")

try:
    yearly = load_yearly()
except Exception as e:
    st.error(f"데이터를 불러오지 못했습니다: {e}")
    st.stop()

x = yearly["연도"].to_numpy(dtype=float)
y = yearly["연평균기온"].to_numpy(dtype=float)
slope, intercept = np.polyfit(x, y, 1)
r = float(np.corrcoef(x, y)[0, 1])

# 사용한 데이터 정보
n_years = len(yearly)
start_year = int(yearly["연도"].min())
end_year = int(yearly["연도"].max())

c1, c2, c3 = st.columns(3)
c1.metric("회귀에 사용한 해의 개수", f"{n_years}개")
c2.metric("시작 연도", f"{start_year}년")
c3.metric("끝 연도", f"{end_year}년")

# 연도 슬라이더와 예상 기온
year = st.slider("예상할 연도를 고르세요", min_value=1900, max_value=2100, value=2050, step=1)
pred = slope * year + intercept

st.markdown(
    f"""
    <div style="text-align:center; padding:1.2rem 0;">
        <div style="font-size:1.2rem; opacity:0.7;">{year}년 예상 연평균기온</div>
        <div style="font-size:4.5rem; font-weight:800; line-height:1.2;">{pred:.2f} °C</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# 산점도 + 회귀 직선
line_x = np.array([1900, 2100], dtype=float)
line_y = slope * line_x + intercept

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["연평균기온"],
        mode="markers",
        name="연평균기온",
        marker=dict(size=7, opacity=0.8),
        hovertemplate="%{x}년<br>%{y:.2f} °C<extra></extra>",
    )
)
fig.add_trace(
    go.Scatter(
        x=line_x,
        y=line_y,
        mode="lines",
        name="회귀 직선",
        line=dict(width=3, color="crimson"),
        hoverinfo="skip",
    )
)
fig.add_trace(
    go.Scatter(
        x=[year],
        y=[pred],
        mode="markers",
        name=f"{year}년 예상",
        marker=dict(size=16, color="gold", symbol="star", line=dict(width=1.5, color="black")),
        hovertemplate="%{x}년 예상<br>%{y:.2f} °C<extra></extra>",
    )
)
fig.update_layout(
    title="서울 연평균기온 산점도와 회귀 직선",
    xaxis_title="연도",
    yaxis_title="연평균기온 (°C)",
    xaxis=dict(range=[1900, 2100]),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=10, r=10, t=80, b=10),
)
st.plotly_chart(fig, width="stretch")

# 기울기를 "100년에 몇 도"로 환산
st.markdown(
    f"""
    <div style="text-align:center; padding:1rem 0;">
        <div style="font-size:1.2rem; opacity:0.7;">전체 기간 추세: 100년에</div>
        <div style="font-size:4rem; font-weight:800; line-height:1.2; color:crimson;">
            {slope * 100:+.2f} °C
        </div>
        <div style="font-size:1.1rem; opacity:0.7;">오르는 속도 ({start_year}~{end_year}년 기준)</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# 전체 기간 vs 최근 20년 기울기 비교
recent = yearly[yearly["연도"] > end_year - 20]
rx = recent["연도"].to_numpy(dtype=float)
ry = recent["연평균기온"].to_numpy(dtype=float)
slope_recent, _ = np.polyfit(rx, ry, 1)
r_recent = float(np.corrcoef(rx, ry)[0, 1])

st.subheader("기울기 비교: 전체 기간 vs 최근 20년")
cmp1, cmp2 = st.columns(2)
with cmp1:
    st.markdown("**전체 기간**")
    st.caption(f"{start_year}~{end_year}년, {n_years}개 해")
    st.metric("100년에", f"{slope * 100:+.2f} °C")
    st.metric("상관계수", f"{r:.3f}")
with cmp2:
    st.markdown("**최근 20년**")
    st.caption(f"{int(recent['연도'].min())}~{int(recent['연도'].max())}년, {len(recent)}개 해")
    st.metric(
        "100년에",
        f"{slope_recent * 100:+.2f} °C",
        delta=f"전체 기간 대비 {(slope_recent - slope) * 100:+.2f} °C",
        delta_color="off",
    )
    st.metric("상관계수", f"{r_recent:.3f}")
st.caption(
    "※ 최근 20년은 해의 개수가 적어 기울기가 연도별 변동에 크게 흔들립니다. "
    "'100년에 몇 도'는 연 단위 기울기에 100을 곱한 환산값이며, 실제로 100년을 관측한 결과는 아닙니다."
)

st.caption(
    f"회귀 직선: 기온 = {slope:.4f} × 연도 + ({intercept:.2f})  |  "
    f"기준 기간 {LAST_YEAR}년까지, 관측일 {MIN_DAYS}일 미만인 해와 {LAST_YEAR}년 이후 데이터는 제외했습니다."
)
st.caption("※ 직선을 만든 기간 밖의 연도는 추세를 그대로 연장한 값이라 실제와 차이가 클 수 있습니다.")
