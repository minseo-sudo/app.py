
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr

st.set_page_config(
    page_title="보유도서수와 영어성취도 분석",
    page_icon="📚",
    layout="wide"
)

st.title("📚 보유도서수와 영어성취도 관계 분석")
st.write("공공데이터를 활용한 회귀분석 및 시각화")

# 1. 데이터 업로드
st.header("1. 데이터 불러오기")

file = st.file_uploader(
    "CSV 또는 Excel 파일 업로드",
    type=["csv", "xlsx", "xls"]
)

if file is None:
    st.info("데이터 파일을 업로드해 주세요.")
    st.stop()

try:
    if file.name.endswith(".csv"):
        try:
            df = pd.read_csv(file, encoding="utf-8-sig")
        except UnicodeDecodeError:
            file.seek(0)
            df = pd.read_csv(file, encoding="cp949")
    else:
        df = pd.read_excel(file)

except Exception as e:
    st.error(f"파일 읽기 오류: {e}")
    st.stop()

st.success("데이터 불러오기 성공!")

# 2. 원본 데이터 확인
st.header("2. 원본 데이터 확인")

st.write("데이터 크기:", df.shape)
st.write("전체 열 이름:")

st.dataframe(
    pd.DataFrame({"열 이름": df.columns}),
    use_container_width=True
)

st.dataframe(df.head(10), use_container_width=True)

# 3. 변수 후보 선택
st.header("3. 분석 변수 설정")

st.info(
    "실제 보유도서수와 영어성취도를 나타내는 열을 "
    "직접 선택해 주세요."
)

def clean_number(s):
    return pd.to_numeric(
        s.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("권", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.strip(),
        errors="coerce"
    )

# 숫자로 변환 가능한 열 확인
numeric_cols = []

for col in df.columns:
    values = clean_number(df[col])

    if values.notna().sum() >= 2:
        numeric_cols.append(col)

if len(numeric_cols) < 2:
    st.error("숫자로 변환 가능한 열이 2개 미만입니다.")
    st.stop()

# 변수 선택
x_col = st.selectbox(
    "독립변수 X - 보유도서수",
    numeric_cols
)

y_options = [c for c in numeric_cols if c != x_col]

y_col = st.selectbox(
    "종속변수 Y - 영어성취도",
    y_options
)

# 4. 데이터 정리
data = pd.DataFrame()

data["X"] = clean_number(df[x_col])
data["Y"] = clean_number(df[y_col])

data = data.replace(
    [np.inf, -np.inf], np.nan
).dropna()

st.header("4. 데이터 분포 확인")

st.write("독립변수:", x_col)
st.write("종속변수:", y_col)
st.write("유효 데이터:", len(data))

if len(data) == 0:
    st.error("분석 가능한 데이터가 없습니다.")
    st.stop()

col1, col2 = st.columns(2)

with col1:
    st.subheader("보유도서수 분포")
    st.write(data["X"].describe())
    st.write("고유값 개수:", data["X"].nunique())

with col2:
    st.subheader("영어성취도 분포")
    st.write(data["Y"].describe())
    st.write("고유값 개수:", data["Y"].nunique())

st.dataframe(data.head(30))

# 5. 독립변수 값 확인
if data["X"].nunique() < 2:

    st.error(
        "선택한 독립변수의 값이 모두 같습니다. "
        "다른 열을 선택하거나 원본 데이터의 구조를 확인해야 합니다."
    )

    st.subheader("독립변수 실제 값")
    st.write(data["X"].value_counts())

    st.stop()

if data["Y"].nunique() < 2:
    st.error("영어성취도 값이 모두 같습니다.")
    st.stop()

if len(data) < 3:
    st.error("회귀분석에는 최소 3개의 유효 데이터가 필요합니다.")
    st.stop()

# 6. 회귀분석
X = data[["X"]]
y = data["Y"]

model = LinearRegression()
model.fit(X, y)

pred = model.predict(X)

r2 = r2_score(y, pred)
rmse = np.sqrt(mean_squared_error(y, pred))
r, p = pearsonr(data["X"], data["Y"])

slope = model.coef_[0]
intercept = model.intercept_

# 7. 분석 결과
st.divider()
st.header("5. 회귀분석 결과")

c1, c2, c3 = st.columns(3)

c1.metric("상관계수", f"{r:.4f}")
c2.metric("결정계수 R²", f"{r2:.4f}")
c3.metric("RMSE", f"{rmse:.4f}")

st.subheader("회귀방정식")

st.latex(
    f"Y = {slope:.4f}X + {intercept:.4f}"
)

st.write("회귀계수:", round(slope, 4))
st.write("절편:", round(intercept, 4))
st.write("상관관계 검정 p-value:", f"{p:.6f}")

# 8. 산점도와 회귀선
st.divider()
st.header("6. 그래프 모델링")

chart = data.copy()
chart["예측값"] = pred

fig = px.scatter(
    chart,
    x="X",
    y="Y",
    title="보유도서수와 영어성취도의 관계",
    labels={
        "X": "보유도서수",
        "Y": "영어성취도"
    },
    opacity=0.6
)

sorted_data = data.sort_values("X")

fig.add_scatter(
    x=sorted_data["X"],
    y=model.predict(sorted_data[["X"]]),
    mode="lines",
    name="회귀선",
    line=dict(color="red", width=3)
)

fig.update_layout(
    template="plotly_white",
    height=550
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# 9. 결과 해석
st.divider()
st.header("7. 결과 해석")

if slope > 0:
    st.success("보유도서수가 증가할수록 영어성취도가 높아지는 양의 관계입니다.")
elif slope < 0:
    st.info("보유도서수가 증가할수록 영어성취도가 낮아지는 음의 관계입니다.")
else:
    st.info("선형적인 변화가 관찰되지 않습니다.")

st.write(
    f"보유도서수가 1단위 증가하면 "
    f"예측 영어성취도는 {slope:.4f}만큼 변합니다."
)

st.write(
    f"모형의 결정계수는 {r2:.4f}입니다."
)

if p < 0.05:
    st.write("상관관계 검정 결과 통계적으로 유의합니다.")
else:
    st.write("상관관계 검정 결과 통계적으로 유의하지 않습니다.")

st.caption(
    "주의: 상관관계는 인과관계를 의미하지 않습니다. "
    "학교급, 학교 규모 등 다른 요인은 통제되지 않았습니다."
)
