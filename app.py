
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr


# ==================================
# 1. 페이지 설정
# ==================================

st.set_page_config(
    page_title="보유도서수와 영어성취도 분석",
    page_icon="📚",
    layout="wide"
)

st.title("📚 보유도서수와 영어성취도의 관계 분석")

st.markdown("""
공공데이터를 활용하여 학교의 보유도서수와
영어성취도 사이의 관계를 분석합니다.
""")

st.divider()


# ==================================
# 2. 데이터 업로드
# ==================================

st.header("1. 데이터 불러오기")

uploaded_file = st.file_uploader(
    "공공데이터 CSV 또는 Excel 파일을 업로드하세요.",
    type=["csv", "xlsx", "xls"]
)

if uploaded_file is None:
    st.info("먼저 공공데이터 파일을 업로드해 주세요.")
    st.stop()


# ==================================
# 3. 데이터 읽기
# ==================================

try:

    if uploaded_file.name.endswith(".csv"):

        try:
            df = pd.read_csv(
                uploaded_file,
                encoding="utf-8-sig"
            )

        except UnicodeDecodeError:
            uploaded_file.seek(0)

            df = pd.read_csv(
                uploaded_file,
                encoding="cp949"
            )

    else:
        df = pd.read_excel(uploaded_file)

    st.success("데이터를 불러왔습니다!")

except Exception as e:

    st.error(f"파일 읽기 오류: {e}")
    st.stop()


# ==================================
# 4. 데이터 미리보기
# ==================================

st.subheader("데이터 미리보기")

st.dataframe(
    df.head(10),
    use_container_width=True
)

st.write("전체 데이터 개수:", len(df))
st.write("전체 열 개수:", len(df.columns))

st.subheader("전체 열 이름")

st.write(list(df.columns))


# ==================================
# 5. 숫자형 데이터 정리
# ==================================

def convert_numeric(series):

    return pd.to_numeric(
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("권", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.strip()
        .replace(["", "nan", "None", "-"], np.nan),
        errors="coerce"
    )


numeric_candidates = []

for col in df.columns:

    converted = convert_numeric(df[col])

    if converted.notna().sum() >= 2:

        numeric_candidates.append(col)


# ==================================
# 6. 분석 변수 설정
# ==================================

st.divider()

st.header("2. 분석 변수 설정")

if len(numeric_candidates) < 2:

    st.error("분석 가능한 숫자형 열이 2개 미만입니다.")
    st.stop()


# 보유도서수 후보 탐색
book_candidates = [
    col for col in numeric_candidates
    if "보유도서수" in str(col)
    or "도서수" in str(col)
    or "장서수" in str(col)
]

# 영어성취도 후보 탐색
english_candidates = [
    col for col in numeric_candidates
    if "영어성취도" in str(col)
    or "영어" in str(col)
]


# 기본 선택값 설정
x_default = (
    book_candidates[0]
    if book_candidates
    else numeric_candidates[0]
)

y_default = (
    english_candidates[0]
    if english_candidates
    else next(
        (
            col for col in numeric_candidates
            if col != x_default
        ),
        numeric_candidates[1]
    )
)


# 독립변수 선택
x_col = st.selectbox(
    "독립변수 X (보유도서수)",
    numeric_candidates,
    index=numeric_candidates.index(x_default)
)


# 종속변수 선택
y_options = [
    col for col in numeric_candidates
    if col != x_col
]

y_default_index = (
    y_options.index(y_default)
    if y_default in y_options
    else 0
)

y_col = st.selectbox(
    "종속변수 Y (영어성취도)",
    y_options,
    index=y_default_index
)


st.write("선택한 독립변수:", x_col)
st.write("선택한 종속변수:", y_col)


# ==================================
# 7. 분석 데이터 정리
# ==================================

data = df[[x_col, y_col]].copy()

data[x_col] = convert_numeric(data[x_col])
data[y_col] = convert_numeric(data[y_col])

data = data.replace(
    [np.inf, -np.inf],
    np.nan
).dropna()


st.divider()

st.header("3. 데이터 점검")

col1, col2, col3 = st.columns(3)

col1.metric(
    "분석 데이터 수",
    len(data)
)

col2.metric(
    "보유도서수 고유값",
    data[x_col].nunique()
)

col3.metric(
    "영어성취도 고유값",
    data[y_col].nunique()
)


st.subheader("분석에 사용되는 실제 데이터")

st.dataframe(
    data.head(30),
    use_container_width=True
)


# ==================================
# 8. 회귀분석 가능 여부 확인
# ==================================

if len(data) < 3:

    st.error("유효한 데이터가 3개 미만입니다.")
    st.stop()


if data[x_col].nunique() < 2:

    st.error(
        "보유도서수 값이 모두 같습니다. "
        "실제 보유도서수 열을 선택했는지 확인하세요."
    )

    st.write("선택된 독립변수의 값:")
    st.write(data[x_col].value_counts())

    st.stop()


if data[y_col].nunique() < 2:

    st.error("영어성취도 값이 모두 같습니다.")
    st.stop()


# ==================================
# 9. 회귀분석 실행
# ==================================

X = data[[x_col]]
y = data[y_col]

model = LinearRegression()

model.fit(X, y)

predictions = model.predict(X)


# ==================================
# 10. 분석 결과 계산
# ==================================

r2 = r2_score(y, predictions)

rmse = np.sqrt(
    mean_squared_error(y, predictions)
)

correlation, p_value = pearsonr(
    data[x_col],
    data[y_col]
)

slope = model.coef_[0]
intercept = model.intercept_


# ==================================
# 11. 분석 결과 출력
# ==================================

st.divider()

st.header("4. 회귀분석 결과")

col1, col2, col3 = st.columns(3)

col1.metric(
    "상관계수 (r)",
    f"{correlation:.4f}"
)

col2.metric(
    "결정계수 (R²)",
    f"{r2:.4f}"
)

col3.metric(
    "RMSE",
    f"{rmse:.4f}"
)


st.subheader("회귀방정식")

st.latex(
    f"Y = {slope:.4f}X + {intercept:.4f}"
)

st.write("회귀계수:", round(slope, 4))
st.write("절편:", round(intercept, 4))
st.write("p-value:", f"{p_value:.6f}")


# ==================================
# 12. 데이터 시각화
# ==================================

st.divider()

st.header("5. 보유도서수와 영어성취도 관계")

sorted_data = data.sort_values(x_col)

fig = px.scatter(
    data,
    x=x_col,
    y=y_col,
    title="보유도서수와 영어성취도의 산점도",
    labels={
        x_col: "보유도서수",
        y_col: "영어성취도"
    },
    opacity=0.65
)

fig.add_scatter(
    x=sorted_data[x_col],
    y=model.predict(
        sorted_data[[x_col]]
    ),
    mode="lines",
    name="회귀선",
    line=dict(
        color="red",
        width=3
    )
)

fig.update_layout(
    template="plotly_white",
    height=550
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# ==================================
# 13. 결과 해석
# ==================================

st.divider()

st.header("6. 분석 결과 해석")

if slope > 0:

    st.success(
        "보유도서수가 증가할수록 "
        "영어성취도가 높아지는 양의 관계가 관찰됩니다."
    )

elif slope < 0:

    st.info(
        "보유도서수가 증가할수록 "
        "영어성취도가 낮아지는 음의 관계가 관찰됩니다."
    )

else:

    st.info(
        "보유도서수와 영어성취도 사이의 "
        "선형적인 변화가 관찰되지 않습니다."
    )


st.write(
    f"보유도서수가 1단위 증가할 때 "
    f"예측 영어성취도는 {slope:.4f}만큼 변합니다."
)

st.write(
    f"결정계수 R²는 {r2:.4f}입니다."
)

if p_value < 0.05:

    st.write(
        "상관관계 검정 결과 통계적으로 유의한 "
        "선형 상관관계가 관찰됩니다 (p < 0.05)."
    )

else:

    st.write(
        "상관관계 검정 결과 통계적으로 유의한 "
        "선형 상관관계를 확인하지 못했습니다 (p ≥ 0.05)."
    )


st.caption(
    "주의: 회귀분석과 상관관계만으로 "
    "보유도서수가 영어성취도의 원인이라고 "
    "단정할 수 없습니다."
)

st.caption(
    "이 분석은 단순 선형회귀모형이며, "
    "학교급이나 학교 규모 등 다른 요인의 영향은 "
    "통제하지 않았습니다."
)
