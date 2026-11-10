
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr


# =====================================
# 1. 페이지 설정
# =====================================

st.set_page_config(
    page_title="보유도서수와 영어성취도 분석",
    page_icon="📚",
    layout="wide"
)

st.title("📚 보유도서수와 영어성취도 관계 분석")

st.markdown("""
공공데이터를 활용하여 학교의 보유도서수와
영어성취도 사이의 관계를 분석합니다.
""")

st.divider()


# =====================================
# 2. 데이터 업로드
# =====================================

st.header("1. 데이터 불러오기")

uploaded_file = st.file_uploader(
    "CSV 또는 Excel 파일 업로드",
    type=["csv", "xlsx", "xls"]
)

if uploaded_file is None:
    st.info("분석할 데이터 파일을 업로드해 주세요.")
    st.stop()


# =====================================
# 3. 데이터 읽기
# =====================================

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

except Exception as e:

    st.error(f"파일 읽기 오류: {e}")
    st.stop()


# =====================================
# 4. 원본 데이터 확인
# =====================================

st.success("데이터 불러오기 성공!")

st.header("2. 원본 데이터 확인")

st.write("전체 데이터 크기:", df.shape)

st.subheader("전체 열 이름")

st.write(list(df.columns))

st.subheader("데이터 미리보기")

st.dataframe(
    df.head(20),
    use_container_width=True
)


# =====================================
# 5. 숫자 변환 함수
# =====================================

def clean_number(series):

    cleaned = (
        series.astype(str)
        .str.strip()
        .str.replace(",", "", regex=False)
        .str.replace("권", "", regex=False)
        .str.replace("%", "", regex=False)
    )

    return pd.to_numeric(
        cleaned,
        errors="coerce"
    )


# =====================================
# 6. 숫자형 열 탐색
# =====================================

numeric_candidates = []

for col in df.columns:

    values = clean_number(df[col])

    if values.notna().sum() >= 2:

        numeric_candidates.append(col)


st.header("3. 분석 변수 설정")

if len(numeric_candidates) < 2:

    st.error("숫자형으로 변환 가능한 열이 부족합니다.")
    st.stop()


# 보유도서수 후보
book_candidates = [
    col for col in df.columns
    if any(
        word in str(col)
        for word in ["보유도서수", "도서수", "장서수"]
    )
]

# 영어성취도 후보
english_candidates = [
    col for col in df.columns
    if any(
        word in str(col)
        for word in ["영어성취도", "영어"]
    )
]


st.subheader("자동 탐색 결과")

st.write("보유도서수 후보:", book_candidates)
st.write("영어성취도 후보:", english_candidates)


# =====================================
# 7. 독립변수 선택
# =====================================

x_options = numeric_candidates

x_default = (
    book_candidates[0]
    if book_candidates and book_candidates[0] in numeric_candidates
    else numeric_candidates[0]
)

x_col = st.selectbox(
    "독립변수 X (보유도서수)",
    x_options,
    index=x_options.index(x_default)
)


# =====================================
# 8. 종속변수 선택
# =====================================

y_options = [
    col for col in numeric_candidates
    if col != x_col
]

if not y_options:

    st.error("종속변수로 선택할 수 있는 열이 없습니다.")
    st.stop()

y_default = next(
    (
        col for col in english_candidates
        if col in y_options
    ),
    y_options[0]
)

y_col = st.selectbox(
    "종속변수 Y (영어성취도)",
    y_options,
    index=y_options.index(y_default)
)


# =====================================
# 9. 선택한 변수 확인
# =====================================

st.info(
    f"독립변수: {x_col} / 종속변수: {y_col}"
)


# =====================================
# 10. 데이터 정리
# =====================================

data = pd.DataFrame()

data["X"] = clean_number(df[x_col])
data["Y"] = clean_number(df[y_col])

data = data.replace(
    [np.inf, -np.inf],
    np.nan
).dropna()


# =====================================
# 11. 데이터 진단
# =====================================

st.divider()

st.header("4. 데이터 진단")

c1, c2, c3 = st.columns(3)

c1.metric("전체 행 개수", len(df))
c2.metric("분석 가능 데이터", len(data))
c3.metric("보유도서수 고유값", data["X"].nunique())

st.subheader("독립변수 X의 통계")

st.write(data["X"].describe())

st.subheader("종속변수 Y의 통계")

st.write(data["Y"].describe())

st.subheader("독립변수 실제 값")

st.write(
    data["X"].value_counts().head(30)
)

st.subheader("분석 데이터 미리보기")

st.dataframe(
    data.head(30),
    use_container_width=True
)


# =====================================
# 12. 원본 열의 값 확인
# =====================================

st.divider()

st.header("5. 원본 변수 확인")

with st.expander("보유도서수 관련 열 확인"):

    for col in book_candidates:

        st.write("열 이름:", col)

        st.write(
            "실제 값:",
            df[col].head(20).tolist()
        )

        st.write(
            "고유값 개수:",
            df[col].nunique()
        )


with st.expander("영어 관련 열 확인"):

    for col in english_candidates:

        st.write("열 이름:", col)

        st.write(
            "실제 값:",
            df[col].head(20).tolist()
        )

        st.write(
            "고유값 개수:",
            df[col].nunique()
        )


# =====================================
# 13. 회귀분석 가능 여부 확인
# =====================================

if len(data) < 3:

    st.error("분석 가능한 데이터가 3개 미만입니다.")
    st.stop()


if data["X"].nunique() < 2:

    st.error(
        "선택한 독립변수의 값이 모두 같습니다. "
        "보유도서수 열을 다시 확인해야 합니다."
    )

    st.subheader("문제 원인 확인")

    st.write("선택한 X:", x_col)

    st.write("실제 값:")
    st.write(data["X"].value_counts())

    st.write("원본 데이터 전체 열:")
    st.write(list(df.columns))

    st.warning(
        "보유도서수 값이 실제로 모두 같다면 "
        "현재 데이터만으로 해당 변수의 회귀분석을 "
        "수행할 수 없습니다."
    )

    st.stop()


if data["Y"].nunique() < 2:

    st.error("영어성취도 값이 모두 같습니다.")
    st.stop()


# =====================================
# 14. 회귀분석
# =====================================

X = data[["X"]]
y = data["Y"]

model = LinearRegression()

model.fit(X, y)

predictions = model.predict(X)


# =====================================
# 15. 평가 지표
# =====================================

r2 = r2_score(y, predictions)

rmse = np.sqrt(
    mean_squared_error(y, predictions)
)

correlation, p_value = pearsonr(
    data["X"],
    data["Y"]
)

slope = model.coef_[0]
intercept = model.intercept_


# =====================================
# 16. 분석 결과
# =====================================

st.divider()

st.header("6. 회귀분석 결과")

c1, c2, c3 = st.columns(3)

c1.metric(
    "상관계수",
    f"{correlation:.4f}"
)

c2.metric(
    "결정계수 R²",
    f"{r2:.4f}"
)

c3.metric(
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


# =====================================
# 17. 산점도 및 회귀선
# =====================================

st.divider()

st.header("7. 그래프 모델링")

fig = px.scatter(
    data,
    x="X",
    y="Y",
    title="보유도서수와 영어성취도의 관계",
    labels={
        "X": "보유도서수",
        "Y": "영어성취도"
    },
    opacity=0.65
)

sorted_data = data.sort_values("X")

fig.add_scatter(
    x=sorted_data["X"],
    y=model.predict(
        sorted_data[["X"]]
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


# =====================================
# 18. 분석 결과 해석
# =====================================

st.divider()

st.header("8. 분석 결과 해석")

if slope > 0:

    st.success(
        "보유도서수가 증가할수록 "
        "영어성취도가 높아지는 양의 관계가 나타납니다."
    )

elif slope < 0:

    st.info(
        "보유도서수가 증가할수록 "
        "영어성취도가 낮아지는 음의 관계가 나타납니다."
    )

else:

    st.info(
        "보유도서수와 영어성취도 사이의 "
        "선형적인 변화가 나타나지 않습니다."
    )


st.write(
    f"보유도서수가 1단위 증가할 때 "
    f"예측 영어성취도는 {slope:.4f}만큼 변합니다."
)

st.write(
    f"결정계수 R²: {r2:.4f}"
)

if p_value < 0.05:

    st.write(
        "통계적으로 유의한 선형 상관관계가 관찰됩니다."
    )

else:

    st.write(
        "통계적으로 유의한 선형 상관관계를 "
        "확인하지 못했습니다."
    )


st.caption(
    "주의: 회귀분석은 상관관계를 분석하는 것이며 "
    "인과관계를 증명하지 않습니다."
)
