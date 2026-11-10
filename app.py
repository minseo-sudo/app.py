
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr

# 페이지 설정
st.set_page_config(
    page_title="보유도서수와 영어성취도 분석",
    page_icon="📚",
    layout="wide"
)

# 제목
st.title("📚 보유도서수와 영어성취도의 관계 분석")
st.markdown("""
공공데이터를 활용하여 학교의 보유도서수와
영어성취도 사이의 관계를 회귀분석으로 살펴봅니다.
""")

st.divider()

# 데이터 업로드
st.header("1. 데이터 불러오기")

uploaded_file = st.file_uploader(
    "공공데이터 CSV 또는 Excel 파일을 업로드하세요.",
    type=["csv", "xlsx", "xls"]
)

if uploaded_file is not None:

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

        st.success("데이터를 성공적으로 불러왔습니다!")

        st.subheader("데이터 미리보기")
        st.dataframe(df.head(10), use_container_width=True)

        st.write("전체 데이터 크기:", df.shape)

        # 숫자형 데이터 후보 찾기
        numeric_candidates = []

        for col in df.columns:
            converted = pd.to_numeric(
                df[col].astype(str).str.replace(",", "", regex=False),
                errors="coerce"
            )

            if converted.notna().sum() >= 2:
                numeric_candidates.append(col)

        st.divider()
        st.header("2. 분석 변수 설정")

        if len(numeric_candidates) < 2:
            st.error("분석에 사용할 숫자형 열이 2개 이상 필요합니다.")
            st.stop()

        x_col = st.selectbox(
            "독립변수 X (보유도서수)",
            numeric_candidates,
            index=0
        )

        y_options = [
            col for col in numeric_candidates
            if col != x_col
        ]

        y_col = st.selectbox(
            "종속변수 Y (영어성취도)",
            y_options,
            index=0
        )

        # 분석 데이터 정리
        data = df[[x_col, y_col]].copy()

        for col in [x_col, y_col]:
            data[col] = pd.to_numeric(
                data[col].astype(str).str.replace(
                    ",", "", regex=False
                ),
                errors="coerce"
            )

        data = data.replace(
            [np.inf, -np.inf], np.nan
        ).dropna()

        st.write(f"분석에 사용되는 데이터 수: {len(data)}개")

        if len(data) < 3:
            st.error("회귀분석을 위해 유효한 데이터가 최소 3개 필요합니다.")
            st.stop()

        if data[x_col].nunique() < 2:
            st.error("독립변수의 값이 모두 같아 회귀분석을 할 수 없습니다.")
            st.stop()

        # 회귀분석
        X = data[[x_col]]
        y = data[y_col]

        model = LinearRegression()
        model.fit(X, y)

        predictions = model.predict(X)

        r2 = r2_score(y, predictions)
        rmse = np.sqrt(mean_squared_error(y, predictions))
        correlation, p_value = pearsonr(
            data[x_col], data[y_col]
        )

        slope = model.coef_[0]
        intercept = model.intercept_

        st.divider()
        st.header("3. 회귀분석 결과")

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
            "평균제곱근오차 (RMSE)",
            f"{rmse:.4f}"
        )

        st.subheader("회귀방정식")

        st.latex(
            f"Y = {slope:.4f}X + {intercept:.4f}"
        )

        st.write(f"회귀계수: {slope:.4f}")
        st.write(f"절편: {intercept:.4f}")
        st.write(f"p-value: {p_value:.6f}")

        # 산점도 및 회귀선
        st.divider()
        st.header("4. 데이터 시각화")

        chart_data = data.copy()
        chart_data["예측 영어성취도"] = predictions

        fig = px.scatter(
            chart_data,
            x=x_col,
            y=y_col,
            title="보유도서수와 영어성취도의 관계",
            labels={
                x_col: "보유도서수",
                y_col: "영어성취도"
            },
            opacity=0.7
        )

        sorted_data = data.sort_values(x_col)

        fig.add_scatter(
            x=sorted_data[x_col],
            y=model.predict(sorted_data[[x_col]]),
            mode="lines",
            name="선형회귀선",
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

        # 결과 해석
        st.divider()
        st.header("5. 분석 결과 해석")

        if slope > 0:
            st.success(
                "보유도서수가 증가할수록 영어성취도가 "
                "높아지는 양의 관계가 관찰됩니다."
            )
        elif slope < 0:
            st.info(
                "보유도서수가 증가할수록 영어성취도가 "
                "낮아지는 음의 관계가 관찰됩니다."
            )
        else:
            st.info("회귀계수가 0에 가까운 관계입니다.")

        st.write(
            f"보유도서수가 1단위 증가할 때 "
            f"예측 영어성취도는 약 {slope:.4f}만큼 변합니다."
        )

        st.write(
            f"결정계수 R²는 {r2:.4f}로, "
            f"이 단순 선형회귀모형이 영어성취도 변동의 "
            f"{r2 * 100:.2f}%를 설명합니다."
        )

        if p_value < 0.05:
            st.write(
                "상관관계 검정에서 통계적으로 유의한 "
                "선형 상관관계가 관찰됩니다 (p < 0.05)."
            )
        else:
            st.write(
                "상관관계 검정에서 통계적으로 유의한 "
                "선형 상관관계를 확인하지 못했습니다 (p ≥ 0.05)."
            )

        st.caption(
            "주의: 상관관계와 회귀분석 결과만으로 "
            "보유도서수가 영어성취도의 원인이라고 단정할 수 없습니다."
        )

    except Exception as e:
        st.error(f"데이터 처리 중 오류가 발생했습니다: {e}")

else:
    st.info("먼저 공공데이터 파일을 업로드해 주세요.")
