
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.inspection import permutation_importance


# ==========================================
# 1. 페이지 설정
# ==========================================

st.set_page_config(
    page_title="영어성취도 요인 분석",
    page_icon="📚",
    layout="wide"
)

st.title("📚 영어성취도에 영향을 주는 요인 분석")

st.markdown("""
학생의 독서 환경과 설문 응답이 영어성취수준과
어떤 관계가 있는지 랜덤 포레스트 모델로 분석합니다.
""")

st.divider()


# ==========================================
# 2. 데이터 업로드
# ==========================================

st.header("1. 데이터 업로드")

uploaded_file = st.file_uploader(
    "공공데이터포털에서 다운로드한 XLSX 파일을 업로드하세요.",
    type=["xlsx", "xls", "csv"]
)

if uploaded_file is None:
    st.info("분석할 파일을 업로드해 주세요.")
    st.stop()


# ==========================================
# 3. 파일 읽기
# ==========================================

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

    st.error(f"파일을 읽을 수 없습니다: {e}")
    st.stop()


df.columns = df.columns.astype(str).str.strip()

# 완전히 빈 열 제거
df = df.dropna(axis=1, how="all")

st.success("데이터를 불러왔습니다.")

st.write("전체 데이터 크기:", df.shape)

with st.expander("원본 데이터 확인"):
    st.write("전체 열 이름:", list(df.columns))
    st.dataframe(df.head(15), use_container_width=True)


# ==========================================
# 4. 열 이름 자동 탐색
# ==========================================

def normalize_name(name):
    return (
        str(name)
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
    )


def find_column(keywords, exclude=None):

    exclude = exclude or []

    for col in df.columns:

        name = normalize_name(col)

        if col in exclude:
            continue

        if any(
            normalize_name(word) in name
            for word in keywords
        ):
            return col

    return None


# 영어성취도 열
target_default = find_column([
    "영어성취수준",
    "영어성취도",
    "영어성취",
    "영어수준",
    "영어"
])

# 보유도서수 열
book_default = find_column([
    "보유도서수",
    "보유도서",
    "도서수",
    "장서수"
])

# SQ1 설문 열
sq1_default = find_column([
    "SQ1"
])


# ==========================================
# 5. 변수 설정
# ==========================================

st.divider()
st.header("2. 분석 변수 설정")

st.caption(
    "데이터의 실제 열 이름을 탐색해 기본값을 설정합니다. "
    "자동 선택이 맞지 않으면 직접 수정해 주세요."
)

all_columns = list(df.columns)

# 종속변수 선택
target_index = (
    all_columns.index(target_default)
    if target_default in all_columns
    else 0
)

target_col = st.selectbox(
    "종속변수 Y - 영어성취수준",
    all_columns,
    index=target_index
)

# 독립변수 선택
feature_options = [
    "선택 안 함"
] + [
    col for col in all_columns
    if col != target_col
]

book_index = (
    feature_options.index(book_default)
    if book_default in feature_options
    else 0
)

sq1_index = (
    feature_options.index(sq1_default)
    if sq1_default in feature_options
    else 0
)

x1_col = st.selectbox(
    "독립변수 X1 - 보유도서수",
    feature_options,
    index=book_index
)

sq1_options = [
    col for col in feature_options
    if col != x1_col or col == "선택 안 함"
]

sq1_default_index = (
    sq1_options.index(sq1_default)
    if sq1_default in sq1_options
    else 0
)

x2_col = st.selectbox(
    "독립변수 X2 - SQ1 설문 응답",
    sq1_options,
    index=sq1_default_index
)

selected_features = [
    col for col in [x1_col, x2_col]
    if col != "선택 안 함"
]

if target_col in selected_features:
    st.error("종속변수와 독립변수는 서로 다른 열이어야 합니다.")
    st.stop()

if not selected_features:
    st.error("독립변수를 하나 이상 선택해 주세요.")
    st.stop()


# ==========================================
# 6. 데이터 정리
# ==========================================

st.divider()
st.header("3. 데이터 전처리")

data = df[[target_col] + selected_features].copy()

# 빈 문자열을 결측값으로 처리
data = data.replace(
    r"^\s*$",
    np.nan,
    regex=True
)

# 종속변수 결측값 제거
data = data.dropna(subset=[target_col])

# 문자열 형태의 종속변수 정리
data[target_col] = (
    data[target_col]
    .astype(str)
    .str.strip()
)

# 종속변수의 고유값 확인
target_counts = data[target_col].value_counts()

st.write("분석 대상 데이터:", len(data))
st.write("영어성취수준 종류:", data[target_col].nunique())

if data[target_col].nunique() < 2:
    st.error(
        "영어성취수준이 한 종류뿐입니다. "
        "다른 성취수준이 존재하는 열을 선택해 주세요."
    )
    st.stop()


# ==========================================
# 7. 독립변수 자동 처리
# ==========================================

X = data[selected_features].copy()
y = data[target_col].copy()

# 실제 값이 변하지 않는 변수 제거
usable_features = []
excluded_features = []

for col in selected_features:

    unique_count = X[col].nunique(dropna=True)

    if unique_count < 2:
        excluded_features.append(col)
    else:
        usable_features.append(col)

if excluded_features:

    st.warning(
        "다음 변수는 값이 모두 같거나 유효값이 없어 "
        "모델에서 제외했습니다: "
        + ", ".join(excluded_features)
    )

if not usable_features:

    st.error(
        "선택한 독립변수 모두 값이 동일합니다. "
        "다양한 값이 존재하는 다른 변수를 선택해야 합니다."
    )

    st.subheader("변수별 실제 데이터")

    for col in selected_features:
        st.write(col)
        st.write(data[col].value_counts(dropna=False))

    st.stop()


X = X[usable_features]


# ==========================================
# 8. 숫자형 / 범주형 자동 구분
# ==========================================

numeric_features = []
categorical_features = []

for col in usable_features:

    converted = pd.to_numeric(
        X[col].astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("권", "", regex=False)
        .str.strip(),
        errors="coerce"
    )

    valid_ratio = converted.notna().mean()

    # 보유도서수처럼 숫자로 해석 가능한 열
    if valid_ratio >= 0.8 and converted.nunique() > 1:

        X[col] = converted
        numeric_features.append(col)

    else:

        # SQ1 등 설문 응답은 범주형으로 처리
        X[col] = X[col].astype("string").astype(object)
        categorical_features.append(col)


st.subheader("변수 처리 결과")

st.write("숫자형 변수:", numeric_features)
st.write("범주형 변수:", categorical_features)
st.write("최종 분석 변수:", usable_features)

st.dataframe(
    X.head(20),
    use_container_width=True
)


# ==========================================
# 9. 학습 / 테스트 데이터 분리
# ==========================================

st.divider()
st.header("4. 모델 학습")

if len(X) < 10:

    st.error("모델 학습을 위해 데이터가 최소 10개 필요합니다.")
    st.stop()

class_counts = y.value_counts()

if len(class_counts) < 2:
    st.error("분류할 영어성취수준이 부족합니다.")
    st.stop()

# 각 성취수준이 학습/테스트에 나뉠 수 있을 때 층화 추출
n_classes = len(class_counts)
n_test = int(np.ceil(len(y) * 0.25))
n_train = len(y) - n_test

can_stratify = (
    class_counts.min() >= 2
    and n_test >= n_classes
    and n_train >= n_classes
)

try:

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y if can_stratify else None
    )

except ValueError as e:

    st.error(f"데이터 분리 오류: {e}")
    st.stop()


# ==========================================
# 10. 전처리 파이프라인
# ==========================================

transformers = []

if numeric_features:

    numeric_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ])

    transformers.append(
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        )
    )

if categorical_features:

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )
    ])

    transformers.append(
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    )


preprocessor = ColumnTransformer(
    transformers=transformers,
    remainder="drop"
)


# ==========================================
# 11. 랜덤 포레스트 모델
# ==========================================

model = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),
    (
        "classifier",
        RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=2,
            random_state=42,
            class_weight="balanced"
        )
    )
])


# ==========================================
# 12. 모델 학습
# ==========================================

try:

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

except Exception as e:

    st.error(f"모델 학습 오류: {e}")
    st.stop()


# ==========================================
# 13. 모델 평가
# ==========================================

accuracy = accuracy_score(
    y_test,
    predictions
)

st.success("모델 학습이 완료되었습니다.")

st.header("5. 모델 평가")

col1, col2, col3 = st.columns(3)

col1.metric(
    "모델 정확도",
    f"{accuracy * 100:.2f}%"
)

col2.metric(
    "학습 데이터",
    f"{len(X_train):,}개"
)

col3.metric(
    "테스트 데이터",
    f"{len(X_test):,}개"
)


# ==========================================
# 14. 혼동행렬
# ==========================================

st.divider()
st.header("6. 영어성취수준 예측 결과")

labels = model.classes_

cm = confusion_matrix(
    y_test,
    predictions,
    labels=labels
)

fig_cm = px.imshow(
    cm,
    x=[str(v) for v in labels],
    y=[str(v) for v in labels],
    text_auto=True,
    color_continuous_scale="Blues",
    labels={
        "x": "예측 성취수준",
        "y": "실제 성취수준",
        "color": "학생 수"
    },
    title="실제 성취수준과 예측 성취수준 비교"
)

st.plotly_chart(
    fig_cm,
    use_container_width=True
)


# ==========================================
# 15. 예측 결과 표
# ==========================================

result = X_test.copy()

result["실제 영어성취수준"] = y_test
result["예측 영어성취수준"] = predictions

st.subheader("예측 결과 데이터")

st.dataframe(
    result.head(50),
    use_container_width=True
)


# ==========================================
# 16. 독립변수 중요도
# ==========================================

st.divider()
st.header("7. 영어성취도에 영향을 주는 변수 분석")

try:

    importance = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=5,
        random_state=42,
        scoring="accuracy"
    )

    importance_df = pd.DataFrame({
        "변수": X_test.columns,
        "중요도": importance.importances_mean
    })

    importance_df = importance_df.sort_values(
        "중요도",
        ascending=False
    )

    fig_importance = px.bar(
        importance_df,
        x="중요도",
        y="변수",
        orientation="h",
        title="독립변수별 예측 중요도",
        color="중요도",
        color_continuous_scale="Blues"
    )

    st.plotly_chart(
        fig_importance,
        use_container_width=True
    )

    st.dataframe(
        importance_df,
        use_container_width=True
    )

except Exception as e:

    st.warning(f"변수 중요도 계산을 완료하지 못했습니다: {e}")


# ==========================================
# 17. 성취수준 분포
# ==========================================

st.divider()
st.header("8. 영어성취수준 분포")

distribution = (
    data[target_col]
    .value_counts()
    .rename_axis("영어성취수준")
    .reset_index(name="학생 수")
)

fig_distribution = px.bar(
    distribution,
    x="영어성취수준",
    y="학생 수",
    title="영어성취수준별 학생 수",
    color="학생 수",
    color_continuous_scale="Blues"
)

st.plotly_chart(
    fig_distribution,
    use_container_width=True
)


# ==========================================
# 18. 최종 해석
# ==========================================

st.divider()
st.header("9. 분석 결과 해석")

st.write(
    f"랜덤 포레스트 모델의 테스트 데이터 정확도는 "
    f"{accuracy * 100:.2f}%입니다."
)

if len(usable_features) > 0:

    best_feature = importance_df.iloc[0]["변수"]

    st.success(
        f"이번 모델에서 가장 높은 예측 중요도를 보인 변수는 "
        f"'{best_feature}'입니다."
    )

st.caption("""
주의사항:
1. 변수 중요도는 예측에 기여한 정도이며 인과관계를 증명하지 않습니다.
2. 이 데이터는 합성데이터이므로 실제 학생 전체에 대한 일반화에 주의해야 합니다.
3. SQ1의 구체적인 질문 내용과 응답 코드 의미는 원본 데이터 설명서를 확인해야 합니다.
4. 테스트 정확도는 학습에 사용하지 않은 데이터에 대한 예측 성능입니다.
""")
