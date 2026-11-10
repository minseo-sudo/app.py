
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
어떤 관계가 있는지 분석합니다.
""")

st.divider()


# ==========================================
# 2. 데이터 업로드
# ==========================================

st.header("1. 데이터 업로드")

uploaded_file = st.file_uploader(
    "공공데이터포털에서 다운로드한 파일을 업로드하세요.",
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

    df.columns = df.columns.astype(str).str.strip()

    df = df.dropna(axis=1, how="all")
    df = df.loc[:, ~df.columns.duplicated()]

except Exception as e:

    st.error(f"파일을 읽을 수 없습니다: {e}")
    st.stop()


if df.empty:
    st.error("데이터가 비어 있습니다.")
    st.stop()


st.success("데이터를 불러왔습니다.")

st.write("전체 데이터 크기:", df.shape)

with st.expander("원본 데이터 확인"):

    st.write("전체 열 이름:", list(df.columns))

    st.dataframe(
        df.head(20),
        use_container_width=True
    )


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


def find_column(keywords):

    for col in df.columns:

        name = normalize_name(col)

        if any(
            normalize_name(word) in name
            for word in keywords
        ):
            return col

    return None


target_default = find_column([
    "영어성취수준",
    "영어성취도",
    "영어성취",
    "영어수준",
    "영어"
])

book_default = find_column([
    "보유도서수",
    "보유도서",
    "도서수",
    "장서수"
])

sq1_default = find_column([
    "SQ1"
])


# ==========================================
# 5. 변수 설정
# ==========================================

st.divider()

st.header("2. 분석 변수 설정")

all_columns = list(df.columns)

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

x1_col = st.selectbox(
    "독립변수 X1 - 보유도서수",
    feature_options,
    index=book_index
)


sq1_options = [
    col for col in feature_options
    if col != x1_col or col == "선택 안 함"
]

sq1_index = (
    sq1_options.index(sq1_default)
    if sq1_default in sq1_options
    else 0
)

x2_col = st.selectbox(
    "독립변수 X2 - SQ1 설문 응답",
    sq1_options,
    index=sq1_index
)


selected_features = [
    col for col in [x1_col, x2_col]
    if col != "선택 안 함"
]


if not selected_features:

    st.error("독립변수를 하나 이상 선택해 주세요.")
    st.stop()


if target_col in selected_features:

    st.error(
        "종속변수와 독립변수는 서로 달라야 합니다."
    )
    st.stop()


# ==========================================
# 6. 데이터 전처리
# ==========================================

st.divider()

st.header("3. 데이터 전처리")

data = df[
    [target_col] + selected_features
].copy()

data = data.replace(
    r"^\s*$",
    np.nan,
    regex=True
)

data = data.dropna(
    subset=[target_col]
)

data[target_col] = (
    data[target_col]
    .astype(str)
    .str.strip()
)

X = data[selected_features].copy()

y = data[target_col].copy()


# ==========================================
# 7. 종속변수 확인
# ==========================================

st.subheader("영어성취수준 분포")

st.write(
    y.value_counts()
)

if y.nunique() < 2:

    st.error(
        "영어성취수준이 한 종류뿐이므로 "
        "분류 모델을 학습할 수 없습니다."
    )
    st.stop()


# ==========================================
# 8. 독립변수 확인 및 자동 제외
# ==========================================

usable_features = []
excluded_features = []

for col in selected_features:

    unique_count = X[col].nunique(
        dropna=True
    )

    if unique_count < 2:

        excluded_features.append(col)

    else:

        usable_features.append(col)


if excluded_features:

    st.warning(
        "다음 변수는 값이 모두 같거나 유효값이 없어 "
        "분석에서 제외했습니다: "
        + ", ".join(excluded_features)
    )


if not usable_features:

    st.error(
        "사용 가능한 독립변수가 없습니다. "
        "다른 변수를 선택해 주세요."
    )

    st.stop()


X = X[usable_features].copy()


# ==========================================
# 9. 숫자형 / 범주형 변수 구분
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

    if valid_ratio >= 0.8:

        X[col] = converted
        numeric_features.append(col)

    else:

        X[col] = X[col].astype("string").astype(object)
        categorical_features.append(col)


st.subheader("최종 분석 변수")

st.write("숫자형 변수:", numeric_features)
st.write("범주형 변수:", categorical_features)

st.write("분석 데이터 수:", len(X))

st.dataframe(
    X.head(20),
    use_container_width=True
)


# ==========================================
# 10. 학습 데이터 분리
# ==========================================

st.divider()

st.header("4. 모델 학습")

if len(X) < 10:

    st.error(
        "모델 학습을 위해 최소 10개의 데이터가 필요합니다."
    )
    st.stop()


class_counts = y.value_counts()

n_classes = len(class_counts)

test_size = 0.25

n_test = int(np.ceil(len(y) * test_size))
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
        test_size=test_size,
        random_state=42,
        stratify=y if can_stratify else None
    )

except ValueError as e:

    st.error(f"데이터 분리 오류: {e}")
    st.stop()


# ==========================================
# 11. 전처리 파이프라인
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
            OneHotEncoder(
                handle_unknown="ignore"
            )
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
# 12. 랜덤 포레스트 모델
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
            class_weight=None
        )
    )
])


# ==========================================
# 13. 모델 학습
# ==========================================

try:

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

except Exception as e:

    st.error(
        f"모델 학습 오류: {e}"
    )

    st.stop()


st.success("모델 학습이 완료되었습니다!")


# ==========================================
# 14. 모델 평가
# ==========================================

st.divider()

st.header("5. 모델 평가")

accuracy = accuracy_score(
    y_test,
    predictions
)

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
# 15. 혼동행렬
# ==========================================

st.header("6. 영어성취수준 예측 결과")

labels = model.named_steps[
    "classifier"
].classes_

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
    title="실제 성취수준과 예측 성취수준 비교",
    labels={
        "x": "예측 성취수준",
        "y": "실제 성취수준",
        "color": "학생 수"
    }
)

st.plotly_chart(
    fig_cm,
    use_container_width=True
)


# ==========================================
# 16. 예측 결과
# ==========================================

st.subheader("예측 결과 데이터")

result = X_test.copy()

result["실제 영어성취수준"] = y_test
result["예측 영어성취수준"] = predictions

st.dataframe(
    result.head(50),
    use_container_width=True
)


# ==========================================
# 17. 변수 중요도
# ==========================================

st.divider()

st.header("7. 영어성취도 영향 요인 분석")

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
        ascending=True
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
        importance_df.sort_values(
            "중요도",
            ascending=False
        ),
        use_container_width=True
    )

except Exception as e:

    st.warning(
        f"변수 중요도 계산 오류: {e}"
    )


# ==========================================
# 18. 영어성취수준 분포
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
# 19. 결과 해석
# ==========================================

st.divider()

st.header("9. 최종 분석 결과")

st.write(
    f"이번 분석에서 사용한 독립변수는 "
    f"{len(usable_features)}개입니다."
)

st.write(
    "사용 변수:",
    ", ".join(usable_features)
)

st.write(
    f"모델의 테스트 데이터 정확도는 "
    f"{accuracy * 100:.2f}%입니다."
)

st.success(
    "영어성취도 요인 분석이 완료되었습니다."
)

st.caption("""
주의사항:
- 변수 중요도는 예측 기여도를 나타내며 인과관계를 증명하지 않습니다.
- 합성데이터의 결과를 실제 학생 전체에 일반화할 수 없습니다.
- SQ1 설문 문항의 정확한 의미는 데이터 설명서를 확인해야 합니다.
- 독립변수의 변동이 없으면 해당 변수는 자동 제외됩니다.
""")
