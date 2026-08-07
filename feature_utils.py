"""
공용 피처 인코딩 함수 (03_modeling.py 재학습 파이프라인 / 04_streamlit_app.py 공통 사용).
raw 컬럼(age, sex, smoking_status, drinking_status, exercise_group)을
모델 입력용 원-핫 컬럼(age, male, smoke_past, smoke_current, drink_current, ex_1, ex_2, ex_3)으로 변환한다.
Pipeline 안에서 SMOTENC 리샘플링 다음 단계로 들어가므로, SMOTENC가 raw 범주형 코드를
그대로 유지한 채(보간 없이) 합성 샘플을 만든 뒤 이 함수가 원-핫으로 바꾼다.
"""
import numpy as np
import pandas as pd

RAW_COLUMNS = ["age", "sex", "smoking_status", "drinking_status", "exercise_group"]
ENCODED_COLUMNS = [
    "age", "male", "smoke_past", "smoke_current",
    "drink_current", "ex_1", "ex_2", "ex_3",
]


def encode_features(X):
    """X: array-like 또는 DataFrame, 컬럼 순서는 RAW_COLUMNS와 동일해야 함."""
    df = pd.DataFrame(X, columns=RAW_COLUMNS).astype(float)

    out = pd.DataFrame(index=df.index)
    out["age"] = df["age"]
    out["male"] = (df["sex"].round() == 1).astype(int)
    out["smoke_past"] = (df["smoking_status"].round() == 1).astype(int)
    out["smoke_current"] = (df["smoking_status"].round() == 2).astype(int)
    out["drink_current"] = (df["drinking_status"].round() == 1).astype(int)
    out["ex_1"] = (df["exercise_group"].round() == 1).astype(int)
    out["ex_2"] = (df["exercise_group"].round() == 2).astype(int)
    out["ex_3"] = (df["exercise_group"].round() == 3).astype(int)
    return out[ENCODED_COLUMNS]
