"""
══════════════════════════════════════════════════════════════
  [프로젝트 종료 후 개인 개선] ML 파이프라인 리팩터링
  ★ 03_modeling.py(팀 작성본)의 모델링 구조를 재검토하고 재학습 ★

  발견한 문제 (03_modeling.py):
    [1] sex/smoking_status/drinking_status/exercise_group을
        원-핫 인코딩 없이 정수 코드 그대로 SMOTE·모델 입력에 투입
        → SMOTE가 범주값을 연속형처럼 보간(예: exercise_group=1.64 같은
          존재하지 않는 값 생성), 모델도 그룹 간 등간격을 가정하게 됨
    [2] SMOTE를 train/test 분리 직후, RandomizedSearchCV(cv=5) 이전에
        학습셋 전체에 한 번만 적용 → CV 각 fold의 validation 쪽에도
        SMOTE로 만든 synthetic sample의 정보가 섞임 (leakage)
    [3] 원 발표자료(38p)를 보면 Test_Data.csv(673명)는 "손대지 않은
        최종 평가셋"이 아니라, 5개 모델 x 3개 보정방식을 비교해서
        최종 모델(Logistic+Isotonic)을 "선택"하는 데 이미 사용됨
        → 모델/보정방식 선택 자체가 test set에 의해 이뤄진 leakage
    [4] 04_streamlit_app.py가 이 모델을 저장/로드하지 않고 앱 실행 시
        전체 데이터로 별도 재학습 (SMOTE 없이, train/test 분리도 없이)
        → README에 보고한 성능과 실제 서비스 모델이 다른 모델이었음

  개선 방향:
    [1] SMOTENC로 범주형 변수를 인식시켜 리샘플링 (raw 코드값 유지)
        → 이후 원-핫 인코딩 → 분류기. 하나의 imblearn Pipeline으로 묶어
        RandomizedSearchCV에 통째로 전달 → 리샘플링이 각 CV train
        fold 내부에서만 수행되도록 구조 변경
    [2] 원본 3,363명을 다시 합쳐 새로 stratified 8:2 분할 (팀이 만든
        기존 Original_Train/Test 분할은 재사용하지 않음 — 위 [3] 이유)
    [3] 5개 후보 모델(Logistic/RF/XGBoost/LightGBM/CatBoost)을 학습
        데이터의 CV ROC-AUC만으로 비교해 최종 모델을 선택 (test set은
        비교에 전혀 사용하지 않음)
    [4] 선택된 모델만 Isotonic 보정 + threshold 선정(학습 OOF 기준)
        까지 마친 뒤, 새 test set으로 마지막 평가 1회만 수행
    [5] 최종 보정 파이프라인 전체를 joblib로 저장 → 04_streamlit_app.py는
        이 파일을 그대로 불러와 predict_proba만 호출 (재학습 없음, raw
        입력 컬럼 그대로 사용 가능)
══════════════════════════════════════════════════════════════
"""
import json
import numpy as np
import pandas as pd
import joblib

from scipy.stats import loguniform, randint, uniform
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import (
    train_test_split, RandomizedSearchCV, cross_val_predict, StratifiedKFold
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, brier_score_loss, roc_curve
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline as ImbPipeline

from feature_utils import encode_features, RAW_COLUMNS

RANDOM_STATE = 42
SPLIT_RANDOM_STATE = 2026  # 팀의 원래 분할(random_state=42)과 겹치지 않는 새 시드
CV_FOLDS = 5
DATA_PATH = "data/0325_hn_all(med).csv"  # 3,363명 전체 (본인 환경에 맞게 수정)
CATEGORICAL_IDX = [1, 2, 3, 4]  # age(0)만 연속형, 나머지 4개는 범주형

# ──────────────────────────────────────────────────────────────
# 데이터 로드 + 새 stratified 8:2 분할 (팀의 기존 분할 재사용 안 함)
# 주의: random_state=42로 다시 나누면 팀이 원래 썼던 것과 동일한 673명이
# 그대로 재생성됨(같은 데이터에 같은 seed → 같은 분할). 실제로 확인해보니
# 100% 일치했음. 그래서 분할에는 다른 시드(SPLIT_RANDOM_STATE)를 사용.
# ──────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH, low_memory=False)
X_all = df[RAW_COLUMNS].astype(float)
y_all = df["metabolic_syndrome"].astype(int)
w_all = df["wt_itvex"].astype(float)

X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    X_all, y_all, w_all, test_size=0.2, random_state=SPLIT_RANDOM_STATE, stratify=y_all
)
print(f"학습셋: {X_train.shape}, 양성 비율 {y_train.mean():.4f}")
print(f"테스트셋: {X_test.shape}, 양성 비율 {y_test.mean():.4f}  (새로 분할, 모델 선택에 미사용)")


def make_pipeline(clf):
    return ImbPipeline([
        ("smotenc", SMOTENC(categorical_features=CATEGORICAL_IDX, random_state=RANDOM_STATE)),
        ("encode", FunctionTransformer(encode_features)),
        ("clf", clf),
    ])


# ──────────────────────────────────────────────────────────────
# STEP 1: 5개 후보 모델을 학습 데이터 CV ROC-AUC만으로 비교
#         (test set은 이 비교에 전혀 사용하지 않음)
# ──────────────────────────────────────────────────────────────
candidates = {
    "Logistic": (
        make_pipeline(LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        {"clf__C": loguniform(1e-3, 1e2), "clf__penalty": ["l1", "l2"], "clf__solver": ["saga"]},
    ),
    "RandomForest": (
        make_pipeline(RandomForestClassifier(random_state=RANDOM_STATE)),
        {"clf__n_estimators": randint(100, 400), "clf__max_depth": randint(3, 12),
         "clf__min_samples_leaf": randint(1, 10)},
    ),
    "XGBoost": (
        make_pipeline(XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss")),
        {"clf__n_estimators": randint(100, 400), "clf__max_depth": randint(2, 8),
         "clf__learning_rate": loguniform(1e-2, 3e-1)},
    ),
    "LightGBM": (
        make_pipeline(LGBMClassifier(random_state=RANDOM_STATE, verbose=-1)),
        {"clf__n_estimators": randint(100, 400), "clf__max_depth": randint(2, 8),
         "clf__learning_rate": loguniform(1e-2, 3e-1)},
    ),
    "CatBoost": (
        make_pipeline(CatBoostClassifier(random_state=RANDOM_STATE, verbose=0)),
        {"clf__iterations": randint(100, 400), "clf__depth": randint(3, 8),
         "clf__learning_rate": loguniform(1e-2, 3e-1)},
    ),
}

print("\n🔍 5개 후보 모델 CV 비교 (SMOTENC -> 인코딩 -> 분류기, 전부 CV fold 내부에서 수행)...")
searches = {}
cv_scores = {}
for name, (pipe, grid) in candidates.items():
    search = RandomizedSearchCV(
        pipe, grid, n_iter=20, scoring="roc_auc",
        cv=CV_FOLDS, n_jobs=-1, random_state=RANDOM_STATE,
    )
    search.fit(X_train, y_train)
    searches[name] = search
    cv_scores[name] = search.best_score_
    print(f"  {name:14s} CV ROC-AUC {search.best_score_:.4f}  (params: {search.best_params_})")

best_name = max(cv_scores, key=cv_scores.get)
best_pipe = searches[best_name].best_estimator_
print(f"\n🏆 학습 CV 기준 최종 선택: {best_name} (CV ROC-AUC {cv_scores[best_name]:.4f})")

with open("cv_model_comparison.json", "w") as f:
    json.dump({k: float(v) for k, v in cv_scores.items()}, f, indent=2, ensure_ascii=False)

# ──────────────────────────────────────────────────────────────
# STEP 2: 선택된 모델만 Isotonic Calibration
# ──────────────────────────────────────────────────────────────
calibrated = CalibratedClassifierCV(estimator=best_pipe, method="isotonic", cv=CV_FOLDS)
calibrated.fit(X_train, y_train)

# ──────────────────────────────────────────────────────────────
# STEP 3: threshold 선정 — 보정된 확률의 OOF 예측(학습 데이터, test 미사용)
# ──────────────────────────────────────────────────────────────
skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
oof_calibrator = CalibratedClassifierCV(estimator=best_pipe, method="isotonic", cv=CV_FOLDS)
oof_proba = cross_val_predict(oof_calibrator, X_train, y_train, cv=skf, method="predict_proba", n_jobs=-1)[:, 1]

fpr, tpr, thr = roc_curve(y_train, oof_proba)
youden = tpr - fpr
best_idx = int(np.argmax(youden))
THRESHOLD = float(thr[best_idx])
print(f"\nYouden's J 최적 threshold (학습 OOF 기준): {THRESHOLD:.4f}  "
      f"(OOF Recall={tpr[best_idx]:.3f}, OOF FPR={fpr[best_idx]:.3f})")

# ──────────────────────────────────────────────────────────────
# STEP 4: 최종 평가 — 새 test set, 여기서 딱 한 번만 사용
# ──────────────────────────────────────────────────────────────
proba = calibrated.predict_proba(X_test)[:, 1]
pred = (proba >= THRESHOLD).astype(int)

metrics = {
    "Model": best_name,
    "Threshold": THRESHOLD,
    "ROC_AUC": roc_auc_score(y_test, proba),
    "Survey_weighted_ROC_AUC": roc_auc_score(y_test, proba, sample_weight=w_test),
    "PR_AUC": average_precision_score(y_test, proba),
    "Accuracy": accuracy_score(y_test, pred),
    "Precision": precision_score(y_test, pred),
    "Recall": recall_score(y_test, pred),
    "F1": f1_score(y_test, pred),
    "Brier": brier_score_loss(y_test, proba),
}
print(f"\n📊 최종 성능 (새 test set, N={len(y_test)}, threshold={THRESHOLD:.3f})")
for k, v in metrics.items():
    print(f"  {k:28s} {v}")

with open("refit_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)

joblib.dump(calibrated, "metabolic_risk_model.joblib")
print("\n✅ 저장 완료: metabolic_risk_model.joblib")
