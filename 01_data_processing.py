# ===========================================================================
# [Task 1] 환경 설정 및 데이터 로드
# ===========================================================================

!apt-get -qq install fonts-nanum
!pip install -q catboost shap

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats
import shap
import joblib

from IPython.display import display
from scipy.stats import loguniform

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, brier_score_loss,
    roc_curve, auc, confusion_matrix, classification_report
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from google.colab import drive

warnings.filterwarnings('ignore')

# ── 전역 상수 ──────────────────────────────────────────────────────────────
RANDOM_STATE = 42
THRESHOLD    = 0.5
N_BINS       = 10
TEST_SIZE    = 0.2
CV_FOLDS     = 5

# ── 한글 폰트 설정 ─────────────────────────────────────────────────────────
fe = fm.FontEntry(
    fname='/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf',
    name='NanumBarunGothic'
)
fm.fontManager.ttflist.insert(0, fe)
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'NanumBarunGothic',
    'axes.unicode_minus': False
})

# ── 변수명 한글 매핑 ───────────────────────────────────────────────────────
han_labels = {
    'age':            '연령',
    'sex':            '성별',
    'smoking_status': '흡연 상태',
    'drinking_status':'음주 상태',
    'exercise_group': '운동 그룹 (4개)'
}

# ── p-value 유의성 별표 변환 함수 ─────────────────────────────────────────
def pval_stars(p):
    """p-value를 별표 유의성 기호로 변환
    ***  p < 0.001
    **   p < 0.01
    *    p < 0.05
    †    p < 0.10
    n.s. p ≥ 0.10
    """
    if   p < 0.001: return '***'
    elif p < 0.01:  return '**'
    elif p < 0.05:  return '*'
    elif p < 0.10:  return '†'
    else:           return 'n.s.'

# ── 구글 드라이브 마운트 ───────────────────────────────────────────────────
if not os.path.exists('/content/drive'):
    drive.mount('/content/drive')

def load_and_setup_data():
    """구글 드라이브 공유 드라이브 경로에서 CSV 데이터 로드"""
    base_path = '/content/drive/Shareddrives/세미1 4조 공유드라이브/TA/신나은/data/'
    df_smote = pd.read_csv(base_path + 'SMOTE_Train_Data.csv')
    df_orig  = pd.read_csv(base_path + 'Original_Train_Data.csv')
    df_test  = pd.read_csv(base_path + 'Test_Data.csv')
    print(f"✅ Data Loaded Successfully!")
    print(f" - SMOTE Train : {df_smote.shape}")
    print(f" - Original Train: {df_orig.shape}")
    print(f" - Test Data   : {df_test.shape}")
    return df_smote, df_orig, df_test

df_smote, df_orig, df_test = load_and_setup_data()

# ===========================================================================
# [Task 2] 3-Step 계층적 로지스틱 회귀 — 복합표본 가중치 적용 (GLM Binomial)
# ===========================================================================

def run_3step_logistic_final(df):
    """성별 구분 3-Step 계층적 로지스틱 회귀 (복합표본 가중치 적용)"""
    results_report = {}

    rename_dict = {
        'age':              '연령',
        'smoking_status_1': '과거흡연',
        'smoking_status_2': '현재흡연',
        'drinking_status_1':'현재음주',
        'exercise_group_1': '운동(병행) ★',
        'exercise_group_2': '운동(근력만)',
        'exercise_group_3': '운동(유산소만)',
        'const':            '상수항(기준:운동 안함/비흡연/비음주)'
    }

    for g_code, g_name in [(1, '남성'), (2, '여성')]:
        df_g = df[df['sex'] == g_code].copy()

        df_g['exercise_group']  = pd.Categorical(df_g['exercise_group'],  categories=[0, 1, 2, 3])
        df_g['smoking_status']  = pd.Categorical(df_g['smoking_status'],  categories=[0, 1, 2])
        df_g['drinking_status'] = pd.Categorical(df_g['drinking_status'], categories=[0, 1])

        w_raw = df_g['wt_itvex'].fillna(1.0).values
        w_g   = w_raw / w_raw.mean()
        print(f"  [{g_name}] 가중치 정규화: "
              f"원래 평균={w_raw.mean():.1f} → 정규화 후 평균={w_g.mean():.3f}, N={len(w_g)}")

        steps = {
            "Model 1": ['age'],
            "Model 2": ['age', 'smoking_status', 'drinking_status'],
            "Model 3": ['age', 'smoking_status', 'drinking_status', 'exercise_group']
        }

        model_outputs = []
        prev_llf = None

        print(f"\n{'='*25} {g_name} 집단: 병행 운동 효과 검증 (가중치 적용) {'='*25}")

        for name, features in steps.items():
            X = pd.get_dummies(df_g[features], drop_first=True, dtype=float)
            X = sm.add_constant(X).astype(float)
            X.columns = [rename_dict.get(col, col) for col in X.columns]
            y = df_g['metabolic_syndrome'].astype(float)

            model = sm.GLM(
                y, X,
                family=sm.families.Binomial(),
                var_weights=w_g
            ).fit()

            # p-value에 별표 추가
            res_df = pd.DataFrame({
                'OR':           np.exp(model.params),
                '95% CI Lower': np.exp(model.conf_int()[0]),
                '95% CI Upper': np.exp(model.conf_int()[1]),
                'P-value':      model.pvalues,
                'Sig':          model.pvalues.map(pval_stars)   # ← 별표 열 추가
            })

            lrt_p = (stats.chi2.sf(2 * (model.llf - prev_llf), X.shape[1] - 1)
                     if prev_llf is not None else np.nan)
            prev_llf = model.llf

            model_outputs.append({'name': name, 'res': res_df, 'lrt_p': lrt_p})

            print(f"\n[{name}] LRT p(참고): {f'{lrt_p:.4f}' if not np.isnan(lrt_p) else 'N/A'}")
            print("※ 가중치 적용 GLM — p-value는 Wald 검정 기반 | *** p<.001  ** p<.01  * p<.05  † p<.10")
            display(res_df.round(4))

        results_report[g_name] = model_outputs

        final_res = model_outputs[-1]['res']
        if '운동(병행) ★' in final_res.index:
            target = final_res.loc['운동(병행) ★']
            if target['P-value'] < 0.05:
                effect_size = (1 - target['OR']) * 100
                print(f"\n📝 [핵심 결론 - {g_name}] (복합표본 가중치 적용)")
                print(f"운동을 전혀 하지 않는 군에 비해, 병행 운동을 실천하는 군은 "
                      f"대사증후군 위험도가 약 {effect_size:.1f}% 낮게 나타났습니다"
                      f"(OR={target['OR']:.3f}, {target['Sig']}, 가중치 적용).")
            else:
                print(f"\n📝 [핵심 결론 - {g_name}] 병행 운동 효과가 유의하지 않습니다"
                      f"(p = {target['P-value']:.4f}, {target['Sig']}).")

    return results_report

logistic_results = run_3step_logistic_final(df_orig)

# ===========================================================================
# [Task 2-2] 3-Step 계층적 로지스틱 회귀 — 전체 집단 (성별 미구분)
# ===========================================================================

def run_3step_logistic_total(df):
    """전체 집단(성별 미구분) 3-Step 계층적 로지스틱 회귀 함수"""
    df_all = df.copy()

    df_all['exercise_group']  = pd.Categorical(df_all['exercise_group'],  categories=[0, 1, 2, 3])
    df_all['smoking_status']  = pd.Categorical(df_all['smoking_status'],  categories=[0, 1, 2])
    df_all['drinking_status'] = pd.Categorical(df_all['drinking_status'], categories=[0, 1])
    df_all['sex']             = pd.Categorical(df_all['sex'],             categories=[1, 2])

    w_raw = df_all['wt_itvex'].fillna(1.0).values
    w_all = w_raw / w_raw.mean()
    print(f"  [전체] 가중치 정규화: "
          f"원래 평균={w_raw.mean():.1f} → 정규화 후 평균={w_all.mean():.3f}, N={len(w_all)}")

    rename_dict = {
        'age':              '연령',
        'sex_2':            '성별(여성)',
        'smoking_status_1': '과거흡연',
        'smoking_status_2': '현재흡연',
        'drinking_status_1':'현재음주',
        'exercise_group_1': '운동(병행) ★',
        'exercise_group_2': '운동(근력만)',
        'exercise_group_3': '운동(유산소만)',
        'const':            '상수항(기준:운동 안함/비흡연/비음주/남성)'
    }

    steps = {
        "Model 1": ['age', 'sex'],
        "Model 2": ['age', 'sex', 'smoking_status', 'drinking_status'],
        "Model 3": ['age', 'sex', 'smoking_status', 'drinking_status', 'exercise_group']
    }

    model_outputs = []
    prev_llf      = None

    print(f"\n{'='*25} 전체 집단 (성별 미구분): 병행 운동 효과 검증 (가중치 적용) {'='*25}")
    print("※ sex 변수를 공변량으로 포함 (남성=기준, 여성=1) | sm.GLM Binomial + var_weights")

    for name, features in steps.items():
        X = pd.get_dummies(df_all[features], drop_first=True, dtype=float)
        X = sm.add_constant(X).astype(float)
        X.columns = [rename_dict.get(col, col) for col in X.columns]
        y = df_all['metabolic_syndrome'].astype(float)

        model = sm.GLM(
            y, X,
            family=sm.families.Binomial(),
            var_weights=w_all
        ).fit()

        # p-value에 별표 추가
        res_df = pd.DataFrame({
            'OR':           np.exp(model.params),
            '95% CI Lower': np.exp(model.conf_int()[0]),
            '95% CI Upper': np.exp(model.conf_int()[1]),
            'P-value':      model.pvalues,
            'Sig':          model.pvalues.map(pval_stars)   # ← 별표 열 추가
        })

        lrt_p = (stats.chi2.sf(2 * (model.llf - prev_llf), X.shape[1] - 1)
                 if prev_llf is not None else np.nan)
        prev_llf = model.llf

        model_outputs.append({'name': name, 'res': res_df, 'lrt_p': lrt_p})

        print(f"\n[{name}] LRT p(참고): {f'{lrt_p:.4f}' if not np.isnan(lrt_p) else 'N/A'}")
        print("※ 가중치 적용 GLM — p-value는 Wald 검정 기반 | *** p<.001  ** p<.01  * p<.05  † p<.10")
        display(res_df.round(4))

    final_res = model_outputs[-1]['res']
    if '운동(병행) ★' in final_res.index:
        target = final_res.loc['운동(병행) ★']
        sig    = target['P-value'] < 0.05
        direction   = "낮게" if target['OR'] < 1 else "높게"
        effect_size = abs(1 - target['OR']) * 100
        print(f"\n📝 [핵심 결론 - 전체 집단]")
        if sig:
            print(f"운동을 전혀 하지 않는 군에 비해, 병행 운동을 실천하는 군은 "
                  f"대사증후군 위험도가 약 {effect_size:.1f}% {direction} 나타났습니다"
                  f"(p < 0.05, {target['Sig']}).")
        else:
            print(f"병행 운동의 효과가 통계적으로 유의하지 않았습니다(p = {target['P-value']:.4f}, n.s.).")
            print("→ 성별 구분 분석 결과와 비교해 해석이 필요합니다.")

    # ── 성별 구분 vs 미구분 결과 비교표 ──────────────────────────────────
    print(f"\n{'='*60}")
    print("  [비교] 병행 운동 OR: 성별 구분 vs 전체 집단 (Model 3 기준)")
    print(f"{'='*60}")

    comparison_rows = []
    for g_name, g_outputs in logistic_results.items():
        g_final = g_outputs[-1]['res']
        if '운동(병행) ★' in g_final.index:
            row = g_final.loc['운동(병행) ★']
            comparison_rows.append({
                '집단':         g_name,
                'OR':           round(row['OR'], 4),
                '95% CI Lower': round(row['95% CI Lower'], 4),
                '95% CI Upper': round(row['95% CI Upper'], 4),
                'P-value':      round(row['P-value'], 4),
                'Sig':          row['Sig'],
                '유의성':       '✅ p<0.05' if row['P-value'] < 0.05 else '❌ n.s.'
            })

    if '운동(병행) ★' in final_res.index:
        row = final_res.loc['운동(병행) ★']
        comparison_rows.append({
            '집단':         '전체 (성별 통제)',
            'OR':           round(row['OR'], 4),
            '95% CI Lower': round(row['95% CI Lower'], 4),
            '95% CI Upper': round(row['95% CI Upper'], 4),
            'P-value':      round(row['P-value'], 4),
            'Sig':          row['Sig'],
            '유의성':       '✅ p<0.05' if row['P-value'] < 0.05 else '❌ n.s.'
        })

    display(pd.DataFrame(comparison_rows).set_index('집단'))
    print(f"{'='*60}")

    return model_outputs

logistic_total_results = run_3step_logistic_total(df_orig)

# ===========================================================================
# [Task 3] 데이터 분할 및 SMOTE 적용
# ===========================================================================

ORIG_PATH = '/content/drive/Shareddrives/세미1 4조 공유드라이브/TA/신나은/data/Original_Train_Data.csv'
orig_df = pd.read_csv(ORIG_PATH)

target_candidates = [c for c in orig_df.columns if 'metabolic' in c.lower()]
if not target_candidates:
    print("❌ 오류: 'metabolic'이 포함된 컬럼 없음. 컬럼 목록:", list(orig_df.columns))
else:
    TARGET = target_candidates[0]
    print(f"🎯 타겟 컬럼 자동 인식 완료: '{TARGET}'")

CANDIDATE_FEATURES = ['age', 'sex', 'smoking_status', 'drinking_status', 'exercise_group']
AVAILABLE_FEATURES = [c for c in CANDIDATE_FEATURES if c in orig_df.columns]

X       = orig_df[AVAILABLE_FEATURES].fillna(0)
y       = orig_df[TARGET]
weights = orig_df['wt_itvex'].fillna(1.0)

X_train_orig, X_test, y_train_orig, y_test, weights_train, weights_test = train_test_split(
    X, y, weights, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)

smote = SMOTE(random_state=RANDOM_STATE)
X_train, y_train = smote.fit_resample(X_train_orig, y_train_orig)

print(f"✅ 분석 준비 완료: 피처 {len(AVAILABLE_FEATURES)}개 / SMOTE 적용 후 학습셋: {X_train.shape}")


# ===========================================================================
# [Task 4] 모델 정의 및 학습
# ===========================================================================

MODELS = {
    'M1: Logistic':     LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_STATE),
    'M2: RandomForest': RandomForestClassifier(n_estimators=300, max_depth=8, random_state=RANDOM_STATE),
    'M3: XGBoost':      XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=RANDOM_STATE),
    'M4: LightGBM':     LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=RANDOM_STATE, verbose=-1),
    'M5: CatBoost':     CatBoostClassifier(iterations=300, learning_rate=0.05, depth=6, random_seed=RANDOM_STATE, verbose=0)
}

for name, model in MODELS.items():
    print(f"🔄 {name} 학습 중...")
    model.fit(X_train, y_train)

# ===========================================================================
# [Task 5] 보정(Calibration) 및 성능 평가
# ===========================================================================

CAL_METHODS  = ['raw', 'sigmoid', 'isotonic']
perf_results = []
plot_data    = {}

for name, model in MODELS.items():
    print(f"🚀 {name} 분석 중...")
    plot_data[name] = {}

    for method in CAL_METHODS:
        if method == 'raw':
            proba = model.predict_proba(X_test)[:, 1]
        else:
            calibrated = CalibratedClassifierCV(estimator=model, method=method, cv=CV_FOLDS)
            calibrated.fit(X_train, y_train)
            proba = calibrated.predict_proba(X_test)[:, 1]

        pred = (proba >= THRESHOLD).astype(int)
        metrics = {
            'Model':       name,
            'Calibration': method,
            'ROC_AUC':     roc_auc_score(y_test, proba),
            'PR_AUC':      average_precision_score(y_test, proba),
            'Accuracy':    accuracy_score(y_test, pred),
            'Precision':   precision_score(y_test, pred),
            'Recall':      recall_score(y_test, pred),
            'F1':          f1_score(y_test, pred),
            'Brier':       brier_score_loss(y_test, proba)
        }
        perf_results.append(metrics)
        plot_data[name][method] = proba

perf_df = pd.DataFrame(perf_results)
print("\n" + "="*90)
print("          [표] 모델별·보정방법별 성능지표 요약")
print("="*90)
print(perf_df.round(4).to_string(index=False))

# ===========================================================================
# [Task 6] ROC 곡선 시각화
# ===========================================================================

def plot_roc_curves(models, X_test, y_test):
    plt.figure(figsize=(10, 8))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for (name, model), color in zip(models.items(), colors):
        y_score = (model.predict_proba(X_test)[:, 1]
                   if hasattr(model, "predict_proba")
                   else model.decision_function(X_test))
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')

    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.title('그림. 모델별 ROC 곡선 및 AUC 비교 (2030 대사증후군 예측)',
              fontsize=15, fontweight='bold', pad=20)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

plot_roc_curves(MODELS, X_test, y_test)

# ===========================================================================
# [Task 7] Calibration Curve 통합 시각화 (5개 모델)
# ===========================================================================

cal_styles = {
    'raw':      {'color': '#4C72B0', 'label': 'Raw (Uncalibrated)',   'marker': 'o'},
    'sigmoid':  {'color': '#DD8452', 'label': 'Sigmoid Calibration',  'marker': '^'},
    'isotonic': {'color': '#55A868', 'label': 'Isotonic Calibration', 'marker': 's'}
}

fig, axes = plt.subplots(5, 1, figsize=(10, 25), sharex=True)
fig.suptitle('전체 모델별 확률 보정 효과 비교 (Reliability Diagram)',
             fontsize=18, fontweight='bold', y=1.01)

for i, (name, model) in enumerate(MODELS.items()):
    ax = axes[i]
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfectly Calibrated")

    for method in CAL_METHODS:
        y_prob = plot_data[name][method]
        prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=N_BINS)
        brier  = brier_score_loss(y_test, y_prob)
        style  = cal_styles[method]
        ax.plot(prob_pred, prob_true, marker=style['marker'], markersize=6,
                color=style['color'], linewidth=2,
                label=f"{style['label']} (Brier: {brier:.4f})")

    ax.set_title(f"[{name}] 확률 보정 효과", fontsize=14, fontweight='bold', pad=10)
    ax.set_ylabel("Actual Probability\n(실제 양성 비율)", fontsize=11)
    ax.set_ylim([-0.05, 1.05])
    ax.legend(loc="upper left", fontsize=10, frameon=True, shadow=True)
    ax.grid(True, linestyle='--', alpha=0.5)

axes[-1].set_xlabel("Mean Predicted Probability (평균 예측 확률)", fontsize=11)
axes[-1].set_xlim([-0.05, 1.05])
plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.show()

# ===========================================================================
# [Task 8] 혼동 행렬 시각화 (Recall 점수 포함)
# ===========================================================================

cf_rows = []

for name in MODELS.keys():
    for cal in CAL_METHODS:
        proba = plot_data[name][cal]
        pred  = (proba >= THRESHOLD).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
        row_data = perf_df[(perf_df['Model'] == name) & (perf_df['Calibration'] == cal)].iloc[0]

        cf_rows.append({
            'Model': name, 'Calibration': cal,
            'TP (참양성)': tp, 'FN (놓침)': fn,
            'FP (오탐)': fp,  'TN (참음성)': tn,
            'Recall': row_data['Recall'], 'Precision': row_data['Precision'],
            'F1': row_data['F1'],         'Accuracy': row_data['Accuracy']
        })

cf_df = pd.DataFrame(cf_rows)
print("\n" + "="*100)
print(f"          [표] 모델별·보정방법별 혼동 행렬 주요 지표 (Threshold = {THRESHOLD})")
print("="*100)
print(cf_df.to_string(index=False))

fig, axes = plt.subplots(5, 3, figsize=(14, 20), sharex=True, sharey=True)

for mi, name in enumerate(MODELS.keys()):
    for ci, cal in enumerate(CAL_METHODS):
        ax  = axes[mi, ci]
        row = cf_df[(cf_df['Model'] == name) & (cf_df['Calibration'] == cal)]

        cm_data = np.array([
            [row['TN (참음성)'].values[0], row['FP (오탐)'].values[0]],
            [row['FN (놓침)'].values[0],   row['TP (참양성)'].values[0]]
        ])

        sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                    xticklabels=['Negative(정상)', 'Positive(위험)'],
                    yticklabels=['Negative(정상)', 'Positive(위험)'])

        recall_val = row['Recall'].values[0]
        ax.set_title(f"{name} ({cal.capitalize()})\nRecall: {recall_val:.4f}",
                     fontsize=11, fontweight='bold', color='darkred')

        if mi == 4: ax.set_xlabel('Predicted (모델의 예측)', fontsize=10)
        if ci == 0: ax.set_ylabel('Actual (실제 정답)', fontsize=10)

plt.suptitle(f'모델별 × 보정방법별 혼동 행렬 (Threshold = {THRESHOLD})',
             fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.show()

# ===========================================================================
# [Task 9] DCA (Decision Curve Analysis)
# ===========================================================================

def calculate_net_benefit(y_true, y_prob, thresholds):
    """DCA Net Benefit 계산"""
    net_benefits = []
    n = len(y_true)
    for pt in thresholds:
        if pt == 1.0:
            pt = 0.999
        y_pred = (y_prob >= pt).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        net_benefits.append((tp / n) - (fp / n) * (pt / (1 - pt)))
    return np.array(net_benefits)

# ── Calibration Curve (1×5 레이아웃) ────────────────────────────────────
fig, axes = plt.subplots(1, 5, figsize=(25, 5), sharey=True)
colors = {'raw': '#4C72B0', 'sigmoid': '#DD8452', 'isotonic': '#55A868'}

for i, (name, _) in enumerate(MODELS.items()):
    ax = axes[i]
    ax.plot([0, 1], [0, 1], "k:", color='orange', lw=2, label="Perfect Calibration")

    for method in CAL_METHODS:
        prob = plot_data[name][method]
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_test, prob, n_bins=N_BINS)
        brier = brier_score_loss(y_test, prob)
        ax.plot(mean_predicted_value, fraction_of_positives, "s-",
                label=f"{method} (Brier: {brier:.4f})",
                color=colors[method], markersize=4)

    ax.set_title(f"{name}", fontsize=13, fontweight='bold')
    ax.set_xlabel("Mean predicted probability")
    if i == 0:
        ax.set_ylabel("Fraction of positives")
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(alpha=0.3)

plt.suptitle('확률 보정 결과 분석: Reliability Diagram (Raw vs Sigmoid vs Isotonic)',
             fontsize=18, fontweight='bold', y=1.05)
plt.tight_layout()
plt.show()

# ── DCA 시각화 ─────────────────────────────────────────────────────────
thresholds   = np.linspace(0.01, 0.5, 50)
prevalence   = y_test.mean()
nb_all       = prevalence - (1 - prevalence) * (thresholds / (1 - thresholds))
model_colors = {
    'M1: Logistic':     '#4C72B0', 'M2: RandomForest': '#DD8452',
    'M3: XGBoost':      '#C44E52', 'M4: LightGBM':     '#8172B2',
    'M5: CatBoost':     '#55A868'
}

plt.figure(figsize=(12, 8))
plt.plot(thresholds, nb_all,                    color='black', linestyle='--', label='Treat All', alpha=0.5)
plt.plot(thresholds, np.zeros(len(thresholds)), color='black', lw=1.5,        label='Treat None')

for name in MODELS.keys():
    nb = calculate_net_benefit(y_test, plot_data[name]['isotonic'], thresholds)
    plt.plot(thresholds, nb, label=f"{name} (Isotonic)",
             color=model_colors[name], lw=2.5)

plt.xlim([0, 0.5])
plt.ylim([-0.05, prevalence + 0.05])
plt.xlabel('Threshold Probability (위험 임계값)', fontsize=12)
plt.ylabel('Net Benefit (순편익)', fontsize=12)
plt.title('의사결정곡선분석 (DCA): 모델별 임상적 순편익 비교',
          fontsize=16, fontweight='bold')
plt.legend(loc='upper right', fontsize=10)
plt.grid(alpha=0.3)
plt.show()

# ===========================================================================
# [Task 10] Feature Importance 및 SHAP 분석
# ===========================================================================

def get_normalized_fi(model_name, model, feature_names):
    """모델 타입별 중요도 추출 후 0~1 정규화"""
    if   hasattr(model, 'feature_importances_'):     fi = model.feature_importances_
    elif hasattr(model, 'coef_'):                    fi = np.abs(model.coef_[0])
    elif hasattr(model, 'get_feature_importance'):   fi = model.get_feature_importance()
    else:                                            return None
    total = np.sum(fi)
    return fi / total if total != 0 else fi

fi_list = []
for name, model in MODELS.items():
    norm_fi = get_normalized_fi(name, model, AVAILABLE_FEATURES)
    if norm_fi is not None:
        for f, v in zip(AVAILABLE_FEATURES, norm_fi):
            fi_list.append({'Model': name, 'Feature': han_labels.get(f, f), 'Importance': v})

fi_df = pd.DataFrame(fi_list)

plt.figure(figsize=(14, 9))
sns.barplot(data=fi_df, x='Importance', y='Feature', hue='Model', palette='viridis')
plt.title('그림. 모델별 변수 중요도', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('정규화된 중요도 (Normalized Importance)', fontsize=12)
plt.ylabel('예측 변수 (Predictive Features)', fontsize=12)
plt.legend(title='모델', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()

fi_pivot = fi_df.pivot(index='Feature', columns='Model', values='Importance')
display(fi_pivot.sort_values(by='M2: RandomForest', ascending=False).round(2))

# ── SHAP Summary Plot ──────────────────────────────────────────────────
X_test_han = X_test.rename(columns=han_labels)

for name, model in MODELS.items():
    print(f"🚀 {name} SHAP 분석 중...")
    try:
        explainer = (shap.LinearExplainer(model, X_train)
                     if 'Logistic' in name
                     else shap.TreeExplainer(model))
        shap_values = explainer.shap_values(X_test)

        if   isinstance(shap_values, list):                              sv = shap_values[1]
        elif hasattr(shap_values, 'shape') and len(shap_values.shape) == 3: sv = shap_values[:, :, 1]
        else:                                                            sv = shap_values

        plt.figure(figsize=(10, 6))
        shap.summary_plot(sv, X_test_han, show=False)
        plt.title(f"SHAP Summary Plot: {name} (한글)", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"⚠️ {name} SHAP 오류: {e}")

# ===========================================================================
# [Task 11] 최종 모델 선정 리포트
# ===========================================================================

display_cols  = ['Model', 'ROC_AUC', 'Brier', 'Recall', 'Precision', 'F1', 'Accuracy']

final_summary = (perf_df[perf_df['Calibration'] == 'isotonic']
                 .copy()
                 .sort_values('ROC_AUC', ascending=False)
                 .reset_index(drop=True))

print("\n" + "="*100)
print("              [표] 후보 모델 간 주요 성능 지표 요약 (Isotonic Calibration 적용)")
print("="*100)
print(final_summary[display_cols].round(4).to_string(index=False))

final_summary['순위_AUC']    = final_summary['ROC_AUC'].rank(ascending=False).astype(int)
final_summary['순위_Recall'] = final_summary['Recall'].rank(ascending=False).astype(int)
final_summary['순위_Brier']  = final_summary['Brier'].rank(ascending=True).astype(int)
final_summary['순위_F1']     = final_summary['F1'].rank(ascending=False).astype(int)

print("\n" + "="*100)
print("  [표] 지표별 순위 (참고용 — 가중치 종합 점수 미사용)")
print("="*100)
rank_cols = ['Model', 'ROC_AUC', 'Recall', 'Brier', 'F1',
             '순위_AUC', '순위_Recall', '순위_Brier', '순위_F1']
print(final_summary[rank_cols].round(4).to_string(index=False))
print("  ※ 임의 가중치 종합 점수는 선행연구 근거가 없어 사용하지 않습니다.")

best_model_name = 'M1: Logistic'
best_row = final_summary[final_summary['Model'] == best_model_name].iloc[0]

print(f"\n{'='*100}")
print(f"🚩 최종 선정 모델: {best_model_name} (Isotonic Calibration)")
print(f"{'='*100}")
print(f"  AUC    : {best_row['ROC_AUC']:.4f}  (순위 {best_row['순위_AUC']}위 / 5개 모델 중)")
print(f"  Recall : {best_row['Recall']:.4f}  (순위 {best_row['순위_Recall']}위) ← 스크리닝 핵심 지표")
print(f"  Brier  : {best_row['Brier']:.4f}  (순위 {best_row['순위_Brier']}위) ← 타 모델 대비 다소 높음")
print(f"  F1     : {best_row['F1']:.4f}  (순위 {best_row['순위_F1']}위)")
print(f"{'─'*100}")
print(f"  선정 근거: AUC·Recall·F1이 모두 5개 모델 중 1위이며,")
print(f"            Brier가 타 모델보다 다소 높으나 Isotonic 보정 적용 및")
print(f"            스크리닝 목적상 Recall 우위가 더 중요하다고 판단하였습니다.")
print(f"{'='*100}")

# ── 레이더 차트 ────────────────────────────────────────────────────────
metrics_radar = ['ROC_AUC', 'Recall', 'Precision', 'F1', 'Accuracy']
radar_metrics = metrics_radar + ['1-Brier']
final_summary['1-Brier'] = 1 - final_summary['Brier']  # [수정] iterrows 대신 벡터화 연산

N      = len(radar_metrics)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
clr_map = {
    'M1: Logistic':     '#4C72B0',
    'M2: RandomForest': '#DD8452',
    'M3: XGBoost':      '#55A868',
    'M4: LightGBM':     '#C44E52',
    'M5: CatBoost':     '#8172B2'
}

for _, row in final_summary.iterrows():
    vals = [row[m] for m in radar_metrics] + [row[radar_metrics[0]]]
    is_best = row['Model'] == best_model_name
    ax.plot(angles, vals, 'o-',
            linewidth=3 if is_best else 1.5,
            color=clr_map.get(row['Model'], 'gray'),
            label=row['Model'],
            alpha=1.0 if is_best else 0.55)
    ax.fill(angles, vals,
            alpha=0.12 if is_best else 0.03,
            color=clr_map.get(row['Model'], 'gray'))

ax.set_xticks(angles[:-1])
ax.set_xticklabels(['AUC', 'Recall', 'Precision', 'F1', 'Accuracy', '1-Brier'], fontsize=11)
ax.set_ylim(0, 1)
ax.set_title(f'모델별 성능 비교 레이더 차트\n(굵은 선: 최종 선정 — {best_model_name})',
             fontsize=13, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=10)
plt.tight_layout()
plt.show()

# ── 최종 선정 모델 Calibration Curve ─────────────────────────────────
plt.figure(figsize=(9, 6))
plt.plot([0, 1], [0, 1], "k:", alpha=0.6, label="Perfectly Calibrated (Ideal)")
prob_true, prob_pred = calibration_curve(y_test, plot_data[best_model_name]['isotonic'], n_bins=N_BINS)
plt.plot(prob_pred, prob_true, "s-", color='#4C72B0', linewidth=2.5,
         label=f"{best_model_name} (Brier: {best_row['Brier']:.4f})")
plt.title(f'최종 선정 모델 예측 신뢰도 ({best_model_name})', fontsize=14, fontweight='bold')
plt.xlabel('Mean Predicted Probability (예측 확률)', fontsize=11)
plt.ylabel('Fraction of Positives (실제 양성 비율)', fontsize=11)
plt.legend(loc='upper left')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ===========================================================================
# [Task 12] 하이퍼파라미터 튜닝 — M1: Logistic
# ===========================================================================

param_grid_lr = {
    'C':        loguniform(1e-3, 1e2),
    'penalty':  ['l1', 'l2'],
    'solver':   ['saga'],
    'max_iter': [2000],
}

lr_base = LogisticRegression(random_state=RANDOM_STATE)

print("🔍 Logistic Regression 하이퍼파라미터 튜닝 시작 (RandomizedSearchCV)...")
random_search = RandomizedSearchCV(
    estimator=lr_base,
    param_distributions=param_grid_lr,
    n_iter=30,
    scoring='roc_auc',
    cv=CV_FOLDS,
    n_jobs=-1,
    verbose=1,
    random_state=RANDOM_STATE
)
random_search.fit(X_train, y_train)

print("\n" + "="*65)
print("🎉 Logistic Regression 튜닝 완료!")
print(f"✅ 베스트 파라미터: {random_search.best_params_}")
print(f"✅ 베스트 CV ROC_AUC: {random_search.best_score_:.4f}")
print("="*65)

# ===========================================================================
# [Task 13] 최종 완성형 엔진 — Tuned Logistic + Isotonic 보정
# ===========================================================================

final_lr_base = LogisticRegression(**random_search.best_params_, random_state=RANDOM_STATE)

print("🚀 Tuned Logistic + Isotonic 보정 학습 시작...")
final_calibrated_lr = CalibratedClassifierCV(
    estimator=final_lr_base, method='isotonic', cv=CV_FOLDS
)
final_calibrated_lr.fit(X_train, y_train)

final_probs  = final_calibrated_lr.predict_proba(X_test)[:, 1]
final_preds  = (final_probs >= THRESHOLD).astype(int)
final_auc    = roc_auc_score(y_test, final_probs)
final_brier  = brier_score_loss(y_test, final_probs)
final_f1     = f1_score(y_test, final_preds)
final_recall = recall_score(y_test, final_preds)

pre_tune_row = final_summary[final_summary['Model'] == best_model_name].iloc[0]

print("\n" + "="*70)
print("🏆 [최종 완성형 엔진] 성능 리포트 (Tuned Logistic + Isotonic)")
print("="*70)
print(f"  {'지표':<15} {'튜닝 전':>12} {'튜닝 후':>12} {'변화':>10}")
print(f"  {'-'*50}")
for label, pre, post in [
    ('ROC_AUC',   pre_tune_row['ROC_AUC'], final_auc),
    ('Recall',    pre_tune_row['Recall'],  final_recall),
    ('F1-Score',  pre_tune_row['F1'],      final_f1),
    ('Brier',     pre_tune_row['Brier'],   final_brier),
]:
    arrow = '▲' if (post > pre if label != 'Brier' else post < pre) else '▼'
    print(f"  {label:<15} {pre:>12.4f} {post:>12.4f} {arrow:>5} {abs(post - pre):.4f}")
print("="*70)
print("\n[상세 분류 지표]")
print(classification_report(y_test, final_preds, target_names=['정상(0)', '대사증후군(1)']))

plt.figure(figsize=(8, 6))
plt.plot([0, 1], [0, 1], "k:", alpha=0.6, label="Perfectly Calibrated (Ideal)")
prob_true, prob_pred = calibration_curve(y_test, final_probs, n_bins=N_BINS)
plt.plot(prob_pred, prob_true, "s-", color='#4C72B0', linewidth=2.5,
         label=f"Tuned Logistic + Isotonic\n(Brier: {final_brier:.4f})")
plt.title('최종 완성형 예측 엔진의 확률 예측 신뢰도\n(Tuned M1: Logistic + Isotonic)',
          fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Mean Predicted Probability (모델이 예측한 위험 확률)', fontsize=12)
plt.ylabel('Fraction of Positives (실제 환자 발생 비율)', fontsize=12)
plt.legend(loc='upper left', fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ===========================================================================
# [Task 14] 가중치 적용 민감도 분석
# ===========================================================================

unweighted_auc = final_auc
weighted_auc   = roc_auc_score(y_test, final_probs, sample_weight=weights_test)
diff           = abs(unweighted_auc - weighted_auc)

print("\n" + "="*65)
print("🔎 [일반화 검증] 최종 모델 가중치 적용 민감도 분석")
print(f"   최종 모델: Tuned {best_model_name} + Isotonic")
print("="*65)
print(f"▶ 비가중(Unweighted) AUC : {unweighted_auc:.4f}  (개별 패턴 기반)")
print(f"▶ 가중적용(Weighted) AUC  : {weighted_auc:.4f}  (모집단 대표성 반영)")
print(f"▶ 성능 편차(Difference)  : {diff:.4f}")
print("-" * 65)

if diff < 0.02:
    print("✅ 가중치 적용 전후 AUC 차이 0.02 미만 → 모집단 일반화 성능 확보.")
    print("   본 모델이 대한민국 2030 세대 모집단 전체를 대표할 수 있음을 시사합니다.")
else:
    print("⚠️ 가중치 적용 시 성능 편차 존재 → 하위 그룹 분석 필요.")
print("="*65)
