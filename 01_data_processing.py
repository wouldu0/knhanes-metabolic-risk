"""
══════════════════════════════════════════════════════════════
  국민건강영양조사 데이터 전처리 통합 파이프라인
  ★ 시나리오 B: 약물 복용군 포함 + 약물 보정 방식 ★

  파이프라인 흐름:
    [1] 원본 데이터 로드 & 연령 필터링 (20~39세)
    [2] 약물 복용군 제외 없음 (전원 포함)
    [3] 가중치 정규화
    [4] 파생변수 생성 (운동그룹, 흡연, 음주, 대사증후군 + 약물 보정)
    [5] 연속형 변수 윈저화 (상·하위 1% 클리핑)
    [6] 연령 표준화 (z-score)
    [7] 필요 칼럼만 추출 & 결측치 정리
    [8] 최종 분석 데이터셋 저장

  약물 보정 매핑 (코드북 확인 완료):
    DI1_2  = 고혈압 의사진단      → ms_bp = 1
    DI2_2  = 이상지질혈증 의사진단 → ms_tg = 1, ms_hdl = 1
    DE1_31 = 인슐린 투여          → ms_glu = 1
    DE1_32 = 경구혈당강하제 복용  → ms_glu = 1

  ⚠️ 주의: 원본 CSV(약물 복용군 제외 전)를 INPUT_PATH에 넣어야 합니다!
══════════════════════════════════════════════════════════════
"""

import pandas as pd
import numpy as np
from scipy import stats

# ──────────────────────────────────────────────────────────────
# ★ CONFIG: 여기만 수정하면 됩니다 ★
# ──────────────────────────────────────────────────────────────

INPUT_PATH  = "data/hn_all.csv"
OUTPUT_PATH = "data/0325_hn_all(med).csv"

WEIGHT_VAR = "wt_itvex"
GROUP_VAR  = "metabolic_syndrome"


# ──────────────────────────────────────────────────────────────
# STEP 1: 원본 데이터 로드 & 연령 필터링 (20~39세)
# ──────────────────────────────────────────────────────────────

print("=" * 60)
print("[STEP 1] 원본 데이터 로드 & 연령 필터링 (20~39세)")
print("=" * 60)

df = pd.read_csv(INPUT_PATH, low_memory=False)
print(f"  원본 데이터 행 수: {len(df):,}명")

df = df[(df["age"] >= 20) & (df["age"] <= 39)].copy()
print(f"  20~39세 필터링 후: {len(df):,}명")


# ──────────────────────────────────────────────────────────────
# STEP 2: 약물 복용군 제외 없음 (전원 포함)
# ──────────────────────────────────────────────────────────────

print("\n[STEP 2] ★ 약물 복용군 제외 없음 (전원 포함) ★")

med_mask = (
    df["DI1_2"].isin([1, 2, 3, 4]) |   # 고혈압 진단
    df["DI2_2"].isin([1, 2, 3, 4]) |   # 이상지질혈증 진단
    df["DE1_31"].eq(1) |                # 인슐린 투여
    df["DE1_32"].eq(1)                  # 경구혈당강하제 복용
)
print(f"  약물 복용/진단자 수: {med_mask.sum():,}명 (제외하지 않고 포함)")
print(f"  현재 총 대상자: {len(df):,}명")


# ──────────────────────────────────────────────────────────────
# STEP 3: 가중치 정규화
# ──────────────────────────────────────────────────────────────

print("\n[STEP 3] 가중치 정규화 (w_norm)")

norm_factor = len(df) / df[WEIGHT_VAR].sum()
df["w_norm"] = df[WEIGHT_VAR] * norm_factor
print(f"  정규화 계수: {norm_factor:.6f}")
print(f"  정규화 후 가중치 합: {df['w_norm'].sum():.1f} (= 표본 수)")


# ──────────────────────────────────────────────────────────────
# STEP 4-1: 운동 그룹 파생변수 생성
# ──────────────────────────────────────────────────────────────
# BE5_1: 1주일간 근력운동 일수 (3~6일 = 근력운동 실천)
# pa_aerobic: 유산소 운동 실천 여부 (1=실천, 0=미실천)
#
# 운동 그룹 분류:
#   1 = 복합 (유산소 + 근력 모두)
#   2 = 근력운동만
#   3 = 유산소운동만
#   4 = 안 함 (둘 다 미실천)

print("\n[STEP 4-1] 운동 그룹 파생변수 생성")

df["strength"] = np.where(df["BE5_1"].isin([3, 4, 5, 6]), 1, 0)
df.loc[df["BE5_1"].isin([8, 9]), "strength"] = np.nan

exercise_conditions = [
    (df["strength"] == 1) & (df["pa_aerobic"] == 1),  # 그룹1: 복합
    (df["strength"] == 1) & (df["pa_aerobic"] == 0),  # 그룹2: 근력만
    (df["strength"] == 0) & (df["pa_aerobic"] == 1),  # 그룹3: 유산소만
    (df["strength"] == 0) & (df["pa_aerobic"] == 0),  # 그룹4: 안 함
]
df["exercise_group"] = np.select(exercise_conditions, [1, 2, 3, 4], default=np.nan)
print(f"  운동 그룹 분포:\n{df['exercise_group'].value_counts(dropna=False).sort_index().to_string()}")


# ──────────────────────────────────────────────────────────────
# STEP 4-2: 흡연 상태 파생변수 생성
# ──────────────────────────────────────────────────────────────
# BS3_1: 현재 흡연 여부 (1,2=현재, 3=과거, 8=비흡연)
# BS12_47: 궐련형 전자담배 사용 (1,2=현재, 3=과거, 8=비해당)
# BS12_2: 액상형 전자담배 사용 (1=현재, 8=비해당)
#
# 흡연 상태 분류:
#   0 = 비흡연 (세 변수 모두 8=비해당)
#   1 = 과거흡연 (현재 피우지 않지만 과거 경험 있음)
#   2 = 현재흡연 (세 가지 중 하나라도 현재 사용)

print("\n[STEP 4-2] 흡연 상태 파생변수 생성")

curr_smoker = (
    df["BS3_1"].isin([1, 2]) |
    df["BS12_47"].isin([1, 2]) |
    df["BS12_2"].eq(1)
)
past_smoker = (df["BS3_1"] == 3) | (df["BS12_47"] == 3)
non_smoker  = (df["BS3_1"] == 8) & (df["BS12_47"] == 8) & (df["BS12_2"] != 1)

df["smoking_status"] = np.select(
    [curr_smoker, past_smoker, non_smoker],
    [2, 1, 0],
    default=np.nan
)
print(f"  흡연 상태 분포:\n{df['smoking_status'].value_counts(dropna=False).sort_index().to_string()}")


# ──────────────────────────────────────────────────────────────
# STEP 4-3: 음주 여부 파생변수 생성
# ──────────────────────────────────────────────────────────────
# BD1_11: 최근 1년간 음주 빈도
#   2~6 = 현재 음주자 (월 1회 이상)
#   그 외 = 비음주 또는 과거 음주 → 0으로 통합

print("\n[STEP 4-3] 음주 여부 파생변수 생성")

curr_drinker = df["BD1_11"].isin([2, 3, 4, 5, 6])
df["drinking_status"] = np.where(curr_drinker, 1, 0)
print(f"  음주 여부 분포:\n{df['drinking_status'].value_counts(dropna=False).sort_index().to_string()}")


# ──────────────────────────────────────────────────────────────
# STEP 4-4: 대사증후군 판정 ★ 약물 보정 포함 ★
# ──────────────────────────────────────────────────────────────
# NCEP ATP III 기준 (아시아인 허리둘레 기준 적용)
#
#   1) 복부비만 (ms_wc): 남성 ≥90cm, 여성 ≥85cm
#   2) 고중성지방 (ms_tg): TG ≥150 mg/dL
#   3) 낮은 HDL (ms_hdl): 남성 <40, 여성 <50 mg/dL
#   4) 높은 혈압 (ms_bp): 수축기 ≥130 또는 이완기 ≥85 mmHg
#   5) 높은 공복혈당 (ms_glu): ≥100 mg/dL
#
# 약물 보정 (코드북 확인 완료):
#   DI1_2  ∈ [1,2,3,4] = 고혈압 의사진단      → ms_bp  = 1
#   DI2_2  ∈ [1,2,3,4] = 이상지질혈증 의사진단 → ms_tg  = 1, ms_hdl = 1
#   DE1_31 == 1         = 인슐린 투여           → ms_glu = 1
#   DE1_32 == 1         = 경구혈당강하제 복용   → ms_glu = 1
#
# → 5개 중 3개 이상 해당 시 대사증후군(metabolic_syndrome = 1)

print("\n[STEP 4-4] 대사증후군 판정 ★ 약물 보정 포함 ★")

# ── 약물 복용 플래그 설정 ──
is_bp_med    = df["DI1_2"].isin([1, 2, 3, 4])          # 고혈압 진단 → 혈압 보정
is_lipid_med = df["DI2_2"].isin([1, 2, 3, 4])          # 이상지질혈증 진단 → 지질 보정
is_glu_med   = (df["DE1_31"] == 1) | (df["DE1_32"] == 1)  # 인슐린/혈당강하제 → 혈당 보정

# ── 5대 구성요소 판정 (수치 + 약물 보정 동시 적용) ──

# 1) 복부비만: 수치로만 판정 (약물 보정 없음)
df["ms_wc"] = np.where(
    df["sex"] == 1,
    (df["HE_wc"] >= 90),      # 남성 기준
    (df["HE_wc"] >= 85)       # 여성 기준
).astype(float)

# 2) 고중성지방: (수치 ≥ 150) OR (이상지질혈증 약 복용 중)
df["ms_tg"] = ((df["HE_TG"] >= 150) | is_lipid_med).astype(float)

# 3) 낮은 HDL: (수치 미달) OR (이상지질혈증 약 복용 중)
df["ms_hdl"] = np.where(
    df["sex"] == 1,
    ((df["HE_HDL_st2"] < 40) | is_lipid_med),   # 남성 기준
    ((df["HE_HDL_st2"] < 50) | is_lipid_med)     # 여성 기준
).astype(float)

# 4) 높은 혈압: (수치 ≥ 130/85) OR (고혈압 약 복용 중)
df["ms_bp"] = ((df["HE_sbp"] >= 130) | (df["HE_dbp"] >= 85) | is_bp_med).astype(float)

# 5) 높은 공복혈당: (수치 ≥ 100) OR (인슐린/혈당강하제 투여 중)
df["ms_glu"] = ((df["HE_glu"] >= 100) | is_glu_med).astype(float)

# ── 대사증후군 최종 판정 ──
ms_components = ["ms_wc", "ms_tg", "ms_hdl", "ms_bp", "ms_glu"]
df["ms_count"] = df[ms_components].sum(axis=1)
df.loc[df[ms_components].isna().any(axis=1), "ms_count"] = np.nan
df[GROUP_VAR] = (df["ms_count"] >= 3).astype(float)

print(f"  대사증후군 구성요소별 이상 비율:")
for comp in ms_components:
    pct = df[comp].mean() * 100
    print(f"    {comp}: {pct:.1f}%")
print(f"\n  대사증후군 판정 결과:")
print(f"    대사증후군 (+): {int(df[GROUP_VAR].sum()):,}명 ({df[GROUP_VAR].mean()*100:.1f}%)")
print(f"    대사증후군 (-): {int((df[GROUP_VAR]==0).sum()):,}명 ({(1-df[GROUP_VAR].mean())*100:.1f}%)")


# ──────────────────────────────────────────────────────────────
# STEP 5: 연속형 변수 윈저화 (상·하위 1% 클리핑)
# ──────────────────────────────────────────────────────────────

print("\n[STEP 5] 연속형 변수 윈저화 (상·하위 1% 클리핑)")

cont_vars = ["HE_wc", "HE_TG", "HE_HDL_st2", "HE_sbp", "HE_dbp", "HE_glu"]
print(f"  {'변수':15s} {'원본 min':>10s} {'원본 max':>10s} → {'윈저 min':>10s} {'윈저 max':>10s}")
print("  " + "-" * 60)

for var in cont_vars:
    orig_min, orig_max = df[var].min(), df[var].max()
    lower_limit = df[var].quantile(0.01)
    upper_limit = df[var].quantile(0.99)
    df[var] = df[var].clip(lower=lower_limit, upper=upper_limit)
    new_min, new_max = df[var].min(), df[var].max()
    print(f"  {var:15s} {orig_min:10.1f} {orig_max:10.1f} → {new_min:10.1f} {new_max:10.1f}")


# ──────────────────────────────────────────────────────────────
# STEP 6: 필요 칼럼 추출 & 결측치 정리
# ──────────────────────────────────────────────────────────────

print("\n[STEP 7] 필요 칼럼 추출 & 결측치 정리")

target_cols = [
    # 식별/조사 정보
    "ID", "year", "age", "sex",
    # 가중치/층화 변수
    "wt_itvex", "kstrata", "psu",
    # 운동 관련 원본 + 파생변수
    "BE5_1", "pa_aerobic", "exercise_group",
    # 대사증후군 관련 검사값 (윈저화 반영됨)
    "HE_wc", "HE_TG", "HE_HDL_st2", "HE_sbp", "HE_dbp", "HE_glu",
    # 대사증후군 5대 요소 판정 결과 (약물 보정 반영됨)
    "ms_wc", "ms_tg", "ms_hdl", "ms_bp", "ms_glu",
    # 대사증후군 최종 판정
    "metabolic_syndrome",
    # 질환 관련 (약물 보정 확인용)
    "DI1_2", "DI2_2", "DE1_31", "DE1_32",
    # 흡연 관련 원본 + 파생변수
    "BS3_1", "BS12_47", "BS12_1", "BS12_2", "smoking_status",
    # 음주 관련 원본 + 파생변수
    "BD1", "BD1_11", "drinking_status",
    # 대사증후군 구성요소 개수
    "ms_count",
]

df_final = df[target_cols].copy()
print(f"  칼럼 추출 후 행 수: {len(df_final):,}명, 칼럼 수: {len(target_cols)}개")

# 모름/무응답(9) → NaN 처리
df_final.replace(9, np.nan, inplace=True)

# 대사증후군 구성요소 중 하나라도 결측이면 ms_count도 결측 처리
df_final.loc[df_final[ms_components].isna().any(axis=1), "ms_count"] = np.nan

# 핵심 분석 변수에 결측이 없는 행만 최종 포함
essential_vars = [
    "exercise_group", "metabolic_syndrome", "ms_count",
    "smoking_status", "drinking_status",
    "HE_wc", "HE_TG", "HE_HDL_st2", "HE_sbp", "HE_dbp", "HE_glu",
]

before_drop = len(df_final)
df_final = df_final.dropna(subset=essential_vars).copy()
df_final["analytic_sample"] = 1  # 분석 대상 플래그

print(f"  필수 변수 결측 제거: {before_drop - len(df_final):,}명 제거됨")
print(f"  ✅ 최종 분석 대상: {len(df_final):,}명")


# ──────────────────────────────────────────────────────────────
# STEP 8: 최종 데이터셋 저장
# ──────────────────────────────────────────────────────────────

print("\n[STEP 8] 최종 데이터셋 저장")
df_final.to_csv(OUTPUT_PATH, index=False)
print(f"  저장 완료: {OUTPUT_PATH}")


# ──────────────────────────────────────────────────────────────
# 최종 요약
# ──────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("📋 시나리오 B (약물 보정) 전처리 완료 요약")
print("=" * 60)
print(f"  최종 분석 대상자: {len(df_final):,}명")
print(f"  대사증후군 (+): {int(df_final[GROUP_VAR].sum()):,}명 ({df_final[GROUP_VAR].mean()*100:.1f}%)")
print(f"  대사증후군 (-): {int((df_final[GROUP_VAR]==0).sum()):,}명 ({(1-df_final[GROUP_VAR].mean())*100:.1f}%)")
print(f"  칼럼 수: {df_final.shape[1]}개")
print(f"  저장 경로: {OUTPUT_PATH}")
print()
print("  약물 보정 매핑 (코드북 확인 완료):")
print("    DI1_2  (고혈압 진단)      → ms_bp  = 1")
print("    DI2_2  (이상지질혈증 진단) → ms_tg = 1, ms_hdl = 1")
print("    DE1_31 (인슐린 투여)       → ms_glu = 1")
print("    DE1_32 (경구혈당강하제)     → ms_glu = 1")
print("=" * 60)