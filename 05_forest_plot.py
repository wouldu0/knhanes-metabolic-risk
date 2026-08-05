"""
══════════════════════════════════════════════════════════════
  Forest Plot — ★ 운동을 마지막에 투입하는 구조 ★

  ★ CONFIG의 DATA_PATH만 본인 경로로 바꿔주세요!

  모델 구조:
    Model 1: 성별, 연령
    Model 2: + 흡연, 음주
    Model 3: + 운동 그룹 (마지막!)

  차트 목록:
    [1] Model 3 전체 변수 Forest Plot
    [2] 모델별 적합도 변화 + Model 3 운동 OR
    [3] 성별 층화 — 운동 그룹 OR 비교
══════════════════════════════════════════════════════════════
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────────
# ★ CONFIG ★
# ──────────────────────────────────────────────────────────────

DATA_PATH = "/content/drive/Shareddrives/세미1 4조 공유드라이브/DA/류주영/0325_hn_all(med).csv"

# Colab: 'NanumGothic' / Windows: 'Malgun Gothic'
plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

BLUE = '#185FA5'
CORAL = '#D85A30'
TEAL = '#0F6E56'
TEAL_LIGHT = '#5DCAA5'
AMBER = '#BA7517'
GRAY = '#888888'
RED_LINE = '#A32D2D'


# ──────────────────────────────────────────────────────────────
# 데이터 로드
# ──────────────────────────────────────────────────────────────

df = pd.read_csv(DATA_PATH, low_memory=False)

df['ex_1'] = (df['exercise_group'] == 1).astype(int)
df['ex_2'] = (df['exercise_group'] == 2).astype(int)
df['ex_3'] = (df['exercise_group'] == 3).astype(int)
df['male'] = (df['sex'] == 1).astype(int)
df['smoke_past'] = (df['smoking_status'] == 1).astype(int)
df['smoke_current'] = (df['smoking_status'] == 2).astype(int)
df['drink_current'] = (df['drinking_status'] == 1).astype(int)

analysis_vars = ['metabolic_syndrome', 'exercise_group', 'age', 'sex',
                 'smoking_status', 'drinking_status', 'kstrata', 'psu', 'wt_itvex']
df = df.dropna(subset=analysis_vars).copy()
df['kstrata'] = df['kstrata'].astype(int)
df['psu_id'] = df['kstrata'].astype(str) + '_' + df['psu'].astype(str)

print(f"데이터 로드 완료: {len(df):,}명")


# ──────────────────────────────────────────────────────────────
# 복합표본 로지스틱 회귀 함수
# ──────────────────────────────────────────────────────────────

def run_survey_logistic(data, x_vars):
    y = data['metabolic_syndrome']
    X = sm.add_constant(data[x_vars])
    model = sm.GLM(y, X,
                   family=sm.families.Binomial(),
                   freq_weights=data['wt_itvex'].values)
    result = model.fit(cov_type='cluster',
                       cov_kwds={'groups': data['psu_id'].values})
    out = []
    for var in result.params.index:
        if var == 'const':
            continue
        out.append({
            'var': var,
            'OR': np.exp(result.params[var]),
            'lo': np.exp(result.conf_int().loc[var, 0]),
            'hi': np.exp(result.conf_int().loc[var, 1]),
            'p': result.pvalues[var],
        })
    return out, result


def sig(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return ''

var_kr = {
    'male': '남성 (vs 여성)',
    'age': '연령 (1세 증가)',
    'smoke_past': '과거흡연 (vs 비흡연)',
    'smoke_current': '현재흡연 (vs 비흡연)',
    'drink_current': '현재음주 (vs 비음주)',
    'ex_1': '복합(유산소+근력)',
    'ex_2': '근력운동만',
    'ex_3': '유산소운동만',
}


# ──────────────────────────────────────────────────────────────
# ★ 운동 마지막 구조로 모델 실행 ★
# ──────────────────────────────────────────────────────────────

print("분석 실행 중...\n")

# Model 1: 성별, 연령
x1 = ['male', 'age']
# Model 2: + 흡연, 음주
x2 = ['male', 'age', 'smoke_past', 'smoke_current', 'drink_current']
# Model 3: + 운동 (마지막!)
x3 = ['male', 'age', 'smoke_past', 'smoke_current', 'drink_current',
      'ex_1', 'ex_2', 'ex_3']

res_m1, mod_m1 = run_survey_logistic(df, x1)
res_m2, mod_m2 = run_survey_logistic(df, x2)
res_m3, mod_m3 = run_survey_logistic(df, x3)

# 성별 층화 (성별 변수 제외)
x_gender = ['age', 'smoke_past', 'smoke_current', 'drink_current',
            'ex_1', 'ex_2', 'ex_3']
res_male, _ = run_survey_logistic(df[df['sex'] == 1].copy(), x_gender)
res_female, _ = run_survey_logistic(df[df['sex'] == 2].copy(), x_gender)

print("분석 완료! Forest Plot 생성 중...\n")


# ══════════════════════════════════════════════════════════════
# [차트 1] Model 3 전체 변수 Forest Plot
# ══════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(10, 6))

labels = [var_kr[r['var']] for r in res_m3][::-1]
ors    = [r['OR'] for r in res_m3][::-1]
ci_lo  = [r['lo'] for r in res_m3][::-1]
ci_hi  = [r['hi'] for r in res_m3][::-1]
pvals  = [r['p']  for r in res_m3][::-1]
y_pos  = np.arange(len(labels))

colors = []
for o, p in zip(ors, pvals):
    if p >= 0.05:   colors.append(GRAY)
    elif o < 1:     colors.append(TEAL)
    else:           colors.append(CORAL)

ax.axvline(x=1, color=RED_LINE, linestyle='--', linewidth=1.5, alpha=0.7, zorder=1)

for i in range(len(y_pos)):
    ax.plot([ci_lo[i], ci_hi[i]], [y_pos[i], y_pos[i]], color=colors[i], linewidth=2, zorder=2)
    ax.plot(ors[i], y_pos[i], 'D', color=colors[i], markersize=8, zorder=3)

text_x = max(ci_hi) + 0.8
for i in range(len(y_pos)):
    txt = f'{ors[i]:.3f} ({ci_lo[i]:.3f}-{ci_hi[i]:.3f}) {sig(pvals[i])}'
    ax.text(text_x, y_pos[i], txt, va='center', fontsize=9, color=colors[i], fontweight='bold')

ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=11)
ax.set_xlabel('Odds Ratio (95% CI)', fontsize=11)
ax.set_title('Forest Plot — Model 3 완전보정 (성별,연령 → 흡연,음주 → 운동)', fontsize=13, fontweight='bold', pad=15)
ax.set_xlim(0, text_x + 2.0)
ax.grid(axis='x', alpha=0.2)
ax.spines[['top', 'right']].set_visible(False)

legend_elements = [
    mpatches.Patch(color=TEAL, label='보호 효과 (OR<1, p<0.05)'),
    mpatches.Patch(color=CORAL, label='위험 증가 (OR>1, p<0.05)'),
    mpatches.Patch(color=GRAY, label='유의하지 않음 (p≥0.05)'),
]
ax.legend(handles=legend_elements, fontsize=9, loc='lower right')

plt.tight_layout()
plt.savefig('forest_plot_1_model3_exercise_last.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.show()
print("✅ [차트 1] 저장 완료: forest_plot_1_model3_exercise_last.png")


# ══════════════════════════════════════════════════════════════
# [차트 2] 모델별 변화 — 운동 추가 효과 시각화
# ══════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# --- 좌측: 모델 적합도 변화 (AIC) ---
ax = axes[0]
model_names = ['Model 1\n(성별,연령)', 'Model 2\n(+흡연,음주)', 'Model 3\n(+운동)']
aics = [mod_m1.aic, mod_m2.aic, mod_m3.aic]
aic_colors = [BLUE, AMBER, TEAL]

bars = ax.bar(model_names, aics, color=aic_colors, width=0.5, zorder=3, edgecolor='white')
for bar, val in zip(bars, aics):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 0.5,
            f'AIC\n{val:,.0f}', ha='center', va='center', fontsize=9,
            fontweight='bold', color='white')

# AIC 감소량 표시
for i in range(1, len(aics)):
    diff = aics[i] - aics[i-1]
    mid_y = (aics[i-1] + aics[i]) / 2
    ax.annotate('', xy=(i, aics[i]), xytext=(i-1, aics[i-1]),
                arrowprops=dict(arrowstyle='->', color='#333', lw=1.5))
    ax.text(i - 0.5, mid_y, f'{diff:,.0f}', ha='center', fontsize=9,
            fontweight='bold', color='#333',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#FFF3CD', edgecolor='#FFD700'))

ax.set_ylabel('AIC', fontsize=11)
ax.set_title('모델 적합도 변화 (AIC 낮을수록 좋음)', fontsize=12, fontweight='bold', pad=10)
ax.grid(axis='y', alpha=0.2, zorder=0)
ax.spines[['top', 'right']].set_visible(False)

# --- 우측: Model 3의 운동 그룹 OR ---
ax = axes[1]
ex_vars = ['ex_1', 'ex_2', 'ex_3']
ex_names = ['복합\n(유산소+근력)', '근력운동만', '유산소운동만', '안 함\n(기준)']
ex_colors_bar = [TEAL, TEAL_LIGHT, AMBER, GRAY]

ex_ors = []
ex_ci_lo = []
ex_ci_hi = []
ex_ps = []
for ev in ex_vars:
    r = [x for x in res_m3 if x['var'] == ev][0]
    ex_ors.append(r['OR'])
    ex_ci_lo.append(r['lo'])
    ex_ci_hi.append(r['hi'])
    ex_ps.append(r['p'])
ex_ors.append(1.0)
ex_ci_lo.append(1.0)
ex_ci_hi.append(1.0)
ex_ps.append(1.0)

bars = ax.bar(ex_names, ex_ors, color=ex_colors_bar, width=0.55, zorder=3, edgecolor='white')

# 에러바
for i in range(len(ex_vars)):
    ax.plot([i, i], [ex_ci_lo[i], ex_ci_hi[i]], color='#333', linewidth=1.5, zorder=4)
    ax.plot([i-0.08, i+0.08], [ex_ci_lo[i], ex_ci_lo[i]], color='#333', linewidth=1.5, zorder=4)
    ax.plot([i-0.08, i+0.08], [ex_ci_hi[i], ex_ci_hi[i]], color='#333', linewidth=1.5, zorder=4)

# 수치 표시
for i, (bar, o, p) in enumerate(zip(bars, ex_ors, ex_ps)):
    label = f'OR={o:.3f} {sig(p)}' if i < 3 else 'ref'
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            label, ha='center', fontsize=9, fontweight='bold')

ax.axhline(y=1.0, color=RED_LINE, linestyle='--', linewidth=1.5, alpha=0.5, zorder=1)
ax.set_ylabel('Odds Ratio', fontsize=11)
ax.set_title('Model 3 — 운동 그룹별 OR (95% CI)', fontsize=12, fontweight='bold', pad=10)
ax.set_ylim(0, max(ex_ci_hi) + 0.3)
ax.grid(axis='y', alpha=0.2, zorder=0)
ax.spines[['top', 'right']].set_visible(False)

plt.suptitle('운동 변수 투입 효과 — 모델 적합도 + OR (복합표본)',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('forest_plot_2_exercise_last_effect.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.show()
print("✅ [차트 2] 저장 완료: forest_plot_2_exercise_last_effect.png")


# ══════════════════════════════════════════════════════════════
# [차트 3] 성별 층화 Forest Plot
# ══════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharey=True)

ex_titles = ['복합(유산소+근력)', '근력운동만', '유산소운동만']
ex_colors = [TEAL, TEAL_LIGHT, AMBER]
grp_results = {'전체': res_m3, '남성': res_male, '여성': res_female}
grp_colors = {'전체': GRAY, '남성': BLUE, '여성': CORAL}
grp_labels = ['전체', '남성', '여성']

for idx, (ev, title, ec) in enumerate(zip(ex_vars, ex_titles, ex_colors)):
    ax = axes[idx]
    s_ors, s_lo, s_hi, s_p, s_labels = [], [], [], [], []

    for grp in grp_labels:
        r = [x for x in grp_results[grp] if x['var'] == ev][0]
        s_ors.append(r['OR'])
        s_lo.append(r['lo'])
        s_hi.append(r['hi'])
        s_p.append(r['p'])
        s_labels.append(grp)

    s_labels_rev = s_labels[::-1]
    s_ors_rev = s_ors[::-1]
    s_lo_rev = s_lo[::-1]
    s_hi_rev = s_hi[::-1]
    s_p_rev = s_p[::-1]
    y = np.arange(3)

    ax.axvline(x=1, color=RED_LINE, linestyle='--', linewidth=1.5, alpha=0.7)

    for i in range(3):
        c = grp_colors[s_labels_rev[i]]
        alpha = 1.0 if s_p_rev[i] < 0.05 else 0.4
        ax.plot([s_lo_rev[i], s_hi_rev[i]], [y[i], y[i]], color=c, linewidth=2.5, alpha=alpha)
        ax.plot(s_ors_rev[i], y[i], 'D', color=c, markersize=9, alpha=alpha)
        s = sig(s_p_rev[i]) if s_p_rev[i] < 0.05 else 'n.s.'
        ax.text(s_ors_rev[i], y[i] + 0.25, f'{s_ors_rev[i]:.3f} {s}',
                ha='center', fontsize=9, fontweight='bold', color=c, alpha=max(alpha, 0.7))

    ax.set_yticks(y)
    ax.set_yticklabels(s_labels_rev, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold', color=ec)
    x_max = max(s_hi_rev) * 1.3
    ax.set_xlim(0, max(x_max, 1.5))
    ax.set_xlabel('OR', fontsize=10)
    ax.grid(axis='x', alpha=0.2)
    ax.spines[['top', 'right']].set_visible(False)

fig.suptitle('성별 층화 Forest Plot — 완전보정, 복합표본 (기준: 운동 안 함)',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('forest_plot_3_gender_stratified.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.show()
print("✅ [차트 3] 저장 완료: forest_plot_3_gender_stratified.png")


# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("Forest Plot 전체 완료!")
print("=" * 50)
print("  forest_plot_1_model3_exercise_last.png  — 전체 변수 OR")
print("  forest_plot_2_exercise_last_effect.png  — 모델 적합도 + 운동 OR")
print("  forest_plot_3_gender_stratified.png     — 성별 층화 OR")
print("=" * 50)