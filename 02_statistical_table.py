import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.weightstats import DescrStatsW

# ══════════════════════════════════════════════════════════════
# ★ CONFIG: 경로 설정 ★
# ══════════════════════════════════════════════════════════════
# VS Code 환경에 맞춰 경로를 수정하세요.
DATA_PATH   = "data/0325_hn_all(med).csv"
OUTPUT_PATH = "data/hn_all_검정통계표.xlsx"

WEIGHT_VAR, GROUP_VAR = 'wt_itvex', 'metabolic_syndrome'

# ══════════════════════════════════════════════════════════════
# STEP 1: 데이터 로드 & 가중치 정규화
# ══════════════════════════════════════════════════════════════
def load_data(data_path):
    df = pd.read_csv(data_path, low_memory=False)

    if 'w_norm' not in df.columns:
        norm_factor = len(df) / df[WEIGHT_VAR].sum()
        df['w_norm'] = df[WEIGHT_VAR] * norm_factor

    df = df.dropna(subset=[GROUP_VAR, 'w_norm']).copy()
    return df

# ══════════════════════════════════════════════════════════════
# STEP 2: 테이블 빌드 함수 (평균±표준편차 반영)
# ══════════════════════════════════════════════════════════════
def build_full_table_korean(df):
    CONFIG = [
        ('continuous', 'age', '연령 (평균±표준편차)'),
        ('categorical', 'sex', '성별', {1: '남성', 2: '여성'}),
        ('categorical', 'exercise_group', '운동 습관', {1: '복합(유산소+근력)', 2: '근력운동만', 3: '유산소운동만', 4: '안 함'}),
        ('categorical', 'smoking_status', '흡연 상태', {0: '비흡연', 1: '과거흡연', 2: '현재흡연'}),
        ('categorical', 'drinking_status', '음주 여부', {1: '현재 음주', 0: '안 함 (비음주/과거음주)'}),
        ('categorical', 'ms_wc', '복부 비만', {0: '정상', 1: '비만'}),
        ('categorical', 'ms_tg', '고중성지방혈증', {0: '정상', 1: '이상'}),
        ('categorical', 'ms_hdl', '낮은 HDL 콜레스테롤', {0: '정상', 1: '이상'}),
        ('categorical', 'ms_bp', '높은 혈압', {0: '정상', 1: '이상'}),
        ('categorical', 'ms_glu', '높은 공복혈당', {0: '정상', 1: '이상'}),
    ]

    rows = []
    ms1, ms0 = df[df[GROUP_VAR]==1], df[df[GROUP_VAR]==0]

    # 기초 정보 행
    rows.append({
        '구분': '전체 대상자',
        '전체 (n, %)': f"N={len(df):,}",
        '대사증후군 (+) (n, 열%)': f"n={len(ms1):,}",
        '대사증후군 (-) (n, 열%)': f"n={len(ms0):,}",
        '유병률 (행%)': '-',
        'p-값': '-'
    })

    for vtype, col, label, *opt in CONFIG:
        if vtype == 'continuous':
            g1, g0 = ms1[col], ms0[col]
            w1, w0 = ms1['w_norm'], ms0['w_norm']
            d1, d0 = DescrStatsW(g1, weights=w1), DescrStatsW(g0, weights=w0)
            try:
                t_stat, p_val, _ = d1.get_compare(d0).ttest_ind()
                stat_label, p_v = f"t={t_stat:.2f}", p_val
            except: stat_label, p_v = "-", np.nan
        else:
            ct = pd.crosstab(df[GROUP_VAR], df[col], values=df['w_norm'], aggfunc='sum')
            chi2, p_v, *_ = stats.chi2_contingency(ct)
            stat_label = f"χ²={chi2:.2f}"

        # p-value 별표 표기
        if pd.isna(p_v): p_str = "-"
        elif p_v < 0.001: p_str = f"{p_v:.4f}***"
        elif p_v < 0.01:  p_str = f"{p_v:.4f}**"
        elif p_v < 0.05:  p_str = f"{p_v:.4f}*"
        else: p_str = f"{p_v:.4f}"

        if vtype == 'continuous':
            # ★ .std_mean(표준오차) 대신 .std(표준편차) 사용 ★
            t_stats = DescrStatsW(df[col], weights=df['w_norm'])
            ms1_stats = DescrStatsW(ms1[col], weights=ms1['w_norm'])
            ms0_stats = DescrStatsW(ms0[col], weights=ms0['w_norm'])
            
            rows.append({
                '구분': label,
                '전체 (n, %)': f"{t_stats.mean:.1f}±{t_stats.std:.2f}",
                '대사증후군 (+) (n, 열%)': f"{ms1_stats.mean:.1f}±{ms1_stats.std:.2f}",
                '대사증후군 (-) (n, 열%)': f"{ms0_stats.mean:.1f}±{ms0_stats.std:.2f}",
                '유병률 (행%)': stat_label,
                'p-값': p_str
            })
        else:
            # 범주형 변수 헤더
            rows.append({'구분': label, '전체 (n, %)': '', '대사증후군 (+) (n, 열%)': '', '대사증후군 (-) (n, 열%)': '', '유병률 (행%)': stat_label, 'p-값': p_str})
            codes = opt[0]
            w_t_v, w_1_v, w_0_v = df['w_norm'].sum(), ms1['w_norm'].sum(), ms0['w_norm'].sum()

            for code, name in codes.items():
                wt_c = df.loc[df[col]==code, 'w_norm'].sum()
                w1_c = ms1.loc[ms1[col]==code, 'w_norm'].sum()
                w0_c = ms0.loc[ms0[col]==code, 'w_norm'].sum()
                n_total, n1, n0 = int((df[col]==code).sum()), int((ms1[col]==code).sum()), int((ms0[col]==code).sum())

                rows.append({
                    '구분': f"  {name}",
                    '전체 (n, %)': f"{n_total:,} ({(wt_c/w_t_v*100):.1f}%)",
                    '대사증후군 (+) (n, 열%)': f"{n1:,} ({(w1_c/w_1_v*100):.1f}%)",
                    '대사증후군 (-) (n, 열%)': f"{n0:,} ({(w0_c/w_0_v*100):.1f}%)",
                    '유병률 (행%)': f"{(w1_c/wt_c*100):.1f}%",
                    'p-값': ''
                })
    return pd.DataFrame(rows)

# ══════════════════════════════════════════════════════════════
# MAIN: 실행
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("🚀 데이터 분석 및 엑셀 생성을 시작합니다...")
    
    try:
        df_ready = load_data(DATA_PATH)
        final_table = build_full_table_korean(df_ready)
        
        # 엑셀 파일 저장
        final_table.to_excel(OUTPUT_PATH, index=False)
        
        print("-" * 50)
        print(f"✅ 분석 완료! 파일이 저장되었습니다.")
        print(f"📁 파일 경로: {OUTPUT_PATH}")
        print("-" * 50)
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")