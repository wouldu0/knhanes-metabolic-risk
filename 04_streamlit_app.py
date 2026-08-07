import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from streamlit_option_menu import option_menu
from feature_utils import RAW_COLUMNS
import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────────
# ★ 초기 설정 및 데이터 로드
# ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="2030 대사증후군 AI 위험도 예측", layout="wide", initial_sidebar_state="collapsed")

# 🎨 [컬러 팔레트 정의]
PRIMARY_LIME = '#c3ee41'
BG_LIME = '#F7FDF0'
POINT_WHITE = '#FFFFFF'

FIGMA_CARD = '#FFFFFF'
FIGMA_TEXT = '#1A1A1A'
FIGMA_SUBTEXT = '#555555'

COLOR_BLUE = '#4A90E2'
COLOR_CORAL = '#FFA8A8'

COLOR_WARN = '#F39C12'

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data/0325_hn_all(med).csv", low_memory=False)
    except FileNotFoundError:
        st.error("⚠️ 데이터 파일을 찾을 수 없습니다. 경로를 확인해주세요: data/0325_hn_all(med).csv")
        return pd.DataFrame()

    df['ex_1'] = (df['exercise_group'] == 1).astype(int)
    df['ex_2'] = (df['exercise_group'] == 2).astype(int)
    df['ex_3'] = (df['exercise_group'] == 3).astype(int)
    df['male'] = (df['sex'] == 1).astype(int)
    df['smoke_past'] = (df['smoking_status'] == 1).astype(int)
    df['smoke_current'] = (df['smoking_status'] == 2).astype(int)
    df['drink_current'] = (df['drinking_status'] == 1).astype(int)

    df['age_group'] = pd.cut(df['age'], bins=[19,24,29,34,39], labels=['20-24세','25-29세','30-34세','35-39세'])
    df['sex_label'] = df['sex'].map({1: '남성', 2: '여성'})
    df['ex_label'] = df['exercise_group'].map({1: '복합(유산소+근력)', 2: '근력운동만', 3: '유산소운동만', 4: '운동 안 함'})
    df['smoke_label'] = df['smoking_status'].map({0: '비흡연', 1: '과거흡연', 2: '현재흡연'})
    df['drink_label'] = df['drinking_status'].map({0: '비음주', 1: '현재음주'})

    analysis_vars = ['metabolic_syndrome', 'age', 'male', 'smoke_past', 'smoke_current', 'drink_current', 'ex_1', 'ex_2', 'ex_3']
    return df.dropna(subset=analysis_vars).copy()

df = load_data()

# 학습·평가까지 끝낸 모델을 그대로 불러온다 (06_model_refactor.py에서 생성).
# 앱에서 재학습하지 않으므로, README에 보고한 성능 지표와 실제 서비스 모델이 항상 같다.
# 파이프라인이 raw 컬럼(age, sex, smoking_status, drinking_status, exercise_group)을
# 그대로 받으므로 앱에서 별도로 원-핫 인코딩할 필요가 없다.
# (참고) 모델 평가용 threshold=0.1143(Youden's J)은 여기서 분류에 쓰지 않는다.
# 앱은 연속 확률만 보여주고, threshold는 06_model_refactor.py의 Recall/F1
# 평가 목적으로만 존재한다.

@st.cache_resource
def load_ml_engine():
    return joblib.load("metabolic_risk_model.joblib")

if not df.empty:
    model = load_ml_engine()

# ==============================================================================
# 🎨 그래프 테마 설정
# ==============================================================================
FIGMA_PLOTLY_THEME_INSIGHT = {
    'layout': dict(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=FIGMA_TEXT, family="Helvetica Neue, sans-serif", size=15, weight='bold'),
        xaxis=dict(showgrid=False, zeroline=False, color=FIGMA_TEXT, tickfont=dict(size=14, color='black'), title=dict(font=dict(size=15))),
        yaxis=dict(showgrid=False, zeroline=False, color=FIGMA_SUBTEXT, tickfont=dict(size=14), title=dict(font=dict(size=15))),
        margin=dict(l=10, r=10, t=35, b=10)
    )
}

# ==============================================================================
# 🎨 글로벌 CSS 설정
# ==============================================================================
st.markdown(f"""
<style>
    /* 전체 배경 */
    [data-testid="stAppViewContainer"] {{ background-color: {BG_LIME} !important; }}
    header[data-testid="stHeader"] {{ background-color: rgba(0,0,0,0) !important; border: none !important; box-shadow: none !important; }}
    section[data-testid="stSidebar"] {{ display: none !important; }}

    /* 카드 디자인 */
    div[data-testid="stVerticalBlock"] > div[style*="border"] {{
        background-color: {FIGMA_CARD} !important;
        border: 1px solid rgba(195, 238, 65, 0.3) !important;
        border-radius: 20px !important;
        box-shadow: 0px 10px 30px rgba(195, 238, 65, 0.05) !important;
        padding: 30px !important;
        margin-bottom: 25px !important;
    }}

    h1, h2, h3, h4, h5, h6 {{ color: {FIGMA_TEXT} !important; font-weight: 700 !important; }}
    p, label {{ color: {FIGMA_SUBTEXT} !important; font-weight: 600 !important; }}

    /* 슬라이더 */
    .stSlider [data-baseweb="slider"] > div > div {{ background-color: #E0E5F2 !important; border: none !important; }}
    .stSlider [data-baseweb="slider"] [role="slider"] {{ background-color: #FFFFFF !important; border: 2px solid #555555 !important; box-shadow: 0px 2px 5px rgba(0,0,0,0.1) !important; }}
    .stSlider [data-testid="stThumbValue"] {{ color: #1A1A1A !important; }}
    /* 슬라이더 나이 선택 바 채워진 부분 (회색으로 유지) */
    .stSlider [data-baseweb="slider"] > div > div > div:first-child {{ background-color: #555555 !important; }}

    /* 체크박스 */
    .stCheckbox [data-baseweb="checkbox"] > div:first-child {{ background-color: #FFFFFF !important; border-color: #CCCCCC !important; }}
    .stCheckbox [data-baseweb="checkbox"] svg {{ fill: #333333 !important; }}

    /* 입력창(Focus) - 파란색 테두리 제거 및 원래 회색 유지 */
    .stTextInput input, .stNumberInput input, .stSelectbox [data-baseweb="select"] {{
        border-radius: 12px !important; border: 1px solid #E0E5F2 !important; background-color: #FFFFFF !important; color: #1A1A1A !important;
    }}
    .stTextInput input:focus, .stNumberInput input:focus, .stSelectbox [data-baseweb="select"]:focus-within {{
        border-color: #555555 !important;
        box-shadow: 0 0 0 2px rgba(85, 85, 85, 0.2) !important;
        outline: none !important; /* 파란색 아웃라인 제거 */
    }}
    /* Selectbox 클릭 시 생기는 내부 파란색 테두리 추가 제거 */
    div[data-baseweb="select"] > div:focus-within {{
        outline: none !important;
        box-shadow: none !important;
    }}

    /* 메인 버튼 */
    .stButton button[kind="primary"] {{
        background-color: {PRIMARY_LIME} !important;
        border-color: {PRIMARY_LIME} !important;
        border-radius: 12px !important; padding: 12px 24px !important; transition: all 0.3s ease;
    }}

    .stButton button[kind="primary"] p {{
        font-size: 20px !important;
        font-weight: 800 !important;
        color: #1A1A1A !important;
    }}

    label[data-testid="stWidgetLabel"] p,
    div[data-testid="stCheckbox"] label p,
    div[data-testid="stCheckbox"] label span {{
        font-size: 20px !important;
    }}

    [data-testid="stMetricLabel"] p, [data-testid="stMetricLabel"] div {{
        font-size: 15px !important;
        color: #000000 !important;
        font-weight: 800 !important;
    }}
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 🚀 상단 타이틀 및 옵션 메뉴
# ==============================================================================
st.markdown(f"<h2 style='text-align: center; margin-top: 15px; margin-bottom: 5px; color: {FIGMA_TEXT} !important;'> 2030 대사증후군 AI 위험도 예측</h2>", unsafe_allow_html=True)
st.markdown(f"<p style='text-align: center; margin-bottom: 25px; color: {FIGMA_SUBTEXT} !important;'>생활습관 정보를 기반으로 대사증후군 위험도를 예측하고, 건강검진 정보 입력 시 임상 기준 충족 여부를 함께 확인합니다.</p>", unsafe_allow_html=True)

selected = option_menu(
    menu_title=None,
    options=["🩺 AI 위험도 예측", "📊 데이터 인사이트", "ℹ️ 프로젝트 정보"], # 이모지 추가
    # icons=["heart-pulse-fill", "bar-chart-fill", "info-circle-fill"], # 기존 아이콘 제거
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {
            "padding": "0!important", "background-color": FIGMA_CARD, "border": "1px solid rgba(195, 238, 65, 0.1)",
            "box-shadow": "0px 8px 20px rgba(195, 238, 65, 0.03)", "border-radius": "25px", "margin-bottom": "35px", "max-width": "900px"
        },
        "icon": {"display": "none"}, # 기본 아이콘 숨김
        "nav-link": {"font-size": "20px", "text-align": "center", "margin":"0px", "--hover-color": BG_LIME, "color": FIGMA_SUBTEXT, "font-weight":"600"},
        "nav-link-selected": {"background-color": PRIMARY_LIME, "color": "#1A1A1A", "border-radius": "25px", "font-weight":"700", "box-shadow": "0px 4px 12px rgba(195, 238, 65, 0.3)"},
    }
)

if df.empty: st.stop()

# ==============================================================================
# 1️⃣ AI 위험도 예측 페이지
# ==============================================================================
if selected == "🩺 AI 위험도 예측": # 메뉴 이름 수정 반영

    with st.container(border=True):
        st.markdown("##### 📝 나의 건강 정보 입력")
        st.caption("라이프스타일 정보는 ML 위험도 예측에 사용되며, 건강검진 정보는 선택 입력으로 임상 기준 확인에 사용됩니다.")
        st.markdown("---")

        col_lifestyle, col_clinical = st.columns(2, gap="large")

        with col_lifestyle:
            st.markdown("<h5 style='color:#333; margin-bottom:18px;'>🏃‍♂️ 라이프스타일 (필수)</h5>", unsafe_allow_html=True)
            u_age = st.slider("📅 연령 (세)", 20, 39, 30)
            st.markdown("<br>", unsafe_allow_html=True)
            u_sex = st.selectbox("👫 성별", ["남성", "여성"])
            st.markdown("<br>", unsafe_allow_html=True)
            u_smoke = st.selectbox("🚬 흡연 상태", ["비흡연", "과거 흡연", "현재 흡연"])
            st.markdown("<br>", unsafe_allow_html=True)
            u_drink = st.selectbox("🍺 음주 여부", ["비음주", "현재 음주"])
            st.markdown("<br>", unsafe_allow_html=True)
            u_ex = st.selectbox("🏋️‍♂️ 운동 습관 (주간)", ["운동 안 함", "유산소 운동만", "근력 운동만", "복합(유산소+근력)"])

        with col_clinical:
            st.markdown("<h5 style='color:#333; margin-bottom:18px;'>🩸 임상 정보 (선택)</h5>", unsafe_allow_html=True)
            st.caption("건강검진 결과가 있다면 입력해주세요. 대사증후군 임상 기준 확인에 사용됩니다.")
            u_waist = st.number_input("📏 허리둘레 (cm)", min_value=0.0, value=0.0)

            # 혈압 입력 (수축기/이완기 나란히 배치)
            bp_col1, bp_col2 = st.columns(2)
            with bp_col1:
                u_sbp = st.number_input("💓 수축기 혈압 (mmHg)", min_value=0, value=0)
            with bp_col2:
                u_dbp = st.number_input("💓 이완기 혈압 (mmHg)", min_value=0, value=0)

            u_fbs = st.number_input("🍬 공복혈당 (mg/dL)", min_value=0, value=0)
            u_tg = st.number_input("🍔 중성지방 (mg/dL)", min_value=0, value=0)
            u_hdl = st.number_input("✨ HDL 콜레스테롤 (mg/dL)", min_value=0, value=0)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**[약물 복용 여부 체크]**")
            med_factors_col1, med_factors_col2 = st.columns(2)
            with med_factors_col1:
                med_bp = st.checkbox("💊 고혈압 약 복용")
                med_sugar = st.checkbox("💊 당뇨약 복용")
            with med_factors_col2:
                med_lipid = st.checkbox("💊 고지혈증 약 복용")

        st.markdown("<br><br>", unsafe_allow_html=True)
        _, btn_col, _ = st.columns([1.2, 2, 1.2])
        run_btn = btn_col.button("🚀 위험도 확인하기", type="primary", use_container_width=True)

    if not run_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown("<div style='text-align:center; padding: 70px 0;'>", unsafe_allow_html=True)
            st.markdown(f"<h3 style='color: {FIGMA_SUBTEXT} !important; font-weight:600 !important;'>👆 위 폼에 정보를 입력하고 위험도를 확인하세요</h3>", unsafe_allow_html=True)
            st.markdown("<p style='font-size:16px;'>생활습관 정보만으로 ML 위험도를 먼저 확인하고, 건강검진 수치가 있다면 대사증후군 임상 기준 충족 여부도 함께 보여드립니다.</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown("<br>", unsafe_allow_html=True)

        # ── ① ML 스크리닝: 생활습관 5개 변수만 모델에 입력 (임상 수치는 사용하지 않음) ──
        raw_input = {
            'age': u_age,
            'sex': 1 if u_sex == "남성" else 2,
            'smoking_status': {'비흡연': 0, '과거 흡연': 1, '현재 흡연': 2}[u_smoke],
            'drinking_status': 1 if u_drink == "현재 음주" else 0,
            'exercise_group': {'복합(유산소+근력)': 1, '근력 운동만': 2, '유산소 운동만': 3, '운동 안 함': 4}[u_ex],
        }
        ai_pred_prob = model.predict_proba(pd.DataFrame([raw_input])[RAW_COLUMNS])[0][1] * 100

        # ── ② 임상 기준 판정: NCEP-ATP III 5개 구성요소, 약물 복용 시 해당 항목 자동 반영 ──
        # (전처리 스크립트 01_data_processing.py와 동일한 매핑: 이상지질혈증 약은
        #  중성지방·HDL 두 항목 모두에 반영)
        components = {}  # name -> True(이상)/False(정상)/None(모름)

        components['복부비만'] = None
        if u_waist > 0:
            components['복부비만'] = (u_sex == "남성" and u_waist >= 90) or (u_sex == "여성" and u_waist >= 85)

        components['높은 혈압'] = None
        if u_sbp > 0 or u_dbp > 0 or med_bp:
            components['높은 혈압'] = (u_sbp >= 130) or (u_dbp >= 85) or med_bp

        components['높은 공복혈당'] = None
        if u_fbs > 0 or med_sugar:
            components['높은 공복혈당'] = (u_fbs >= 100) or med_sugar

        components['높은 중성지방'] = None
        if u_tg > 0 or med_lipid:
            components['높은 중성지방'] = (u_tg >= 150) or med_lipid

        components['낮은 HDL'] = None
        if u_hdl > 0 or med_lipid:
            components['낮은 HDL'] = ((u_sex == "남성" and 0 < u_hdl < 40) or (u_sex == "여성" and 0 < u_hdl < 50)) or med_lipid

        known_items = {k: v for k, v in components.items() if v is not None}
        known_count = len(known_items)
        risk_count = sum(1 for v in known_items.values() if v)
        risk_factors = [k for k, v in known_items.items() if v]
        med_factors = []
        if med_bp: med_factors.append("혈압")
        if med_sugar: med_factors.append("혈당")
        if med_lipid: med_factors.append("중성지방/HDL")

        with st.container(border=True):
            st.markdown("#### 💡 예측 리포트")
            st.markdown("---")

            col_ml, col_clinical = st.columns([1, 1.2], gap="large")

            # ── ML 스크리닝 결과 (항상 표시) ──
            # 20%/50% 같은 색상 구간을 두지 않는다 — 이 값들은 임상적으로 검증된
            # 저/중/고위험 cut-off가 아니라 무근거 판단 기준을 암시하게 됨. 게이지는
            # 단일 색으로 확률만 보여준다 (모델 평가용 threshold=0.1143도 여기서는
            # 쓰지 않음 — 06_model_refactor.py의 Recall/F1 평가 목적으로만 존재).
            with col_ml:
                st.markdown("<h5>🤖 생활습관 기반 ML 스크리닝</h5>", unsafe_allow_html=True)
                gauge_color = COLOR_BLUE

                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=ai_pred_prob,
                    number={'suffix': "%", 'font': {'size': 40, 'color': gauge_color, 'weight': 'bold'}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white", 'visible': False},
                        'bar': {'color': gauge_color},
                        'bgcolor': "#F0F2F6",
                        'borderwidth': 0,
                    }
                ))
                fig_gauge.update_layout(height=250, margin=dict(l=25, r=25, t=15, b=15), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=FIGMA_TEXT, family="Helvetica Neue, sans-serif"))
                st.plotly_chart(fig_gauge, use_container_width=True)
                st.caption("생활습관 5개 변수를 기반으로 산출한 통계적 예측 확률이며, 임상적 진단 확률이나 진단 기준을 의미하지 않습니다.")

            # ── 임상 기준 판정 결과 (검진 정보 입력량에 따라 분기) ──
            with col_clinical:
                st.markdown("<h5>🩺 대사증후군 임상 기준 확인</h5>", unsafe_allow_html=True)
                if known_count == 0:
                    st.info("건강검진 수치를 입력하면 대사증후군 5개 임상 기준 충족 여부를 확인할 수 있습니다.")
                else:
                    if known_count == 5:
                        if risk_count >= 3:
                            st.error(f"🚨 **구성요소 {risk_count}/5개 충족 → 대사증후군 임상 기준을 충족합니다.**")
                        else:
                            st.success(f"✅ **구성요소 {risk_count}/5개 충족 → 대사증후군 임상 기준에 해당하지 않습니다.**")
                    else:
                        st.warning(f"🟡 **5개 구성요소 중 {known_count}개 입력됨 (그중 위험요인 {risk_count}개).** "
                                   f"나머지 {5 - known_count}개를 몰라 기준 충족 여부는 확인 보류입니다.")

                    for name in ['복부비만', '높은 혈압', '높은 공복혈당', '높은 중성지방', '낮은 HDL']:
                        val = components[name]
                        icon = "✅" if val is True else ("⬜" if val is False else "❔")
                        st.write(f"{icon} {name}")

                    if med_factors:
                        st.caption(f"💊 복용 중인 약물({', '.join(med_factors)}) 정보가 해당 항목 판정에 반영되었습니다.")

            st.markdown("---")
            st.markdown("<h5 style='margin-bottom:18px;'>🏃‍♂️ 맞춤형 솔루션 가이드</h5>", unsafe_allow_html=True)

            if known_count == 5 and risk_count >= 3:
                st.write("🚨 **[총평]**: 대사증후군 임상 기준을 충족합니다. 의료진 상담을 통해 현재 상태를 확인하고, 생활습관 관리를 함께 고려해보세요.")

            # 운동 안내: 본인 데이터의 OR 순위(복합 0.398 vs 근력만 0.378)로 특정
            # 운동을 서로 비교해 추천하지 않는다 — 각 그룹은 '운동 안 함'과만 비교된
            # 값이라 그룹 간 우열을 매길 근거가 없음. 대신 공식 신체활동 가이드라인
            # 원칙(부족한 활동을 보완)에 따라 "현재 상태에서 무엇이 부족한지"를 안내.
            EXERCISE_GUIDE = {
                "운동 안 함": "규칙적인 신체활동을 시작해보세요. 유산소 활동과 근력운동을 균형 있게 실천하는 방법을 고려할 수 있습니다.",
                "유산소 운동만": "현재의 유산소 활동을 유지하면서, 근력운동도 함께 실천하는 것을 고려해보세요.",
                "근력 운동만": "현재의 근력운동을 유지하면서, 유산소 활동도 함께 고려해보세요.",
                "복합(유산소+근력)": "현재의 규칙적인 운동습관을 꾸준히 유지해보세요.",
            }
            st.write(f"✔️ **[운동]**: {EXERCISE_GUIDE[u_ex]}")
            st.caption(
                "📊 참고: 본 프로젝트 분석에서 운동을 하지 않는 군 대비 복합운동군·근력운동군은 "
                "대사증후군 odds가 유의하게 낮았습니다(유산소만 하는 군은 유의하지 않음). "
                "각 운동 유형을 서로 직접 비교한 결과는 아니라서 이 중 어떤 운동이 더 낫다고는 "
                "말할 수 없고, 구체적인 운동 방법은 공식 신체활동 가이드라인을 따르는 것이 안전합니다."
            )

            if risk_count > 0:
                guideline_text = "✔️ **[집중 관리]**: "
                if "복부비만" in risk_factors: guideline_text += "복부비만 관리(정제 탄수화물 감소), "
                if "높은 혈압" in risk_factors: guideline_text += "나트륨 섭취 감소, "
                if "높은 공복혈당" in risk_factors: guideline_text += "식후 30분 산책 권장, "
                if "높은 중성지방" in risk_factors: guideline_text += "야식 및 음주 제한, "
                if "낮은 HDL" in risk_factors: guideline_text += "꾸준한 유산소 운동 실천, "
                st.write(guideline_text.strip(", "))

# ==============================================================================
# 2️⃣ 데이터 인사이트 페이지
# ==============================================================================
elif selected == "📊 데이터 인사이트": # 메뉴 이름 수정 반영
    st.markdown("### 📊 청년층 건강 데이터 인사이트")
    st.caption("국민건강영양조사 데이터를 기반으로 한 2030세대 통계입니다.")
    st.caption("※ 아래 유병률은 분석 표본의 비가중 단순 집계입니다.")
    st.markdown("<br>", unsafe_allow_html=True)

    overall_rate = df['metabolic_syndrome'].mean() * 100
    male_rate = df[df['male'] == 1]['metabolic_syndrome'].mean() * 100
    female_rate = df[df['male'] == 0]['metabolic_syndrome'].mean() * 100
    none_rate = df[df['ex_label'] == '운동 안 함']['metabolic_syndrome'].mean() * 100

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        with st.container(border=True): st.metric(label="총 분석 표본", value=f"{len(df):,}명", delta="20~39세")
    with kpi2:
        with st.container(border=True): st.metric(label="전체 평균 유병률", value=f"{overall_rate:.1f}%")
    with kpi3:
        with st.container(border=True): st.metric(label="남성 유병률", value=f"{male_rate:.1f}%", delta=f"여성의 약 {male_rate / female_rate:.1f}배", delta_color="inverse")
    with kpi4:
        with st.container(border=True): st.metric(label="운동 안하는 그룹 유병률", value=f"{none_rate:.1f}%")
    st.markdown("<br>", unsafe_allow_html=True)

    # --- 1번째 줄 ---
    c1, c2 = st.columns(2, gap="large")
    with c1:
        with st.container(border=True):
            st.markdown("#### [연령대별 대사증후군 유병률]")
            age_data = df.groupby('age_group')['metabolic_syndrome'].mean() * 100
            fig_age = px.bar(x=age_data.index, y=age_data.values, text=age_data.apply(lambda x: f'{x:.1f}%'))
            fig_age.update_traces(marker_color=COLOR_WARN, textposition='outside', textfont=dict(color='#555555', size=14), cliponaxis=False)
            fig_age.update_yaxes(range=[0, max(age_data.values) * 1.2])
            fig_age.update_layout(**FIGMA_PLOTLY_THEME_INSIGHT['layout'], xaxis_title="", yaxis_title="유병률 (%)", height=380)
            st.plotly_chart(fig_age, use_container_width=True)
    with c2:
        with st.container(border=True):
            st.markdown("#### [성별 유병률 비교]")
            sex_rates = df.groupby('sex_label')['metabolic_syndrome'].mean() * 100
            fig_sex = px.bar(x=sex_rates.index, y=sex_rates.values, text=sex_rates.apply(lambda x: f'{x:.1f}%'), color=sex_rates.index, color_discrete_map={'남성': COLOR_BLUE, '여성': COLOR_CORAL})
            fig_sex.update_layout(**FIGMA_PLOTLY_THEME_INSIGHT['layout'])
            fig_sex.update_traces(textposition='outside', textfont=dict(color='#555555', size=14), cliponaxis=False)
            fig_sex.update_yaxes(range=[0, max(sex_rates.values) * 1.2])
            fig_sex.update_layout(xaxis_title="", yaxis_title="유병률 (%)", height=380, showlegend=False)
            st.plotly_chart(fig_sex, use_container_width=True)

    # --- 2번째 줄 ---
    c3, c4 = st.columns(2, gap="large")
    with c3:
        with st.container(border=True):
            st.markdown("#### [음주 상태별 유병률]")
            drink_rates = df.groupby('drink_label')['metabolic_syndrome'].mean() * 100
            fig_drink = px.bar(x=drink_rates.index, y=drink_rates.values, text=drink_rates.apply(lambda x: f'{x:.1f}%'))
            fig_drink.update_traces(marker_color=COLOR_BLUE, textposition='outside', textfont=dict(color='#555555', size=14), cliponaxis=False)
            fig_drink.update_yaxes(range=[0, max(drink_rates.values) * 1.2])
            fig_drink.update_layout(**FIGMA_PLOTLY_THEME_INSIGHT['layout'])
            fig_drink.update_layout(xaxis_title="", yaxis_title="유병률 (%)", height=380)
            st.plotly_chart(fig_drink, use_container_width=True)
    with c4:
        with st.container(border=True):
            st.markdown("#### [흡연 상태별 유병률]")
            smoke_rates = df.groupby('smoke_label')['metabolic_syndrome'].mean() * 100
            smoke_rates = smoke_rates.reindex(['비흡연', '과거흡연', '현재흡연'])
            fig_smoke = px.bar(x=smoke_rates.index, y=smoke_rates.values, text=smoke_rates.apply(lambda x: f'{x:.1f}%'))
            fig_smoke.update_traces(marker_color=COLOR_WARN, textposition='outside', textfont=dict(color='#555555', size=14), cliponaxis=False)
            fig_smoke.update_yaxes(range=[0, max(smoke_rates.values) * 1.2])
            fig_smoke.update_layout(**FIGMA_PLOTLY_THEME_INSIGHT['layout'])
            fig_smoke.update_layout(xaxis_title="", yaxis_title="유병률 (%)", height=380)
            st.plotly_chart(fig_smoke, use_container_width=True)

    # --- 3번째 줄 ---
    with st.container(border=True):
        st.markdown("#### 🏋️ 운동 습관별 대사증후군 유병률 (비가중)")
        ex_order = ['복합(유산소+근력)', '근력운동만', '유산소운동만', '운동 안 함']
        ex_rates_df = pd.DataFrame([{'운동 그룹': label, '유병률': df[df['ex_label'] == label]['metabolic_syndrome'].mean() * 100} for label in ex_order if label in df['ex_label'].unique()])
        fig_ex = px.bar(ex_rates_df, x='운동 그룹', y='유병률', color='운동 그룹',
                      color_discrete_map={'복합(유산소+근력)': COLOR_BLUE, '운동 안 함': COLOR_CORAL, '유산소운동만': COLOR_BLUE, '근력운동만': COLOR_WARN},
                      text=ex_rates_df['유병률'].apply(lambda x: f'{x:.1f}%'))
        fig_ex.update_layout(**FIGMA_PLOTLY_THEME_INSIGHT['layout'])
        fig_ex.update_traces(textposition='outside', textfont=dict(color='#555555', size=14), cliponaxis=False)
        fig_ex.update_yaxes(range=[0, ex_rates_df['유병률'].max() * 1.2])
        fig_ex.update_layout(xaxis_title="", yaxis_title="유병률 (%)", height=380, showlegend=False)
        st.plotly_chart(fig_ex, use_container_width=True)

# ==============================================================================
# 3️⃣ 프로젝트 정보 페이지
# ==============================================================================
elif selected == "ℹ️ 프로젝트 정보": # 메뉴 이름 수정 반영
    with st.container(border=True):
        st.markdown("### 🎯 프로젝트 목표 및 개요")
        st.write("20~30대 청년층에서의 생활습관 및 위험요인은 누적되어 향후 만성질환으로 발전할 수 있습니다. 본 AI 모델은 청년층의 생활습관 데이터를 분석하여 대사증후군 위험도를 조기에 예측하고, 개인 맞춤형 예방 가이드를 제공하기 위해 개발되었습니다.")
        st.info("데이터 출처: 국민건강영양조사 제9기 (KNHANES 2022–2024) - 20~39세 청년층 남녀 3,363명 데이터")

    c_pie, c_bar = st.columns(2, gap="large")
    with c_pie:
        with st.container(border=True):
            st.markdown("#### [분석 표본 성별 분포]")
            fig_pie = px.pie(df, names='sex_label', color='sex_label', color_discrete_map={'남성': COLOR_BLUE, '여성': COLOR_CORAL}, hole=0.55)
            fig_pie.update_layout(**FIGMA_PLOTLY_THEME_INSIGHT['layout'])
            fig_pie.add_annotation(text=f"{len(df):,}명", x=0.5, y=0.5, showarrow=False, font_size=17, font_color=FIGMA_TEXT)
            fig_pie.update_layout(height=410, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5))
            st.plotly_chart(fig_pie, use_container_width=True)

    with c_bar:
        with st.container(border=True):
            st.markdown("#### [분석 표본 연령 분포]")
            ct = pd.crosstab(df['age_group'], df['sex_label'])
            fig_bar = px.bar(ct, barmode='group', color_discrete_map={'남성': COLOR_BLUE, '여성': COLOR_CORAL})
            fig_bar.update_layout(**FIGMA_PLOTLY_THEME_INSIGHT['layout'])
            fig_bar.update_layout(xaxis_title="", yaxis_title="인원 수 (명)", height=410, legend_title="")
            st.plotly_chart(fig_bar, use_container_width=True)