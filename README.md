# 🩺 AI 기반 청년층(MZ세대) 대사증후군 위험 예측 모델

> **국민건강영양조사(KNHANES) 기반 20–39세 청년층의 대사증후군 위험을 조기에 식별하는 머신러닝 예측 모델**

---

## 🎬 시연 영상

[![시연 영상](https://img.youtube.com/vi/TcNgaddpYXQ/0.jpg)](https://youtube.com/shorts/TcNgaddpYXQ?feature=share)

---

## 📌 프로젝트 개요

청년층(20–39세) 대사증후군 유병률이 빠르게 증가하고 있음에도 기존 관리 시스템은 중장년층 위주로 설계되어 있습니다.  
본 프로젝트는 라이프스타일(운동, 흡연, 음주)과 임상 정보를 통합하여 청년층 맞춤형 대사증후군 위험도를 예측하고, Streamlit 웹 인터페이스로 제공합니다.

| 항목 | 내용 |
|------|------|
| 데이터 출처 | 국민건강영양조사(KNHANES) 제9기 (2022–2024) |
| 분석 대상 | 20–39세 청년층 N=3,363명 |
| 학습 데이터 | SMOTE 증강 후 N=4,728명 |
| 검증 데이터 | N=673명 (독립 검증셋) |
| 최종 모델 | Tuned Logistic Regression + Isotonic Calibration |
| 성능 (Test) | ROC-AUC 0.7347 · Recall 0.6923 · F1 0.3285 |

---

## 🗂️ 프로젝트 구조

```
.
├── data/                         # 데이터 디렉토리 (gitignore 처리)
│   ├── hn_all.csv                # 원본 KNHANES 데이터 (직접 다운로드 필요)
│   └── 0325_hn_all(med).csv      # 전처리 완료 데이터 (전처리 스크립트 실행 후 생성)
│
├── 01_data_processing.py      # [STEP 1] 원본 데이터 → 분석용 데이터셋 생성
├── 02_statistical_table.py    # [STEP 2] 기술통계 및 검정통계표 생성 (.xlsx)
├── 03_modeling.py             # [STEP 3] 모델 학습, 평가, SHAP 분석 (Google Colab)
├── 04_streamlit_app.py        # [STEP 4] 웹 서비스 구현
├── 05_forest_plot.py          # [STEP 5] 계층적 로지스틱 회귀 OR Forest Plot
│
└── README.md
```

> ℹ️ `01_data_processing.py` ↔ `03_modeling.py`는 이전 업로드 시 파일명과 내용이 뒤바뀌어 있었습니다. 지금은 파일명 그대로 내용이 일치하도록 정정된 상태입니다 (01=전처리, 03=모델링).

---

## ⚙️ 실행 방법

### 사전 준비

KNHANES 원본 데이터는 [질병관리청 국민건강영양조사](https://knhanes.kdca.go.kr) 에서 직접 다운로드 후 `data/` 폴더에 위치시켜야 합니다.

```bash
pip install pandas numpy scipy statsmodels scikit-learn xgboost lightgbm catboost imbalanced-learn shap streamlit plotly streamlit-option-menu
```

### 1단계: 데이터 전처리

```bash
python 01_data_processing.py
```

- 20–39세 필터링, 약물 복용군 포함(약물 보정 방식)
- 연속형 변수 윈저화(상하위 1%), 가중치 정규화
- 운동그룹 / 흡연상태 / 음주상태 파생변수 생성
- 출력: `data/0325_hn_all(med).csv`

### 2단계: 검정통계표 생성

```bash
python 02_statistical_table.py
```

- 복합표본 가중치(`wt_itvex`) 적용
- 연속형: 가중 t-검정 / 범주형: 카이제곱 검정
- 출력: `data/hn_all_검정통계표.xlsx`

### 3단계: 모델 학습 (Google Colab 권장)

`03_modeling.py`를 Google Colab에서 실행합니다.

```
Google Colab → 공유 드라이브 경로 설정 → 전체 실행
```

학습되는 모델: Logistic / RandomForest / XGBoost / LightGBM / CatBoost  
평가 지표: ROC-AUC, PR-AUC, Recall, F1, Brier Score, DCA

### 4단계: 웹 서비스 실행

```bash
streamlit run 04_streamlit_app.py
```

### 5단계: Forest Plot 생성 (선택)

```bash
python 05_forest_plot.py
```

- 3단계 계층적 로지스틱 회귀(Model 1→2→3) 각 변수의 OR을 Forest Plot으로 시각화
- Model 3 전체 변수 Forest Plot, 모델별 적합도 변화 + 운동 그룹 OR, 성별 층화 비교 등 3종 차트 생성

---

## 🔬 핵심 방법론

### 데이터 설계

- **대사증후군 진단 기준**: 대한비만학회 NCEP-ATP III 기준 (5개 구성요소 중 3개 이상)
  - 복부비만(허리둘레), 고중성지방혈증, 저HDL콜레스테롤, 고혈압, 고공복혈당
- **약물 보정**: 고혈압·이상지질혈증·당뇨 약복용자를 해당 구성요소 '이상'으로 판정
- **클래스 불균형 보정**: 학습 데이터에 SMOTE 적용 (0:1 비율 1:1 균등화)

### 모델 선정 근거

통계적 설명력 확보를 위해 머신러닝 모델링 전 **3단계 계층적 로지스틱 회귀(LRT 검정)** 수행:

| 모델 | 추가 변수 | LRT p-value |
|------|-----------|-------------|
| Model 1 | 연령, 성별 | N/A |
| Model 2 | + 흡연, 음주 | 0.0001 |
| Model 3 | + 운동그룹 | < 0.0001 |

운동 그룹의 독립 효과(OR 0.36, p<0.001) 통계적으로 입증 → 핵심 예측 변수로 확정

> 📈 위 3단계 모델의 변수별 OR은 `05_forest_plot.py`로 Forest Plot 시각화하여 확인할 수 있습니다.

### 최종 모델 성능

| 구분 | 값 |
|------|----|
| ROC-AUC (Unweighted) | 0.7345 |
| ROC-AUC (Weighted, 복합표본 반영) | 0.7323 |
| AUC 편차 | 0.0022 (< 0.02 기준 충족) |
| Recall | 0.6923 |
| F1 Score | 0.3285 |

---

## 📊 주요 분석 결과

- **성별·연령**: 5개 모델 공통 1순위 예측 변수 (SHAP 일관)
- **운동 효과**: 복합운동(유산소+근력) 그룹 유병률 8.3% vs 미실천 그룹 16.4% (약 2배 차이)
- **흡연**: 현재흡연 OR 1.80 (+79.9% 위험 증가, p<0.001)
- **음주**: p=0.289로 유의하지 않음 (J-curve 현상 및 금주자 편향 영향)

---

## 🌐 Streamlit 웹 서비스

사용자 입력 흐름:

```
라이프스타일 입력 (연령·성별·흡연·음주·운동)
    ↓
임상 정보 선택 입력 (허리둘레·혈압·혈당·중성지방·HDL·약복용 여부)
    ↓
AI 예측 위험도 + 임상 기준 위험 요인 병합
    ↓
대사증후군 위험도(%) + 맞춤형 솔루션 가이드 출력
```

- 임상 정보 미입력 시: ML 모델 예측 확률만 표시
- 임상 기준 위험 요인 3개 이상 해당 시: 무조건 100%(확진 수준) 표시

---

## ⚠️ 한계점 및 향후 과제

**한계점**
- 설문·검진 데이터 기반으로 활동량·수면 등 실시간 데이터 미반영
- 식단 정보 미포함으로 영양 관련 케어 기능 제한
- 장기적 개선 효과 모니터링 및 공공기관 사업 연계 미비

**향후 과제**
- 스마트워치 실시간 데이터 통합으로 예측 정밀도 향상
- 고위험군 대상 보건소 등 공공기관 사업 연계
- 식단 데이터 추가 입력 및 분석으로 관리 체계 고도화

---


## 📚 참고 문헌

- 이광인. "머신러닝을 활용한 대사증후군 발생 예측 모델 개발." 고려대학교 대학원, 2026.
- 김윤성, 박영민, & 김동일 (2026). 유산소운동 및 근력운동의 규칙적인 참여가 대사건강 지표에 미치는 영향. *운동과학, 35*(1), 41-48.
- Liang M, et al. Effects of aerobic, resistance, and combined exercise on metabolic syndrome parameters. *Rev Cardiovasc Med.* 2021;22(4):1523-33.
- Myers J, et al. Physical activity, cardiorespiratory fitness, and the metabolic syndrome. *Nutrients.* 2019;11(7).

---

## 📄 라이선스

본 프로젝트는 학술 및 교육 목적으로 작성되었습니다.  
KNHANES 원본 데이터는 [질병관리청 이용 정책](https://knhanes.kdca.go.kr)을 따릅니다.
