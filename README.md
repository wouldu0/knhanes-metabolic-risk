# 🩺 KNHANES 기반 청년층 대사증후군 위험 분석 및 예측

> **국민건강영양조사(KNHANES) 기반 20–39세 청년층의 생활습관 및 임상 데이터를 분석하고 대사증후군 위험을 예측한 헬스케어 데이터 프로젝트**

---

## 🎬 시연 영상

[![시연 영상](https://img.youtube.com/vi/TcNgaddpYXQ/0.jpg)](https://youtube.com/shorts/TcNgaddpYXQ?feature=share)

---

## 📌 프로젝트 개요

청년층(20–39세) 대사증후군 유병률이 빠르게 증가하고 있음에도 기존 관리 시스템은 중장년층 위주로 설계되어 있습니다.  
본 프로젝트는 라이프스타일(연령·성별·흡연·음주·운동) 기반 ML 위험도 예측과 임상 수치 기반 대사증후군 기준 확인을 결합하여, Streamlit 웹 인터페이스로 제공합니다.

| 항목 | 내용 |
|------|------|
| 데이터 출처 | 국민건강영양조사(KNHANES) 제9기 (2022–2024) |
| 분석 대상 | 20–39세 청년층 N=3,363명 |
| 학습/테스트 분할 | Stratified 8:2 (N=2,690 / N=673), SMOTENC는 학습 데이터 교차검증 내부에서만 적용 |
| 최종 모델 | Tuned RandomForest + Isotonic Calibration |
| 성능 (Test) | ROC-AUC 0.7122 · Recall 0.654 · F1 0.2812 |

> ℹ️ 위 성능은 프로젝트 종료 후 리팩터링한 파이프라인(`06_model_refactor.py`) 기준입니다. 자세한 배경은 아래 "🔧 ML 파이프라인 리팩터링" 절 참고.

![연령대별 성별 분포](results/chart1_age_gender_distribution.png)

---

## 👥 팀 구성 및 담당 역할

| 항목 | 내용 |
|------|------|
| 개발 기간 | 2026.04 (약 1개월) |
| 팀 구성 | 6인 팀 — 기획(AA) 2인, 데이터 분석(DA) 2인, 모델링/구현(TA) 2인 |

**프로젝트 당시 담당 (DA)**
데이터 전처리(`01_data_processing.py`), 검정통계표 작성(`02_statistical_table.py`), 3단계 계층적 로지스틱 회귀 및 Forest Plot(OR) 분석(`05_forest_plot.py`)을 단독 수행. ML 모델링(`03_modeling.py`)과 Streamlit 웹 구현(`04_streamlit_app.py`)은 TA 팀원이 각 1인씩 담당.

**🔧 프로젝트 종료 후 개인 개선**
포트폴리오 정리 과정에서 두 부분을 다시 검토했습니다. TA가 작성한 `03_modeling.py`/`04_streamlit_app.py`에서 모델링 파이프라인의 구조적 문제(범주형 변수 처리, 교차검증 leakage, 서비스에 배포된 모델과 평가 모델의 불일치)를 발견해 `06_model_refactor.py`로 직접 리팩터링·재학습하고 `04_streamlit_app.py`의 모델 로딩 구조·임상 판정 로직도 함께 수정했습니다. 또한 본인이 작성한 3단계 계층적 로지스틱 회귀(`05_forest_plot.py`)도 KNHANES 복합표본 설계(가중치+층화+집락)를 완전히 반영하지 못했던 부분을 발견해 R `survey` 패키지로 재검증했습니다(`07_survey_analysis.R`). 자세한 내용은 "🔧 ML 파이프라인 리팩터링"과 "통계적 가설 검증" 절 참고.

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
├── 03_modeling.py             # [STEP 3] 모델 학습, 평가, SHAP 분석 (Google Colab, 팀 작성본 — 원본 보존)
├── 04_streamlit_app.py        # [STEP 4] 웹 서비스 구현
├── 05_forest_plot.py          # [STEP 5] 계층적 로지스틱 회귀 OR Forest Plot
├── 06_model_refactor.py       # [STEP 6] (프로젝트 종료 후 개선) ML 파이프라인 리팩터링 + 재학습
├── 07_survey_analysis.R       # [STEP 7] (프로젝트 종료 후 개선) 복합표본 설계(R survey) 재검증
├── feature_utils.py           # 06/04에서 공용으로 쓰는 피처 인코딩 함수
├── metabolic_risk_model.joblib  # 06에서 저장한 최종 모델 (04가 그대로 로드)
│
├── results/                   # README에 삽입된 결과 차트 (EDA, Forest Plot, 앱 화면)
├── requirements.txt
└── README.md
```

---

## ⚙️ 실행 방법

### 사전 준비

KNHANES 원본 데이터는 [질병관리청 국민건강영양조사](https://knhanes.kdca.go.kr) 에서 직접 다운로드 후 `data/` 폴더에 위치시켜야 합니다.

```bash
pip install -r requirements.txt
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

> ⚠️ 파일 상단 `load_and_setup_data()`의 `base_path`와 `ORIG_PATH`는 팀 공유 드라이브 경로로 되어 있습니다. 본인 환경의 경로로 수정한 뒤 실행하세요.

학습되는 모델: Logistic / RandomForest / XGBoost / LightGBM / CatBoost  
평가 지표: ROC-AUC, PR-AUC, Recall, F1, Brier Score, DCA

### 4단계: 웹 서비스 실행

```bash
streamlit run 04_streamlit_app.py
```

`metabolic_risk_model.joblib`(6단계 산출물)과 `feature_utils.py`가 같은 폴더에 있어야 합니다.

### 5단계: Forest Plot 생성 (선택)

```bash
python 05_forest_plot.py
```

> ⚠️ 파일 상단 `DATA_PATH`는 개인 Google Drive 경로로 되어 있습니다. 본인 환경의 `0325_hn_all(med).csv` 경로로 수정한 뒤 실행하세요.

- 3단계 계층적 로지스틱 회귀(Model 1→2→3) 각 변수의 OR을 Forest Plot으로 시각화
- Model 3 전체 변수 Forest Plot, 모델별 적합도 변화 + 운동 그룹 OR, 성별 층화 비교 등 3종 차트 생성

### 6단계: ML 파이프라인 재학습 (프로젝트 종료 후 개선, 선택)

```bash
python 06_model_refactor.py
```

- `data/0325_hn_all(med).csv`(3,363명 전체)를 새로 stratified 8:2 분할해 실행합니다 (팀이 만든 기존 `Original_Train_Data.csv`/`Test_Data.csv` 분할은 재사용하지 않음 — 이유는 아래 방법론 참고).
- 5개 후보 모델(Logistic/RandomForest/XGBoost/LightGBM/CatBoost)을 학습 데이터 CV로 비교 → 최종 모델 선정 → Isotonic 보정 → threshold 산출 → test set 평가까지 한 번에 수행하고 `metabolic_risk_model.joblib`을 저장합니다.
- macOS에서 XGBoost가 `libomp.dylib` 로드 오류를 낼 수 있습니다. `brew install libomp`로 해결됩니다.

### 7단계: 복합표본 설계 재검증 (프로젝트 종료 후 개선, 선택, R 필요)

```bash
Rscript 07_survey_analysis.R
```

- R과 `survey` 패키지가 필요합니다: `install.packages("survey")`
- 입력 파일은 `data/0325_hn_all(med).csv`가 아니라 `data/hn_all_full_merged.csv`입니다 — 원본 KNHANES 전체 표본(전 연령, `hn_all.csv`)에 `01_data_processing.py`가 만든 파생변수(`metabolic_syndrome` 등, 20-39세 analytic sample만 값 존재)를 `ID` 기준으로 합친 파일입니다. 20-39세만 먼저 잘라서 `svydesign()`을 만들면 그 하위표본에 없는 strata/PSU의 설계 정보가 사라져 분산추정이 편향될 수 있어, **원표본 전체로 design을 정의한 뒤 `subset()`으로 20-39세를 지정**하는 방식을 씁니다(survey 패키지 공식 권장 방식). 병합 방법:
  ```python
  full = pd.read_csv("data/hn_all.csv")[["ID","psu","kstrata","wt_itvex","age","sex"]]
  sub = pd.read_csv("data/0325_hn_all(med).csv")[["ID","metabolic_syndrome","exercise_group","smoking_status","drinking_status"]]
  full.merge(sub, on="ID", how="left").to_csv("data/hn_all_full_merged.csv", index=False)
  ```
- `svydesign`(weight+strata+PSU, 전 연령) → `subset(!is.na(metabolic_syndrome))`으로 20-39세 analytic sample 지정 → `svyglm(quasibinomial)`로 3단계 계층적 로지스틱 회귀를 재추정하고, `regTermTest`로 운동그룹 전체 효과를 Wald test로 확인합니다.
- 분석 대상 3,363명 전체를 사용합니다 (ML용 train/test 분할과 무관).

---

## 🔬 핵심 방법론

### 데이터 설계

- **대사증후군 진단 기준**: 대한비만학회 NCEP-ATP III 기준 (5개 구성요소 중 3개 이상)
  - 복부비만(허리둘레), 고중성지방혈증, 저HDL콜레스테롤, 고혈압, 고공복혈당
- **약물 보정**: 고혈압·이상지질혈증·당뇨 약복용자를 해당 구성요소 '이상'으로 판정
- **클래스 불균형 보정**: 팀 원본 파이프라인(`03_modeling.py`)은 학습 데이터에 SMOTE를 적용 (0:1 비율 1:1 균등화). 이 방식의 한계와 개선은 아래 "🔧 ML 파이프라인 리팩터링" 절 참고

### 통계적 가설 검증 — 운동 변수의 독립적 연관성

연구 질문은 처음부터 끝까지 동일합니다 — *연령·성별·흡연·음주를 보정했을 때 운동그룹과 대사증후군은 어떤 연관성을 보이는가?* 이를 **3단계 계층적 로지스틱 회귀**(연령·성별 → 흡연·음주 → 운동그룹 순차 투입)로 검증했습니다.

당시(`05_forest_plot.py`)에는 KNHANES 표본가중치(`freq_weights`)와 PSU 단위 cluster-robust 표준오차로 이를 추정했는데, 층화(`kstrata`)까지는 반영하지 못한 상태였습니다. **프로젝트 종료 후**, 동일한 모델 구조를 R `survey` 패키지의 `svydesign`(weight + strata + PSU)·`svyglm(quasibinomial)`로 다시 추정했습니다(`07_survey_analysis.R`). 이때 20-39세만 먼저 잘라서 design을 만들지 않고, KNHANES 원표본 전체(전 연령, 20,191명)로 design을 정의한 뒤 `subset()`으로 20-39세 analytic sample(3,363명)을 지정했습니다 — subset 대상만으로 design을 만들면 그 하위표본에 없는 strata/PSU의 설계 정보가 사라져 분산추정이 편향될 수 있어, survey 패키지가 공식적으로 권장하는 방식입니다. LRT·AIC는 var_weights 기반 Binomial GLM에서 log-likelihood가 정의되지 않아 부적절하다는 statsmodels 문서의 지적과 같은 이유로 사용하지 않고, 운동그룹 전체 효과는 design-based Wald test(`regTermTest`)로 확인했습니다: **F=11.06 (df=3, 495), p<0.001**.

Train/Test 분할은 ML 모델의 "새로운 사람에 대한 예측력"을 보기 위한 것이고, 이 통계 분석은 "분석 표본 내 연관성"을 보는 것이므로 분석 대상 전체 3,363명을 그대로 사용했습니다.

재추정한 결과는 이전 방법과 사실상 일치했습니다 — 복합운동군의 대사증후군 odds는 운동을 하지 않는 군 대비 약 60% 낮게 나타났습니다(OR 0.40, 95% CI 0.28–0.57, p<0.001). 관찰연구 기반 분석이므로 인과관계로 해석하지 않았습니다.

![Model 3 전체 변수 Forest Plot (복합표본 설계 반영)](results/forest_plot_1_model3_exercise_last.png)

### 🔧 ML 파이프라인 리팩터링 (프로젝트 종료 후 개선)

포트폴리오 정리 과정에서 팀이 작성한 `03_modeling.py`를 다시 검토하다 아래 문제를 발견했습니다.

1. **범주형 변수를 인코딩 없이 그대로 투입** — `sex`/`smoking_status`/`drinking_status`/`exercise_group`을 정수 코드 그대로 SMOTE와 분류기에 넣었습니다. SMOTE가 이 값들을 연속형처럼 보간해 `exercise_group=1.64` 같은 존재하지 않는 값을 만들고, 모델도 그룹 간 간격이 동일하다고 가정하게 됩니다.
2. **SMOTE를 교차검증 밖에서 한 번만 적용** — `train_test_split` 직후 학습셋 전체에 SMOTE를 적용한 뒤 `RandomizedSearchCV(cv=5)`에 넣어, 하이퍼파라미터 탐색의 각 validation fold에도 synthetic sample의 정보가 섞였습니다(leakage).
3. **평가에 쓴 hold-out을 모델 선택에도 재사용** — 원 발표자료를 보면 673명 test set은 최종 평가 1회가 아니라 5개 모델 × 3개 보정방식을 비교해 최종 모델(Logistic+Isotonic)을 **선택**하는 데 이미 쓰였습니다.
4. **평가한 모델과 배포된 모델이 다름** — `04_streamlit_app.py`는 이 모델을 저장·로드하지 않고 앱 실행 시 전체 데이터로 별도 재학습(SMOTE 없이, train/test 분리도 없이)했습니다. README에 보고한 성능과 실제 서비스 모델이 같은 모델이 아니었습니다.

**개선(`06_model_refactor.py`)**: `SMOTENC`(범주형을 원래 코드 그대로 유지한 채 리샘플링) → 원-핫 인코딩 → 분류기를 하나의 `imblearn.Pipeline`으로 묶어 `RandomizedSearchCV`에 통째로 전달했습니다. 3,363명 전체를 새로 stratified 8:2 분할하고(팀의 기존 673명은 이미 모델 선택에 쓰였으므로 재사용하지 않음), 5개 후보 모델을 **학습 데이터 CV ROC-AUC만으로** 비교했습니다 — test set은 최종 평가 1회에만 사용:

| 모델 | CV ROC-AUC |
|---|---|
| RandomForest | 0.7707 |
| CatBoost | 0.7704 |
| Logistic | 0.7686 |
| XGBoost | 0.7661 |
| LightGBM | 0.7647 |

상위 3개 모델은 사실상 오차범위 내 차이지만, 사전에 정한 절차(CV 1위 모델 채택)를 그대로 따라 **RandomForest**를 최종 모델로 선정했습니다. Isotonic 보정 후 threshold는 학습 데이터의 교차검증 예측값(OOF)만으로 결정하고, 새 test set(N=673)에서 마지막 평가 1회를 수행했습니다.

| 구분 | 값 |
|------|----|
| ROC-AUC | 0.7122 |
| Survey-weighted ROC-AUC | 0.7281 |
| Recall | 0.654 (threshold=0.114) |
| F1 | 0.2812 |
| Brier | 0.0994 |

기존(리팩터링 전) 수치 대비 다소 낮지만, leakage 없이 정직하게 재현한 결과입니다. 최종 모델이 RandomForest로 바뀌면서 회귀계수 기반 해석은 할 수 없게 됐지만, 운동 변수의 통계적 유의성은 위 3단계 계층적 로지스틱 회귀·Forest Plot 분석이 별도로 담당하고 있어 예측 모델의 선택과는 무관하게 유지됩니다.

`04_streamlit_app.py`도 이 모델을 `joblib.load()`로 그대로 불러오도록 수정해, README에 보고한 성능과 실제 서비스 모델이 같은 모델이 되도록 통일했습니다.

---

## 📊 주요 분석 결과

- **성별·연령**: 원 팀 분석(`03_modeling.py`)의 SHAP 기준 5개 모델 공통 1순위 예측 변수
- **운동 효과**: 복합운동(유산소+근력) 그룹 유병률 8.3% vs 미실천 그룹 16.4% (약 2배 차이)
- **흡연**: 현재흡연 OR 1.80 (비흡연 대비 odds 약 79.9% 높음, p<0.001)
- **음주**: p=0.289로 유의하지 않음 (J-curve 현상 및 금주자 편향 영향)

![운동 그룹별 대사증후군 유병률](results/chart2_exercise_group_prevalence.png)

---

## 🌐 Streamlit 웹 서비스

ML 예측과 임상 기준 판정을 하나의 확률로 섞지 않고, 역할이 다른 두 결과로 분리해서 보여줍니다 (원래 서비스 설계도 "① AI 모델 예측"과 "② 임상정보 기반 위험요인 계산"을 별개 로직으로 두고 있었고, 이번 개선은 화면에서도 이 둘을 분리해서 보여주도록 정리한 것입니다).

```
① 라이프스타일 입력 (필수: 연령·성별·흡연·음주·운동)
    → ML 모델(RandomForest)로 예측 확률 계산 (건강검진 수치는 사용하지 않음)
    → "생활습관 기반 ML 스크리닝 — 예측 확률 n%"

② 임상 정보 입력 (선택: 허리둘레·혈압·혈당·중성지방·HDL·약물 복용)
    → NCEP-ATP III 5개 구성요소를 rule-based로 판정 (약물 복용 시 해당 항목 자동 반영)
    → 5개 모두 입력 시: "n/5개 충족 → 기준 충족 / 미충족"
    → 일부만 입력 시: "n개 확인됨, 나머지는 확인 보류" (모르는 항목을 정상으로 간주하지 않음)
    → 미입력 시: 임상 기준 확인 결과 없음
```

두 결과는 항상 별도 카드로 표시되며, 임상 기준 3개 이상 충족을 ML 확률에 강제로 덮어씌우는 로직은 없습니다. 약물 복용 매핑은 `01_data_processing.py`와 동일하게 이상지질혈증 약 복용 시 중성지방·HDL 두 항목 모두에 반영됩니다.

![생활습관 기반 ML 위험도 예측과 임상정보 입력 상태에 따른 기준 확인 결과를 분리 제공](results/app_screenshot_risk_check.png)

---

## 🔧 트러블슈팅

### 1) 약물 복용군 처리 방식 결정 — 배제 vs 보정

**문제**: 고혈압·이상지질혈증·당뇨 약을 복용 중인 사람은 약물 효과로 검사 수치가 정상 범위에 들어오는 경우가 많습니다. 원본 수치만으로 대사증후군을 판정하면 실제로는 위험군인 사람을 정상으로 오분류하게 됩니다.

**시도**: 약물 복용군을 진단 기준 산정에서 제외하는 방식과, 전원을 포함하되 약물 복용 시 해당 구성요소를 자동으로 '이상'으로 판정하는 방식(약물 보정)을 각각 전처리해서 비교했습니다.

**결정**: 약물 보정 방식을 채택했습니다. 표본을 잃지 않으면서, 치료 중인 사람도 실질적인 위험군으로 반영하는 것이 임상적으로 더 타당하다고 판단했습니다. `01_data_processing.py` STEP 4-4에 약물 보정 매핑(`DI1_2`→`ms_bp`, `DI2_2`→`ms_tg`/`ms_hdl`, `DE1_31`·`DE1_32`→`ms_glu`)으로 남아 있습니다.

### 2) 연속형 변수 전처리 방식 시행착오

**문제**: 허리둘레·혈압·혈당·중성지방·HDL 같은 임상 수치는 극단치가 섞여 있어, 어떻게 다듬어야 통계 검정과 모델 성능 양쪽에서 무리가 없을지 바로 정해지지 않았습니다.

**시도**: 윈저화 적용 전/후, 음주 변수를 다범주로 둘지 이분화할지 등 여러 버전을 각각 만들어 검정통계표 결과를 비교하며 최종 방식을 결정했습니다.

**결정**: 연속형 변수는 상·하위 1% 윈저화, 음주는 '월 1회 이상 현재 음주 여부'로 이분화하는 방식으로 정착했습니다(`01_data_processing.py` STEP 4-3, STEP 5).

---

## 💡 프로젝트 경험

**임상 기준의 데이터 로직 전환**
NCEP-ATP III 대사증후군 진단 기준과 약물 복용 여부를 판정 규칙(`ms_wc`, `ms_tg`, `ms_hdl`, `ms_bp`, `ms_glu`)으로 코드화하여, 임상 지식을 분석 가능한 파생변수로 변환했습니다.

**KNHANES 가중 분석 설계**
국민건강영양조사의 표본 가중치를 기술통계(`DescrStatsW` 가중 t-검정) 및 카이제곱 검정에 적용했습니다. `02_statistical_table.py`의 이 기술통계표는 가중치만 반영하고 층화·집락까지 포함하는 완전한 survey-design 검정은 아니라, 탐색적 수준으로 해석했습니다.

**가설 검증을 통한 변수 효과 분석**
연령·성별 → 흡연·음주 → 운동그룹을 순차적으로 추가하는 3단계 계층적 로지스틱 회귀로 모델 적합도 변화와 OR을 비교하고, Forest Plot으로 시각화해 통계적 근거를 가진 핵심 예측 변수를 팀에 제시했습니다.

**(프로젝트 종료 후) 복합표본 설계 재검증 및 ML 파이프라인 리팩터링**
포트폴리오 정리 과정에서 두 가지를 다시 검토했습니다. 첫째, 본인이 작성한 3단계 계층적 로지스틱 회귀(`05_forest_plot.py`)가 표본가중치·집락은 반영했지만 층화까지는 반영하지 못했다는 걸 발견해, R `survey` 패키지(`svydesign`+`svyglm`+`regTermTest`)로 weight·strata·PSU를 모두 반영해 재추정했습니다(`07_survey_analysis.R`) — 결과는 기존과 거의 일치해 원래 분석이 견고했음을 재확인했습니다. 둘째, 팀이 작성한 ML 모델링 코드(`03_modeling.py`)에서 범주형 변수 인코딩 누락, SMOTE-CV leakage, 모델 선택에 hold-out이 재사용된 점, 평가 모델과 배포 모델의 불일치까지 구조적 문제를 발견해 `SMOTENC` 기반 `imblearn.Pipeline`으로 재구성하고(`06_model_refactor.py`) Streamlit 앱이 이 검증된 모델을 그대로 불러오도록 통일했습니다. 처음 만든 코드가 아니어도 구조를 끝까지 따라가며 문제를 진단하고, 통계적으로도 방법론적으로도 더 정확한 근거를 가진 결과로 마무리하는 경험이었습니다.

---

## 📖 변수 정의 (데이터 사전)

<details>
<summary>전체 변수 정의 펼쳐보기</summary>

### 기본 인구학적 변수

| 변수명 | 설명 | 내용 |
|---|---|---|
| `sex` | 성별 | 1=남자, 2=여자 |
| `age` | 나이 | 만 나이(세), 20~39세만 사용 |

### 복합표본설계 변수

| 변수명 | 설명 | 내용 |
|---|---|---|
| `wt_itvex` | 건강설문·검진조사 가중치 | 복합표본 분석용 원본 가중치 |
| `w_norm` | 정규화 가중치 (파생) | `wt_itvex`를 표본 수에 맞춰 정규화 |
| `kstrata` | 층화변수 | 복합표본설계 층화 |
| `psu` | 1차 추출단위 | 복합표본설계 집락(PSU) |

### 운동 관련 변수

| 변수명 | 설명 | 내용 |
|---|---|---|
| `BE5_1` | 1주일간 근력운동 일수(원시) | 1~6=0~5일 이상, 8/9=비해당·무응답 → 3(2일) 이상이면 근력운동 실천 |
| `pa_aerobic` | 유산소 신체활동 실천율(원시) | 0=권장수준 미달, 1=충족(중강도 150분 또는 고강도 75분 이상) |
| `exercise_group` | 운동 그룹 분류(파생) | 1=복합(근력+유산소), 2=근력만, 3=유산소만, 4=둘 다 안 함 |

### 대사증후군 관련 변수

| 변수명 | 설명 | 측정단위 | 이상 기준 |
|---|---|---|---|
| `HE_wc` | 허리둘레(원시) | cm | 남 ≥90, 여 ≥85 |
| `HE_TG` | 중성지방(원시) | mg/dL | ≥150 |
| `HE_HDL_st2` | HDL 콜레스테롤(원시) | mg/dL | 남 <40, 여 <50 |
| `HE_sbp` | 수축기혈압(원시) | mmHg | ≥130 |
| `HE_dbp` | 이완기혈압(원시) | mmHg | ≥85 |
| `HE_glu` | 공복혈당(원시) | mg/dL | ≥100 |

| 변수명 | 설명 | 판정 기준(파생, 약물 보정 포함) |
|---|---|---|
| `ms_wc` | 복부비만 | `HE_wc` 기준 초과 |
| `ms_tg` | 고중성지방혈증 | `HE_TG`≥150 또는 이상지질혈증 약 복용(`DI2_2`) |
| `ms_hdl` | 낮은 HDL 콜레스테롤 | `HE_HDL_st2` 기준 미달 또는 이상지질혈증 약 복용(`DI2_2`) |
| `ms_bp` | 높은 혈압 | `HE_sbp`/`HE_dbp` 기준 초과 또는 고혈압 약 복용(`DI1_2`) |
| `ms_glu` | 높은 공복혈당 | `HE_glu`≥100 또는 인슐린·혈당강하제 사용(`DE1_31`/`DE1_32`) |
| `metabolic_syndrome` | 대사증후군 최종 판정 | 위 5개 구성요소 중 3개 이상 |
| `ms_count` | 대사증후군 구성요소 개수 | 0~5점 |

### 약물 복용 관련 변수

| 변수명 | 설명 | 내용 |
|---|---|---|
| `DI1_2` | 혈압조절제 복용 | 1~4=복용, 5=미복용, 8/9=비해당·무응답 |
| `DI2_2` | 이상지질혈증약 복용 | 1~4=복용, 5=미복용, 8/9=비해당·무응답 |
| `DE1_31` | 인슐린 주사 투여 | 0=아니오, 1=예 |
| `DE1_32` | 당뇨병약 복용 | 0=아니오, 1=예 |

### 흡연 관련 변수

| 변수명 | 설명 | 내용 |
|---|---|---|
| `BS3_1` | 일반담배(궐련) 흡연(원시) | 1,2=현재, 3=과거, 8=비해당(평생비흡연) |
| `BS12_47` | 궐련형 전자담배 흡연(원시) | 1,2=현재, 3=과거, 8=비해당 |
| `BS12_2` | 액상형 전자담배 현재사용(원시) | 1=예, 8=비해당 |
| `smoking_status` | 흡연 상태(파생) | 0=비흡연, 1=과거흡연, 2=현재흡연 |

### 음주 관련 변수

| 변수명 | 설명 | 내용 |
|---|---|---|
| `BD1_11` | 1년간 음주빈도(원시) | 1=전혀 안 마심, 2~6=월1회 미만~주4회 이상 |
| `drinking_status` | 음주 상태(파생) | 0=현재비음주(`BD1_11`=1), 1=현재음주(`BD1_11`=2~6) |

### 분석 대상자 선정 기준

| 조건 | 기준 |
|---|---|
| 연령 | 20~39세 |
| 주요 변수 유효성 | 운동·대사증후군·흡연·음주 변수 결측치 없는 경우만 포함 |
| 최종 대상자 | 3,363명 |

### 윈저화(Winsorization)

`HE_wc`, `HE_TG`, `HE_HDL_st2`, `HE_sbp`, `HE_dbp`, `HE_glu` 6개 연속형 변수에 상·하위 1% 절단을 적용해 극단값 영향을 최소화했습니다(정보 손실 없이 분포만 완화).

</details>

---

## ⚠️ 한계점 및 향후 과제

**한계점**
- 설문·검진 데이터 기반으로 활동량·수면 등 실시간 데이터 미반영
- 식단 정보 미포함으로 영양 관련 케어 기능 제한
- 장기적 개선 효과 모니터링 및 공공기관 사업 연계 미비
- 리팩터링 후 5개 후보 모델의 CV ROC-AUC가 상위 3개(RandomForest·CatBoost·Logistic) 모두 0.77 안팎으로 오차범위 내 차이 — 사전에 정한 절차(CV 1위 채택)를 따랐지만, 다른 시드/분할에서는 순위가 바뀔 수 있음
- 최종 모델이 RandomForest라 회귀계수 기반 해석은 어려움 (변수 간 연관성 해석은 별도의 통계 분석(Forest Plot)이 담당)

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

## 📄 데이터 이용 안내

본 프로젝트는 학술 및 교육 목적으로 작성되었습니다. KNHANES 원본 데이터는 저장소에 포함하지 않으며, [질병관리청 국민건강영양조사 이용 정책](https://knhanes.kdca.go.kr)을 따릅니다.
