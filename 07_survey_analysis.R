# ══════════════════════════════════════════════════════════════
#  [프로젝트 종료 후 개선] DA 통계 분석 재검증
#  ★ 05_forest_plot.py(freq_weights + cluster-robust)를
#    KNHANES 복합표본 설계(weight + strata + PSU)를 반영한
#    survey::svyglm 기반 분석으로 재수행 ★
#
#  연구 질문과 모델 구조(Model 1→2→3)는 원래 그대로 유지하고,
#  계산 방식만 아래처럼 정석에 맞게 바꿈:
#    - 일반 GLM+weight        → svydesign(weight+strata+PSU) 기반 svyglm(quasibinomial)
#    - 일반 LRT/AIC            → regTermTest(Wald) / AIC 사용 안 함
#      (statsmodels 문서상 var_weights를 쓴 Binomial GLM은 log-likelihood가
#       제대로 정의되지 않아 AIC·LRT가 부적절 — 같은 이유로 여기서도 배제)
#    - 20-39세만 먼저 잘라서 design 생성 → 원표본(20,191명, 전 연령) 전체로
#      design을 정의한 뒤 subset()으로 20-39세 analytic sample을 지정.
#      복합표본에서 subset 대상만으로 design을 만들면 그 하위 표본에
#      없는 strata/PSU의 표본설계 정보가 사라져 분산추정이 편향될 수
#      있음 — subset()은 design 정보를 보존한 채 해당 subpopulation의
#      분산만 올바르게 추정함 (survey 패키지 공식 권장 방식).
#
#  Train/Test 분할(ML용)과는 무관하게, 분석 대상 전체 3,363명을 사용함
#  ("새로운 사람을 잘 예측하는가"가 아니라 "표본 내 연관성이 어떠한가"를
#  보는 분석이므로).
#
#  data/hn_all_full_merged.csv: 원본 KNHANES 전체 표본(20,191명, 전 연령)에
#  01_data_processing.py가 만든 파생변수(metabolic_syndrome 등, 20-39세
#  analytic sample 3,363명만 값 존재, 나머지는 NA)를 ID 기준으로 합친 파일.
# ══════════════════════════════════════════════════════════════

suppressMessages(library(survey))

df <- read.csv("data/hn_all_full_merged.csv")

cat(sprintf("원표본(전 연령): %d명 / 20-39세 analytic sample: %d명\n",
            nrow(df), sum(!is.na(df$metabolic_syndrome))))

df$male          <- ifelse(df$sex == 1, 1, 0)
df$smoke_past    <- ifelse(df$smoking_status == 1, 1, 0)
df$smoke_current <- ifelse(df$smoking_status == 2, 1, 0)
df$drink_current <- ifelse(df$drinking_status == 1, 1, 0)
df$ex_1          <- ifelse(df$exercise_group == 1, 1, 0)
df$ex_2          <- ifelse(df$exercise_group == 2, 1, 0)
df$ex_3          <- ifelse(df$exercise_group == 3, 1, 0)

# ── design은 원표본 전체(전 연령)로 정의 ──
design_full <- svydesign(
  id      = ~psu,
  strata  = ~kstrata,
  weights = ~wt_itvex,
  data    = df,
  nest    = TRUE
)

# ── 20-39세 analytic sample은 subset()으로 지정 (design 정보 보존) ──
design <- subset(design_full, !is.na(metabolic_syndrome))
cat(sprintf("subset() 적용 후 분석 대상: %d명\n", nrow(design)))

model1 <- svyglm(metabolic_syndrome ~ age + male,
                  design = design, family = quasibinomial())
model2 <- svyglm(metabolic_syndrome ~ age + male + smoke_past + smoke_current + drink_current,
                  design = design, family = quasibinomial())
model3 <- svyglm(metabolic_syndrome ~ age + male + smoke_past + smoke_current + drink_current +
                    ex_1 + ex_2 + ex_3,
                  design = design, family = quasibinomial())

extract_or <- function(model) {
  co <- summary(model)$coefficients
  ci <- confint(model)
  data.frame(
    term     = rownames(co),
    OR       = exp(co[, 1]),
    CI_lower = exp(ci[, 1]),
    CI_upper = exp(ci[, 2]),
    p_value  = co[, 4],
    row.names = NULL
  )
}

res1 <- extract_or(model1); res1$model <- "Model 1"
res2 <- extract_or(model2); res2$model <- "Model 2"
res3 <- extract_or(model3); res3$model <- "Model 3"

cat("\n=== Model 3 (전체 변수, survey-weighted) ===\n")
print(res3)

write.csv(res3, "survey_model3_or.csv", row.names = FALSE)
write.csv(rbind(res1, res2, res3), "survey_all_models_or.csv", row.names = FALSE)

# ── 운동그룹 전체 효과 Wald test (LRT/AIC 대체) ──
wald_ex <- regTermTest(model3, ~ex_1 + ex_2 + ex_3, method = "Wald")
cat("\n=== 운동그룹 전체 효과 Wald test (Model 3) ===\n")
print(wald_ex)
sink("survey_wald_test.txt")
print(wald_ex)
sink()

# ── 성별 층화 (Model 3, 성별 변수 제외) ──
design_male   <- subset(design, male == 1)
design_female <- subset(design, male == 0)

model3_male <- svyglm(metabolic_syndrome ~ age + smoke_past + smoke_current + drink_current +
                         ex_1 + ex_2 + ex_3,
                       design = design_male, family = quasibinomial())
model3_female <- svyglm(metabolic_syndrome ~ age + smoke_past + smoke_current + drink_current +
                           ex_1 + ex_2 + ex_3,
                         design = design_female, family = quasibinomial())

res_male   <- extract_or(model3_male);   res_male$group   <- "남성"
res_female <- extract_or(model3_female); res_female$group <- "여성"

cat("\n=== 성별 층화 (남성) ===\n"); print(res_male)
cat("\n=== 성별 층화 (여성) ===\n"); print(res_female)

write.csv(rbind(res_male, res_female), "survey_gender_stratified_or.csv", row.names = FALSE)

cat("\n✅ 저장 완료: survey_model3_or.csv, survey_all_models_or.csv, survey_gender_stratified_or.csv, survey_wald_test.txt\n")
