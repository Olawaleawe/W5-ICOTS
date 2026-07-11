## =========================================================================
## SESSION 2 PRACTICE (11:00-12:30)  --  EXPLAIN
## ICOTS-12 Pre-Conference Workshop, Brisbane
## Instructor: Olawale Awe
## Mirrors Session2_Explain.pdf exactly, section for section:
##   2.1  Training five models
##   2.2  How stable is a result? (cross-validation)
##   2.3  Explaining with icarm (global + local)
##   2.4  A model-agnostic view (DALEX)
##   2.5  Beyond icarm's own models (caret + DALEX)
##   2.6  A second opinion (LIME)
##   2.7  Open exploration (modelStudio)
##
## This script is SELF-CONTAINED: it does not depend on any earlier script
## having been run first. If you're jumping straight into Session 2 without
## having run Session 1, this still works -- it rebuilds train_bin and
## test_bin itself, from the raw data, in the first few lines. Try with student-por data
## =========================================================================


## -------------------------------------------------------------------------
## SETUP
## -------------------------------------------------------------------------

required_pkgs <- c("icarm", "dplyr", "ggplot2", "caret", "gbm", "DALEX",
                   "lime", "modelStudio", "patchwork")
missing_pkgs <- required_pkgs[!sapply(required_pkgs, requireNamespace, quietly = TRUE)]
if (length(missing_pkgs) > 0) {
  stop("\n\nMissing package(s): ", paste(missing_pkgs, collapse = ", "),
       "\nInstall them first:\n  install.packages(c(",
       paste0('"', missing_pkgs, '"', collapse = ", "), "))\n", call. = FALSE)
}

suppressMessages({
  library(icarm); library(dplyr); library(ggplot2)
  library(caret); library(gbm); library(DALEX); library(lime)
  library(modelStudio); library(patchwork)
})

if (!file.exists("student-mat.csv")) {
  stop("\n\nCan't find 'data/student-mat.csv' from the current working directory:\n  ",
       getwd(), "\nRun this from the repository root, or open QUEST.Rproj in RStudio.", call. = FALSE)
}

## dir.create("figures", showWarnings = FALSE, recursive = TRUE)
## dir.create("outputs", showWarnings = FALSE, recursive = TRUE)

## Rebuild train_bin / test_bin from scratch (identical to Session 1's split,
## same seed, so results match the slides exactly even if run in isolation).
students <- read.csv2("student-mat.csv", stringsAsFactors = TRUE) %>%
  mutate(pass = factor(ifelse(G3 >= 10, "Pass", "Fail")))
sp <- icarm_split(students, prop = 0.75, seed = 2025, stratify = "pass")
train_bin <- sp$train %>% select(-G1, -G2, -G3)
test_bin  <- sp$test  %>% select(-G1, -G2, -G3)

cat(sprintf("Ready: %d training students, %d test students.\n", nrow(train_bin), nrow(test_bin)))


## =========================================================================
## 2.1  TRAINING FIVE MODELS
## =========================================================================
## PROBLEM: five model families exist for this problem. Which one should
## this workshop actually use?
## METHOD: one interface, five model types.

m_cart  <- icarm_fit(pass ~ ., train_bin, model = "cart",          positive = "Pass")
m_logit <- icarm_fit(pass ~ ., train_bin, model = "logistic",      positive = "Pass")
m_l1    <- icarm_fit(pass ~ ., train_bin, model = "logistic_l1",   positive = "Pass")
m_rf    <- icarm_fit(pass ~ ., train_bin, model = "random_forest", positive = "Pass")
m_svm   <- icarm_fit(pass ~ ., train_bin, model = "svm",           positive = "Pass")

## RESULT
cmp <- icarm_compare(
  list(CART = m_cart, Logistic = m_logit, `Logistic (L1)` = m_l1,
       RandomForest = m_rf, SVM = m_svm),
  test_data = test_bin, outcome = "pass", positive = "Pass")
print(cmp)
icarm_plot_comparison(cmp, metrics = c("accuracy", "f1"))

## INSIGHT: is there any accuracy reason left to prefer the black-box models
## over the fully interpretable Logistic (L1)? Write your answer:
##
## ___________________________________________________________________


## =========================================================================
## 2.2  HOW STABLE IS A RESULT?
## =========================================================================
## PROBLEM: a single train/test split gives exactly one number. How much
## would that number move under a different split?
## METHOD: 5-fold cross-validation, entirely within the training data.

ctrl <- caret::trainControl(method = "cv", number = 5,
                            classProbs = TRUE, summaryFunction = twoClassSummary)
cv_logit <- caret::train(pass ~ ., data = train_bin, method = "glm", family = "binomial",
                         trControl = ctrl, metric = "ROC")

## RESULT
print(cv_logit)
cv_logit$resample   # one row per fold

## INSIGHT: what's the range (max - min) of the ROC column across the five
## folds? Write it here, then compare with your neighbour's:
##
## ___________________________________________________________________

## TRY IT YOURSELF: repeat this for the random forest instead. Change
## method = "glm" to method = "rf" and drop the family argument. Does the
## fold-to-fold spread get better or worse?
# cv_rf <- caret::train(pass ~ ., train_bin, method = "rf", trControl = ctrl, metric = "ROC")
# cv_rf$resample


## =========================================================================
## 2.3  EXPLAINING WITH icarm
## =========================================================================
## PROBLEM: we have a working model. We don't yet know what it's actually
## using to make its decisions -- in general, or for any one student.
## METHOD: global explanation (the whole model) and local explanation
## (one prediction) are two different questions, with two different tools.

## ---- Global ----
ex_rf <- icarm_explain(m_rf, data = test_bin, label = "Random Forest")

## RESULT
ex_rf$importance %>% arrange(desc(importance)) %>% head(8)
icarm_plot_importance(ex_rf, n_features = 12)

## ---- Local ----
pred_rf  <- predict(m_rf, test_bin, type = "class")
case_id  <- which(pred_rf == "Fail")[1]   # a student predicted to fail

## RESULT
test_bin[case_id, c("failures", "goout", "absences")]
predict(m_rf, test_bin[case_id, ], type = "prob")
icarm_explain_local(ex_rf, newdata = test_bin[case_id, ], n_features = 6)

## INSIGHT: this student was predicted to fail with 87% confidence, driven
## almost entirely by failures = 3. Check the ACTUAL outcome:
test_bin$pass[case_id]
##
## Was the model right? What does that tell you about treating even a
## confident, well-supported prediction as certain for any one person?


## =========================================================================
## 2.4  A MODEL-AGNOSTIC VIEW
## =========================================================================
## PROBLEM: icarm_explain() is built specifically for icarm models. What
## if you're handed a model icarm never fit?
## METHOD: DALEX wraps ANY model with a predict() method.

explainer_rf <- DALEX::explain(
  model = m_rf$fit, data = test_bin %>% select(-pass), y = test_bin$pass == "Pass",
  predict_function = function(m, d) predict(m, d, type = "prob")[, "Pass"],
  label = "Random Forest", type = "classification", verbose = FALSE
)

## RESULT: performance, global importance, local break-down, partial dependence
DALEX::model_performance(explainer_rf)

vi_rf <- DALEX::model_parts(explainer_rf, B = 15)
plot(vi_rf, max_vars = 10)

bd_case <- DALEX::predict_parts(
  explainer_rf, new_observation = test_bin[case_id, ] %>% select(-pass), type = "break_down")
print(bd_case)
plot(bd_case)

## Partial dependence, computed by hand (no extra package needed):
pdp_failures <- sapply(sort(unique(test_bin$failures)), function(fval) {
  grid <- test_bin %>% select(-pass) %>% mutate(failures = fval)
  mean(predict(m_rf, grid, type = "prob")[, "Pass"])
})
data.frame(failures = sort(unique(test_bin$failures)), mean_P_Pass = pdp_failures)

## INSIGHT: DALEX's break-down and icarm's local explanation both point at
## the same student, the same variable. Two different pieces of software,
## the same answer -- how much should that increase your confidence?


## =========================================================================
## 2.5  BEYOND icarm's OWN MODELS
## =========================================================================
## PROBLEM: does the "model-agnostic" claim actually hold, or does DALEX
## secretly only work on models icarm fit?
## METHOD: train a gradient-boosted model the standard way (caret), and
## hand it to the exact same explanation machinery.

gbm_caret <- caret::train(pass ~ ., train_bin, method = "gbm", verbose = FALSE,
                          trControl = ctrl, metric = "ROC")
pred_gbm <- predict(gbm_caret, test_bin)

## RESULT
cat("GBM test accuracy:", round(mean(pred_gbm == test_bin$pass), 3), "\n")
table(Predicted = pred_gbm, Actual = test_bin$pass)

explainer_gbm <- DALEX::explain(
  gbm_caret, data = test_bin %>% select(-pass), y = test_bin$pass == "Pass",
  predict_function = function(m, d) predict(m, d, type = "prob")[, "Pass"],
  label = "GBM", type = "classification", verbose = FALSE
)
DALEX::model_performance(explainer_gbm)

vi_gbm <- DALEX::model_parts(explainer_gbm, B = 15)
plot(vi_rf) + plot(vi_gbm)   # patchwork: side by side, one figure

## INSIGHT: icarm never fit this GBM. Everything above worked anyway.
## icarm's five built-in models are a convenient starting set, not a
## ceiling on what you can explain.


## =========================================================================
## 2.6  A SECOND OPINION
## =========================================================================
## PROBLEM: DALEX's break-down is one explanation method, built on one
## particular piece of mathematics. What if it's misleading us?
## METHOD: LIME explains a prediction by fitting a small, simple model in
## the immediate neighbourhood of that one point -- a completely different
## approach from decomposing the original model.

## LIME ships built-in support for `ranger` random forests but not the
## base `randomForest` package -- two small S3 methods teach it how:
model_type.randomForest <- function(x, ...) "classification"
predict_model.randomForest <- function(x, newdata, type, ...) {
  as.data.frame(predict(x, newdata, type = "prob"))
}

lime_explainer <- lime::lime(train_bin %>% select(-pass), m_rf$fit)
lime_explanation <- lime::explain(
  test_bin[case_id, ] %>% select(-pass), lime_explainer,
  labels = "Pass", n_features = 6, n_permutations = 500
)

## RESULT
lime_explanation %>% select(feature, feature_value, feature_weight)
plot_features(lime_explanation)

## INSIGHT: compare this plot against the DALEX break-down from Section
## 2.4, for the SAME student. List the top three variables each method
## found:
##
## DALEX top 3: ___________________________________________________
## LIME top 3:  ___________________________________________________
##
## Do they agree? Where they disagree, which would you trust more, and why?


## =========================================================================
## 2.7  OPEN EXPLORATION
## =========================================================================
## An interactive dashboard combining break-down, importance, and partial
## dependence for MANY students at once. Uncomment and run this yourself
## in RStudio -- it opens a real browser window, so it needs an interactive
## session (it will not run inside a headless/batch script). Take five
## minutes with it once it opens: pick a student, see if you can find
## something that surprises you.

# ms <- modelStudio::modelStudio(explainer_rf, new_observation = test_bin[1:6, ] %>% select(-pass))
# ms   # opens in your default browser


## =========================================================================
## SYNTHESIS
## =========================================================================

cat("\n=========================================================================\n")
cat("Session 2 complete. Three explanation methods, five models, one\n")
cat("real case where the model's most confident prediction was wrong.\n")
cat("Carry train_bin, test_bin, m_rf, and explainer_rf into Session 3.\n")
cat("=========================================================================\n")
