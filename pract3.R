## =========================================================================
## SESSION 3 PRACTICE (1:15-3:00)  --  STORY AND TRANSFORM
## ICOTS-12 Pre-Conference Workshop, Brisbane
##
## Mirrors Session3_Story_Transform.pdf exactly, section for section:
##   3.1  Who does the model fail? (fairness: icarm + fairmodels)
##   3.2  Telling the story honestly (the Four C's)
##   3.3  Does this generalise? (Mathematics vs. Portuguese)
##   3.4  The cut-off is a policy choice (thresholds)
##   3.5  The actual deliverable (audit trail)
##   3.6  Your turn (capstone)
##
## This script is SELF-CONTAINED: it does not depend on Sessions 1 or 2
## having been run first. It rebuilds train_bin/test_bin and refits the
## models it needs from the raw data, in the first few lines.
## =========================================================================


## -------------------------------------------------------------------------
## SETUP
## -------------------------------------------------------------------------

required_pkgs <- c("icarm", "dplyr", "ggplot2", "caret", "gbm", "DALEX", "fairmodels")
missing_pkgs <- required_pkgs[!sapply(required_pkgs, requireNamespace, quietly = TRUE)]
if (length(missing_pkgs) > 0) {
  stop("\n\nMissing package(s): ", paste(missing_pkgs, collapse = ", "),
       "\nInstall them first:\n  install.packages(c(",
       paste0('"', missing_pkgs, '"', collapse = ", "), "))\n", call. = FALSE)
}

suppressMessages({
  library(icarm); library(dplyr); library(ggplot2)
  library(caret); library(gbm); library(DALEX); library(fairmodels)
})

if (!file.exists("student-mat.csv")) {
  stop("\n\nCan't find 'data/student-mat.csv' from the current working directory:\n  ",
       getwd(), "\nRun this from the repository root, or open QUEST.Rproj in RStudio.", call. = FALSE)
}
if (!file.exists("data/student-por.csv")) {
  stop("\n\nCan't find 'student-por.csv' -- Section 3.3 needs it. Check you're\n",
       "running from the repository root.", call. = FALSE)
}



## Rebuild everything Session 3 needs, from scratch, same seed as Sessions 1-2.
students <- read.csv2("data/student-mat.csv", stringsAsFactors = TRUE) %>%
  mutate(pass = factor(ifelse(G3 >= 10, "Pass", "Fail")))
sp <- icarm_split(students, prop = 0.75, seed = 2025, stratify = "pass")
train_bin <- sp$train %>% select(-G1, -G2, -G3)
test_bin  <- sp$test  %>% select(-G1, -G2, -G3)

m_cart <- icarm_fit(pass ~ ., train_bin, model = "cart",          positive = "Pass")
m_rf   <- icarm_fit(pass ~ ., train_bin, model = "random_forest", positive = "Pass")

ctrl <- caret::trainControl(method = "cv", number = 5,
                            classProbs = TRUE, summaryFunction = twoClassSummary)
gbm_caret <- caret::train(pass ~ ., train_bin, method = "gbm", verbose = FALSE,
                          trControl = ctrl, metric = "ROC")

explainer_rf <- DALEX::explain(
  m_rf$fit, data = test_bin %>% select(-pass), y = test_bin$pass == "Pass",
  predict_function = function(m, d) predict(m, d, type = "prob")[, "Pass"],
  label = "Random Forest", type = "classification", verbose = FALSE)
explainer_gbm <- DALEX::explain(
  gbm_caret, data = test_bin %>% select(-pass), y = test_bin$pass == "Pass",
  predict_function = function(m, d) predict(m, d, type = "prob")[, "Pass"],
  label = "GBM", type = "classification", verbose = FALSE)

cat("Ready: models fitted, explainers built. Starting Session 3.\n")


## =========================================================================
## 3.1  WHO DOES THE MODEL FAIL?
## =========================================================================
## PROBLEM: Session 2 asked whether we could trust the model's reasoning.
## It didn't ask whether that reasoning treats everyone the same way.
## METHOD: a group fairness audit, split by sex.

fair_sex <- icarm_fairness(m_rf, test_bin, outcome = "pass", protected = "sex", positive = "Pass")

## RESULT
print(fair_sex %>% select(grp, n, acc, tnr))
icarm_plot_fairness(fair_sex, metric = "acc")

## INSIGHT: if this triggered a real support intervention, how many of the
## female students who actually needed it would be missed? Write the
## fraction here (as a sentence, not just a number):
##
## ___________________________________________________________________

## A second, independent view: compare TWO models on identical metrics.
fobject <- fairmodels::fairness_check(explainer_rf, explainer_gbm,
                                      protected = test_bin$sex, privileged = "M")
print(fobject)
plot(fobject)

## INSIGHT: which model has the better (lower) total fairness loss? Is
## that the same model that had the better raw accuracy in Session 2?
##
## ___________________________________________________________________


## =========================================================================
## 3.2  TELLING THE STORY HONESTLY
## =========================================================================
## FRAMEWORK: Claim, Context, Case, Caveat.

## Applied to the 3.1 finding above:
##   CLAIM:   The model catches 10% of actually-failing female students,
##            vs. 38.5% of male students.
##   CONTEXT: 9 of 10 female students who need support would never be
##            flagged.
##   CASE:    2 of 20 actually-failing female students caught, vs. 5 of 13
##            male.
##   CAVEAT:  One 100-student test split -- worth confirming on a fresh
##            sample.

## YOUR TURN: using the fairmodels comparison above, write your own Four
## C's for the Random-Forest-vs-GBM fairness difference.
##
## CLAIM:   ____________________________________________________________
## CONTEXT: ____________________________________________________________
## CASE:    ____________________________________________________________
## CAVEAT:  ____________________________________________________________

## Bad / Better / Best -- try rewriting one of your own results this way:
## BAD:    "[Model] produced [X]% accuracy."
## BETTER: "[Plain-language restatement of what the model actually finds]"
## BEST:   "[The same finding, with a concrete real-world implication]"


## =========================================================================
## 3.3  DOES THIS GENERALISE?
## =========================================================================
## PROBLEM: everything so far comes from one course, Mathematics. Does any
## of it hold for a different subject?
## METHOD: the identical pipeline, run on a different file.

run_subject <- function(file, label) {
  df <- read.csv2(file, stringsAsFactors = TRUE)
  df$pass <- factor(ifelse(df$G3 >= 10, "Pass", "Fail"))
  sp <- icarm_split(df, prop = 0.75, seed = 2025, stratify = "pass")
  tr <- sp$train %>% select(-G1, -G2, -G3)
  te <- sp$test  %>% select(-G1, -G2, -G3)
  
  m_cart_s <- icarm_fit(pass ~ ., tr, model = "cart", positive = "Pass")
  m_rf_s   <- icarm_fit(pass ~ ., tr, model = "random_forest", positive = "Pass")
  met      <- icarm_metrics(te$pass, predict(m_cart_s, te, type = "class"), positive = "Pass")
  fair     <- icarm_fairness(m_rf_s, te, outcome = "pass", protected = "sex", positive = "Pass")
  
  cat("\n==", label, "==\n")
  cat("n =", nrow(df), " pass rate =", round(mean(df$pass == "Pass"), 3), "\n")
  cat("Tree's first question:", as.character(m_cart_s$fit$frame$var[1]), "\n")
  cat("Accuracy:", round(met["accuracy"], 3), " Specificity:", round(met["specificity"], 3), "\n")
  print(fair %>% select(grp, n, acc, tnr))
  invisible(list(metrics = met, fairness = fair))
}

## RESULT
res_math <- run_subject("data/student-mat.csv", "Mathematics")
res_por  <- run_subject("data/student-por.csv", "Portuguese")

## INSIGHT: does the tree ask the same first question in both subjects?
## Does the fairness gap look the same, better, or worse in Portuguese?
## What does that tell you about trusting a finding from ONE dataset?
##
## ___________________________________________________________________


## =========================================================================
## 3.4  THE CUT-OFF IS A POLICY CHOICE
## =========================================================================
## PROBLEM: every model so far used the default rule -- predict Pass if
## predicted probability > 0.50. Who decided that number was right?
## METHOD: sweep the threshold and watch every metric move.

probs <- predict(m_rf, test_bin, type = "prob")[, "Pass"]
thr <- icarm_thresholds(test_bin$pass, probs, positive = "Pass")

## RESULT
print(thr, n = 20)
icarm_plot_thresholds(thr, metrics = c("accuracy", "recall", "precision", "f1"))

## INSIGHT: find the threshold with the highest accuracy in the table
## above. Is it 0.50? If not, would you actually recommend changing the
## default -- and if so, what would you need to know first?
##
## ___________________________________________________________________


## =========================================================================
## 3.5  THE ACTUAL DELIVERABLE
## =========================================================================
## PROBLEM: every finding in this workshop lives in a slide or a console.
## Neither is evidence anyone else can check later.
## METHOD: a reproducible, timestamped record.

metrics_rf <- icarm_metrics(test_bin$pass, predict(m_rf, test_bin, type = "class"),
                            y_prob = probs, positive = "Pass")

icarm_audit(m_rf, metrics = metrics_rf, fairness = fair_sex, analyst = "Your Name",
            notes = "ICOTS-12 workshop case study; not for real deployment.",
            path = "outputs/session3_audit_trail.json")
icarm_scorecard(m_rf, test_bin, outcome = "pass", protected = "sex", positive = "Pass",
                analyst = "Your Name", project = "ICOTS-12 Workshop",
                path = "outputs/session3_scorecard.json")

## RESULT: open outputs/session3_scorecard.json and look at it directly --
## every number in Sections 3.1-3.4 should be traceable back to this file.
cat("\nScorecard written to outputs/session3_scorecard.json -- open it now.\n")


## =========================================================================
## 3.6  YOUR TURN
## =========================================================================
## Pick ONE attribute -- different from sex -- and repeat the fairness
## audit yourself. No two pairs in the room should choose the same one.

ATTR <- "internet"   # change to your assigned attribute:
# Pstatus, internet, schoolsup, or guardian

team_fair <- icarm_fairness(m_rf, test_bin, outcome = "pass", protected = ATTR, positive = "Pass")
print(team_fair)
icarm_plot_fairness(team_fair, metric = "acc", title = paste("Accuracy by", ATTR))

## Prepare a 3-minute summary using this structure:
##   CLAIM:          your exact fairness numbers
##   CONTEXT:        what it means for a real decision
##   CASE:           one specific student from your test set
##   CAVEAT:         a genuine limitation
##   RECOMMENDATION: deploy as-is / with safeguards / not at all


cat("\n=========================================================================\n")
cat("Session 3 complete. Six real findings: a fairness gap, a framework for\n")
cat("telling it honestly, a generalisation test, a policy choice hiding in\n")
cat("a default number, a reproducible record, and your own audited case.\n")
cat("=========================================================================\n")