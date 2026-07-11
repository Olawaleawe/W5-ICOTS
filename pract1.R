## =========================================================================
## MODULE 1 PRACTICE SESSION -- A First Tour of the Toolkit
## ICOTS-12 Pre-Conference Workshop, Brisbane
##
## Purpose: before the clock starts on Session 1 (Question & Understand),
## this is a guided, hands-on tour of EVERY exploratory and plotting
## function in icarm AND its civic-education predecessor, civic.icarm ---
## run on real data, so you have a working feel for the whole toolkit
## before we go deep on any one part of it.
##
## Three parts:
##   A. Real exploratory data analysis (ggplot2 / GGally / plotly) on the
##      Mathematics and Portuguese student datasets
##   B. Every icarm exploratory/plotting function, on the student data
##   C. Every civic.icarm exploratory/plotting function, on civic.icarm's
##      own built-in civic-education dataset (civic_voting)
##
## A note on honesty: icarm and civic.icarm are MODELLING and EXPLANATION
## packages -- neither ships a dedicated histogram/boxplot function. Part A
## below is genuine EDA using standard R plotting packages; Parts B and C
## are the two packages' own native "explore the model" functions. Being
## clear about that distinction is itself a small critical-data-literacy
## lesson: know which tool is doing which job.
## =========================================================================

# install.packages(c("icarm", "civic.icarm", "dplyr", "ggplot2", "GGally", "plotly"))

suppressMessages({
  library(icarm)
  library(civic.icarm)
  library(dplyr)
  library(ggplot2)
  library(GGally)
  library(plotly)
})

## dir.create("figures", showWarnings = FALSE)
## dir.create("outputs", showWarnings = FALSE)


## =========================================================================
## PART A -- REAL EXPLORATORY DATA ANALYSIS (ggplot2 / GGally / plotly)
## =========================================================================
## This is genuine EDA, using general-purpose R plotting tools -- NOT
## icarm functions. It comes first because you always look at data before
## you model it.
getwd()
math <- read.csv2("student-mat.csv", stringsAsFactors = TRUE) %>%
  mutate(pass = factor(ifelse(G3 >= 10, "Pass", "Fail")), subject = "Mathematics")
por  <- read.csv2("student-por.csv", stringsAsFactors = TRUE) %>%
  mutate(pass = factor(ifelse(G3 >= 10, "Pass", "Fail")), subject = "Portuguese")

cat("Mathematics:", nrow(math), "students | mean G3 =", round(mean(math$G3), 2), "\n")
cat("Portuguese: ", nrow(por),  "students | mean G3 =", round(mean(por$G3), 2), "\n")

## A.1 -- Distributions, side by side
both <- bind_rows(math, por)
p_dist <- ggplot(both, aes(x = G3, fill = pass)) +
  geom_histogram(binwidth = 1, color = "white") +
  facet_wrap(~subject) +
  scale_fill_manual(values = c(Fail = "#8C2F26", Pass = "#2E7D32")) +
  labs(title = "Final Grade Distribution, Both Subjects", x = "G3 (0-20)", y = "Students") +
  theme_minimal(base_size = 13)
p_dist
ggsave("figures/a1_grade_distributions.png", p_dist, width = 10, height = 5, dpi = 150)

## A.2 -- Group comparisons (patchwork combines three plots into one figure)
library(patchwork)
p_sex   <- ggplot(math, aes(sex, G3, fill = sex)) + geom_boxplot() +
  theme_minimal() + theme(legend.position = "none") + labs(title = "By Sex")
p_study <- ggplot(math, aes(factor(studytime), G3)) + geom_boxplot(fill = "#9C7A2E") +
  theme_minimal() + labs(title = "By Study Time", x = "studytime")
p_fail  <- ggplot(math, aes(factor(failures), G3)) + geom_boxplot(fill = "#8C2F26") +
  theme_minimal() + labs(title = "By Past Failures", x = "failures")
p_sex + p_study + p_fail
ggsave("figures/a2_group_comparisons.png", width = 12, height = 4.5, dpi = 150)

## A.3 -- Correlation with the outcome
numeric_vars <- math %>% select(where(is.numeric))
sort(cor(numeric_vars, use = "pairwise.complete.obs")["G3", ], decreasing = TRUE)

## A.4 -- Pairwise relationships
GGally::ggpairs(math %>% select(G3, studytime, failures, absences, goout))

## A.5 -- An interactive plot: hover to see who each point is
p_base <- ggplot(math, aes(studytime, G3, color = pass,
                           text = paste0("age: ", age, "  failures: ", failures))) +
  geom_jitter(width = 0.15, alpha = 0.7) +
  theme_minimal() + labs(title = "Study Time vs. Grade (Mathematics)")
plotly::ggplotly(p_base, tooltip = "text")

cat("\nPart A complete: five real EDA views, using ggplot2/GGally/plotly.\n")
cat("None of the functions above are icarm or civic.icarm functions --\n")
cat("that distinction matters, and Parts B and C show you why.\n")
## What do ou notice from the plots? Can you try more EDA or plots? What would be interesting to look at?   Discuss

## =========================================================================
## PART B -- EVERY icarm EXPLORATORY / PLOTTING FUNCTION (student data)
## =========================================================================
## icarm supports BOTH interpretable models (cart, logistic, logistic_l1)
## AND extended "black-box" models (random_forest, svm) -- that's the
## first thing to notice, and it's different from Part C.

sp <- icarm_split(math %>% select(-subject), prop = 0.75, seed = 2025, stratify = "pass")
train_bin <- sp$train %>% select(-G1, -G2, -G3)
test_bin  <- sp$test  %>% select(-G1, -G2, -G3)

## B.1 -- icarm_fit(): one interface, five model types
m_cart <- icarm_fit(pass ~ ., train_bin, model = "cart",          positive = "Pass")
m_rf   <- icarm_fit(pass ~ ., train_bin, model = "random_forest", positive = "Pass")
print(m_cart)             # print.icarm_model()
summary(m_cart)            # summary.icarm_model()

## B.2 -- icarm_metrics() + icarm_compare() + icarm_plot_comparison()
pred_cart <- predict(m_cart, test_bin, type = "class")   # predict.icarm_model()
icarm_metrics(test_bin$pass, pred_cart, positive = "Pass")

cmp <- icarm_compare(list(CART = m_cart, RandomForest = m_rf),
                     test_data = test_bin, outcome = "pass", positive = "Pass")
print(cmp)
icarm_plot_comparison(cmp, metrics = c("accuracy", "f1"))

## B.3 -- icarm_plot_confusion()
icarm_plot_confusion(test_bin$pass, pred_cart, title = "CART: Confusion Matrix")

## B.4 -- icarm_explain() + icarm_plot_importance() + icarm_explain_local()
ex_rf <- icarm_explain(m_rf, data = test_bin, label = "Random Forest")
ex_rf$importance %>% arrange(desc(importance)) %>% head(8)
icarm_plot_importance(ex_rf, n_features = 12)

case_id <- which(predict(m_rf, test_bin, type = "class") == "Fail")[1]
icarm_explain_local(ex_rf, newdata = test_bin[case_id, ], n_features = 6)

## B.5 -- icarm_fairness() + icarm_plot_fairness() + icarm_equity_summary()
fair_sex <- icarm_fairness(m_rf, test_bin, outcome = "pass", protected = "sex", positive = "Pass")
print(fair_sex)
icarm_plot_fairness(fair_sex, metric = "acc")
icarm_equity_summary(fair_sex)

## B.6 -- icarm_equalized_odds_curve() + icarm_plot_roc_groups()
eoc <- icarm_equalized_odds_curve(m_rf, test_bin, outcome = "pass", protected = "sex", positive = "Pass")
icarm_plot_roc_groups(eoc)

## B.7 -- icarm_calibrate() + icarm_plot_calibration()
probs <- predict(m_rf, test_bin, type = "prob")[, "Pass"]
cal <- icarm_calibrate(m_rf, test_bin, outcome = "pass", positive = "Pass")
print(cal)
icarm_plot_calibration(cal)

## B.8 -- icarm_thresholds() + icarm_plot_thresholds()
thr <- icarm_thresholds(test_bin$pass, probs, positive = "Pass")
icarm_plot_thresholds(thr)

## B.9 -- icarm_audit() + icarm_scorecard()
met_full <- icarm_metrics(test_bin$pass, pred_cart, y_prob = probs, positive = "Pass")
icarm_audit(m_rf, metrics = met_full, fairness = fair_sex, analyst = "Your Name"
            )
icarm_scorecard(m_rf, test_bin, outcome = "pass", protected = "sex", positive = "Pass",
                analyst = "Your Name", project = "Module 1 Practice"
                )

cat("\nPart B complete: every icarm exploratory/plotting function has now\n")
cat("run on real Mathematics data.\n")


## =========================================================================
## PART C -- EVERY civic.icarm EXPLORATORY / PLOTTING FUNCTION
## (civic.icarm's own built-in civic-education dataset: civic_voting)
## =========================================================================
## civic.icarm was built FIRST, specifically for civic and political
## education. Its own built-in dataset -- 1,000 simulated respondents on
## voter turnout -- is exactly the kind of civic data it was designed for.
##
## The key structural difference from icarm, worth noticing as you go:
## civic_fit() only supports INTERPRETABLE models -- "cart", "logistic",
## "logistic_l1". There is no random-forest or SVM option. That is a
## deliberate design choice, not a missing feature: a package built for
## civic and political education treats transparency as non-negotiable.

data(civic_voting)
cat(sprintf("\ncivic_voting: %d respondents, %d variables\n", nrow(civic_voting), ncol(civic_voting)))
table(civic_voting$voted)

sp_civic <- civic_split(civic_voting, prop = 0.75, seed = 2025, stratify = "voted")
train_civic <- sp_civic$train
test_civic  <- sp_civic$test

## C.1 -- civic_fit(): interpretable models only, by design
m_civic_cart  <- civic_fit(voted ~ ., train_civic, model = "cart",        positive = "yes")
m_civic_logit <- civic_fit(voted ~ ., train_civic, model = "logistic",    positive = "yes")
m_civic_l1    <- civic_fit(voted ~ ., train_civic, model = "logistic_l1", positive = "yes")
print(m_civic_cart)
summary(m_civic_cart)

## C.2 -- civic_metrics() + civic_compare() + civic_plot_comparison()
pred_civic <- predict(m_civic_cart, test_civic, type = "class")
civic_metrics(test_civic$voted, pred_civic, positive = "yes")

cmp_civic <- civic_compare(
  list(CART = m_civic_cart, Logistic = m_civic_logit, `Logistic (L1)` = m_civic_l1),
  test_data = test_civic, outcome = "voted", positive = "yes")
print(cmp_civic)
civic_plot_comparison(cmp_civic)

## C.3 -- civic_plot_confusion()
civic_plot_confusion(test_civic$voted, pred_civic, title = "Voting Model: Confusion Matrix")

## C.4 -- civic_explain() + civic_plot_importance() + civic_explain_local()
ex_civic <- civic_explain(m_civic_cart, data = test_civic)
ex_civic$importance %>% arrange(desc(importance)) %>% head(5)
civic_plot_importance(ex_civic)

case_civic <- which(pred_civic == "no")[1]
civic_explain_local(ex_civic, newdata = test_civic[case_civic, ], n_features = 5)

## C.5 -- civic_fairness() + civic_plot_fairness() + civic_equity_summary()
fair_civic <- civic_fairness(m_civic_cart, test_civic, outcome = "voted",
                             protected = "education", positive = "yes")
print(fair_civic)
civic_plot_fairness(fair_civic, metric = "acc")
civic_equity_summary(fair_civic)

## C.6 -- civic_equalized_odds_curve() + civic_plot_roc_groups()
eoc_civic <- civic_equalized_odds_curve(m_civic_cart, test_civic, outcome = "voted",
                                        protected = "education", positive = "yes")
civic_plot_roc_groups(eoc_civic)

## C.7 -- civic_calibrate() + civic_plot_calibration()
probs_civic <- predict(m_civic_cart, test_civic, type = "prob")[, "yes"]
cal_civic <- civic_calibrate(m_civic_cart, test_civic, outcome = "voted", positive = "yes")
print(cal_civic)
civic_plot_calibration(cal_civic)

## C.8 -- civic_thresholds() + civic_plot_thresholds()
thr_civic <- civic_thresholds(test_civic$voted, probs_civic, positive = "yes")
civic_plot_thresholds(thr_civic)

## C.9 -- civic_audit() + civic_scorecard()
met_civic_full <- civic_metrics(test_civic$voted, pred_civic, y_prob = probs_civic, positive = "yes")
civic_audit(m_civic_cart, metrics = met_civic_full, fairness = fair_civic, analyst = "Your Name")
civic_scorecard(m_civic_cart, test_civic, outcome = "voted", protected = "education", positive = "yes",
                analyst = "Your Name", project = "Module 1 Practice",
          )

