setwd(normalizePath(dirname(rstudioapi::getActiveDocumentContext()$path)))

library(tidyverse)
library(brms)
library(tidyverse)
library(hypr)
library(MASS)
library(bayestestR)

# Experiment 1: Emergence----------------------

#epochs <- read.csv('exp1/epochs_training.csv')
accs <- read.csv('exp1/accuracies.csv')
ml <- read.csv('exp1/message_length.csv')
entr <- read.csv('exp1/entropies.csv')
entr_hier <- read.csv('exp1/entropies_hierarchical.csv')

## 1) Accuracy-------------------
accs$condition <- factor(accs$condition, levels = c("context_unaware", "context_aware"))
accs_val <- accs %>% filter(acc_type == 'val')
contrasts(accs_val$condition) <- contr.sum(2)
fit_acc_val <- brm(accuracy ~ condition + (1|dataset),
               data=accs_val,
               prior =
                 c(prior(uniform(0, 1), class = Intercept, lb = 0, ub = 1),
                   prior(uniform(0, 0.1), class = sigma, lb = 0, ub = 0.1)),
               # increasing adapt_delta because of few divergent transitions
               # after warmup
               control = list(adapt_delta = 0.99),
               file = "exp1/fit_acc"
               )
summary(fit_acc_val)
describe_posterior(fit_acc_val, rope_range = rope_range(fit_acc_val))
# main effect condition: context-aware > context-unaware

## 2) Message length--------------
ml$condition <- factor(ml$condition, levels = c("context_unaware", "context_aware"))
contrasts(ml$condition) <- contr.sum(2)
fit_ml <- brm(ml ~ condition * fixed + (1|dataset),
              data=ml,
              prior =
                c(prior(uniform(0, 20), class = Intercept, lb = 0, ub = 20),
                  prior(uniform(0, 5), class = sigma, lb = 0, ub = 5)),
              # increasing adapt_delta because of few divergent transitions
              # after warmup
              control = list(adapt_delta = 0.9),
              file = "exp1/fit_ml2"
              )
summary(fit_ml)
describe_posterior(fit_ml, rope_range = rope_range(fit_ml))
# main effect condition: context-aware < context-unaware
# probable main effect for fixed
# interaction condition:fixed

## 3) Entropy---------------------
### Means--------------------
#### NMI-----------------
entr$condition <- factor(entr$condition, levels = c("context_unaware", "context_aware"))
contrasts(entr$condition) <- contr.sum(2)
NMI <- entr %>% filter(score == 'NMI')
fit_NMI <- brm(entropy ~ condition + (1|dataset),
               data=NMI,
               prior =
                 c(prior(uniform(0, 1), class = Intercept, lb = 0, ub = 1),
                   prior(uniform(0, 0.1), class = sigma, lb = 0, ub = 0.1)),
               # increasing adapt_delta because of few divergent transitions
               # after warmup
               control = list(adapt_delta = 0.99),
               file = "exp1/fit_NMI")
summary(fit_NMI)
describe_posterior(fit_NMI, rope_range = rope_range(fit_NMI))
# main effect condition: context-aware < context-unaware

#### effectiveness-----------------
eff <- entr %>% filter(score == 'effectiveness')
fit_eff <- brm(entropy ~ condition + (1|dataset),
               data=eff,
               prior =
                 c(prior(uniform(0, 1), class = Intercept, lb = 0, ub = 1),
                   prior(uniform(0, 0.1), class = sigma, lb = 0, ub = 0.1)),
               # increasing adapt_delta because of few divergent transitions
               # after warmup
               control = list(adapt_delta = 0.99),
               file = "exp1/fit_eff"
               )
summary(fit_eff)
describe_posterior(fit_eff, rope_range = rope_range(fit_eff))
# main effect condition: context-aware < context-unaware

#### consistency-----------------
cons <- entr %>% filter(score == 'consistency')
fit_cons <- brm(entropy ~ condition + (1|dataset),
                data=cons,
                prior =
                  c(prior(uniform(0, 1), class = Intercept, lb = 0, ub = 1),
                    prior(uniform(0, 0.1), class = sigma, lb = 0, ub = 0.1)),
                # increasing adapt_delta because of few divergent transitions
                # after warmup
                control = list(adapt_delta = 0.99),
                file = "exp1/fit_cons"
                )
summary(fit_cons)
describe_posterior(fit_cons, rope_range = rope_range(fit_cons))
# main effect condition: context-aware < context-unaware

### hierarchical------------

#### NMI-----------------
entr_hier$condition <- factor(entr_hier$condition, levels = c("context_unaware", "context_aware"))
contrasts(entr_hier$condition) <- contr.sum(2)
NMI_hier <- entr_hier %>% filter(score == 'NMI_hierarchical')
fit_NMI_hier <- brm(entropy ~ condition * fixed + (1|dataset),
                    data=NMI_hier,
                    prior =
                      c(prior(uniform(0, 1), class = Intercept, lb = 0, ub = 1),
                        prior(uniform(0, 0.1), class = sigma, lb = 0, ub = 0.1)),
                    # increasing adapt_delta because of few divergent transitions
                    # after warmup
                    control = list(adapt_delta = 0.99),
                    file = "exp1/fit_NMI_hier"
                    )
summary(fit_NMI_hier)
describe_posterior(fit_NMI_hier, rope_range = rope_range(fit_NMI_hier))
# probably main effect condition: context-unaware > context-unaware
# main effect fixed (negative slope)
# interaction condition:fixed

#### effectiveness-----------------
eff_hier <- entr_hier %>% filter(score == 'effectiveness_hierarchical')
fit_eff_hier <- brm(entropy ~ condition * fixed + (1|dataset),
                    data=eff_hier,
                    prior =
                      c(prior(uniform(0, 1), class = Intercept, lb = 0, ub = 1),
                        prior(uniform(0, 0.1), class = sigma, lb = 0, ub = 0.1)),
                    # increasing adapt_delta because of few divergent transitions
                    # after warmup
                    control = list(adapt_delta = 0.99),
                    file = "exp1/fit_eff_hier"
                    )
summary(fit_eff_hier)
describe_posterior(fit_eff_hier, rope_range = rope_range(fit_eff_hier))
# main effect condition: context-aware < context-unaware
# main effect fixed (negative slope)
# interaction condition:fixed

#### consistency-----------------
cons_hier <- entr_hier %>% filter(score == 'consistency_hierarchical')
fit_cons_hier <- brm(entropy ~ condition * fixed + (1|dataset),
                     data=cons_hier,
                     prior =
                       c(prior(uniform(0, 1), class = Intercept, lb = 0, ub = 1),
                         prior(uniform(0, 0.1), class = sigma, lb = 0, ub = 0.1)),
                     # increasing adapt_delta because of few divergent transitions
                     # after warmup
                     control = list(adapt_delta = 0.99),
                     file = "exp1/fit_cons_hier"
                     )
summary(fit_cons_hier)
describe_posterior(fit_cons_hier, rope_range = rope_range(fit_cons_hier))
# main effect condition: context-aware > context-unaware
# probable main effect fixed (positive)
# interaction condition:fixed

# Experiment 2: Inference----------------------

accs2 <- read.csv('exp2/accuracies.csv')
ml2 <- read.csv('exp2/message_length.csv')
lex_props <- read.csv('exp2/lexicon_props.csv')

## 1) Test accuracy---------------------
accs2$condition <- factor(accs2$condition, levels = c("context_unaware", "context_aware"))
accs2$rsa <- factor(accs2$rsa, levels = c("test", "rsa"))
contrasts(accs2$condition) <- contr.sum(2)
contrasts(accs2$rsa) <- contr.sum(2)
fit_acc2 <- brm(accuracy ~ condition * rsa + (1|dataset),
               data=accs2,
               prior =
                 c(prior(uniform(0, 1), class = Intercept, lb = 0, ub = 1),
                   prior(uniform(0, 0.1), class = sigma, lb = 0, ub = 0.1)),
               # increasing adapt_delta because of few divergent transitions
               # after warmup
               control = list(adapt_delta = 0.99),
               file = "exp2/fit_acc")
summary(fit_acc2)
describe_posterior(fit_acc2, rope_range = rope_range(fit_acc2))
# main effect rsa: rsa < test

## 2) Message length--------------
ml2$condition <- factor(ml2$condition, levels = c("context_unaware", "context_aware"))
ml2$rsa <- factor(ml2$rsa, levels = c("test", "rsa"))
contrasts(ml2$condition) <- contr.sum(2)
contrasts(ml2$rsa) <- contr.sum(2)
fit_ml2 <- brm(ml ~ condition * rsa * fixed + (1|dataset),
              data=ml2,
              prior =
                c(prior(uniform(0, 20), class = Intercept, lb = 0, ub = 20),
                  prior(uniform(0, 5), class = sigma, lb = 0, ub = 5)),
              file = "exp2/fit_ml"
              )
summary(fit_ml2)
describe_posterior(fit_ml2, rope_range = rope_range(fit_ml2))
# main effect condition: context-aware < context-unaware
# main effect rsa: rsa < test
# interaction condition:rsa

## 3) Lexicon size
lex_props$condition <- factor(lex_props$condition, levels = c("context_unaware", "context_aware"))
lex_props$rsa <- factor(lex_props$rsa, levels = c("test", "rsa"))
contrasts(lex_props$condition) <- contr.sum(2)
contrasts(lex_props$rsa) <- contr.sum(2)
fit_lex_size <- brm(lexsize ~ condition * rsa + (1|dataset),
                    data=lex_props,
                    prior =
                      c(prior(uniform(0, 1460), class = Intercept, lb = 0, ub = 1460),
                        prior(uniform(0, 100), class = sigma, lb = 0, ub = 100)),
                    # increasing adapt_delta because of few divergent transitions
                    # after warmup
                    control = list(adapt_delta = 0.99),
                    file = "exp2/fit_lexsize")
summary(fit_lex_size)
describe_posterior(fit_lex_size, rope_range = rope_range(fit_lex_size))
# main effect rsa: rsa < test
# interaction condition:rsa

## 4) Lexicon informativeness
fit_lex_info <- brm(lexinfo ~ condition * rsa + (1|dataset),
                    data=lex_props,
                    prior =
                      c(prior(uniform(0, 9), class = Intercept, lb = 0, ub = 9),
                        prior(uniform(0, 2), class = sigma, lb = 0, ub = 2)),
                    # increasing adapt_delta because of few divergent transitions
                    # after warmup
                    control = list(adapt_delta = 0.99),
                    file = "exp2/fit_lexinfo"
                    )
summary(fit_lex_info)
describe_posterior(fit_lex_info, rope_range = rope_range(fit_lex_info))
# main effect condition context-aware < context-unaware
# main effect rsa: rsa < test
# interaction condition:rsa
