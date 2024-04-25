#####  LIMPANDO OBJETOS SALVOS  #####

rm(list=ls(all=TRUE))


#####  PACOTES  #####

library(tidyverse)
library(tidymodels)
library(stacks)
library(plotly)
library(plsmod)
library(pls)
library(kknn)
library(baguette)
library(themis)
library(timetk)
library(tictoc)



#####  CARREGAR OS DADOS  #####

df.train<- read.csv("dens_api_treino.csv",head=T)
df.test<- read.csv("dens_api_teste.csv",head=T)


# Standard Normal Variate (SNV)

df.train.snv<- df.train %>% 
  rowwise() %>% 
  mutate(m=mean())


##### SPLIT #####

set.seed(0)
#split<- initial_split(df.train, strata=y)

#df.train<- training(split)
#df.test<- testing(split)

folds<- vfold_cv(df.train, v=3, strata=y)



##### PRÉ-PROCESSAMENTO #####

recipe_pls<- recipe(y ~ . , data = df.train) %>%
  #step_filter_missing(all_predictors(),threshold = 0.4) %>% 
  #step_novel(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors()) %>% 
  #step_impute_knn(all_predictors()) %>%
  #step_dummy(all_nominal_predictors()) %>% 
  #step_zv(all_predictors()) %>% 
  step_pls(all_predictors(), outcome = "y", num_comp = tune())

recipe_norm<- recipe(y ~ . , data = df.train) %>%
  #step_filter_missing(all_predictors(),threshold = 0.4) %>% 
  #step_novel(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors()) #%>% 
#step_impute_knn(all_predictors()) %>%
#step_dummy(all_nominal_predictors()) %>% 
#step_zv(all_predictors())






##### MODELOS #####

model_knn<- nearest_neighbor(neighbors = tune()) %>%
  set_engine("kknn") %>%
  set_mode("regression")

model_net<- linear_reg(penalty = tune(), mixture = tune()) %>%
  set_engine("glmnet") %>%
  set_mode("regression")






##### WORKFLOW #####

wf_knn<- workflow() %>%
  add_recipe(recipe_pls) %>%
  add_model(model_knn)

wf_net1<- workflow() %>%
  add_recipe(recipe_pls) %>%
  add_model(model_net)

wf_net2<- workflow() %>%
  add_recipe(recipe_norm) %>%
  add_model(model_net)




##### HIPERPARAMETERS TUNING - BAYESIAN SEARCH #####

tic()
tune_knn<- tune_bayes(wf_knn,
                      resamples = folds,
                      initial = 10,
                      #control = control_stack_bayes(),
                      control = control_bayes(save_pred=TRUE,
                                              save_workflow=TRUE,
                                              seed=0),
                      metrics = metric_set(rmse),
                      param_info = parameters(num_comp(range(1,30)),
                                              neighbors(range=c(1,30)))
)
toc()
# 315.97 sec elapsed (~ 5 min)


tic()
tune_net1<- tune_bayes(wf_net1,
                       resamples = folds,
                       initial = 10,
                       #control = control_stack_bayes(),
                       control = control_bayes(save_pred=TRUE,
                                               save_workflow=TRUE,
                                               seed=0),
                       metrics = metric_set(rmse),
                       param_info = parameters(num_comp(range=c(1,30)),
                                               penalty(range=c(-10,10)),
                                               mixture(range=c(0,1)))
)
toc()
# 397.43 sec elapsed (~ 6.5 min)


tic()
tune_net2<- tune_bayes(wf_net2,
                       resamples = folds,
                       initial = 10,
                       #control = control_stack_bayes(),
                       control = control_bayes(save_pred=TRUE,
                                               save_workflow=TRUE,
                                               seed=0),
                       metrics = metric_set(rmse),
                       param_info = parameters(penalty(range=c(-10,10)),
                                               mixture(range=c(0,1)))
)
toc()
# 276.72 sec elapsed (~ 4.5 min)



## ESCOLHENDO O MELHOR (BEST roc_auc)

show_best(tune_knn,n=3)
show_best(tune_net1,n=3)
show_best(tune_net2,n=3)



##### STACKING ENSEMBLE  #####

stack_ensemble_data<- stacks() %>% 
  add_candidates(tune_knn) %>% 
  add_candidates(tune_net1) %>% 
  add_candidates(tune_net2) 


stack_ensemble_data


set.seed(0)
stack_ensemble_model<- stack_ensemble_data %>% 
  blend_predictions(penalty = 10^(-9:-1),
                    mixture = 1, # 0=RIDGE; 1=LASSO
                    control = control_grid(),
                    non_negative = TRUE,
                    metric = metric_set(rmse))

autoplot(stack_ensemble_model)
autoplot(stack_ensemble_model,type = "weights")

stack_ensemble_model




##### FINALIZANDO O MODELO #####

stack_ensemble_model<- stack_ensemble_model %>% 
  fit_members()

stack_ensemble_model




##### FINALIZANDO MODELOS INDIVIDUAIS #####

wf_train_knn<- wf_knn %>% finalize_workflow(select_best(tune_knn)) %>% fit(df.train)
wf_train_net1<- wf_net1 %>% finalize_workflow(select_best(tune_net1)) %>% fit(df.train)
wf_train_net2<- wf_net2 %>% finalize_workflow(select_best(tune_net2)) %>% fit(df.train)




### VALIDATION ###

# PREDIZENDO DADOS TESTE

pred.knn<- predict(wf_train_knn, df.test)
pred.net1<- predict(wf_train_net1, df.test)
pred.net2<- predict(wf_train_net2, df.test)
pred.stc<- predict(stack_ensemble_model, df.test)


predicao<- data.frame(df.test$y,
                      pred.knn,
                      pred.net1,
                      pred.net2,
                      pred.stc)

colnames(predicao)<- c("y",
                       "knn",
                       "net1",
                       "net2",
                       "stc")
head(predicao)

RMSE<- cbind(rmse(predicao,y,knn)$.estimate,
             rmse(predicao,y,net1)$.estimate,
             rmse(predicao,y,net2)$.estimate,
             rmse(predicao,y,stc)$.estimate)

colnames(RMSE)<- c("knn",
                   "net1",
                   "net2",
                   "stc")
RMSE


# 
#          knn     net1     net2      stc
#[1,] 2.321754 1.136127 1.410666 1.225641
