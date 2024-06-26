#####  LIMPANDO OBJETOS SALVOS  #####

rm(list=ls(all=TRUE))


#####  PACOTES  #####

library(tidyverse)
library(tidymodels)
library(stacks)
library(plsmod)
library(kknn)
library(baguette)
library(themis)
library(timetk)
library(quantreg)
library(tictoc)



#####  CARREGAR OS DADOS  #####

df.train<- read.csv("dens_api_treino.csv",head=T)
df.test<- read.csv("dens_api_teste.csv",head=T)



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



##### CRIANDO MODELO QUANTÍLICO #####


set_new_model("quant_reg")
set_model_mode(model = "quant_reg",
               mode = "regression")
set_model_engine("quant_reg",
                 mode = "regression",
                 eng = "quantreg")
set_dependency("quant_reg", eng = "quantreg", pkg = "quantreg")

show_model_info("quant_reg")

quant_reg <- function(mode = "regression",  sub_classes = NULL) {
  # Check for correct mode
  if (mode  != "regression") {
    rlang::abort("`mode` should be 'regression'")
  }
  
  # Capture the arguments in quosures
  args <- list(sub_classes = rlang::enquo(sub_classes))
  
  # Save some empty slots for future parts of the specification
  new_model_spec(
    "quant_reg",
    args = args,
    eng_args = NULL,
    mode = mode,
    method = NULL,
    engine = NULL
  )
}

set_fit(
  model = "quant_reg",
  eng = "quantreg",
  mode = "regression",
  value = list(
    interface = "formula",
    protect = c("formula", "data"),
    func = c(pkg = "quantreg", fun = "rq"),
    defaults = list()
  )
)

show_model_info("quant_reg")

set_encoding(
  model = "quant_reg",
  eng = "quantreg",
  mode = "regression",
  options = list(
    predictor_indicators = "traditional",
    compute_intercept = TRUE,
    remove_intercept = TRUE,
    allow_sparse_x = FALSE
  )
)

class_info <- 
  list(
    pre = NULL,
    post = NULL,
    func = c(fun = "predict"),
    args =
      # These lists should be of the form:
      # {predict.mda argument name} = {values provided from parsnip objects}
      list(
        # We don't want the first two arguments evaluated right now
        # since they don't exist yet. `type` is a simple object that
        # doesn't need to have its evaluation deferred. 
        object = quote(object$fit),
        newdata = quote(new_data),
        type = "numeric"
      )
  )

set_pred(
  model = "quant_reg",
  eng = "quantreg",
  mode = "regression",
  type = "numeric",
  #value = class_info
  value = list(
    pre = NULL,
    post = NULL,
    func = c(fun = "predict.rq"),
    args =
      list(
        object = expr(object$fit),
        newdata = expr(new_data),
        type = "none")
))

show_model_info("quant_reg")




##### MODELOS #####

model_knn<- nearest_neighbor(neighbors = tune()) %>%
  set_engine("kknn") %>%
  set_mode("regression")

model_qr<- quant_reg() %>%
  set_engine("quantreg") %>%
  set_mode("regression")

model_net<- linear_reg(penalty = tune(), mixture = tune()) %>%
  set_engine("glmnet") %>%
  set_mode("regression")






##### WORKFLOW #####

wf_knn<- workflow() %>%
  add_recipe(recipe_pls) %>%
  add_model(model_knn)

wf_qr<- workflow() %>%
  add_recipe(recipe_pls) %>%
  add_model(model_qr)

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
                      param_info = parameters(num_comp(range(1,100)),
                                              neighbors(range=c(1,30)))
)
toc()
# 315.97 sec elapsed (~ 5 min)


tic()
tune_qr<- tune_bayes(wf_qr,
                     resamples = folds,
                     initial = 10,
                     #control = control_stack_bayes(),
                     control = control_bayes(save_pred=TRUE,
                                             save_workflow=TRUE,
                                             seed=0),
                     metrics = metric_set(rmse),
                     param_info = parameters(num_comp(range=c(1,100)))
)
toc()
# 327 sec elapsed (~ 5.5 min)


tic()
tune_net1<- tune_bayes(wf_net1,
                       resamples = folds,
                       initial = 10,
                       #control = control_stack_bayes(),
                       control = control_bayes(save_pred=TRUE,
                                               save_workflow=TRUE,
                                               seed=0),
                       metrics = metric_set(rmse),
                       param_info = parameters(num_comp(range=c(1,100)),
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
show_best(tune_qr,n=3)
show_best(tune_net1,n=3)
show_best(tune_net2,n=3)



##### STACKING ENSEMBLE  #####

stack_ensemble_data<- stacks() %>% 
  add_candidates(tune_knn) %>% 
  add_candidates(tune_qr) %>% 
  add_candidates(tune_net1) %>% 
  add_candidates(tune_net2) 


stack_ensemble_data


set.seed(0)
stack_ensemble_model<- stack_ensemble_data %>% 
  blend_predictions(penalty = 10^(-9:-1),
                    mixture = 0, # 0=RIDGE; 1=LASSO
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
wf_train_qr<- wf_qr %>% finalize_workflow(select_best(tune_qr)) %>% fit(df.train)
wf_train_net1<- wf_net1 %>% finalize_workflow(select_best(tune_net1)) %>% fit(df.train)
wf_train_net2<- wf_net2 %>% finalize_workflow(select_best(tune_net2)) %>% fit(df.train)




### VALIDATION ###

# PREDIZENDO DADOS TESTE

pred.knn<- predict(wf_train_knn, df.test)
#pred.qr<- predict(wf_train_qr, df.test)
pred.net1<- predict(wf_train_net1, df.test)
pred.net2<- predict(wf_train_net2, df.test)



predicao<- data.frame(df.test$y,
                      pred.knn,
                      #pred.qr,
                      pred.net1,
                      pred.net2)#,
                      #pred.stc)
colnames(predicao)<- c("y",
                       "knn",
                       #"qr",
                       "net1",
                       "net2")#,
                       #"stc")
head(predicao)

RMSE<- cbind(rmse(predicao,y,knn)$.estimate,
             #rmse(predicao,y,qr)$.estimate,
             rmse(predicao,y,net1)$.estimate,
             rmse(predicao,y,net2)$.estimate)#,
             #rmse(predicao,y,stc)$.estimate)

colnames(RMSE)<- c("knn",
                   #"qr",
                   "net1",
                   "net2")#,
                   #"stc")
RMSE


