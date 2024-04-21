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

df.train<- read.csv("dens_api.csv",head=T)
df.test<- read.csv("dens_api_teste.csv",head=T)



##### SPLIT #####

set.seed(0)
#split<- initial_split(df.train, strata=y)

#df.train<- training(split)
#df.test<- testing(split)

folds<- vfold_cv(df.train, v=5, strata=y)



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
  value =   value = list(
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

model_las<- linear_reg(penalty = tune(), mixture = tune()) %>%
  set_engine("glmnet") %>%
  set_mode("regression")

model_rf<- rand_forest(mtry = tune(), trees = tune(), min_n = tune()) %>%
  set_engine("ranger") %>%
  set_mode("regression")





##### WORKFLOW #####

wf_knn<- workflow() %>%
  add_recipe(recipe_pls) %>%
  add_model(model_knn)

wf_qr<- workflow() %>%
  add_recipe(recipe_pls) %>%
  add_model(model_qr)

wf_las<- workflow() %>%
  add_recipe(recipe_norm) %>%
  add_model(model_las)

wf_rf<- workflow() %>%
  add_recipe(recipe_norm) %>%
  add_model(model_rf)




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
                      param_info = parameters(num_comp(range(1,50)),
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
                     param_info = parameters(num_comp(range=c(1,50)))
)
toc()
# 327 sec elapsed (~ 5.5 min)


tic()
tune_las<- tune_bayes(wf_las,
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
# 260.82 sec elapsed (~ 4.5 min)


tic()
tune_rf<- tune_bayes(wf_rf,
                     resamples = folds,
                     initial = 10,
                     #control = control_stack_bayes(),
                     control = control_bayes(save_pred=TRUE,
                                             save_workflow=TRUE,
                                             seed=0),
                     metrics = metric_set(rmse),
                     param_info = parameters(mtry(range=c(1,300)),
                                             trees(range=c(10,10000)),
                                             min_n(range=c(1,10))
                     )
)
toc()
# 699.35 sec elapsed (~ 11 min)



## ESCOLHENDO O MELHOR (BEST roc_auc)

show_best(tune_knn,n=3)
show_best(tune_qr,n=3)
show_best(tune_las,n=3)
show_best(tune_rf,n=3)



##### STACKING ENSEMBLE  #####

stack_ensemble_data<- stacks() %>% 
  add_candidates(tune_knn) %>% 
  add_candidates(tune_qr) %>% 
  add_candidates(tune_las) %>% 
  add_candidates(tune_rf) 


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


stack_ensemble_model<- stack_ensemble_model %>% 
  fit_members()

stack_ensemble_model




### VALIDANDO O(S) MODELO(S) ###

# PREDIÇÃO DE DADOS TESTE
pred.test<- predict(stack_ensemble_model, df.test)

predicao<- data.frame(df.test$y,pred.test)
colnames(predicao)<- c("y","pred")
head(predicao)
rmse(predicao,y,pred)
#26.5 num_comp=30
#17.3 num_comp=21
#27.4 Log


# PREDIÇÃO DE DADOS NOVOS

pred.new<- predict(stack_ensemble_model, new_data = df.new)

predicao<- data.frame(df.new$y,pred.new)
colnames(predicao)<- c("y","pred")
head(predicao)
rmse(predicao,y,pred)
#1.33 num_comp=30
#1.33 num_comp=21
#1.65 Log

# SALVANDO O MODELO

saveRDS(stack_ensemble_model,"G:/Meu Drive/estatistica/LabPetro/stackEnsemble/stack_dens_api.rds")





#############################################
### STACKING THE BEST MODELS OF EACH TYPE ###
#############################################

##### WORKFLOW #####

wf.knn2<- workflow() %>%
  add_recipe(recipe.pls) %>%
  add_model(fit.knn)

wf.pls2<- workflow() %>%
  add_recipe(recipe.norm) %>%
  add_model(fit.pls)

wf.las2<- workflow() %>%
  add_recipe(recipe.pls) %>%
  add_model(fit.las)

wf.rf2<- workflow() %>%
  add_recipe(recipe.pls) %>%
  add_model(fit.rf)

wf.xgb2<- workflow() %>%
  add_recipe(recipe.pls) %>%
  add_model(fit.xgb)

wf.svm2<- workflow() %>%
  add_recipe(recipe.pls) %>%
  add_model(fit.svm)





##### HIPERPARAMETERS TUNING - GRID SEARCH #####

set.seed(0)
tune.knn2<- tune_grid(wf.knn2,
                      resamples = folds,
                      grid = show_best(tune.knn,n=1)[,1],
                      control = control_stack_grid(),
                      metrics = metric_set(rmse))


set.seed(0)
tune.pls2<- tune_grid(wf.pls2,
                      resamples = folds,
                      grid = show_best(tune.pls,n=1)[,1],
                      control = control_stack_grid(),
                      metrics = metric_set(rmse))


set.seed(0)
tune.las2<- tune_grid(wf.las2,
                      resamples = folds,
                      grid = show_best(tune.las,n=1)[,1:2],
                      control = control_stack_grid(),
                      metrics = metric_set(rmse))


set.seed(0)
tune.rf2<- tune_grid(wf.rf2,
                     resamples = folds,
                     grid = show_best(tune.rf,n=1)[,1:3],
                     control = control_stack_grid(),
                     metrics = metric_set(rmse))


set.seed(0)
tune.xgb2<- tune_grid(wf.xgb2,
                      resamples = folds,
                      grid = show_best(tune.xgb,n=1)[,1:3],
                      control = control_stack_grid(),
                      metrics = metric_set(rmse))


set.seed(0)
tune.svm2<- tune_grid(wf.svm2,
                      resamples = folds,
                      grid = show_best(tune.svm,n=1)[,1:3],
                      control = control_stack_grid(),
                      metrics = metric_set(rmse))




##### STACKING ENSEMBLE  #####

stack_ensemble_data2<- stacks() %>% 
  add_candidates(tune.knn2) %>% 
  add_candidates(tune.pls2) %>% 
  add_candidates(tune.las2) %>% 
  add_candidates(tune.rf2) %>% 
  add_candidates(tune.xgb2) %>% 
  add_candidates(tune.svm2)

stack_ensemble_data2


set.seed(0)
stack_ensemble_model2<- stack_ensemble_data2 %>% 
  blend_predictions(penalty = 10^(-9:-1),
                    mixture = seq(0,1,by=0.1),
                    control = control_grid(),
                    non_negative = FALSE,
                    metric = metric_set(rmse))

stack_ensemble_model2$penalty

# p=penalty and m=mixture
p<- as.numeric(stack_ensemble_model2$penalty[1])
grid.p<- c(0.5*p,0.75*p,p,1.25*p,1.5*p)

m<- as.numeric(stack_ensemble_model2$penalty[2])
if(m==0) {
  grid.m <- c(0, 0.01, 0.025, 0.05, 0.075)
} else{
  if (m == 1) {
    grid.m <- c(0.925, 0.95, 0.975, 0.99, 1)
  } else{
    inv.m <- log(m / (1 - m))
    grid.m <-
      1 / (1 + exp(-c(
        0.5 * inv.m, 0.75 * inv.m, inv.m, 1.25 * inv.m, 1.5 * inv.m
      )))
  }
}

set.seed(0)
stack_ensemble_model2<- stack_ensemble_data2 %>% 
  blend_predictions(penalty = grid.p,
                    mixture = grid.m,
                    control = control_grid(),
                    non_negative = FALSE,
                    metric = metric_set(rmse))

autoplot(stack_ensemble_model2)
autoplot(stack_ensemble_model2,type="members")
autoplot(stack_ensemble_model2,type = "weights")

stack_ensemble_model2$penalty

stack_ensemble_model2<- stack_ensemble_model2 %>% 
  fit_members()

stack_ensemble_model2
#── A stacked ensemble model ─────────────────────────────────────

#Out of 6 possible candidate members, the ensemble retained 4.
#Penalty: 0.15.
#Mixture: 0.75.

#The 4 highest weighted members are:

#   member        type             weight

# 1 tune.pls2_1_1 pls              0.580 
# 2 tune.xgb2_1_1 boost_tree       0.224 
# 3 tune.las2_1_1 linear_reg       0.203 
# 4 tune.knn2_1_1 nearest_neighbor 0.0673



### VALIDANDO O(S) MODELO(S) ###

# PREDIÇÃO DE DADOS TESTE
pred.test<- predict(stack_ensemble_model2, df.test)

predicao<- data.frame(df.test$y,pred.test)
colnames(predicao)<- c("y","pred")
head(predicao)
rmse(predicao,y,pred)
#28.1 num_comp=30
#14.2 num_comp=21



# PREDIÇÃO DE DADOS NOVOS

pred.new<- predict(stack_ensemble_model2, new_data = df.new)

predicao<- data.frame(df.new$y,pred.new)
colnames(predicao)<- c("y","pred")
head(predicao)
rmse(predicao,y,pred)
#1.54 num_comp=30
#1.51 num_comp=21



# SALVANDO O MODELO

saveRDS(stack_ensemble_model2,"G:/Meu Drive/estatistica/LabPetro/stackEnsemble/stackBest_dens_api.rds")

