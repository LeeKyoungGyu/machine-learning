## 0. Packages

library(ggplot2)
library(dplyr)
library(rpart)
library(partykit)
library(gridExtra)
library(h2o)
library(ModelMetrics)
library(mlbench)
library(pROC)

## Data EDA
##Data Load
##Data Preprocessing

data <- read.csv("./HR.csv")
data$left <- factor(data$left)


## Train set (70%) , Test set (30%)

train_idx <- sample(nrow(data) , 0.7*nrow(data), replace = F)

train <- data[train_idx,]
test <- data[-train_idx,]

## Change frame by 'h2o'

h2o.no_progress()
h2o.init(nthreads = -1)
train.hex <- as.h2o(train , destination_frame =  "train.hex")
test.hex <- as.h2o(test , destination_frame =  "test.hex")

## Modeling (Logistic Regression)

Y <- "left"
X <- setdiff( names(train.hex) , Y)

logit_model <- h2o.glm(x = X,y = Y , 
                       training_frame = train.hex, family = "binomial")
h2o.performance(logit_model, test.hex)

## Check the lowest logloss by Grid_Search

alphas <- seq(0, 1, 0.1)

logit.grid <- h2o.grid(
  algorithm = "glm",
  hyper_params = list(alpha = alphas),
  x = X, y = Y,
  grid_id = "logit_grid1",
  training_frame = train.hex,
  stopping_metric = "logloss",
  stopping_tolerance = 0.0001,
  family = "binomial",
  standardize = TRUE , 
  nfolds = 10 ,
  max_runtime_secs = 20
)

logit_gridperf <- h2o.getGrid(grid_id = "logit_grid1",
                              sort_by = "logloss",
                              decreasing = FALSE)

best_logit <- h2o.getModel(logit_gridperf@model_ids[[1]])
best_logit@model$coefficients_table
best_logit@parameters
best_logit@model$model_summary

## Evaluate Model with test data

perf <- h2o.performance(best_logit , test.hex)

h2o.confusionMatrix(perf)

## p1 density plot

logit.pred <- predict(object = best_logit, newdata = test.hex)

logit.pred.D <- as.data.frame(logit.pred)
logit.pred.D$left <- test$left 


logit.pred.D  %>% ggplot( aes(p1)) + geom_density(aes(fill = predict), alpha =0.5) +ggtitle("P1 Distribution") + theme_classic() 

## Check MSE, RMSE, Logloss, Mean Per-Class Error, AUC, Gini

perf <- h2o.performance(best_logit , test.hex)

mse <- perf@metrics$MSE 
rmse <- perf@metrics$RMSE
logloss <- perf@metrics$logloss
mean_per_class_error <- perf@metrics$mean_per_class_error
AUC <- perf@metrics$AUC
Gini <- perf@metrics$Gini
output <- data.frame(mse , rmse , logloss , mean_per_class_error , AUC , Gini) %>% tidyr::gather("metric", "value")

ggplot(output , aes(metric , value , fill = metric)) + 
  geom_bar( stat= "identity" , color ="black" , alpha = 0.5) + coord_flip()

## AUC value


target <- logit.pred.D$left
score <- logit.pred.D$p1

auc(target,score)

## ROC curve

roc_data=roc(target,score,ci=TRUE)
plot.roc(roc_data,
         col='black',
         print.auc=TRUE,
         print.auc.col = 'red',
         print.thres=TRUE,
         print.thres.pch=19,
         print.thres.col="red",
         grid=c(0.2,0.2),
         cex.lab=1,
         legacy.axes=TRUE,
         print.auc.adj = c(0,1))


## Gradient Boosting Machine

Y <- "left"
X <- setdiff( names(train.hex) , Y)

gbm_model <- h2o.gbm(x = X , y = Y , training_frame = train.hex)

h2o.performance(gbm_model , test.hex)

## Check the lowest logloss by Grid_Search

Y <- "left"
X <- setdiff( names(train.hex) , Y)

gbm_params1 <- list(learn_rate = c(0.001 , 0.05  , 0.01),
                    max_depth = c(3, 5, 7),
                    sample_rate = c(0.8, 0.9 ,1.0),
                    col_sample_rate = c(0.5, 0.7, 0.9) ,
                    min_split_improvement  = 10^c(-3 , -4 , -5 ))

search_criteria <- list(strategy = "RandomDiscrete",
                        max_models = 70 , seed = 1, 
                        max_runtime_secs = 20 ,
                        stopping_metric = "logloss", 
                        stopping_tolerance = 0.0001 )

gbm_grid <- h2o.grid("gbm", x = X, y = Y,
                     grid_id = "gbm_grid1",
                     training_frame = train.hex,
                     ntrees = 100,
                     hyper_params = gbm_params1 , 
                     sample_rate_per_class = c( 0.6 , 1)  , 
                     search_criteria = search_criteria)


gbm_gridperf <- h2o.getGrid(grid_id = "gbm_grid1",
                            sort_by = "logloss",
                            decreasing = FALSE)

best_gbm <- h2o.getModel(gbm_gridperf@model_ids[[1]])
best_gbm@parameters
best_gbm@model$model_summary


## Evaluate Model with test data

perf2 <- h2o.performance(best_gbm , test.hex)

h2o.confusionMatrix(perf2)

## Relative importance of value

h2o.varimp(best_gbm)
h2o.varimp_plot(best_gbm)

## p1 density plot

gbm.pred <- predict(object = best_gbm, newdata = test.hex)

gbm.pred.D <- as.data.frame(gbm.pred)
gbm.pred.D$left <- test$left 


gbm.pred.D  %>% ggplot( aes(p1)) + geom_density(aes(fill = predict), alpha =0.5) +ggtitle("P1 Distribution") + theme_classic() 

## Check MSE, RMSE, Logloss, Mean Per-Class Error, AUC, Gini

perf <- h2o.performance(best_gbm , test.hex)

mse <- perf@metrics$MSE 
rmse <- perf@metrics$RMSE
logloss <- perf@metrics$logloss
mean_per_class_error <- perf@metrics$mean_per_class_error
AUC <- perf@metrics$AUC
Gini <- perf@metrics$Gini
output <- data.frame(mse , rmse , logloss , mean_per_class_error , AUC , Gini) %>% tidyr::gather("metric", "value")

ggplot(output , aes(metric , value , fill = metric)) + 
  geom_bar( stat= "identity" , color ="black" , alpha = 0.5) + coord_flip()

## AUC value

target <- gbm.pred.D$left
score <- gbm.pred.D$p1

auc(target, score)

## ROC curve

roc_data=roc(target,score,ci=TRUE)
plot.roc(roc_data,
         col='black',
         print.auc=TRUE,
         print.auc.col = 'red',
         print.thres=TRUE,
         print.thres.pch=19,
         print.thres.col="red",
         grid=c(0.2,0.2),
         cex.lab=1,
         legacy.axes=TRUE,
         print.auc.adj = c(0,1))

## Random Forest

Y <- "left"
X <- setdiff( names(train.hex) , Y)

rf_model <- h2o.randomForest(x = X , y = Y , training_frame = train.hex)

h2o.performance(rf_model , test.hex)

## Check the lowest logloss by Grid_Search

Y <- "left"
X <- setdiff( names(train.hex) , Y)

drf_param <- list(max_depth = c(3, 5, 7),
                  sample_rate = c(0.5 , 0.6, 0.7 ),
                  min_split_improvement  = 10^c(-3 , -4 , -5  , -6))

search_criteria <- list(max_runtime_secs = 20 ,
                        strategy = "RandomDiscrete" , 
                        stopping_metric = "logloss", 
                        stopping_tolerance = 0.0001 )

rf_grid <- h2o.grid("drf", x = X, y = Y,
                    grid_id = "drf_grid",
                    balance_classes = TRUE ,
                    training_frame = train.hex,
                    ntrees = 100,
                    seed = 1  , 
                    
                    hyper_params = drf_param , 
                    search_criteria = search_criteria)


rf_gridperf <- h2o.getGrid(grid_id = "drf_grid",
                           sort_by = "logloss",
                           decreasing = FALSE)

best_rf <- h2o.getModel(rf_gridperf@model_ids[[1]])

h2o.performance(best_rf , test.hex)
best_rf@parameters
best_rf@model$model_summary


## Evaluate Model with test data


perf3 <- h2o.performance(best_rf , test.hex)

h2o.confusionMatrix(perf3)

## Relative importance of value

h2o.varimp(best_rf)
h2o.varimp_plot(best_rf)

## p1 density plot

rf.pred <- predict(object = best_rf, newdata = test.hex)

rf.pred.D <- as.data.frame(rf.pred)
rf.pred.D$left <- test$left 

rf.pred.D  %>% ggplot( aes(p1)) + geom_density(aes(fill = predict), alpha =0.5) +ggtitle("P1 Distribution") + theme_classic() 

## Check MSE, RMSE, Logloss, Mean Per-Class Error, AUC, Gini

perf <- h2o.performance(best_rf , test.hex)

mse <- perf@metrics$MSE 
rmse <- perf@metrics$RMSE
logloss <- perf@metrics$logloss
mean_per_class_error <- perf@metrics$mean_per_class_error
AUC <- perf@metrics$AUC
Gini <- perf@metrics$Gini
output <- data.frame(mse , rmse , logloss , mean_per_class_error , AUC , Gini) %>% tidyr::gather("metric", "value")

ggplot(output , aes(metric , value , fill = metric)) + 
  geom_bar( stat= "identity" , color ="black" , alpha = 0.5) + coord_flip()

## AUC value

target <- rf.pred.D$left
score <- rf.pred.D$p1


auc(target, score)

## ROC curve

roc_data=roc(target,score,ci=TRUE)
plot.roc(roc_data,
         col='black',
         print.auc=TRUE,
         print.auc.col = 'red',
         print.thres=TRUE,
         print.thres.pch=19,
         print.thres.col="red",
         grid=c(0.2,0.2),
         cex.lab=1,
         legacy.axes=TRUE,
         print.auc.adj = c(0,1))

## Choose p1 of each logit, GBM, RF then check Score Density Plot

log.1 = logit.pred.D %>% filter(left == 1 ) %>% dplyr::select( p1 , left) %>% mutate(model ="log")
gbm.1 = gbm.pred.D %>% filter(left == 1 ) %>% dplyr::select( p1 , left) %>% mutate(model ="GBM")
rf.1 = rf.pred.D %>% filter(left == 1 ) %>% dplyr::select( p1 , left) %>% mutate(model ="RF")

output1 <- rbind(log.1 , gbm.1)
output2 <- rbind(output1 , rf.1)


ggplot( output2 , aes(x = p1 , fill = model)) + geom_density(alpha = 0.5 ) + labs( main = "Three Model Left = 1 Density Plot") + xlim( 0 , 1 )
