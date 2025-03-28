#Necessary libraries for this project
library(rpart)
library(rpart.plot)
library(viridis)
library(patchwork)
library(GGally)
library(skimr)
library(tidyverse) 
library(dslabs)
library(ggplot2)
library(readxl)
library(caret)
library(MASS)
library(e1071)
library(glmnet)
library(Matrix)
library(randomForest)
library(xgboost)
library(SHAPforxgboost)
library(ROCR)
library(DMwR2)
library(pROC)
library(writexl)

# Loading the data
data=read_csv(file.choose())

#Examining the raw data
skim(data)
head(data)
boxplot(data$NumWebVisitsMonth)



# assuming typos in 1893 and 1899 should be 1993 and 1999
data <- data %>%
  mutate(Year_Birth = case_when(
    Year_Birth == 1893 ~ 1993,
    Year_Birth == 1899 ~ 1999,
    TRUE ~ Year_Birth
  ))

# Dropping the row where birth year is 1900
data <- data %>%
  filter(Year_Birth != 1900)


## converting 'Alone', 'Absurd', 'Yolo' in Marital_Status
## to 'Single 
data <- data %>%
  mutate(Marital_Status = case_when(
    Marital_Status == 'Alone' ~ 'Single',
    Marital_Status == 'YOLO' ~ 'Single',
    Marital_Status == 'Absurd' ~ 'Single',
    TRUE ~ Marital_Status
  ))


## creating a new age  column
data <- data %>%
  mutate(Age = 2015- Year_Birth)

## Clustering the age by percentile
data <- data %>%
  mutate(Age_Group = case_when(
    Age < 16 ~ "P25",
    Age < 38 ~ "P50",
    Age < 45 ~ "P75",
    Age >= 45 ~ ">P75",
    TRUE ~ NA_character_
  ))

skim(data)

##finding median for each age group and education level

mediandata <- data %>%
  group_by(Age_Group, Education) %>%
  summarise(median_value = median(Income, na.rm = TRUE)) ##na.rm negates Nulls

mediandata

##joining that median value to the original data

data <- data %>%
  left_join(mediandata, by = c("Age_Group", "Education"))

## replacing nulls in the the original income data with the median
## for their age group and education level

data <- data %>%
  mutate(Income = ifelse(is.na(Income), median_value, Income))  

#drop the median income data column (package mass also has a select function, thats why dplyr was specified)
data <- dplyr::select(data, -median_value)



# Change category "Together" in marital_status to "Married"
data <- data %>%
  mutate(Marital_Status = case_when(
    Marital_Status == 'Together' ~ 'Married',
    Marital_Status == 'Widow' ~ 'Single',
    TRUE ~ Marital_Status
  ))


data <- data %>%
  mutate(Income_Group = case_when(
    Income< 35733 ~ 'P25',
    Income<51518 ~ 'P50',
    Income<68688 ~ 'P75',
    Income>=68688 ~ '>P75',
    
    TRUE ~ NA_character_
  ))



# Turn categorical variables into factors
data$Response <- as.factor(data$Response)
data$Education <- as.factor(data$Education)
data$Marital_Status <- as.factor(data$Marital_Status)
data$Dt_Customer <- as.factor(data$Dt_Customer)







#drop the original column (package mass also has a select function, that's why dplyr was specified)
## we are dropping this column, because there is no reference point
data <- dplyr::select(data, -Dt_Customer)


# Create Dummy Variables for all but Response
dummies <- model.matrix(Response ~ ., data = data)
predictors_dummy <- data.frame(dummies[,-1])
data <- cbind(Response = data$Response, predictors_dummy) 



#should we leave this part or not? we can't have both this and the
# uncombined education variables (double count)

# Combine education dummy variables into broader categories
data$Education_Higher <- data$EducationGraduation + data$EducationMaster + data$EducationPhD

# Drop extra columns
data <- dplyr::select(data, -EducationGraduation)
data <- dplyr::select(data, -EducationMaster)
data <- dplyr::select(data, -EducationPhD)
data <- dplyr::select(data, -Year_Birth)
#data <- dplyr::select(data, -Age)
data <- dplyr::select(data, -Income)


# Lasso Model
skim(data)
head(data)


set.seed(99)
index <- createDataPartition(data$Response, p = .8,list = FALSE)
data_train <- data[index,]
data_test <- data[-index,]

data_train$Response <- factor(data_train$Response, levels = c(0,1), labels = c("No", "Yes"))
data_test$Response <- factor(data_test$Response, levels = c(0,1), labels = c("No", "Yes"))


data_train$Response <- relevel(data_train$Response, ref = "Yes")
data_test$Response <- factor(data_test$Response, levels = levels(data_train$Response))


set.seed(99)
lasso_model <- train(Response ~ .,
                     data = data_train,
                     method = "glmnet",
                     standardize = T,
                     tuneGrid = expand.grid(alpha = 1,
                                            lambda = seq(0.001, 0.01, length = 200)),
                     trControl = trainControl(method = "cv",
                                              number = 10,
                                              classProbs = TRUE,
                                              summaryFunction = twoClassSummary),
                     metric = "ROC")

lasso_model

coef(lasso_model$finalModel, lasso_model$bestTune$lambda)

predprob_lasso <- predict(lasso_model , data_test, type="prob")
predprob_lasso

pred_lasso <- prediction(predprob_lasso$Yes, data_test$Response, label.ordering = c("No", "Yes"))
perf_lasso <- performance(pred_lasso, "tpr", "fpr")
plot(perf_lasso, colorize=TRUE)
grid()

auc_lasso <- unlist(slot(performance(pred_lasso, "auc"), "y.values"))
auc_lasso



#XGB Boost
set.seed(99)
model_gbm <- train(Response ~ .,
                   data = data_train,
                   method = "xgbTree",
                   trControl =trainControl(method = "cv", 
                                           number = 9),
                   # provide a grid of parameters
                   tuneGrid = expand.grid(
                     nrounds = c(50,200),
                     eta = c(0.025,0.3 ,0.05, 0.7),
                     max_depth = c(2,4,6,8, 10),
                     gamma = 0,
                     colsample_bytree = 1,
                     min_child_weight = 1,
                     subsample = 1),
                   verbose=FALSE)


model_gbm
model_gbm$results
plot(model_gbm)

significant <- varImp(model_gbm, scale = FALSE)
plot(significant,
     main = "Important Variables for XGBoost",
     color = "red")

predprod_gbm <- predict(model_gbm, data_test, type = "prob")


# Receiver Operating Characteristic Measures Sensistivity v Specificity
pred_gbm <- prediction(predprod_gbm$Yes, data_test$Response, label.ordering = c("No", "Yes"))
perf_gbm <- performance(pred_gbm, "tpr", "fpr")
plot(perf_gbm, colorize=TRUE)
grid()

auc_gbm <- unlist(slot(performance(pred_gbm, "auc"), "y.values"))
auc_gbm

confusionMatrix(predict(model_gbm, data_test), data_test$Response)


colnames(predprod_gbm)

Xdata <- as.matrix(data_train[,-which(names(data_train)== "Response")]) # change data to matrix for plots
# Crunch SHAP values
shap <- shap.prep(model_gbm$finalModel, X_train = Xdata)

# SHAP importance plot
shap.plot.summary(shap)

top4 <- shap.importance(shap, names_only = TRUE)[1:30]

for (x in top4) {
  p <- shap.plot.dependence(
    shap, 
    x = x, 
    color_feature = "auto", 
    smooth = FALSE, 
    jitter_width = 0.01, 
    alpha = 0.4
  ) +
    ggtitle(x)
  print(p)
}


#####################################
####################################
#####################################
## Random Forest





set.seed(99)
model_rf <- train(Response ~ .,
                  data = data_train,
                  method = "rf",
                  tuneGrid= expand.grid(mtry = c(1,2,3, 4,6, 10)),
                  trControl =trainControl(method = "cv", 
                                          number = 10,
                                          #Estimate class probabilities
                                          classProbs = TRUE,
                                          ## Evaluate performance using the following function
                                          summaryFunction = twoClassSummary,
                                          sampling = "smote"),
                  metric="ROC",
                  ntree = 2000)

model_rf

model_rf$bestTune
plot(varImp(model_rf))
plot(model_rf)

predprod_rf <-  predict(model_rf, data_test, type="prob")
pred_rf <- prediction(predprod_rf$Yes, data_test$Response, label.ordering =c("No","Yes")) 
perf_rf = performance(pred_rf, "tpr", "fpr")
plot(perf_rf, colorize=TRUE)
grid()

auc_rf <- unlist(slot(performance(pred_rf, "auc"), "y.values"))
auc_rf


significant_rf <- varImp(model_rf, scale = FALSE)
plot(significant_rf)

confusionMatrix(predict(model_rf, data_test), data_test$Response)

colnames(predprod_rf)



#####################
####################
# Final EDA

# Get chart comparing marital status with average age
skim(data)
ggplot(data, aes(x = Response, y = Marital_StatusSingle)) +
  geom_bar()

write_xlsx(data, "case_data.xlsx")
