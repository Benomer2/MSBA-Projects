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
library(pROC)
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


## creating a new age group column

data <- data %>%
  mutate(Age_Group = 2015 - Year_Birth) 

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
    TRUE ~ Marital_Status
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


# Lasso Model
skim(data)
head(data)

data$Response <- fct_recode(data$Response, Yes = "1", No = "0")
data$Response <- relevel(data$Response, ref = "Yes")
levels(data$Response)
data$Marital_StatusMarried <- relevel(data$Marital_StatusMarried, ref = "1")
levels(data$Marital_StatusMarried)

set.seed(99)
index <- createDataPartition(data$Response, p = .8,list = FALSE)
data_train <- data[index,]
data_test <- data[-index,]

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

auc_lasso <- unlist(slot(performance(pred_lasso, "auc"), "y.values"))
auc_lasso



library(xgboost)


data$Response = factor(make.names(data$Response))

#Split Set
set.seed(99)
index <-createDataPartition(data$Response, p = .8,list = FALSE)
data_train <- data[index,]
data_test <- data[-index,]


train_Control<- trainControl(method = "cv", 
                             number = 10,
                             verboseIter = TRUE,
                             classProbs = TRUE,
                             summaryFunction = defaultSummary,
                             sampling = "smote")


tuneGrid<-expand.grid(
  nrounds = c(465),
  eta = c( .047, .05),
  max_depth = c(6,10),
  gamma = c(0),
  colsample_bytree = c(.9, .5),
  min_child_weight = c(1, 2),
  subsample = c(1, .6, .1))

#XGB Boost
set.seed(99)
model_gbm <- train(Response ~ .,
                   data = data_train,
                   method = "xgbTree",
                   trControl = train_Control,
                   tuneGrid= tuneGrid)


model_gbm
model_gbm$results
plot(model_gbm)

Significant <- varImp(model_gbm, scale = FALSE)
plot(Significant)

bc_prob <- predict(model_gbm, data_test, type = "prob")


# Receiver Operating Characteristic Measures Sensistivity v Specificity
roc_data <- roc(data_test$Response, bc_prob[,1])


Area_Under_the_Curve <- auc(roc_data)
Area_Under_the_Curve

plot(roc_data)

confusionMatrix(predict(model_gbm, data_test), data_test$Response)

colnames(bc_prob)

Xdata<-as.matrix(data_train[,-which(names(data_train)== "Response")]) # change data to matrix for plots
# Crunch SHAP values
shap <- shap.prep(model_gbm$finalModel, X_train = Xdata)

# SHAP importance plot
shap.plot.summary(shap)

top4<-shap.importance(shap, names_only = TRUE)[1:4]

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





library(xgboost)


data$Response = factor(make.names(data$Response))

#Split Set
set.seed(99)
index <-createDataPartition(data$Response, p = .8,list = FALSE)
data_train <- data[index,]
data_test <- data[-index,]


train_Control<- trainControl(method = "cv", 
                             number = 10,
                             verboseIter = TRUE,
                             classProbs = TRUE,
                             summaryFunction = defaultSummary,
                             sampling = "smote")


tuneGrid<-expand.grid(
  nrounds = c(465),
  eta = c( .047, .05),
  max_depth = c(6,10),
  gamma = c(0),
  colsample_bytree = c(.9, .5),
  min_child_weight = c(1, 2),
  subsample = c(1, .6, .1))

#XGB Boost
set.seed(99)
model_gbm <- train(Response ~ .,
                   data = data_train,
                   method = "xgbTree",
                   trControl = train_Control,
                   tuneGrid= tuneGrid)


model_gbm
model_gbm$results
plot(model_gbm)

Significant <- varImp(model_gbm, scale = FALSE)
plot(Significant)

bc_prob <- predict(model_gbm, data_test, type = "prob")


# Receiver Operating Characteristic Measures Sensistivity v Specificity
roc_data <- roc(data_test$Response, bc_prob[,1])

b


Area_Under_the_Curve <- auc(roc_data)
Area_Under_the_Curve

plot(roc_data)

confusionMatrix(predict(model_gbm, data_test), data_test$Response)

colnames(bc_prob)

Xdata<-as.matrix(data_train[,-which(names(data_train)== "Response")]) # change data to matrix for plots
# Crunch SHAP values
shap <- shap.prep(model_gbm$finalModel, X_train = Xdata)

# SHAP importance plot
shap.plot.summary(shap)

top4<-shap.importance(shap, names_only = TRUE)[1:4]

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


skim(data$Age_Group)
