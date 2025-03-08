#UofL MBB Worksheet
library(tidyverse) 
library(dslabs)
library(readxl)
library(dplyr)
library(caret)
library(skimr)
library(GGally)
library(ggplot2)
library(patchwork)
library(e1071)
library(glmnet)
library(Matrix)
library(ROCR)
options(scipen=999)

# Part 1: Loading/Changing Data
mbb_data <- read_excel("C:/Users/benom/OneDrive - University of Louisville/Coding Projects/UofL MBB.xlsx",
                          sheet = "Net Points for UofL",
                          range = "C2:N19")

factors_uofl <- read_excel("C:/Users/benom/OneDrive - University of Louisville/Coding Projects/UofL MBB.xlsx",
                           sheet = "Factors",
                           range = "C2:H19")

factors_uofl <- factors_uofl %>% rename("WL" = "W/L") # Change Win Loss column name
factors_uofl <- factors_uofl %>% rename("TwoPoint" = "2PT")
factors_uofl <- factors_uofl %>% rename("ThreePoint" = "3PT")

factors_uofl <- factors_uofl %>% mutate_at(1, as.factor)
factors_uofl$WL <- relevel(factors_uofl$WL, ref = "W")
levels(factors_uofl$WL)
table(factors_uofl$WL)


# Part 2: Exploratory Data Analytics
skim(factors_uofl)
summary(factors_uofl)
aggregate(.~WL, factors_uofl, mean) 
aggregate(.~WL, factors_uofl, sd)


# Density per factor
plot_2pt <- plot(density(factors_uofl$'TwoPoint'), xlim = c(-30,30))
abline(v= mean(factors_uofl$'TwoPoint'),
       lty = 2,
       lwd = 1.5)

plot_3pt <- plot(density(factors_uofl$'ThreePoint'), xlim = c(-30,30))
abline(v= mean(factors_uofl$'ThreePoint'),
       lty = 2,
       lwd = 1.5)

plot_to <- plot(density(factors_uofl$'Turnover'), xlim = c(-30,30))
abline(v= mean(factors_uofl$'Turnover'),
       lty = 2,
       lwd = 1.5)

plot_reb <- plot(density(factors_uofl$'Rebound'), xlim = c(-30,30))
abline(v= mean(factors_uofl$'Rebound'),
       lty = 2,
       lwd = 1.5)

plot_ft <- plot(density(factors_uofl$'FreeThrow'), xlim = c(-30,30))
abline(v= mean(factors_uofl$'FreeThrow'),
       lty = 2,
       lwd = 1.5)


ggplot(data = factors_uofl) +
  aes(x = TwoPoint, fill = WL) +
  geom_density() + 
  facet_wrap(~WL, nrow = 3)


# Boxplots to identify outliers
boxplot(factors_uofl$TwoPoint)
boxplot(factors_uofl$ThreePoint)
boxplot(factors_uofl$Turnover)
boxplot(factors_uofl$Rebound)
boxplot(factors_uofl$FreeThrow)

boxplot(factors_uofl[,2:6])

boxplot(factors_uofl$TwoPoint ~factors_uofl$WL, xlab = "Win or Loss", ylab = "2pt Factor")
boxplot(factors_uofl$ThreePoint ~factors_uofl$WL, xlab = "Win or Loss", ylab = "3pt Factor")
boxplot(factors_uofl$Turnover ~factors_uofl$WL, xlab = "Win or Loss", ylab = "Turnover Factor")
boxplot(factors_uofl$Rebound ~factors_uofl$WL, xlab = "Win or Loss", ylab = "Rebound Factor")
boxplot(factors_uofl$FreeThrow ~factors_uofl$WL, xlab = "Win or Loss", ylab = "Free Throw Factor")

#ggplot(data = factors_uofl) +
  #aes(x = WL, y = TwoPoint, color = WL) +
  #geom_boxplot()


# Scatter plots
#ggpairs(factors_uofl, columns = c(2:6), aes(color = WL), legend = 1 
#) +
  #theme(legend.position = "bottom") 

ggplot(factors_uofl, aes(x = `Turnover`, y = `ThreePoint`, color = `WL`)) +
  geom_point(size = 3, alpha = 0.7) +
  labs(title = "Scatter Plot of 2PT vs. 3PT by W/L Outcome",
       x = "2PT Factor",
       y = "3PT Factor",
       color = "Game Outcome") +
  theme_minimal() + 
  scale_color_manual(values = c("W" = "green", "L" = "red"))


# Part 3: Partition the Data
set.seed(99)
index <- createDataPartition(factors_uofl$WL, p = .8, list = FALSE)
wl_train <-factors_uofl[index,]
wl_test <- factors_uofl[-index,]


# Step 4: Train or Fit LASSO Logistic Regression Model
set.seed(10)
wl_model <- train(WL ~ .,
                      data = wl_train,
                      method = "glmnet",
                      standardize = T,
                      tuneGrid = expand.grid(alpha = 1,
                                             lambda = seq(0.0001, 1, length = 200)),
                      trControl =trainControl(method = "cv",
                                              number = 10,
                                              classProbs = TRUE,
                                              summaryFunction = twoClassSummary),
                      metric="ROC")
wl_model                           

wl_forward_model <- train(WL ~ .,
                       data = wl_train,
                       method = "glmStepAIC",
                       direction = "forward",
                       trControl = trainControl(method = "cv",
                                                number = 3,
                                                classProbs = TRUE,
                                                summaryFunction = twoClassSummary),
                       metric="ROC")

coef(wl_model$finalModel, wl_model$bestTune$lambda)
coef(wl_forward_model$finalModel)


# Step 5: Get Predictions using Testing Set Data
predprob_lasso <- predict(wl_model , wl_test, type = "prob")
predprob_forward <- predict(wl_forward_model , wl_test, type = "prob")

pred_lasso <- prediction(predprob_lasso$W, wl_test$WL, label.ordering = c("L","W"))
perf_lasso <- performance(pred_lasso, "tpr", "fpr")
plot(perf_lasso, colorize = TRUE)

pred_forward <- prediction(predprob_forward$W, wl_test$WL, label.ordering = c("L","W"))
perf_forward <- performance(pred_forward, "tpr", "fpr")
plot(perf_forward, colorize=TRUE)

auc_lasso <- unlist(slot(performance(pred_lasso, "auc"), "y.values"))
auc_lasso

auc_forward <- unlist(slot(performance(pred_forward, "auc"), "y.values"))
auc_forward