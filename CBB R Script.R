# Final Project
library(tidyverse) 
library(dslabs)
library(readxl)
library(shadowtext)

# Read the file to generate a dataset
cbb_data <- read_excel("C:/Users/benom/OneDrive - University of Louisville/MSBA/R/kenpom_dec.xls")
head(cbb_data)

# Part 1 - Clean the Data

# Filter data for only relevant information
only_top25 <- cbb_data |> 
  select(TeamName, AdjEM, RankAdjEM, AdjTempo, RankAdjTempo,
         AdjOE, RankAdjOE, AdjDE, RankAdjDE) |>
  filter(RankAdjEM <= 25) |>
  arrange(RankAdjEM)
only_top25


# Part 2 - Data Visualization

# Visualize Defensive Efficiency and Offensive Efficiency and where the best teams relate
only_top25 |> ggplot(aes(AdjOE, AdjDE)) +
  geom_shadowtext(aes(label = TeamName, color = TeamName),
                  bg.color = "black",
                  bg.r = 0.2,
                  size = 3) +
  scale_color_manual(values = c("Gonzaga" = "steelblue",
                                "Auburn" = "darkorange",
                                "Kentucky" = "blue",
                                "Duke" = "blue",
                                "Michigan" = "yellow",
                                "Tennesse" = "orange",
                                "Marquette" = "gold",
                                "Kansas" = "blue",
                                "St. John's" = "red",
                                "Clemson" = "orange",
                                "UCLA" = "cadetblue",
                                "Cincinnati" = "red",
                                "Texas Tech" = "red",
                                "Oregon" = "green",
                                "Maryland" = "tomato",
                                "North Carolina" = "skyblue",
                                "Mississippi St." = "brown",
                                "Illinois" = "orangered1",
                                "Purdue" = "gold4",
                                "Houston" = "red",
                                "Iowa St." = "brown2",
                                "Baylor" = "darkgreen",
                                "Connecticut" = "darkslateblue",
                                "Alabama" = "red1",
                                "Florida" = "green3",
                                "Tennessee" = "orange")) +
  scale_y_reverse() +
  labs(title = "Offensive Efficiency vs Defensive Efficiency",
       x = "Offensive Efficiency",
       y = "Defensive Efficiency") +
  theme(legend.position = "none")


# Based on our chart, we see a few teams that are top of the pack. We want to
# look specifically at Auburn, Gonzaga, Tennessee, Duke, and Houston
teams <- c("Auburn", "Gonzaga", "Tennessee", "Duke", "Houston")

comparison <- only_top25 |>
  filter(TeamName %in% teams) |>
  select(TeamName, AdjOE, AdjDE) |>
  pivot_longer(cols = c(AdjOE, AdjDE),
               names_to  = "Metric",
               values_to = "Efficiency")

ggplot(comparison, aes(x = TeamName, y = Efficiency, fill = TeamName)) + 
  geom_col() +
  scale_fill_manual(values = c("Gonzaga" = "steelblue4",
                               "Auburn" = "darkorange2",
                               "Duke" = "blue",
                               "Tennessee" = "orange",
                               "Houston" = "red")) +
  facet_wrap(~ Metric, ncol = 1, 
             labeller = as_labeller(c("AdjOE" = "Offensive Efficiency",
                                  "AdjDE" = "Defensive Efficiency"))) +
  scale_y_continuous(limits = c(80, 140), oob = scales::squish) +
  labs(title = "Efficiency Compared",
       x = "Team Name",
       y = "Efficiency Margins") +
  theme(legend.position = "none")


# Based on the plots we've seen comparing the top teams, we want to see if the
# top 25 teams have high or low tempo. In other words, is there a high 
# correlation between the speed played and the offensive efficiency of the team?
ggplot(only_top25, aes(AdjOE, AdjTempo)) +
  geom_point(size = 1) + 
  geom_smooth(method = "lm", color = "black") + 
  geom_shadowtext(aes(label = TeamName, color = TeamName),
                  bg.color = "black",
                  bg.r = 0.2,
                  size = 3) +
  scale_color_manual(values = c("Gonzaga" = "steelblue",
                                "Auburn" = "darkorange",
                                "Kentucky" = "blue",
                                "Duke" = "blue",
                                "Michigan" = "yellow",
                                "Tennesse" = "orange",
                                "Marquette" = "gold",
                                "Kansas" = "blue",
                                "St. John's" = "red",
                                "Clemson" = "orange",
                                "UCLA" = "cadetblue",
                                "Cincinnati" = "red",
                                "Texas Tech" = "red",
                                "Oregon" = "green",
                                "Maryland" = "tomato",
                                "North Carolina" = "skyblue",
                                "Mississippi St." = "brown",
                                "Illinois" = "orangered1",
                                "Purdue" = "gold4",
                                "Houston" = "red",
                                "Iowa St." = "brown2",
                                "Baylor" = "darkgreen",
                                "Connecticut" = "darkslateblue",
                                "Alabama" = "red1",
                                "Florida" = "green3",
                                "Tennessee" = "orange")) +
  scale_x_continuous(name = "Efficiency Margin") + 
  scale_y_continuous(name = "Tempo") + 
  labs(title = "Tempo compared to Efficiency Margin") +
  theme(legend.position = "none")

# Conclusion: of the 5 teams we looked at, only Gonzaga is above the regression
# line which separates the top 25. There is also only a slight increase in the
# regression line. So, the best teams can win by either picking up the pace and
# hurrying teams up, or they can slow their opponents down by wearing them down
# defensively


# Part 3 - Machine Learning

# Prediction problem 
# Which is a better predictor of a team in the top 25: Offensive Efficiency, or
# Defensive Efficiency?


# Step 1 - Transform EM column into binary
cbb_data$top_25 <- ifelse(cbb_data$RankAdjEM <= 25, "1", "0")
head(cbb_data$top_25)

# Step 2 - EDA
summary(cbb_data)

ggplot(cbb_data, aes(x = AdjOE, y = AdjDE, color = top_25)) +
  geom_point(size = 2) +
  scale_y_reverse() +
  labs(title = "Relationship between Offensive Efficiency & Defensive Efficiency,
       on a Teams rank")

# Step 3 - Train-Test Split
set.seed(123) # For reproducibility
library(caret)

# Create training and testing datasets
train_index <- createDataPartition(cbb_data$top_25, p = 0.9, list = FALSE)
train_data <- cbb_data[train_index, ]
test_data <- cbb_data[-train_index, ]

# Step 4 - Fit Logistic Regression Model

# Fit logistic regression model
train_data$top_25 <- as.factor(train_data$top_25)
logit_model <- glm(top_25 ~ AdjOE  + AdjDE, 
                   data = train_data, family = "binomial")

# Display model summary
summary(logit_model)

# Interpreting:
# A positive in the estimate column indicates that an increase in that estimate
# produces a higher odds of being in the top 25 (for defense, since it is 
# measured backwards, indicates the same thing in reverse)

# Although, the model does not seem to produce statistically significant
# predictors or show a good model fit in the aic


# Part 5 - Predict and Evaluate the model

# Predict probabilities on the test set
test_data$predicted_prob <- predict(logit_model, newdata = test_data, type = "response")

# Classify based on a threshold of 0.5
test_data$predicted_class <- ifelse(test_data$predicted_prob > 0.9, 1, 0)

# Confusion matrix
conf_matrix <- confusionMatrix(as.factor(test_data$predicted_class), as.factor(test_data$top_25))

# Print the confusion matrix and accuracy
conf_matrix


# Step 6 - Visualize model performance
ggplot(test_data, aes(x = predicted_prob, fill = top_25)) +
  geom_histogram(binwidth = 0.1, alpha = 0.9, position = "identity") +
  labs(title = "Predicted Probabilities by Transmission Type",
       x = "Predicted Probability (Manual)",
       y = "Count") +
  theme_minimal()

