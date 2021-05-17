# Joel, Lichen
# Statistical Models & Computing (01:954:567)
# Professor Wang
# April 20, 2021

### Preliminaries
library(ggplot2)
library(caret)
library(tidyverse)
library(boot)
library(broom)
library(tidytext)
library(rvest)
library(purrr)
library(gridExtra)
library(grid)
library(lattice)
library(bnpa) # check.na
library(plyr) 
library(tree)
library(pROC)
library(mgcv)
library(cvTools)
library(randomForest)
library(gam)
library(gbm)
library(glmnet)
library(rpart)

### Checking & cleaning data
## loading data
data = read.csv(file = 'Titanic_Data.csv') # choose working directory first
temp_data = data # used to compare original data
head(data)
dim(data) # 1309 rows x 12 columns
names(data)
## cleaning data
# imputing missing values for Age & Fare with mean (reference: https://www.guru99.com/r-replace-missing-values.html)
check.na(data) # missing approx. 20% of data, so cannot drop observations/rows
age_mean = mean(data$Age, na.rm = TRUE)
fare_mean = mean(data$Fare, na.rm = TRUE)
data$Age[is.na(data$Age)] = age_mean # note: don't do data$Age[data$Age == NA] = age_mean
data$Fare[is.na(data$Fare)] = fare_mean
check.na(data) # should be no more missing values! (note: there are 2 more left...)
# dropping NA's for Embarked
levels(data$Embarked)# "" = implies missing values, but not marked with NA
data$Embarked[data$Embarked == ""] = NA # marks them as such; doing count(data$Embarked) shows 2 NA's
check.na(data) # still 2 left, let's drop them (can't impute means, since is multinomial variable)
data = na.omit(data)
count(data$Embarked)
data$Embarked = factor(data$Embarked) # eliminates "" level
levels(data$Embarked) # no more NAs/"" level
# Converting all values to numeric
# data = map_df(data,as.numeric) %>% as.data.frame() ### not using this, since converts all numeric values
data[, c(6, 7, 8, 10)] = map_df(data[, c(6, 7, 8, 10)], as.numeric)
# Converting nominal and ordinal categorical variables
data[, c(2, 5)] = map_df(data[, c(2, 5)], as.factor)
data$Pclass = factor(data$Pclass, ordered = TRUE) # is 3 column
# checking categorical variables (which will be used)
levels(data$Survived) # binary
levels(data$Pclass) # ordinal
levels(data$Sex) # binary
levels(data$Embarked) # multinomial
# imputing 0'& 1's for sex
levels(data$Sex)
levels(data$Sex)[1] = 0 # female = 0
levels(data$Sex)[2] = 1 # male = 1
levels(data$Sex)
# imputing 1's, 2's, & 3's for Embarked
levels(data$Embarked)
levels(data$Embarked)[1] = 1 # C = 1
levels(data$Embarked)[2] = 2 # Q = 2
levels(data$Embarked)[3] = 3 # S = 3
levels(data$Embarked) 
data$Embarked = factor(data$Embarked, ordered = FALSE) # is 12 column; NEEDED to let R recognize it as multinomial

### Descriptive Statistics (EDA)
## Summaries
dim(data)
head(data)
summary(data)
## Plots 
# Bar charts (since variables are categorical)
p1 = ggplot(data, aes(x = Survived)) + 
  geom_bar() + ggtitle("Bar chart of survived")
p2 = ggplot(data, aes(x = Pclass)) +
  geom_bar() + ggtitle("Bar chart of pclass") 
p3 = ggplot(data, aes(x = Sex)) +
  geom_bar() + ggtitle("Bar chart of sex")
p4 = ggplot(data, aes(x = Embarked)) +
  geom_bar() + ggtitle("Bar chart of embarked")
grid.arrange(p1, p2, p3, p4, nrow = 2, ncol = 2) 
# Histograms (since variables are continuous)
par(mfrow = c(2, 2))
hist(data$Age)
hist(data$SibSp)
hist(data$Parch)
hist(data$Fare)
par(mfrow = c(1, 1)) # must run, resets par()
# Plots (reference: https://r4ds.had.co.nz/exploratory-data-analysis.html)
ggplot(data = data) + # plot b/w 2 categorical variables
  geom_count(mapping = aes(x = Sex, y = Survived))
plot(data$Age, data$Survived, main = "Scatterplot of Age & Survived", # plot cont. variable against categorical
     xlab = "Age", ylab = "Survived", pch=19)

### Model building
## splitting and check data
set.seed(1)
set = sample(1:nrow(data), nrow(data)*0.5) # 0.5 = 1/2
train = data[set, ] # training set
test = data[-set, ] # test set
Survived.train = train$Survived
Survived.test = test$Survived
dim(train) # 653 x 12
dim(test) # 654 x 12
dim(data)
count(Survived.test) # load plyr library first before running this
count(Survived.train) # seems to be a nice ratio of both 0's and 1's for both sets

## Logistic regression
# logit models (2nd & 3rd models = reduced based on most statistically significant p-values)
glm.fits1 = glm(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked, family = binomial, data = train)
summary(glm.fits1)
glm.fits2 = glm(Survived ~ Pclass + SibSp + Parch + Fare + Embarked, family = binomial, data = train)
summary(glm.fits2)
glm.fits3 = glm(Survived ~ SibSp + Parch + Fare, family = binomial, data = train)
summary(glm.fits3) # note that AIC values = decreasing
# predicting 1st model
nrow(test)
glm.probs1 = predict(glm.fits1, test, type = "response")
glm.pred1 = rep(0, 654) # 654 = comes from nrow(test)
glm.pred1[glm.probs1 > 0.5] = 1
table(glm.pred1, Survived.test)
100*mean(glm.pred1 == Survived.test) # gives 85.47 test accuracy
# predicting 2nd model
glm.probs2 = predict(glm.fits2, test, type = "response")
glm.pred2 = rep(0, 654) # 654 = comes from nrow(test)
glm.pred2[glm.probs2 > 0.5] = 1
table(glm.pred2, Survived.test)
100*mean(glm.pred2 == Survived.test) # gives 62.39 test accuracy
# predicting 3rd model
glm.probs3 = predict(glm.fits3, test, type = "response")
glm.pred3 = rep(0, 654) # 654 = comes from nrow(test)
glm.pred3[glm.probs3 > 0.5] = 1
table(glm.pred3, Survived.test)
100*mean(glm.pred3 == Survived.test) # gives 64.07 test accuracy
# LOOCV
cv.err1 = cv.glm(train, glm.fits1)
cv.err1$delta
cv.err2 = cv.glm(train, glm.fits2)
cv.err2$delta
cv.err3 = cv.glm(train, glm.fits3)
cv.err3$delta
# Given the fact that the 1st model has the lowest AIC value, highest
# prediction accuracy, and lowest CV error, we choose model 1
# AUC 
glm.fits1_roc = roc((as.integer(Survived.test)-1),glm.pred1, plot=T,print.auc=T,
                    col="red",lwd=4,legacy.axes=T, main="ROC curves for Logistic Regression (Model 1)")
# check multicollinearity b/w continous variables
chart.Correlation(data[,c(6, 7, 8, 10)] ,method="spearman")

## Classification tree
set.seed(1)
tree.data = tree(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked, data = train) # tree.data = R gives output corresponding to each branch of tree
tree.pred = predict(tree.data, test, type = "class")
table(tree.pred, Survived.test) # evaluating performance on test data
100*round((374 + 192)/566, 4) # 100% test observations are correctly classified
# performing CV on tree
set.seed(1)
cv.data = cv.tree(tree.data, FUN = prune.misclass) # let's see if pruning tree leads = improved results
names(cv.data) # cv.tree = determines optimal level of tree complexity
cv.data # dev = corresponds to CV error rate; tree with 2 terminal nodes = has lowest CV error
# plot error rate as function of size & k
par(mfrow = c(1, 2))
plot(cv.data$size, cv.data$dev, type = "b")
plot(cv.data$k, cv.data$dev, type = "b")
par(mfrow = c(1, 1)) # reset par
# pruning & plotting tree of 5 nodes
prune.data = prune.misclass(tree.data, best = 4) # based on cv.data, 4 or 6 nodes is best
plot(prune.data)
text(prune.data, pretty = 0)
tree.preds = predict(prune.data, test, type = "class") # check how well pruned tree performs on test set
table(tree.preds, Survived.test)
100*round((374 + 192)/566, 4) # 100% of test observations are correctly classified
# AUC
tree_roc = roc((as.integer(Survived.test)-1), (as.integer(tree.preds)-1), plot=T,print.auc=T,
               col="red",lwd=4,legacy.axes=T, main="ROC curves for Classification Tree")

## Generalized Additive Model (GAM)
# fit the train data using the generalized additive model
# s is the smooth term
# Pclass, Sex and Embarked are facors, we do not need to use the smooth terms
# SibSp and Parch are discrete variables, we do not have enough degrees to fit the data so we do not use the smooth terms
# 1st GAM
set.seed(1)
gam_fit <- gam(Survived ~ Pclass + Sex + s(Age) + SibSp + Parch + s(Fare) + Embarked, family = binomial, data = train)
summary(gam_fit)
layout(matrix(1:2, nrow = 1))
plot(gam_fit, shade = T)
# predict function returns the linear prediction
gam_pred <- 1 - 1 / (1 + exp(predict(gam_fit, test))) > 0.5
table(gam_pred, Survived.test) # evaluating performance on test data
mean(gam_pred == as.numeric(Survived.test) - 1) # gives 86.39% test accuracy
##### Failed attempt for 2nd & 3rd GAMs ######
# 2nd GAM
#gam_fit1 <- gam(Survived ~ Pclass + Sex + s(Age,2) + SibSp + Parch + s(Fare) + Embarked, family = binomial, data = train)
#summary(gam_fit1)
#layout(matrix(1:2, nrow = 1))
#plot(gam_fit1, shade = T)
# predict function returns the linear prediction
#gam_pred1 <- 1 - 1 / (1 + exp(predict(gam_fit1, test))) > 0.5
#table(gam_pred1, Survived.test) # evaluating performance on test data
#mean(gam_pred1 == as.numeric(Survived.test) - 1) # gives 86.39% test accuracy
# 3rd GAM
#gam_fit2 <- gam(Survived ~ Pclass + Sex + s(Age) + SibSp + Parch + s(Fare, 2) + Embarked, family = binomial, data = train)
#summary(gam_fit2)
#layout(matrix(1:2, nrow = 1))
#plot(gam_fit2, shade = T)
# predict function returns the linear prediction
#gam_pred2 <- 1 - 1 / (1 + exp(predict(gam_fit2, test))) > 0.5
#table(gam_pred2, Survived.test) # evaluating performance on test data
#mean(gam_pred2 == as.numeric(Survived.test) - 1) # gives 86.39% test accuracy
##### End of failed attempt #####
# use cross validation to evaluate the model
# we perform 50 times CV
cvK <- 5  # number of CV folds
cv_gam_res <- cv_acc <- c()
for (i in 1:50) {
  cvSets <- cvTools::cvFolds(nrow(train), cvK)  # permute all the data, into cvK folds
  cv_acc <- NA  # initialise results vector
  for (j in 1:cvK) {
    test_id <- cvSets$subsets[cvSets$which == j]
    df_test <- train[test_id, ]
    df_train <- train[-test_id, ]
    fit <- gam(Survived ~ Pclass + Sex + s(Age) + SibSp + Parch + s(Fare) + Embarked, family = binomial, data = df_train)
    pred <- 1 - 1 / (1 + exp(predict(fit, df_test))) > 0.5
    cv_acc[j] <- mean(pred == as.numeric(df_test$Survived) - 1)
  }
  cv_gam_res <- append(cv_gam_res, mean(cv_acc))
}
par(mfrow = c(1, 1))
boxplot(cv_gam_res, horizontal = TRUE, xlab = "Accuracy")
mean(cv_gam_res)*100
# AUC 
par(mfrow = c(1, 1))
gam_roc = roc((as.numeric(Survived.test)-1),as.numeric(gam_pred), plot=T,print.auc=T,
              col="red",lwd=4,legacy.axes=T, main="ROC curves for the generalized additive model")

## Random Forest
# fit the train data using Random Forest
# ntree: Number of trees to grow
# mtry: Number of variables randomly sampled as candidates at each split
rf_fit <- randomForest(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked, data = train,
                       ntree = 500, mtry = 3, proximity = TRUE, importance = TRUE)
# the importance of variables
# 0: the influence of variable replacement on the data classified as 0
# 1: Represents the impact of variable replacement on data classified as 0
# Mean decrease accuracy: the decrease of accuracy after variable replacement
# Mean decrease Gini: the decrease of Gini coefficient after variable replacement.
# The larger the value, the more important the variable is.
rf_fit$importance
# predict the test data
rf_pred <- predict(rf_fit, test)
table(rf_pred, Survived.test) # evaluating performance on test data
mean(rf_pred == Survived.test) # give test accuracy
# use cross validation to evaluate the model
# we perform 50 times CV
cvK <- 5  # number of CV folds
cv_rf_res <- cv_acc <- c()
for (i in 1:50) {
  cvSets <- cvTools::cvFolds(nrow(train), cvK)  # permute all the data, into cvK folds
  cv_acc <- NA  # initialise results vector
  for (j in 1:cvK) {
    test_id <- cvSets$subsets[cvSets$which == j]
    df_test <- train[test_id, ]
    df_train <- train[-test_id, ]
    fit <- randomForest(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked, data = df_train,
                        ntree = 500, mtry = 3, proximity = TRUE, importance = TRUE)
    pred <- predict(fit, df_test)
    cv_acc[j] <- mean(pred == df_test$Survived)
  }
  cv_rf_res <- append(cv_rf_res, mean(cv_acc))
}
par(mfrow = c(1, 1))
boxplot(cv_rf_res, horizontal = TRUE, xlab = "Accuracy")
mean(cv_rf_res)*100
# AUC 
rf_roc = roc((as.numeric(Survived.test)-1),as.numeric(rf_pred), plot=T,print.auc=T,
             col="red",lwd=4,legacy.axes=T, main="ROC curves for the Random Forest")

## KNN
# fit the train data using knn
# consider 3 neighbours
idx <- c(3, 5:8, 10, 12)
knn_pred <- class::knn(train[, idx], test[, idx], cl = train[, 2], k = 3)
table(knn_pred, Survived.test) # evaluating performance on test data
mean(knn_pred == Survived.test) # give test accuracy
# use cross validation to evaluate the model
# we perform 50 times CV
cvK <- 5  # number of CV folds
cv_knn_res <- cv_acc <- c()
for (i in 1:50) {
  cvSets <- cvTools::cvFolds(nrow(train), cvK)  # permute all the data, into cvK folds
  cv_acc <- NA  # initialise results vector
  for (j in 1:cvK) {
    test_id <- cvSets$subsets[cvSets$which == j]
    df_test <- train[test_id, ]
    df_train <- train[-test_id, ]
    pred <- class::knn(df_train[, idx], df_test[, idx], cl = df_train[, 2], k = 3)
    cv_acc[j] <- mean(pred == df_test$Survived)
  }
  cv_knn_res <- append(cv_knn_res, mean(cv_acc))
}
par(mfrow = c(1, 1))
boxplot(cv_knn_res, horizontal = TRUE, xlab = "Accuracy")
mean(cv_knn_res)*100
# AUC 
knn_roc = roc((as.numeric(Survived.test)-1),as.numeric(knn_pred), plot=T,print.auc=T,
              col="red",lwd=4,legacy.axes=T, main="ROC curves for knn")