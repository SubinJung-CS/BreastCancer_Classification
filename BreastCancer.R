#install.packages("nclSLR",repos="http://R-Forge.R-project.org")
# Loading packages
library(nclSLR)
library(mlbench)
library(dplyr)
library(tidyverse)
library(corrplot)
library(leaps)
library(glmnet)
library(MASS)

# Loading the data
data(BreastCancer)
# Checking size
dim(BreastCancer)

### Data Preparation ###

# Converting factor column into numeric
## Checking the class of each column
sapply(BreastCancer, class)
## Identifying the factor columns only
sapply(BreastCancer, is.factor)
## Converting factor to quantitative variables (without 'Class' column)
BreastCancer[,2:10] <- as.data.frame(apply(BreastCancer[,2:10], 2, as.numeric))
## Checking the class of each column
sapply(BreastCancer, class)

# Removing missing values (NA)
BreastCancer <- BreastCancer %>% drop_na()



### Data Exploration ###

# Exploring a numerical summaries
summary(BreastCancer)
## Creating a new dataframe only with predictor variables
pred_var = BreastCancer[,2:10]
## Creating a new dataframe only with response variable
resp_var = BreastCancer[,11]

# Exploring a graphical summaries
## Plotting a scatter plot matrix
pairs(pred_var)
## Correlation Matrix
cor_mat <- cor(pred_var)
corrplot(cor_mat, method="number")

# Standardising the predictor variables only 
X = scale(pred_var)
y = as.integer(resp_var)
y2 = resp_var
## New dataframe with standardised predictor variables & response variable.
BreastCancer_St = data.frame(X,y)
BreastCancer_lr = data.frame(X,y2)



### Split Training/Validation ###

## Create random values in every run
set.seed(1)
## Training/Validation Set
train_set = sample(c(TRUE, FALSE), nrow(BreastCancer_St), replace=TRUE)
validation_set = (!train_set)
## Indices of training/validation data
training_indices = which(train_set)
validation_indices = which(!train_set)



### Subset Selection ### 

# Best Subset Selection
bss = regsubsets(y ~ ., data=BreastCancer_St[train_set, ], method="exhaustive", nvmax=8)
## Whole Summary
(bss_summary = summary(bss))
## Models Summary
bss_summary$adjr2 ## Adjusted R2
bss_summary$cp ## Mallow's Cp statistic
bss_summary$bic ## Bayes Information Criterion (BIC)
## k value
(best_adjr2 = which.max(bss_summary$adjr2)) ## Adjusted R2
(best_cp = which.min(bss_summary$cp)) ## Mallow's Cp statistic
(best_bic = which.min(bss_summary$bic)) ## Bayes Information Criterion (BIC)

## Plot graphs with the k point
par(mfrow=c(1,3))
### Adjusted R2
plot(1:8, bss_summary$adjr2, xlab="Number of predictors", ylab="Adjusted Rsq",
     type="b")
points(best_adjr2, bss_summary$adjr2[best_adjr2], col="red", pch=16)
### Mallow's Cp statistic
plot(1:8, bss_summary$cp, xlab="Number of predictors", ylab="Cp", type="b")
points(best_cp, bss_summary$cp[best_cp], col="red", pch=16)
### Bayes Information Criterion (BIC)
plot(1:8, bss_summary$bic, xlab="Number of predictors", ylab="BIC", type="b")
points(best_bic, bss_summary$bic[best_bic], col="red", pch=16)

# Cross-validation for Best Subset Selection
k = 10
set.seed(1)
folds = sample(1:k, nrow(BreastCancer_St), replace = TRUE)
cv.errors = matrix(NA, k, 8, dimnames = list(NULL, paste(1:8)))
## Build Predict Method
predict.regsubsets = function(object, newdata, id, ...){
  form = as.formula(object$call[[2]])
  mat = model.matrix(form, newdata)
  coefi = coef(object, id = id)
  xvars = names(coefi)
  mat[, xvars]%*%coefi
}
## Perform Cross-Validation
for(i in 1:k){
  best.fit = regsubsets(y ~ ., data = BreastCancer_St[folds != i, ], nvmax = 8)
  for(j in 1:8){
    pred = predict(best.fit, BreastCancer_St[folds == i, ], id = j)
    cv.errors[i, j] = mean((BreastCancer_St$y[folds == i] - pred)^2)
  }
}
## Take the mean of over all folds for each model size
mean.cv.errors = apply(cv.errors, 2, mean)
mean.cv.errors
## Find the model size with the smallest cross-validation error
min = which.min(mean.cv.errors)
min
## Plot the cross-validation error for each model size, highlight the min
plot(1:8, mean.cv.errors, xlab="Number of predictors", type="b")
points(min, mean.cv.errors[min][1], col = "red", cex = 2, pch = 20)
## Check the final variables
reg_best = regsubsets(y~., data = BreastCancer_St, nvmax = 8)
coef(reg_best, 5)



# Regularisation - Ridge Regression #

# Ridge Regression
## Set the variables
x = model.matrix(y ~ ., data = BreastCancer_St)[, -1]
y = BreastCancer_St$y
## Split train/validation set
set.seed(1)
train = sample(1:nrow(x), nrow(x)/2)
test = (-train)
y.test = y[test]

## Choosing grid of values for the tuning parameter
grid = 10^seq(5, -3, length=100)
## Fit a ridge regression model
ridge.mod = glmnet(x[train, ], y[train], alpha = 0, standardize=FALSE, lambda = grid)
ridge.pred = predict(ridge.mod, s = 4, newx = x[test, ])
mean((ridge.pred - y.test)^2)

## Coefficients plot
plot(ridge.mod, xvar="lambda", label=TRUE)

# Find the best lambda value with performing Cross Validation
## Plot a graph
set.seed(1)
cv.out = cv.glmnet(x[train, ], y[train], alpha = 0, standardize=FALSE)
plot(cv.out)
## Actual lambda value
bestlam = cv.out$lambda.min
bestlam
## Compute test mean squared error with best lambda value
ridge.pred = predict(ridge.mod, s = bestlam, newx = x[test, ])
mean((ridge.pred - y.test)^2)
## Coefficient from Ridge Regression
out = glmnet(x, y, alpha = 0, standardize=FALSE)
predict(out, type = "coefficients", s = bestlam)[1:10, ]




# Regularisation - the LASSO #

# The LASSO
lasso.mod = glmnet(x[train, ], y[train], alpha = 1, standardize=FALSE, lambda = grid)
# Coefficient Plot
plot(lasso.mod, xvar="lambda", label=TRUE)

# Find the best lambda value with performing Cross Validation
set.seed(1)
cv.out = cv.glmnet(x[train, ], y[train], alpha = 1, standardize=FALSE)
plot(cv.out)
## Actual lambda value
bestlam = cv.out$lambda.min
bestlam
## Compute test mean squared error with best lambda value
lasso.pred = predict(lasso.mod, s = bestlam, newx = x[test, ])
mean((lasso.pred - y.test)^2)
## Coefficient from the LASSO
out = glmnet(x, y, alpha = 1, standardize=FALSE)
predict(out, type = "coefficients", s = bestlam)[1:10, ]



### LDA ###

# LDA
(lda_train = lda(y~., data=BreastCancer_St[train_set,]))
## Compute fitted values for the validation data
lda_test = predict(lda_train, BreastCancer_St[!train_set,])
names(lda_test)
yhat_test = lda_test$class
head(yhat_test)
# Compute test error
1 - mean(BreastCancer_St$y[!train_set] == yhat_test)

# Cross-Validation
## Training set Accuracy
lda_cross <- lda(y~., data=BreastCancer_St[train_set,], CV=TRUE)
table(BreastCancer_St[train_set,]$y, lda_cross$class, dnn = c('Actual Group','Predicted Group'))


### QDA ###

## Perform QDA on the training data
(qda_train = qda(y~., data=BreastCancer_St[train_set,]))
## Compute fitted values for the validation data:
qda_test = predict(qda_train, BreastCancer_St[!train_set,])
yhat_test = qda_test$class
## Compute test error
1 - mean(BreastCancer_St$y[!train_set] == yhat_test)

# Cross-Validation
qda_cross <- qda(y~., data=BreastCancer_St[train_set,], CV=TRUE)
table(BreastCancer_St[train_set,]$y, qda_cross$class, dnn = c('Actual Group','Predicted Group'))













