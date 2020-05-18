# Group Project for TO567 @ UMich Ross
##############################################################################################################
# wine quality classification
# training with C5.0 with 16 trials is the best model to classify wine quality (71.6% accuracy)
# - read red wine quality data with quality as categorical variable
#=============================================================================================================
# check the red wine quality with kNN algorithm
# 
library(dplyr)
library(class)
library(rpart)
library(caret)
library(e1071)
library(gmodels)
library(rpart)
library(C50)
library(partykit)
library(rJava)
library(RWeka)
library(kernlab)
library(cluster)
library(gridExtra)
library(mlbench)
library(caret)
library(caretEnsemble)

# load wine quality data
red_wine_quality = read.csv("winequality-red.csv", header = TRUE)

# change the order of variables, y variable first
red_wine_quality = dplyr::select(red_wine_quality, quality, dplyr::everything())
range(red_wine_quality$quality)

# set the quality variable as a factor variable
red_wine_quality$quality = factor(red_wine_quality$quality, labels = c("3", "4", "5", "6", "7", "8"), levels = (3:8))
str(red_wine_quality$quality)

# check the NA data and if there is NA, remove it
if(sum(!complete.cases(red_wine_quality)) != 0)
{
  red_wine_quality = na.omit(red_wine_quality)
}

# set tthe sample size 80% of total data
row_size = nrow(red_wine_quality)
col_size = ncol(red_wine_quality)
sample_size = floor(row_size * 0.8)

# get sampling index
set.seed(567)
sample_index = sample(row_size, sample_size)
validation_size = row_size - sample_size

# set training and validation data set after normalizing all the values of x variables
training_x_scaled = as.data.frame(scale(red_wine_quality[sample_index, 2:col_size]))
training_y = red_wine_quality[sample_index, 1]
validation_x_scaled = as.data.frame(scale(red_wine_quality[-sample_index, 2:col_size]))
validation_y = red_wine_quality[-sample_index, 1]

# optimize k for knn algorithm
index = 1
trials = 30
optimized_k = 1
optimized_accuracy = 0

# find the optimized k between 1 and 30 based on accuracy
for(index in 1:trials) 
{
  temp_predicted_class_knn = knn(train = training_x_scaled, cl = training_y, test = validation_x_scaled, k = index, prob = TRUE)

  temp_accuracy = sum(temp_predicted_class_knn == validation_y) / validation_size
  
  if(temp_accuracy > optimized_accuracy)
  {
    optimized_k = index
    optimized_accuracy = temp_accuracy
  }
}

optimized_k
optimized_accuracy

# run kNN algorithm with the optimized k
predicted_class_knn = knn(train = training_x_scaled, test = validation_x_scaled, cl = training_y, k = optimized_k, prob = TRUE)

# result of the final categorization output
# - accuracy = 0.622 > NIR (0.456), Kappa = 0.39
# - lack of data which have quality 3 and 8, accuracy is not good
CrossTable(predicted_class_knn, validation_y, prop.chisq = FALSE)
confusionMatrix(predicted_class_knn, validation_y)
#=============================================================================================================


#=============================================================================================================
# check the red wine quality with rpart algorithm
# 
# set training and validation data set without normalization of x variables
training_data = red_wine_quality[sample_index, ]
validation_x = red_wine_quality[-sample_index, 2:col_size]
validation_y = red_wine_quality[-sample_index, 1]

# classification prediction
red_wine_model_rpart = rpart(formula = quality ~ ., data = training_data, method = "class", control = rpart.control(minsplit = 100, cp = 0.01))
summary(red_wine_model_rpart)

plot(red_wine_model_rpart, uniform = TRUE)
text(red_wine_model_rpart, use.n = TRUE, all = TRUE, cex = 0.7)

# check the cp table
# - 0.02 could be the point to be pruned
red_wine_model_rpart$cptable
plot(red_wine_model_rpart$cptable, col = "red")

# check the importance of each x variable (smaller # of x variables is better)
# - alcohol > volatile.acidity > total.sulfur.dioxide are the most important three variables
varImp(red_wine_model_rpart, scale = TRUE)
red_wine_model_rpart$variable.importance

# predict the wine quality
predicted_class_rpart = predict(red_wine_model_rpart, newdata = validation_x, type = "class")
str(predicted_class_rpart)

# check the performance via CrossTable and confusion matrix
# - Accuracy: 0.6 > NIR (0.456), Kappa = 0.33
CrossTable(predicted_class_rpart, validation_y)
confusionMatrix(predicted_class_rpart, validation_y)
#=============================================================================================================


#=============================================================================================================
# check the red wine quality with C5.0 package for boosting technique
#
# run C5.0 algorithm with 10 trials
red_wine_model_C50 = C5.0(formula = quality ~ ., data = training_data, trials = 20, rules = FALSE)

# check the importance of x variables
red_wine_model_C50

# check the importance of x variables
varImp(red_wine_model_C50, scale = TRUE)

# predict the output of validation x data based on boost model(C5.0) by weighted voting of all trials' predictions
predicted_class_C50 = predict(red_wine_model_C50, newdata = validation_x, type = "class")

# check the performance of classification tree with C5.0 package
# - Accuracy = 0.70.3 > NIR (0.456), Kappa = 0.51.5
CrossTable(predicted_class_C50, validation_y)
confusionMatrix(predicted_class_C50, validation_y)
#=============================================================================================================


#=============================================================================================================
# check the red wine quality with OneR and JRip
#
# run the OneR algorithm
red_wine_model_1r = OneR(quality ~ ., data = training_data)

# check the result with a single rule
# one simple rule to classify the wine quality is alcohol.
#  - About 56.8% of training data are correctly classified based on single alcohol variable
summary(red_wine_model_1r)
red_wine_model_1r

# check the performance of one R
# - Accuracy: 0.572 > NIR (0.456), Kappa = 0.272
predicted_class_1r = predict(red_wine_model_1r, validation_x)
confusionMatrix(predicted_class_1r, validation_y)


# run the JRip algorithm
red_wine_model_jrip = JRip(quality ~ ., data = training_data)

# check the result with major rules
#  - About 65.1% of training data are correctly classified based on 11 rules
summary(red_wine_model_jrip)
red_wine_model_jrip

# check the performance of one R
# - Accuracy: 0.575 > NIR (0.456), Kappa = 0.315
predicted_class_jrip = predict(red_wine_model_jrip, validation_x)
confusionMatrix(predicted_class_jrip, validation_y)
#=============================================================================================================


#=============================================================================================================
# check the red wine quality with naive bayes algorithm
#
# run naive Bayes algorithm with 10 trials
red_wine_model_NB = naiveBayes(formula = quality ~ ., data = training_data)

# predict the output of validation x data based on naive bayes algorithm
predicted_class_NB = predict(red_wine_model_NB, validation_x, type = "class")

# check the performance of naive bayes
# - Accuracy = 0.569 > NIR (0.456), Kappa = 0.336
CrossTable(predicted_class_NB, validation_y)
confusionMatrix(predicted_class_NB, validation_y)
#=============================================================================================================


#=============================================================================================================
# check the red wine quality with svm algorithm
#
# run the support vector machine
red_wine_model_svm_linear = svm(quality ~ ., data = training_data, method = "C-classification", kernel = "linear")
red_wine_model_svm_radial = svm(quality ~ ., data = training_data, method = "C-classification", kernel = "radial")
red_wine_model_svm_polynomial = svm(quality ~ ., data = training_data, method = "C-classification", kernel = "polynomial")

# check the basic summary of all three kernels
red_wine_model_svm_linear
red_wine_model_svm_radial
red_wine_model_svm_polynomial

# check the performance of support vector machine
# - Accuracy: 0.61 > NIR (0.456), Kappa = 0.32
predicted_class_svm_linear = predict(red_wine_model_svm_linear, validation_x)
confusionMatrix(predicted_class_svm_linear, validation_y)

# - Accuracy: 0.644 > NIR (0.456), Kappa = 0.397
predicted_class_svm_radial = predict(red_wine_model_svm_radial, validation_x)
confusionMatrix(predicted_class_svm_radial, validation_y)

# - Accuracy: 0.638 > NIR (0.456), Kappa = 0.39
predicted_class_svm_polynomial = predict(red_wine_model_svm_polynomial, validation_x)
confusionMatrix(predicted_class_svm_polynomial, validation_y)
#=============================================================================================================


#=============================================================================================================
# check the red wine quality with ksvm algorithm
red_wine_model_ksvm_linear = ksvm(quality ~ ., data = training_data, type = "C-svc", kernel = "vanilladot")
red_wine_model_ksvm_radial = ksvm(quality ~ ., data = training_data, type = "C-svc", kernel = "rbfdot")
red_wine_model_ksvm_polynomial = ksvm(quality ~ ., data = training_data, type = "C-svc", kernel = "polydot")

# check the # of support vectors that are located in the edge of marginal area for each hyperplane type
red_wine_model_ksvm_linear
red_wine_model_ksvm_radial
red_wine_model_ksvm_polynomial

# check the predicted results for each kernel
# - Accuracy: 0.61 > NIR (0.456), Kappa = 0.334
predicted_class_ksvm_linear = predict(red_wine_model_ksvm_linear, validation_x)
confusionMatrix(predicted_class_ksvm_linear, validation_y)

# - Accuracy: 0.653 > NIR (0.456), Kappa = 0.411
predicted_class_ksvm_radial = predict(red_wine_model_ksvm_radial, validation_x)
confusionMatrix(predicted_class_ksvm_radial, validation_y)

# - Accuracy: 0.61 > NIR (0.456), Kappa = 0.324
predicted_class_ksvm_polynomial = predict(red_wine_model_ksvm_polynomial, validation_x)
confusionMatrix(predicted_class_ksvm_polynomial, validation_y)
#=============================================================================================================


#=============================================================================================================
# training with best algorithm (C5.0)
#
# set training control with booting + cross validation
red_wine_train_control = trainControl(method = "repeatedcv", 
                                      number = 10, index = createFolds(training_data$quality, 10), 
                                      selectionFunction = "best", savePredictions = TRUE)

# set tune grid with boosting
# - found that trials = 16 is the optimal trials
red_wine_tune_grid = expand.grid(.trials = c(5, 10, 16, 20, 25), .model = "tree", .winnow = FALSE)

# training with C5.0 boosting
red_wine_quality_training_repeatedcv = train(quality ~ ., data = training_data, 
                                             method = "C5.0", 
                                             metric = "Accuracy", 
                                             trControl = red_wine_train_control, 
                                             tuneGrid = red_wine_tune_grid)
# check the training result
red_wine_quality_training_repeatedcv

# prediction accuracy is 71.6% (Kappa 53.4%)
predicted_class_C50 = predict(red_wine_quality_training_repeatedcv, validation_x)
confusionMatrix(predicted_class_C50, validation_y)
#=============================================================================================================


##############################################################################################################
# wine quality numerical prediction
# - svm and ksmv with radial kernel is the best numerical prediction (+/-0.3 quality point)
# - read red wine quality data with quality as numerical variable
#=============================================================================================================
# load wine quality data
red_wine_quality = read.csv("winequality-red.csv", header = TRUE)

# change the order of variables, y variable first
red_wine_quality = dplyr::select(red_wine_quality, quality, dplyr::everything())
range(red_wine_quality$quality)

red_wine_quality$quality = as.numeric(red_wine_quality$quality)

# check the NA data and if there is NA, remove it
if(sum(!complete.cases(red_wine_quality)) != 0)
{
  red_wine_quality = na.omit(red_wine_quality)
}

# set tthe sample size 80% of total data
row_size = nrow(red_wine_quality)
col_size = ncol(red_wine_quality)
sample_size = floor(row_size * 0.8)

# get sampling index
set.seed(567)
sample_index = sample(row_size, sample_size)
validation_size = row_size - sample_size

# set training and validation data set without normalization of x variables
# - create normalized training and validation x
training_data = red_wine_quality[sample_index, ]
validation_x = red_wine_quality[-sample_index, 2:col_size]
validation_y = red_wine_quality[-sample_index, 1]

training_x_scaled = as.data.frame(scale(training_data[ , 2:col_size]))
validation_x_scaled = as.data.frame(scale(validation_x))
training_y = training_data$quality
#=============================================================================================================


#=============================================================================================================
# knn regression
#
index = 1
trials = 50
optimized_k = 1
# set the initial optimized error to maximum dummy number
optimized_error = 1000000

# start with k = 1 and run kNN algorithm to find optimized k by comparing Root of Mean Squared Error
for(index in 1:trials) 
{
  temp_predicted_regression_knn = knn(train = training_x_scaled, cl = training_y, test = validation_x_scaled, k = index, prob = TRUE)
  
  # change the factor data to numeric data
  temp_predicted_regression_knn = as.numeric(as.character(temp_predicted_regression_knn))
  
  # compare RMSE = standard error of differences
  temp_error = RMSE(temp_predicted_regression_knn, validation_y)
  
  if(temp_error < optimized_error)
  {
    optimized_error = temp_error
    optimized_k = index
  }
}

# run knn with optimized k
predicted_regression_knn = knn(train = training_x_scaled, cl = training_y, test = validation_x_scaled, k = optimized_k)

# change factor variable predicted_y_knn as numeric variable
predicted_regression_knn = as.numeric(as.character(predicted_regression_knn))

# Mean difference between predicted y and validation y is around 0.69 point in terms of quality(+/- 0.35 point)
RMSE(predicted_regression_knn, validation_y)
#=============================================================================================================


#=============================================================================================================
# rpart regression
#
# numeric prediction with rpart algorithm
red_wine_model_rpart = rpart(formula = quality ~ ., data = training_data, method = "anova")
summary(red_wine_model_rpart)
red_wine_model_rpart$variable.importance
plot(red_wine_model_rpart)
text(red_wine_model_rpart, use.n = TRUE, all = TRUE, cex = 0.7)

# show the RMSE = Root Mean Squared Error = standard deviation of differences between two data set
# Mean difference between predicted y and validation y is around 0.66 point in terms of quality(+/- 0.33 point)
predicted_regression_rpart = predict(red_wine_model_rpart, newdata = validation_x)
RMSE(predicted_regression_rpart, validation_y)
#=============================================================================================================


#=============================================================================================================
# M5P regression
#
red_wine_model_m5p = M5P(quality ~ ., data = training_data)
# plot(red_wine_model_m5p)

# show the RMSE = Root Mean Squared Error = standard deviation of differences between two data set
# Mean difference between predicted y and validation y is around 29 point in terms of quality(+/- 14.5 point)
predicted_regression_m5p = predict(red_wine_model_m5p, newdata = validation_x)
RMSE(predicted_regression_m5p, validation_y)
#=============================================================================================================


#=============================================================================================================
red_wine_model_svm_linear = svm(quality ~ ., data = training_data, method = "eps-regression", kernel = "linear")
red_wine_model_svm_radial = svm(quality ~ ., data = training_data, method = "eps-regression", kernel = "radial")
red_wine_model_svm_polynomial = svm(quality ~ ., data = training_data, method = "eps-regression", kernel = "polynomial")

# check the # of support vectors that are located in the edge of marginal area for each hyperplane type
summary(red_wine_model_svm_linear)
summary(red_wine_model_svm_radial)
summary(red_wine_model_svm_polynomial)

# (7) check the predicted results for each kernel
predicted_regression_svm_linear = predict(red_wine_model_svm_linear, validation_x)

# Mean difference between predicted y and validation y is around 0.66 point in terms of quality(+/- 0.33 point)
RMSE(predicted_regression_svm_linear, validation_y)

predicted_regression_svm_radial = predict(red_wine_model_svm_radial, validation_x)

# Mean difference between predicted y and validation y is around 0.6 point in terms of quality(+/- 0.3 point)
RMSE(predicted_regression_svm_radial, validation_y)

predicted_regression_svm_polynomial = predict(red_wine_model_svm_polynomial, validation_x)

# Mean difference between predicted y and validation y is around 0.93 point in terms of quality(+/- 0.46 point)
RMSE(predicted_regression_svm_polynomial, validation_y)
#=============================================================================================================


#=============================================================================================================
red_wine_model_ksvm_linear = ksvm(quality ~ ., data = training_data, type = "eps-svr", kernel = "vanilladot")
red_wine_model_ksvm_radial = ksvm(quality ~ ., data = training_data, type = "eps-svr", kernel = "rbfdot")
red_wine_model_ksvm_anova = ksvm(quality ~ ., data = training_data, type = "eps-svr", kernel = "anovadot")

# check the # of support vectors that are located in the edge of marginal area for each hyperplane type
red_wine_model_ksvm_linear
red_wine_model_ksvm_radial
red_wine_model_ksvm_anova

# check the predicted results for each kernel
predicted_regression_ksvm_linear = predict(red_wine_model_ksvm_linear, validation_x)

# Mean difference between predicted y and validation y is around 0.66 point in terms of quality(+/- 0.33 point)
RMSE(predicted_regression_ksvm_linear, validation_y)

predicted_regression_ksvm_radial = predict(red_wine_model_ksvm_radial, validation_x)

# Mean difference between predicted y and validation y is around 0.6 point in terms of quality(+/- 0.3 point)
RMSE(predicted_regression_ksvm_radial, validation_y)

predicted_regression_ksvm_anova = predict(red_wine_model_ksvm_anova, validation_x)

# Mean difference between predicted y and validation y is around 8.84 point in terms of quality(+/- 4.42 point)
RMSE(predicted_regression_ksvm_anova, validation_y)
#=============================================================================================================


#=============================================================================================================
# stacking with best 3 algorithm C5.0 > svm Radial > knn
#
# set training control with booting + cross validation
red_wine_train_control = trainControl(method = "repeatedcv", 
                                      number = 20, index = createFolds(training_data$quality, 10), 
                                      selectionFunction = "best", savePredictions = TRUE)

red_wine_quality_models_list = caretList(quality ~ ., training_data,
                                         methodList = c("knn", "rpart", "svmRadial"),
                                         trControl = red_wine_train_control,
                                         metric = "RMSE")

red_wine_quality_models = caretEnsemble(red_wine_quality_models_list)

# Mean difference between predicted y and validation y is around 0.61 point in terms of quality(+/- 0.3 point)
predicted_regression_models = predict(red_wine_quality_models, validation_x)
RMSE(predicted_regression_models, validation_y)
#=============================================================================================================

