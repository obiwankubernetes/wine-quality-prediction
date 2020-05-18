# Group Project for TO567 @ UMich Ross
##############################################################################################################
# wine quality classification
# C5.0 with 21 trials is the best model to classify wine quality (71% accuracy)
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
white_wine_quality = read.csv("winequality-white.csv", header = TRUE)

# change the order of variables, y variable first
white_wine_quality = dplyr::select(white_wine_quality, quality, everything())
range(white_wine_quality$quality)

# set the quality variable as a factor variable
white_wine_quality$quality = factor(white_wine_quality$quality, labels = c(3:9), levels = (3:9))
str(white_wine_quality$quality)

# check the NA data and if there is NA, remove it
if(sum(!complete.cases(white_wine_quality)) != 0)
{
  white_wine_quality = na.omit(white_wine_quality)
}

# set tthe sample size 80% of total data
row_size = nrow(white_wine_quality)
col_size = ncol(white_wine_quality)
sample_size = floor(row_size * 0.9)

# get sampling index
set.seed(567)
sample_index = sample(row_size, sample_size)
validation_size = row_size - sample_size

# set training and validation data set after normalizing all the values of x variables
training_x_scaled = as.data.frame(scale(white_wine_quality[sample_index, 2:col_size]))
training_y = white_wine_quality[sample_index, 1]
validation_x_scaled = as.data.frame(scale(white_wine_quality[-sample_index, 2:col_size]))
validation_y = white_wine_quality[-sample_index, 1]

#=============================================================================================================
# check the white wine quality with knn algorithm
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
# - accuracy = 0.663 > NIR (0.437), Kappa = 0.507
# - lack of data which have quality 3 and 9, accuracy is not good
CrossTable(predicted_class_knn, validation_y, prop.chisq = FALSE)
confusionMatrix(predicted_class_knn, validation_y)
#=============================================================================================================


#=============================================================================================================
# check the white wine quality with rpart algorithm
# 
# set training and validation data set without normalization of x variables
training_data = white_wine_quality[sample_index, ]
validation_x = white_wine_quality[-sample_index, 2:col_size]
validation_y = white_wine_quality[-sample_index, 1]

# classification prediction
white_wine_model_rpart = rpart(formula = quality ~ ., data = training_data, method = "class", control = rpart.control(minsplit = 100, cp = 0.01))
summary(white_wine_model_rpart)

plot(white_wine_model_rpart, uniform = TRUE)
text(white_wine_model_rpart, use.n = TRUE, all = TRUE, cex = 0.7)

# check the cp table
# - 0.02 could be the point to be pruned
white_wine_model_rpart$cptable
plot(white_wine_model_rpart$cptable, col = "red")

# check the importance of each x variable (smaller # of x variables is better)
# - alcohol > volatile.acidity > total.sulfur.dioxide are the most important three variables
varImp(white_wine_model_rpart, scale = TRUE)
white_wine_model_rpart$variable.importance

# predict the wine quality
predicted_class_rpart = predict(white_wine_model_rpart, newdata = validation_x, type = "class")
str(predicted_class_rpart)

# check the performance via CrossTable and confusion matrix
# - Accuracy: 0.567 > NIR (0.437), Kappa = 0.309
CrossTable(predicted_class_rpart, validation_y)
confusionMatrix(predicted_class_rpart, validation_y)
#=============================================================================================================


#=============================================================================================================
# check the white wine quality with C5.0 package for boosting technique
#
# run C5.0 algorithm with 10 trials
white_wine_model_C50 = C5.0(formula = quality ~ ., data = training_data, trials = 21, rules = FALSE)

# check the importance of x variables
white_wine_model_C50

# check the importance of x variables
varImp(white_wine_model_C50, scale = TRUE)

# predict the output of validation x data based on boost model(C5.0) by weighted voting of all trials' predictions
predicted_class_C50 = predict(white_wine_model_C50, newdata = validation_x, type = "class")

# check the performance of classification tree with C5.0 package
# - Accuracy = 0.706 > NIR (0.437), Kappa = 0.556
CrossTable(predicted_class_C50, validation_y)
confusionMatrix(predicted_class_C50, validation_y)
#=============================================================================================================


#=============================================================================================================
# check the white wine quality with OneR and JRip
#
# run the OneR algorithm
white_wine_model_1r = OneR(quality ~ ., data = training_data)

# check the result with a single rule
# one simple rule to classify the wine quality is alcohol.
#  - About 56.8% of training data are correctly classified based on single alcohol variable
summary(white_wine_model_1r)
white_wine_model_1r

# check the performance of one R
# - Accuracy: 0.437 > NIR (0.437), Kappa = 0.102
predicted_class_1r = predict(white_wine_model_1r, validation_x)
confusionMatrix(predicted_class_1r, validation_y)


# run the JRip algorithm
white_wine_model_jrip = JRip(quality ~ ., data = training_data)

# check the result with major rules
#  - About 65.1% of training data are correctly classified based on 11 rules
summary(white_wine_model_jrip)
white_wine_model_jrip

# check the performance of one R
# - Accuracy: 0.569 > NIR (0.437), Kappa = 0.32
predicted_class_jrip = predict(white_wine_model_jrip, validation_x)
confusionMatrix(predicted_class_jrip, validation_y)
#=============================================================================================================


#=============================================================================================================
# check the red wine quality with naive bayes algorithm
#
# run naive Bayes algorithm with 10 trials
white_wine_model_NB = naiveBayes(formula = quality ~ ., data = training_data)

# predict the output of validation x data based on naive bayes algorithm
predicted_class_NB = predict(white_wine_model_NB, validation_x, type = "class")

# check the performance of naive bayes
# - Accuracy = 0.48 > NIR (0.437), Kappa = 0.27
CrossTable(predicted_class_NB, validation_y)
confusionMatrix(predicted_class_NB, validation_y)
#=============================================================================================================


#=============================================================================================================
# check the white wine quality with support vector machine algorithm
#
# run the support vector machine
white_wine_model_svm_linear = svm(quality ~ ., data = training_data, method = "C-classification", kernel = "linear")
white_wine_model_svm_radial = svm(quality ~ ., data = training_data, method = "C-classification", kernel = "radial")
white_wine_model_svm_polynomial = svm(quality ~ ., data = training_data, method = "C-classification", kernel = "polynomial")

# check the basic summary of all three kernels
white_wine_model_svm_linear
white_wine_model_svm_radial
white_wine_model_svm_polynomial

# check the performance of support vector machine
# - Accuracy: 0.525 > NIR (0.437), Kappa = 0.207
predicted_class_svm_linear = predict(white_wine_model_svm_linear, validation_x)
confusionMatrix(predicted_class_svm_linear, validation_y)

# - Accuracy: 0.560 > NIR (0.437), Kappa = 0.359
predicted_class_svm_radial = predict(white_wine_model_svm_radial, validation_x)
confusionMatrix(predicted_class_svm_radial, validation_y)

# - Accuracy: 0.537 > NIR (0.437), Kappa = 0.245
predicted_class_svm_polynomial = predict(white_wine_model_svm_polynomial, validation_x)
confusionMatrix(predicted_class_svm_polynomial, validation_y)
#=============================================================================================================


#=============================================================================================================
# check the white wine quality with ksvm algorithm
white_wine_model_ksvm_linear = ksvm(quality ~ ., data = training_data, type = "C-svc", kernel = "vanilladot")
white_wine_model_ksvm_radial = ksvm(quality ~ ., data = training_data, type = "C-svc", kernel = "rbfdot")
white_wine_model_ksvm_polynomial = ksvm(quality ~ ., data = training_data, type = "C-svc", kernel = "polydot")

# check the # of support vectors that are located in the edge of marginal area for each hyperplane type
white_wine_model_ksvm_linear
white_wine_model_ksvm_radial
white_wine_model_ksvm_polynomial

# check the predicted results for each kernel
# - Accuracy: 0.525 > NIR (0.437), Kappa = 0.207
predicted_class_ksvm_linear = predict(white_wine_model_ksvm_linear, validation_x)
confusionMatrix(predicted_class_ksvm_linear, validation_y)

# - Accuracy: 0.588 > NIR (0.437), Kappa = 0.344
predicted_class_ksvm_radial = predict(white_wine_model_ksvm_radial, validation_x)
confusionMatrix(predicted_class_ksvm_radial, validation_y)

# - Accuracy: 0.525 > NIR (0.437), Kappa = 0.207
predicted_class_ksvm_polynomial = predict(white_wine_model_ksvm_polynomial, validation_x)
confusionMatrix(predicted_class_ksvm_polynomial, validation_y)
#=============================================================================================================


#=============================================================================================================
# training with best algorithm (C5.0)
#
# set training control with booting + cross validation
white_wine_train_control = trainControl(method = "repeatedcv", 
                                        number = 10, index = createFolds(training_data$quality, 10), 
                                        selectionFunction = "best", savePredictions = TRUE)

# set tune grid with boosting
# - found that trials = 16 is the optimal trials
white_wine_tune_grid = expand.grid(.trials = c(20:30), .model = "tree", .winnow = FALSE)

# training with C5.0 boosting
white_wine_quality_training_repeatedcv = train(quality ~ ., data = training_data, 
                                               method = "C5.0", 
                                               metric = "Accuracy", 
                                               trControl = white_wine_train_control, 
                                               tuneGrid = white_wine_tune_grid)
# check the training result
white_wine_quality_training_repeatedcv

# prediction accuracy is 70.6% (Kappa 55.7%)
predicted_class_C50 = predict(white_wine_quality_training_repeatedcv, validation_x)
confusionMatrix(predicted_class_C50, validation_y)
#=============================================================================================================



##############################################################################################################
# wine quality numerical prediction
# svm and ksmv with radial kernel is the best numerical prediction (+/-0.35 quality point)
# - read white wine quality data with quality as numerical variable
#=============================================================================================================
# load wine quality data
white_wine_quality = read.csv("winequality-white.csv", header = TRUE)

# change the order of variables, y variable first
white_wine_quality = dplyr::select(white_wine_quality, quality, dplyr::everything())
range(white_wine_quality$quality)

white_wine_quality$quality = as.numeric(white_wine_quality$quality)

# check the NA data and if there is NA, remove it
if(sum(!complete.cases(white_wine_quality)) != 0)
{
  white_wine_quality = na.omit(white_wine_quality)
}

# set tthe sample size 80% of total data
row_size = nrow(white_wine_quality)
col_size = ncol(white_wine_quality)
sample_size = floor(row_size * 0.8)

# get sampling index
set.seed(567)
sample_index = sample(row_size, sample_size)
validation_size = row_size - sample_size

# set training and validation data set without normalization of x variables
# - create normalized training and validation x
training_data = white_wine_quality[sample_index, ]
validation_x = white_wine_quality[-sample_index, 2:col_size]
validation_y = white_wine_quality[-sample_index, 1]

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

# Mean difference between predicted y and validation y is around 0.78 point in terms of quality(+/- 0.39 point)
RMSE(predicted_regression_knn, validation_y)
#=============================================================================================================


#=============================================================================================================
# rpart regression
#
# numeric prediction with rpart algorithm
white_wine_model_rpart = rpart(formula = quality ~ ., data = training_data, method = "anova")
summary(white_wine_model_rpart)
white_wine_model_rpart$variable.importance
plot(white_wine_model_rpart)
text(white_wine_model_rpart, use.n = TRUE, all = TRUE, cex = 0.7)

# show the RMSE = Root Mean Squared Error = standard deviation of differences between two data set
# Mean difference between predicted y and validation y is around 0.78 point in terms of quality(+/- 0.39 point)
predicted_regression_rpart = predict(white_wine_model_rpart, newdata = validation_x)
RMSE(predicted_regression_rpart, validation_y)
#=============================================================================================================


#=============================================================================================================
# M5P regression
#
white_wine_model_m5p = M5P(quality ~ ., data = training_data)
# plot(white_wine_model_m5p)

# show the RMSE = Root Mean Squared Error = standard deviation of differences between two data set
# Mean difference between predicted y and validation y is around 118.39 point in terms of quality(+/- 59.2 point)
predicted_regression_m5p = predict(white_wine_model_m5p, newdata = validation_x)
RMSE(predicted_regression_m5p, validation_y)
#=============================================================================================================


#=============================================================================================================
white_wine_model_svm_linear = svm(quality ~ ., data = training_data, method = "eps-regression", kernel = "linear")
white_wine_model_svm_radial = svm(quality ~ ., data = training_data, method = "eps-regression", kernel = "radial")
white_wine_model_svm_polynomial = svm(quality ~ ., data = training_data, method = "eps-regression", kernel = "polynomial")

# check the # of support vectors that are located in the edge of marginal area for each hyperplane type
summary(white_wine_model_svm_linear)
summary(white_wine_model_svm_radial)
summary(white_wine_model_svm_polynomial)

# (7) check the predicted results for each kernel
predicted_regression_svm_linear = predict(white_wine_model_svm_linear, validation_x)

# Mean difference between predicted y and validation y is around 0.79 point in terms of quality(+/- 0.39 point)
RMSE(predicted_regression_svm_linear, validation_y)

predicted_regression_svm_radial = predict(white_wine_model_svm_radial, validation_x)

# Mean difference between predicted y and validation y is around 0.71 point in terms of quality(+/- 0.35 point)
RMSE(predicted_regression_svm_radial, validation_y)

predicted_regression_svm_polynomial = predict(white_wine_model_svm_polynomial, validation_x)

# Mean difference between predicted y and validation y is around 0.83 point in terms of quality(+/- 0.41 point)
RMSE(predicted_regression_svm_polynomial, validation_y)
#=============================================================================================================


#=============================================================================================================
white_wine_model_ksvm_linear = ksvm(quality ~ ., data = training_data, type = "eps-svr", kernel = "vanilladot")
white_wine_model_ksvm_radial = ksvm(quality ~ ., data = training_data, type = "eps-svr", kernel = "rbfdot")
white_wine_model_ksvm_anova = ksvm(quality ~ ., data = training_data, type = "eps-svr", kernel = "anovadot")

# check the # of support vectors that are located in the edge of marginal area for each hyperplane type
white_wine_model_ksvm_linear
white_wine_model_ksvm_radial
white_wine_model_ksvm_anova

# check the predicted results for each kernel
predicted_regression_ksvm_linear = predict(white_wine_model_ksvm_linear, validation_x)

# Mean difference between predicted y and validation y is around 0.79 point in terms of quality(+/- 0.39 point)
RMSE(predicted_regression_ksvm_linear, validation_y)

predicted_regression_ksvm_radial = predict(white_wine_model_ksvm_radial, validation_x)

# Mean difference between predicted y and validation y is around 0.72 point in terms of quality(+/- 0.36 point)
RMSE(predicted_regression_ksvm_radial, validation_y)

predicted_regression_ksvm_anova = predict(white_wine_model_ksvm_anova, validation_x)

# Mean difference between predicted y and validation y is around 13.6 point in terms of quality(+/- 6.8 point)
RMSE(predicted_regression_ksvm_anova, validation_y)
#=============================================================================================================


#=============================================================================================================
# stacking with best 3 algorithm C5.0 > svm Radial > knn
#
# set training control with booting + cross validation
white_wine_train_control = trainControl(method = "repeatedcv", 
                                        number = 20, index = createFolds(training_data$quality, 10), 
                                        selectionFunction = "best", savePredictions = TRUE)

white_wine_quality_models_list = caretList(quality ~ ., training_data,
                                           methodList = c("knn", "rpart", "svmRadial"),
                                           trControl = white_wine_train_control,
                                           metric = "RMSE")

white_wine_quality_models = caretEnsemble(white_wine_quality_models_list)

# Mean difference between predicted y and validation y is around 0.72 point in terms of quality(+/- 0.36 point)
predicted_regression_models = predict(white_wine_quality_models, validation_x)
RMSE(predicted_regression_models, validation_y)
#=============================================================================================================

