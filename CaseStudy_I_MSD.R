# Case Study I: MSD

# Install and load required packages
if (!require(caret)) install.packages("caret")
if (!require(neuralnet)) install.packages("neuralnet")
if (!require(ResourceSelection)) install.packages("ResourceSelection")
if (!require(pROC)) install.packages("pROC")

library(caret)
library(neuralnet)
library(ResourceSelection)
library(pROC)

# STEP 1 - Dataset for the MSD

Input = "
Gender	Marital_status	Education	Handed	Sports_activity	Position_working	Work_environment	MSD	AgeN	ExperienceN	Sleep_hourN	Working_daysN	Working_hoursN	Musculo_problem	BMI
1	1	2	2	1	3	2	2	30	10	6	6	9	2	2.28
1	1	2	2	2	3	2	2	34	6	8	7	7	2	2.90
1	1	2	1	2	2	2	2	29	4	6	3	10	2	1.13
⋮	⋮	⋮	⋮	⋮	⋮	⋮	⋮	⋮	⋮	⋮	⋮	⋮	⋮	⋮
1	1	1	1	2		2	1	35	2	7	6	6	2	2.01
2	1	1	1	1	1	1	2	30	8	4	7	8	2	3.11
2	2	2	1	1	1	2	2	32	14	5	3	5	2	2.28
"
data <- read.table(textConnection(Input), header=TRUE)

# STEP 2 - Checking for Missing Values
apply(data, 2, function(x) sum(is.na(x)))

# STEP 3 - Normalization (if needed)
normalize <- function(x) { return ((x - min(x)) / (max(x) - min(x))) }
maxmindf <- as.data.frame(lapply(data, normalize))

# STEP 4 - Split the Dataset (70% for Training and 30% for Testing)
set.seed(123)  # Set seed for reproducibility
index <- sample(1:nrow(maxmindf), 0.7 * nrow(maxmindf))
Training <- maxmindf[index, ]
Testing <- maxmindf[-index, ]

# Logistic Regression
# STEP 5 - Apply Multiple Logistic Regression

# Use 'MSD' as the dependent variable for logistic regression
Training$MSD <- as.factor(Training$MSD)  # Convert 'MSD' to factor for logistic regression

# Apply the logistic regression model
logistic_model <- glm(MSD ~ Gender + Marital_status + Education + Sports_activity + Work_environment + 
                        AgeN + Position_working + ExperienceN + Sleep_hourN + 
                        Working_daysN + Working_hoursN + Musculo_problem+BMI, 
                      data = Training, family = binomial)

# Summarize the logistic regression model
summary(logistic_model)

# STEP 6 - Predict the Outcomes on the Testing Set
logistic_predictions <- predict(logistic_model, newdata = Testing, type = "response")

# STEP 7 - Convert Probabilities to Binary Outcomes (Threshold: 0.5)
predicted_classes <- ifelse(logistic_predictions > 0.5, 1, 0)

# STEP 8 - Model Accuracy: Compare Predicted vs Actual Results
actual_classes <- Testing$MSD

# Generate confusion matrix
confusion_matrix <- table(Predicted = predicted_classes, Actual = actual_classes)
print(confusion_matrix)

# STEP 9 - Calculate Accuracy
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
print(paste("Logistic Regression Accuracy: ", round(accuracy * 100, 2), "%", sep = ""))

# Mean Squared Error for Logistic Regression Predictions
# Calculate the Mean Squared Error using the predicted probabilities
MSE_logistic <- mean((Testing$MSD - logistic_predictions)^2)
print(paste("Logistic Regression MSE: ", MSE_logistic))

# You can also calculate other metrics like Precision, Recall, and F1-Score
conf_matrix <- confusionMatrix(as.factor(predicted_classes), as.factor(actual_classes))
print(conf_matrix)

# AUC for Logistic Regression
roc_logistic <- roc(Testing$MSD, logistic_predictions)
auc_logistic <- auc(roc_logistic)
print(paste("Logistic Regression AUC: ", auc_logistic))

# Perform Hosmer-Lemeshow Test
hl_test <- hoslem.test(Training$MSD, fitted(logistic_model), g=10)
print(hl_test)

# Perform Brier Score for Logistic Regression
brier_score_logistic <- mean((logistic_predictions - Testing$MSD)^2)
print(paste("Logistic Regression Brier Score: ", brier_score_logistic))

#################### Multilayer Perceptron (Neural Network) ####################

# STEP 10 - Max-Min Data Normalization
maxmindf <- as.data.frame(lapply(data, normalize))

# STEP 11-Determine the Training and Testing of the Dataset
Training <- maxmindf[1:round(0.7 * nrow(maxmindf)), ]
Testing <- maxmindf[(round(0.7 * nrow(maxmindf)) + 1):nrow(maxmindf), ]

# STEP 12-MLR-ENHANCED MLP MODEL
    enhanced_train_cv <- train_data_cv
    enhanced_test_cv  <- test_data_cv
    
    # Inner 3-fold CV for out-of-fold MLR probability
    inner_folds <- caret::createFolds(
      Training_mlr_cv$MSD,
      k = 3,
      list = TRUE
    )
    inner_oof_prob <- rep(NA, nrow(Training_mlr_cv))
    for (j in seq_along(inner_folds)) {
      inner_valid_idx <- inner_folds[[j]]
      inner_train_idx <- setdiff(
        seq_len(nrow(Training_mlr_cv)),
        inner_valid_idx
      )
      inner_train <- Training_mlr_cv[inner_train_idx, , drop = FALSE]
      inner_valid <- Training_mlr_cv[inner_valid_idx, , drop = FALSE]
       inner_mlr_model <- glm(
        formula = mlr_formula,
        data = inner_train,
        family = binomial
      )
         inner_pred <- safe_predict_glm(
        fitted_glm = inner_mlr_model,
        newdata = inner_valid
      )
      
      inner_oof_prob[inner_valid_idx] <- inner_pred
    }
    if (any(is.na(inner_oof_prob))) {
      inner_oof_prob[is.na(inner_oof_prob)] <- mean(inner_oof_prob, na.rm = TRUE)
    }
     enhanced_train_cv$MLR_probability <- inner_oof_prob
     enhanced_test_cv$MLR_probability <- safe_predict_glm(
      fitted_glm = mlr_cv_model,
      newdata = Testing_mlr_cv
    )
    
        enhanced_cv_predictors <- setdiff(
      names(enhanced_train_cv),
      "MSD"
    )
    for (col in enhanced_cv_predictors) {
      enhanced_train_cv[[col]] <- as.numeric(enhanced_train_cv[[col]])
      enhanced_test_cv[[col]]  <- as.numeric(enhanced_test_cv[[col]])
    }
    for (col in enhanced_cv_predictors) {
      finite_values <- enhanced_train_cv[[col]][
        is.finite(enhanced_train_cv[[col]])
      ]
      replacement_value <- ifelse(
        length(finite_values) > 0,
        median(finite_values, na.rm = TRUE),
        0
      )
      
      enhanced_train_cv[[col]][
        !is.finite(enhanced_train_cv[[col]])
      ] <- replacement_value
      enhanced_test_cv[[col]][
        !is.finite(enhanced_test_cv[[col]])
      ] <- replacement_value
    }
    predictor_variance <- sapply(
      enhanced_train_cv[, enhanced_cv_predictors, drop = FALSE],
      function(x) var(x, na.rm = TRUE)
    )
    zero_variance_cols <- names(predictor_variance)[
      is.na(predictor_variance) | predictor_variance == 0
    ]
    if (length(zero_variance_cols) > 0) {
      enhanced_train_cv <- enhanced_train_cv[
        ,
        !names(enhanced_train_cv) %in% zero_variance_cols,
        drop = FALSE
      ]
      
      enhanced_test_cv <- enhanced_test_cv[
        ,
        !names(enhanced_test_cv) %in% zero_variance_cols,
        drop = FALSE
      ]
    }
    enhanced_cv_predictors <- setdiff(
      names(enhanced_train_cv),
      "MSD"
    )
    enhanced_cv_formula <- as.formula(
      paste("MSD ~", paste(enhanced_cv_predictors, collapse = " + "))
    )
    # Train MLR-enhanced MLP using hidden = c(5,5)
    enhanced_fit <- safe_train_nn(
      formula = enhanced_cv_formula,
      data = enhanced_train_cv,
      seed = 321 + i
    )
    enhanced_cv_prob <- safe_compute_nn(
      nn_fit = enhanced_fit,
      newdata = enhanced_test_cv[, enhanced_cv_predictors, drop = FALSE]
    )
    if (all(is.na(enhanced_cv_prob))) {
      enhanced_accuracy <- NA
      
    } else {
      enhanced_cv_eval <- get_binary_metrics(
        actual = enhanced_test_cv$MSD,
        predicted_prob = enhanced_cv_prob,
        threshold = 0.5,
        model_name = "MLR-enhanced MLP CV"
      )
      enhanced_accuracy <- enhanced_cv_eval$metrics$Accuracy
    }    
    
    cv_results <- rbind(
      cv_results,
      data.frame(
        Resample = resample_name,
        MLR_Accuracy = mlr_cv_eval$metrics$Accuracy,
        Standalone_MLP_Accuracy = mlp_accuracy,
        MLR_Enhanced_MLP_Accuracy = enhanced_accuracy
      )
    )
  }
  return(cv_results)
}


cat("\nObjects containing 'hidden':\n")
print(ls(pattern = "hidden"))
if (exists("best_hidden")) {
  best_hidden_msd <- best_hidden
} else {
  stop("Object 'best_hidden' does not exist. Please run your MSD hidden-layer tuning code first, or manually set best_hidden_msd to the selected best hidden structure.")
}
cat("\nBest hidden structure used for MSD repeated CV:\n")
print(best_hidden_msd)

# STEP 13-RUN REPEATED 5-FOLD CV COMPARISON FOR MSD
cv_results_5x10_msd <- run_repeated_cv_comparison_msd(
  raw_data = data_msd,
  data_mlr_ready = data_mlr_msd,
  k = 5,
  repeats = 10,
  hidden_structure = best_hidden_msd
)
print(cv_results_5x10_msd)
if (!exists("cv_results_5x10_msd")) {
  stop("cv_results_5x10_msd does not exist. Please run Section 19B successfully first.")
}
cat("\nPreview of cv_results_5x10_msd:\n")
print(head(cv_results_5x10_msd))
# Remove failed folds, if any neural network fold failed
cv_results_final_msd <- cv_results_5x10_msd[
  complete.cases(cv_results_5x10_msd),
]
cat("\nNumber of total repeated CV folds:", nrow(cv_results_5x10_msd), "\n")
cat("Number of completed repeated CV folds:", nrow(cv_results_final_msd), "\n")
cat("Number of removed failed folds:", nrow(cv_results_5x10_msd) - nrow(cv_results_final_msd), "\n")
if (nrow(cv_results_final_msd) < 2) {
  stop("Too few completed folds for paired t-test. Check neural network convergence or hidden structure.")
}
print(cv_results_final_msd)
cv_results_final_msd <- cv_results_5x10_msd[
  complete.cases(cv_results_5x10_msd),
]
cat("\nNumber of completed CV folds:", nrow(cv_results_final_msd), "\n")
if (nrow(cv_results_final_msd) < 2) {
  stop("Too few completed folds for paired t-test.")
}

# STEP 14 - Model Validation and Accuracy Calculation for Neural Network

# Rescale the predictions back to the original range (if necessary)
predicted1 <- nn.results$net.result * abs(diff(range(data$MSD))) + min(data$MSD)

# Use the actual values from the Testing set
actual1 <- Testing$MSD * abs(diff(range(data$MSD))) + min(data$MSD)

# Calculate the deviation and mean absolute error
deviation <- (actual1 - predicted1)
value <- abs(mean(deviation))

# Calculate accuracy in percentage
accuracy_in_percent <- (1 - value) * 100
print(paste("Neural Network Accuracy: ", accuracy_in_percent, "%"))

# STEP 15 - Mean Squared Error for Neural Network Predictions
predicted <- nn.results$net.result
MSE.net <- sum((Testing$MSD - predicted)^2) / nrow(Testing)
print(paste("Neural Network MSE: ", MSE.net))

# Calculate Precision, Recall, and F1-Score using confusionMatrix from the caret package
conf_matrix_mlp <- confusionMatrix(as.factor(predicted_classes_mlp), as.factor(actual_classes_mlp))
print(conf_matrix_mlp)

# AUC for Neural Network
roc_nn <- roc(Testing$MSD, nn.results$net.result)
auc_nn <- auc(roc_nn)
print(paste("Neural Network AUC: ", auc_nn))

# Perform Brier Score for Neural Network
brier_score_mlp <- mean((predicted_probabilities_mlp - Testing$MSD)^2)
print(paste("Neural Network Brier Score: ", brier_score_mlp))

#AIC
# Number of parameters in the neural network (weights + biases)
num_hidden_layers <- c(5, 5)  # This is the structure of the hidden layers
num_inputs <- ncol(Temp_test)  # Number of input features
num_outputs <- 1  # Binary output (MSD)

# Total number of parameters (weights + biases)
num_weights <- num_inputs * num_hidden_layers[1] + 
  num_hidden_layers[1] * num_hidden_layers[2] + 
  num_hidden_layers[2] * num_outputs +
  sum(num_hidden_layers)  # Adding the biases

# Sum of Squared Errors (SSE)
SSE <- sum((Testing$MSD - predicted_probabilities_mlp)^2)

# Log-likelihood approximation (since MSE is closely related to SSE, we use this for a rough estimate)
log_likelihood <- -SSE / 2

# Calculate AIC for Neural Network
aic_nn <- 2 * num_weights - 2 * log_likelihood
print(paste("Neural Network AIC: ", aic_nn))

#Step 16-ROC Curve

# Plot ROC Curve for Logistic Regression
plot(roc_logistic, col = "blue", lwd = 2, main = "ROC Curve for Logistic Regression")
text(0.6, 0.4, labels = paste("AUC:", round(auc_logistic, 2)), col = "blue")

# Plot ROC Curve for Neural Network
plot(roc_nn, col = "red", lwd = 2, main = "ROC Curve for Neural Network")
text(0.6, 0.4, labels = paste("AUC:", round(auc_nn, 2)), col = "red")

# Overlay both ROC Curves
plot(roc_logistic, col = "blue", lwd = 2, main = "ROC Curves for Logistic Regression and Neural Network-MSD")
plot(roc_nn, col = "red", lwd = 2, add = TRUE)
legend("bottomright", legend = c("Logistic Regression-MSD", "Neural Network-MSD"), col = c("blue", "red"), lwd = 2)

#Step 17-apply McNemar's and Paired T-Test to determine whether the differences in prediction accuracy between the Multiple Logistic Regression (MLR) and the MLR + Multilayer Perceptron (MLP) models are statistically significant#
# Assuming you already have predicted_classes_mlr and predicted_classes_mlp
threshold <- 0.5  # Change this to optimize
predicted_classes_mlr <- ifelse(logistic_predictions > threshold, 1, 0)
predicted_classes_mlp <- ifelse(predicted_probabilities_mlp > threshold, 1, 0)

#Create the McNemar's table
mc_table <- table(MLR = predicted_classes_mlr, MLP = predicted_classes_mlp)
print(mc_table)

#Apply McNemar's test
mcnemar_result <- mcnemar.test(mc_table)

# Print the test result
print(mcnemar_result)

# Ensure the MSD column is a factor for classification
Training$MSD <- as.factor(Training$MSD)

# Set up 3-fold cross-validation
set.seed(123)
train_control <- trainControl(method = "cv", number = 3)

# Perform 3-fold cross-validation for MLR using caret
mlr_cv_model <- train(MSD ~ Gender + Marital_status + Education + Sports_activity + Work_environment + 
                        AgeN + Position_working + ExperienceN + Sleep_hourN + 
                        Working_daysN + Working_hoursN + Musculo_problem + BMI, 
                      data = Training, method = "glm", family = "binomial", 
                      trControl = train_control)

# Perform 3-fold cross-validation for MLP using caret
mlp_cv_model <- train(MSD ~ Gender + Marital_status + Education + Sports_activity + Work_environment + 
                        AgeN + Position_working + ExperienceN + Sleep_hourN + 
                        Working_daysN + Working_hoursN + Musculo_problem + BMI, 
                      data = Training, method = "nnet", linout = FALSE, trace = FALSE, 
                      trControl = train_control)

# Extract accuracy from resampling results
accuracy_mlr_cv <- mlr_cv_model$resample$Accuracy
accuracy_mlp_cv <- mlp_cv_model$resample$Accuracy

# Check the length of the accuracy vectors to ensure they match
print(length(accuracy_mlr_cv))
print(length(accuracy_mlp_cv))

# Perform paired t-test on cross-validated accuracies
paired_ttest_result_cv <- t.test(accuracy_mlr_cv, accuracy_mlp_cv, paired = TRUE)

# Print the paired t-test result
print(paired_ttest_result_cv)
