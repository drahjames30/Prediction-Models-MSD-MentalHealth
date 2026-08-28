# Case Study II: Mental health

# Install and load required packages
if (!require(caret)) install.packages("caret")
if (!require(neuralnet)) install.packages("neuralnet")
if (!require(ResourceSelection)) install.packages("ResourceSelection")
if (!require(pROC)) install.packages("pROC")

library(caret)
library(neuralnet)
library(ResourceSelection)
library(pROC)

# STEP 1 - Dataset for the Mental Health
Input = "
Gender Age MS EL PS SES Religion Residence Income Mental History1 History2 Duration
2	2	2	2	1	2	1	1	1	2	2	2	2
1	2	1	2	1	2	1	1	4	2	1	1	3
1	3	1	3	3	2	1	2	1	2	2	2	3
⋮	⋮	⋮	⋮	⋮	⋮	⋮	⋮	⋮	⋮	⋮	⋮	⋮
1	2	1	4	1	2	1	4	4	2	2	2	2
1	3	1	4	1	2	1	3	4	2	2	2	2
2	2	2	4	1	2	3	3	2	1	2	1	2
"
data <- read.table(textConnection(Input), header=TRUE)

# STEP 2 - Checking for Missing Values
apply(data, 2, function(x) sum(is.na(x)))

# STEP 3 - Normalization (if needed)
normalize <- function(x) {return ((x - min(x)) / (max(x) - min(x)))}
maxmindf <- as.data.frame(lapply(data, normalize))

# STEP 4 - Split the Dataset (70% for Training and 30% for Testing)
set.seed(123)  # Set seed for reproducibility
index <- sample(1:nrow(maxmindf), 0.7 * nrow(maxmindf))
Training <- maxmindf[index, ]
Testing <- maxmindf[-index, ]

# Logistic Regression
# STEP 5 - Apply Multiple Logistic Regression

# Use 'Mental' as the dependent variable for logistic regression
Training$Mental <- as.factor(Training$Mental)  # Convert 'Mental' to factor for logistic regression
# Apply the logistic regression model
logistic_model <- glm(Mental ~ Gender + Age + MS + EL + PS + SES + Religion + Residence + Income + History1 + History2 + Duration, 
                      data = Training, family = binomial)
# Summarize the logistic regression model
summary(logistic_model)

# STEP 6 - Predict the Outcomes on the Testing Set
logistic_predictions <- predict(logistic_model, newdata = Testing, type = "response")

# STEP 7 - Convert Probabilities to Binary Outcomes (Threshold: 0.5)
predicted_classes <- ifelse(logistic_predictions > 0.5, 1, 0)

# STEP 8 - Model Accuracy: Compare Predicted vs Actual Results
actual_classes <- Testing$Mental

# Generate confusion matrix
confusion_matrix <- table(Predicted = predicted_classes, Actual = actual_classes)
print(confusion_matrix)

# STEP 9 - Calculate Accuracy
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
print(paste("Logistic Regression Accuracy: ", round(accuracy * 100, 2), "%", sep = ""))

# Mean Squared Error for Logistic Regression Predictions
# Calculate the Mean Squared Error using the predicted probabilities
MSE_logistic <- mean((Testing$Mental - logistic_predictions)^2)
print(paste("Logistic Regression MSE: ", MSE_logistic))

# You can also calculate other metrics like Precision, Recall, and F1-Score
conf_matrix <- confusionMatrix(as.factor(predicted_classes), as.factor(actual_classes))
print(conf_matrix)

# AUC for Logistic Regression
roc_logistic <- roc(Testing$Mental, logistic_predictions)
auc_logistic <- auc(roc_logistic)
print(paste("Logistic Regression AUC: ", auc_logistic))

# Perform Hosmer-Lemeshow Test
hl_test <- hoslem.test(Training$Mental, fitted(logistic_model), g=10)
print(hl_test)

# Perform Brier Score for Logistic Regression
brier_score_logistic <- mean((logistic_predictions - Testing$Mental)^2)
print(paste("Logistic Regression Brier Score: ", brier_score_logistic))


#################### Multilayer Perceptron (Neural Network) ####################

# STEP 10 - Max-Min Data Normalization
maxmindf <- as.data.frame(lapply(data, normalize))

# STEP 11-Determine the Training and Testing of the Dataset
Training <- maxmindf[1:round(0.7 * nrow(maxmindf)), ]
Testing <- maxmindf[(round(0.7 * nrow(maxmindf)) + 1):nrow(maxmindf), ]

# STEP 12- CREATE MLR-ENHANCED MLP TRAINING AND TESTING DATA
enhanced_train_data <- train_data
enhanced_test_data  <- test_data
set.seed(123)
folds <- caret::createFolds(
  Training_mlr[[outcome_var]],
  k = 3,
  list = TRUE
)
oof_mlr_prob <- rep(NA, nrow(Training_mlr))
for (i in seq_along(folds)) {
  valid_idx <- folds[[i]]
  train_idx <- setdiff(seq_len(nrow(Training_mlr)), valid_idx)
  fold_train <- Training_mlr[train_idx, , drop = FALSE]
  fold_valid <- Training_mlr[valid_idx, , drop = FALSE]
  for (var in mental_predictors) {
    if (is.factor(fold_train[[var]])) {
      train_levels <- levels(droplevels(fold_train[[var]]))
      fold_train[[var]] <- factor(
        as.character(fold_train[[var]]),
        levels = train_levels
      )
      fold_valid[[var]] <- factor(
        ifelse(
          as.character(fold_valid[[var]]) %in% train_levels,
          as.character(fold_valid[[var]]),
          NA
        ),
        levels = train_levels
      )
    }
  }
  fold_mlr_model <- glm(
    formula = mental_formula,
    data = fold_train,
    family = binomial
  )
  fold_pred <- predict(
    fold_mlr_model,
    newdata = fold_valid,
    type = "response"
  )
  if (any(is.na(fold_pred))) {
    fold_mean_prob <- mean(
      predict(
        fold_mlr_model,
        newdata = fold_train,
        type = "response"
      ),
      na.rm = TRUE
    )
    fold_pred[is.na(fold_pred)] <- fold_mean_prob
  }
  oof_mlr_prob[valid_idx] <- fold_pred
}
enhanced_train_data$MLR_probability <- oof_mlr_prob
cat("Number of missing OOF MLR probabilities:", 
    sum(is.na(enhanced_train_data$MLR_probability)), "\n")
enhanced_test_data$MLR_probability <- predict(
  logistic_model,
  newdata = Testing_mlr,
  type = "response"
)
print(sum(is.na(enhanced_test_data$MLR_probability)))

enhanced_train_nn <- enhanced_train_data
enhanced_test_nn  <- enhanced_test_data
enhanced_train_nn[[outcome_var]] <- as.numeric(as.character(enhanced_train_nn[[outcome_var]]))
enhanced_test_nn[[outcome_var]]  <- as.numeric(as.character(enhanced_test_nn[[outcome_var]]))
all_predictor_cols <- setdiff(names(enhanced_train_nn), outcome_var)
for (col in all_predictor_cols) {
  enhanced_train_nn[[col]] <- as.numeric(enhanced_train_nn[[col]])
  enhanced_test_nn[[col]]  <- as.numeric(enhanced_test_nn[[col]])
}
for (col in all_predictor_cols) {
  train_col <- enhanced_train_nn[[col]]
  finite_values <- train_col[is.finite(train_col)]
  replacement_value <- ifelse(
    length(finite_values) > 0,
    median(finite_values, na.rm = TRUE),
    0
  )
  enhanced_train_nn[[col]][!is.finite(enhanced_train_nn[[col]])] <- replacement_value
  enhanced_test_nn[[col]][!is.finite(enhanced_test_nn[[col]])]   <- replacement_value
}

cat("\nOutcome distribution in enhanced training data:\n")
print(table(enhanced_train_nn[[outcome_var]], useNA = "ifany"))
if (length(unique(enhanced_train_nn[[outcome_var]])) < 2) {
  stop("Training outcome has only one class. MLR-enhanced MLP cannot be trained.")
}
predictor_variance <- sapply(
  enhanced_train_nn[, all_predictor_cols, drop = FALSE],
  function(x) var(x, na.rm = TRUE)
)
zero_variance_cols <- names(predictor_variance)[
  is.na(predictor_variance) | predictor_variance == 0
]
if (length(zero_variance_cols) > 0) {
  cat("\nRemoving zero-variance predictors:\n")
  print(zero_variance_cols)
  enhanced_train_nn <- enhanced_train_nn[
    ,
    !names(enhanced_train_nn) %in% zero_variance_cols,
    drop = FALSE
  ]
  enhanced_test_nn <- enhanced_test_nn[
    ,
    !names(enhanced_test_nn) %in% zero_variance_cols,
    drop = FALSE
  ]
}
enhanced_train_data <- enhanced_train_nn
enhanced_test_data  <- enhanced_test_nn
enhanced_predictor_names <- setdiff(names(enhanced_train_data), outcome_var)

  paste(outcome_var, "~", paste(enhanced_predictor_names, collapse = " + "))
)
fixed_hidden <- c(5,5)
enhanced_threshold <- 0.5
cat("\nNumber of training rows:", nrow(enhanced_train_data), "\n")
cat("Number of predictors:", length(enhanced_predictor_names), "\n")
cat("Number of missing values in enhanced_train_data:", sum(is.na(enhanced_train_data)), "\n")

# Train model
set.seed(456)
enhanced_model <- neuralnet::neuralnet(
  formula = formula_enhanced,
  data = enhanced_train_data,
  hidden = fixed_hidden,
  act.fct = "logistic",
  err.fct = "ce",
  linear.output = FALSE,
  rep = 10,
  stepmax = 1e7,
  threshold = 0.1
)
best_rep_enhanced <- which.min(enhanced_model$result.matrix["error", ])
attr(enhanced_model, "best_rep") <- best_rep_enhanced
attr(enhanced_model, "predictor_names") <- enhanced_predictor_names
cat("Best final repetition:", best_rep_enhanced, "\n")
plot(enhanced_model, rep = best_rep_enhanced)

# EVALUATE MLR-ENHANCED MLP MODEL
enhanced_test_inputs <- enhanced_test_data[, enhanced_predictor_names, drop = FALSE]

enhanced_result <- neuralnet::compute(
  enhanced_model,
  enhanced_test_inputs,
  rep = best_rep_enhanced
)
enhanced_predicted_prob <- as.vector(enhanced_result$net.result)
enhanced_predicted_prob <- pmin(pmax(enhanced_predicted_prob, 1e-6), 1 - 1e-6)
enhanced_actual_class <- enhanced_test_data[[outcome_var]]
enhanced_eval <- get_binary_metrics(
  actual = enhanced_actual_class,
  predicted_prob = enhanced_predicted_prob,
  threshold = enhanced_threshold,
  model_name = "MLR-enhanced MLP"
)
print(enhanced_eval$metrics)
print(enhanced_eval$confusion_matrix)
enhanced_predicted_class <- enhanced_eval$predicted_class

# STEP 13-LIME AND SHAP FOR MLR-ENHANCED MLP
attr(enhanced_model, "best_rep") <- best_rep_enhanced
attr(enhanced_model, "predictor_names") <- enhanced_predictor_names

# ---------- LIME for MLR-enhanced MLP ----------
model_type.nn <- function(x, ...) {
  "classification"
}
predict_model.nn <- function(model, newdata, ...) {
  rep_id <- attr(model, "best_rep")
  predictor_names <- attr(model, "predictor_names")
  if (is.null(rep_id)) {
    rep_id <- 1
  }
  if (is.null(predictor_names)) {
    stop("Predictor names were not stored in the model object.")
  }
  newdata <- as.data.frame(newdata)
  
  missing_cols <- setdiff(predictor_names, names(newdata))
  if (length(missing_cols) > 0) {
    stop(
      paste(
        "Missing predictor columns in newdata:",
        paste(missing_cols, collapse = ", ")
      )
    )
  }
  
  newdata <- newdata[, predictor_names, drop = FALSE]
  newdata[] <- lapply(newdata, as.numeric)
  pred <- as.vector(
    neuralnet::compute(
      model,
      newdata,
      rep = rep_id
    )$net.result
  )
  pred <- pmin(pmax(pred, 1e-6), 1 - 1e-6)
  data.frame(
    `0` = 1 - pred,
    `1` = pred,
    check.names = FALSE
  )
}
x_train_enhanced <- enhanced_train_data[, enhanced_predictor_names, drop = FALSE]
x_test_enhanced  <- enhanced_test_data[, enhanced_predictor_names, drop = FALSE]

x_train_enhanced[] <- lapply(x_train_enhanced, as.numeric)
x_test_enhanced[]  <- lapply(x_test_enhanced, as.numeric)
enhanced_prob_check <- predict_model.nn(
  enhanced_model,
  x_test_enhanced
)[["1"]]

print(summary(enhanced_prob_check))

# Select several cases for LIME and SHAP explanation
mental_case_ids <- select_explanation_cases(
  predicted_prob = enhanced_predicted_prob,
  n_cases = 5
)
lime_case_ids <- mental_case_ids
mental_case_summary <- data.frame(
  Case_Row_In_Test_Set = mental_case_ids,
  Predicted_Probability = round(enhanced_predicted_prob[mental_case_ids], 4),
  Predicted_Class = ifelse(
    enhanced_predicted_prob[mental_case_ids] >= 0.5,
    "Mental Health Yes",
    "Mental Health No"
  ),
  Actual_Class = enhanced_actual_class[mental_case_ids]
)
print(mental_case_summary)
write.csv(
  mental_case_summary,
  "Mental_Selected_Cases_for_Explanation.csv",
  row.names = FALSE
)
print(lime_case_ids)
print(enhanced_prob_check[lime_case_ids])
enhanced_explainer <- lime::lime(
  x = x_train_enhanced,
  model = enhanced_model,
  bin_continuous = FALSE
)

# Run LIME case-by-case
enhanced_lime_list <- list()
for (case_id in lime_case_ids) {
  cat("\nRunning LIME for testing case:", case_id, "\n")
  lime_result <- tryCatch(
    {
      lime::explain(
        x = x_test_enhanced[case_id, , drop = FALSE],
        explainer = enhanced_explainer,
        labels = "1",
        n_features = min(5, ncol(x_test_enhanced)),
        n_permutations = 5000,
        kernel_width = 1
      )
    },
    error = function(e) {
      cat("LIME skipped for case", case_id, ":", conditionMessage(e), "\n")
      return(NULL)
    }
  )
  if (!is.null(lime_result)) {
    enhanced_lime_list[[length(enhanced_lime_list) + 1]] <- lime_result
  }
}
f (length(enhanced_lime_list) > 0) {
  
  enhanced_lime_exp <- dplyr::bind_rows(enhanced_lime_list)
  
  print(enhanced_lime_exp)
  
  plot_features(enhanced_lime_exp)
  enhanced_lime_export <- as.data.frame(enhanced_lime_exp)
  list_columns <- vapply(
    enhanced_lime_export,
    is.list,
    logical(1)
  )
  enhanced_lime_export <- enhanced_lime_export[, !list_columns, drop = FALSE]
  selected_lime_columns <- c(
    "model_type",
    "case",
    "label",
    "label_prob",
    "model_r2",
    "model_intercept",
    "model_prediction",
    "feature",
    "feature_value",
    "feature_weight",
    "feature_desc"
  )
  selected_lime_columns <- intersect(
    selected_lime_columns,
    names(enhanced_lime_export)
  )
  enhanced_lime_export <- enhanced_lime_export[, selected_lime_columns, drop = FALSE]
  print(enhanced_lime_export)
  
  write.csv(
    enhanced_lime_export,
    "MLR_Enhanced_MLP_LIME_Explanation_Mental.csv",
    row.names = FALSE
  )
} else {
  cat("\nLIME could not be generated for the selected MLR-enhanced MLP cases.\n")
  cat("Reason: the model returned constant predictions across LIME perturbations.\n")
  cat("Use SHAP interpretation for this model, or select another case.\n")
}


# ---------- SHAP for MLR-enhanced MLP ----------

predict_enhanced_prob <- function(model, newdata) {
  
  rep_id <- attr(model, "best_rep")
  predictor_names <- attr(model, "predictor_names")
  
  if (is.null(rep_id)) {
    rep_id <- 1
  }
  
  if (is.null(predictor_names)) {
    predictor_names <- enhanced_predictor_names
  }
  
  newdata <- as.data.frame(newdata)
  newdata <- newdata[, predictor_names, drop = FALSE]
  newdata[] <- lapply(newdata, as.numeric)
  
  pred <- as.vector(
    neuralnet::compute(
      model,
      newdata,
      rep = rep_id
    )$net.result
  )
  
  pred <- pmin(pmax(pred, 1e-6), 1 - 1e-6)
  return(pred)
}
predictor_enhanced <- iml::Predictor$new(
  model = enhanced_model,
  data = x_train_enhanced,
  y = enhanced_train_data[[outcome_var]],
  predict.function = predict_enhanced_prob
)

# MLR-ENHANCED MLP: SELECT CASES FOR SHAP EXPLANATIONS
enhanced_case_ids <- select_explanation_cases(
  predicted_prob = enhanced_predicted_prob,
  n_cases = 5
)
enhanced_case_summary <- data.frame(
  Case_Row_In_Test_Set = enhanced_case_ids,
  Predicted_Probability = round(
    enhanced_predicted_prob[enhanced_case_ids],
    4
  ),
  Predicted_Class = ifelse(
    enhanced_predicted_prob[enhanced_case_ids] >= enhanced_threshold,
    "Mental Health Yes",
    "Mental Health No"
  ),
  Actual_Class = ifelse(
    enhanced_actual_class[enhanced_case_ids] == 1,
    "Mental Health Yes",
    "Mental Health No"
  ),
  stringsAsFactors = FALSE
)
write.csv(
  enhanced_case_summary,
  "MLR_Enhanced_MLP_Selected_Cases_for_Explanation.csv",
  row.names = FALSE
)

# MLR-ENHANCED MLP: SHAP FOR SELECTED TEST OBSERVATIONS
shap_list_enhanced <- list()

set.seed(123)
for (row_id in enhanced_case_ids) {
  cat(
    "\nCalculating SHAP for MLR-enhanced MLP test case:",
    row_id,
    "\n"
  )
  shap_temp <- tryCatch(
    {
      iml::Shapley$new(
        predictor = predictor_enhanced,
        x.interest = x_test_enhanced[
          row_id,
          ,
          drop = FALSE
        ],
        sample.size = 100
      )
    },
    error = function(e) {
      cat(
        row_id,
        ":",
        conditionMessage(e),
        "\n"
      )
      return(NULL)
    }
  )
  if (is.null(shap_temp)) {
    next
  }
  temp_results <- shap_temp$results
  temp_results$case_id <- row_id
  temp_results$predicted_probability <-
    enhanced_predicted_prob[row_id]
  temp_results$predicted_class <- ifelse(
    enhanced_predicted_prob[row_id] >= enhanced_threshold,
    "Mental Health Yes",
    "Mental Health No"
  )
  temp_results$actual_class <- ifelse(
    enhanced_actual_class[row_id] == 1,
    "Mental Health Yes",
    "Mental Health No"
  )
  temp_results$feature_value_label <- vapply(
    temp_results$feature,
    function(f) {
      if (f %in% colnames(x_test_enhanced)) {
        return(
          as.character(
            x_test_enhanced[[f]][row_id]
          )
        )
      }
      return(NA_character_)
    },
    character(1)
  )
  temp_results$feature_value_numeric <- vapply(
    temp_results$feature,
    function(f) {
      if (f %in% colnames(x_test_enhanced)) {
        return(
          as.numeric(
            x_test_enhanced[[f]][row_id]
          )
        )
      }
      
      return(NA_real_)
    },
    numeric(1)
  )
  shap_list_enhanced[[as.character(row_id)]] <-
    temp_results
}
if (length(shap_list_enhanced) == 0) {
  stop(
    paste(
      "No SHAP results were generated for the",
      "MLR-enhanced MLP model."
    )
  )
}
# Combine SHAP results from all selected observations
shap_long_enhanced <- dplyr::bind_rows(
  shap_list_enhanced
)

print(head(shap_long_enhanced))

write.csv(
  shap_long_enhanced,
  "MLR_Enhanced_MLP_SHAP_Long_Several_Test_Cases.csv",
  row.names = FALSE
)

# MLR-ENHANCED MLP: GLOBAL SHAP IMPORTANCE TABLE
feature_order_enhanced <- shap_long_enhanced %>%
  dplyr::group_by(feature) %>%
  dplyr::summarise(
    Mean_Absolute_SHAP = mean(
      abs(phi),
      na.rm = TRUE
    ),
    Mean_SHAP = mean(
      phi,
      na.rm = TRUE
    ),
    Number_of_Explanations = dplyr::n(),
    .groups = "drop"
  ) %>%
  dplyr::arrange(
    dplyr::desc(Mean_Absolute_SHAP)
  )
print(feature_order_enhanced)

write.csv(
  feature_order_enhanced,
  "MLR_Enhanced_MLP_SHAP_Global_Importance.csv",
  row.names = FALSE
)
# Set feature order for the beeswarm plot
# Lowest importance appears at the bottom;
# highest importance appears at the top.
shap_long_enhanced$feature <- factor(
  shap_long_enhanced$feature,
  levels = rev(
    feature_order_enhanced$feature
  )
)

# MLR-ENHANCED MLP: SHAP BEESWARM PLOT
p_beeswarm_enhanced <- ggplot2::ggplot(
  shap_long_enhanced,
  ggplot2::aes(
    x = phi,
    y = feature,
    color = feature_value_numeric
  )
) +
  ggplot2::geom_point(
    position = ggplot2::position_jitter(
      height = 0.20,
      width = 0
    ),
    alpha = 0.75,
    size = 2
  ) +
  ggplot2::geom_vline(
    xintercept = 0,
    linetype = "dashed"
  ) +
  ggplot2::labs(
    title = paste(
      "SHAP Beeswarm Plot for",
      "MLR-Enhanced MLP Model: Mental Health"
    ),
    subtitle = paste(
      "SHAP explanations for",
      length(shap_list_enhanced),
      "selected testing observations"
    ),
    x = "SHAP value",
    y = "Predictor",
    color = "Feature value"
  ) +
  ggplot2::theme_minimal() +
  ggplot2::theme(
    plot.title = ggplot2::element_text(
      face = "bold"
    ),
    axis.text.y = ggplot2::element_text(
      size = 8
    )
  )
print(p_beeswarm_enhanced)

ggplot2::ggsave(
  filename =
    "Mental_SHAP_Beeswarm_Plot_MLR_Enhanced_MLP.png",
  plot = p_beeswarm_enhanced,
  width = 11,
  height = 8,
  dpi = 300
)


# MLR-ENHANCED MLP: DEPENDENCE-PLOT FUNCTION
plot_shap_dependence_enhanced <- function(
    variable_name
) {
  available_features <- unique(
    as.character(
      shap_long_enhanced$feature
    )
  )
  
  if (!variable_name %in% available_features) {
    warning(
      paste(
        "Variable was not found in the",
        "MLR-enhanced MLP SHAP results:",
        variable_name
      )
    )
    
    return(NULL)
  }
  dep_data <- shap_long_enhanced[
    as.character(shap_long_enhanced$feature) ==
      variable_name,
    ,
    drop = FALSE
  ]
  
  
  p_dep <- ggplot2::ggplot(
    dep_data,
    ggplot2::aes(
      x = feature_value_numeric,
      y = phi
    )
  ) +
    ggplot2::geom_point(
      position = ggplot2::position_jitter(
        width = 0.02,
        height = 0
      ),
      alpha = 0.80,
      size = 2.5
    ) +
    ggplot2::geom_hline(
      yintercept = 0,
      linetype = "dashed"
    ) +
    ggplot2::labs(
      title = paste(
        "MLR-Enhanced MLP SHAP Dependence Plot for",
        variable_name
      ),
      subtitle = paste(
        "Positive SHAP values increase the predicted",
        "probability of Mental Health Yes"
      ),
      x = paste(
        variable_name,
        "(dummy-coded or normalised value)"
      ),
      y = "SHAP value"
    ) +
    ggplot2::theme_minimal() +
    ggplot2::theme(
      plot.title = ggplot2::element_text(
        face = "bold"
      )
    )
  
  print(p_dep)
  
  
  safe_name <- gsub(
    "[^A-Za-z0-9_]",
    "_",
    variable_name
  )
  
  
  ggplot2::ggsave(
    filename = paste0(
      "Mental_SHAP_Dependence_MLR_Enhanced_MLP_",
      safe_name,
      ".png"
    ),
    plot = p_dep,
    width = 8,
    height = 6,
    dpi = 300
  )
  
  
  invisible(p_dep)
}

# MLR-ENHANCED MLP: SELECT VARIABLES FOR DEPENDENCE PLOTS
dependence_vars_enhanced <- head(
  feature_order_enhanced$feature,
  min(
    6,
    nrow(feature_order_enhanced)
  )
)
dependence_vars_enhanced <- as.character(
  dependence_vars_enhanced
)
if (
  "MLR_probability" %in%
  feature_order_enhanced$feature &&
  !"MLR_probability" %in%
  dependence_vars_enhanced
) {
  if (length(dependence_vars_enhanced) >= 6) {
    dependence_vars_enhanced[
      length(dependence_vars_enhanced)
    ] <- "MLR_probability"
    
  } else {
    dependence_vars_enhanced <- c(
      dependence_vars_enhanced,
      "MLR_probability"
    )
  }
}

# Remove any duplicated variable names
dependence_vars_enhanced <- unique(
  dependence_vars_enhanced
)
cat(
  "\nMLR-enhanced MLP variables selected",
  "for SHAP dependence plots:\n"
)
print(dependence_vars_enhanced)

# Generate the dependence plots
for (v in dependence_vars_enhanced) {
  plot_shap_dependence_enhanced(v)
}

# MLR-ENHANCED MLP: INSTANCE-LEVEL SHAP-PLOT FUNCTION
plot_shap_instance_enhanced <- function(
    selected_case_id,
    top_n = 8
) {
  
  instance_data <- shap_long_enhanced[
    shap_long_enhanced$case_id ==
      selected_case_id,
    ,
    drop = FALSE
  ]
  if (nrow(instance_data) == 0) {
        warning(
      paste(
        "No SHAP results were found for",
        "MLR-enhanced MLP case",
        selected_case_id
      )
    )
    
    return(NULL)
  }
    
  instance_data <- instance_data[
    order(
      abs(instance_data$phi),
      decreasing = TRUE
    ),
    ,
    drop = FALSE
  ]
  
  instance_data <- instance_data[
    seq_len(
      min(
        top_n,
        nrow(instance_data)
      )
    ),
    ,
    drop = FALSE
  ]
  
  instance_data$feature_display <- paste0(
    as.character(instance_data$feature),
    " = ",
    round(
      instance_data$feature_value_numeric,
      3
    )
  )
  predicted_probability <- enhanced_predicted_prob[
    selected_case_id
  ]
  predicted_class <- ifelse(
    predicted_probability >= enhanced_threshold,
    "Mental Health Yes",
    "Mental Health No"
  )
  actual_class <- ifelse(
    enhanced_actual_class[selected_case_id] == 1,
    "Mental Health Yes",
    "Mental Health No"
  )
  p_instance <- ggplot2::ggplot(
    instance_data,
    ggplot2::aes(
      x = reorder(
        feature_display,
        phi
      ),
      y = phi
    )
  ) +
    ggplot2::geom_col() +
    ggplot2::coord_flip() +
    ggplot2::geom_hline(
      yintercept = 0,
      linetype = "dashed"
    ) +
    ggplot2::labs(
      title = paste(
        "MLR-Enhanced MLP SHAP Instance-Level",
        "Explanation: Case",
        selected_case_id
      ),
      subtitle = paste0(
        "Predicted probability = ",
        round(
          predicted_probability,
          4
        ),
        "; predicted class = ",
        predicted_class,
        "; actual class = ",
        actual_class
      ),
      x = "Predictor and observed value",
      y = "SHAP value"
    ) +
    ggplot2::theme_minimal() +
    ggplot2::theme(
      plot.title = ggplot2::element_text(
        face = "bold"
      )
    )
  
  print(p_instance)
  
  
  ggplot2::ggsave(
    filename = paste0(
      "Mental_SHAP_Instance_MLR_Enhanced_MLP_Case_",
      selected_case_id,
      ".png"
    ),
    plot = p_instance,
    width = 10,
    height = 7,
    dpi = 300
  )
  invisible(p_instance)
}

# GENERATE INSTANCE-LEVEL PLOT FOR EVERY SELECTED CASE
for (case_id in enhanced_case_ids) {
  plot_shap_instance_enhanced(
    selected_case_id = case_id,
    top_n = 8
  )
}

# STEP 14 - Model Validation and Accuracy Calculation for Neural Network
predicted1 <- nn.results$net.result * abs(diff(range(data$Mental))) + min(data$Mental)
actual1 <- Testing$Mental * abs(diff(range(data$Mental))) + min(data$Mental)
deviation <- (actual1 - predicted1)
value <- abs(mean(deviation))
accuracy_in_percent <- (1 - value) * 100
print(paste("Neural Network Accuracy: ", accuracy_in_percent, "%"))

# STEP 15 - Mean Squared Error for Neural Network Predictions
predicted <- nn.results$net.result
MSE.net <- sum((Testing$Mental - predicted)^2) / nrow(Testing)
print(paste("Neural Network MSE: ", MSE.net))


# AUC for Neural Network
roc_nn <- roc(Testing$Mental, nn.results$net.result)
auc_nn <- auc(roc_nn)
print(paste("Neural Network AUC: ", auc_nn))

# Perform Brier Score for Neural Network
brier_score_mlp <- mean((predicted_probabilities_mlp - Testing$Mental)^2)
print(paste("Neural Network Brier Score: ", brier_score_mlp))

#Step 16-ROC Curve

# Plot ROC Curve for Logistic Regression
plot(roc_logistic, col = "blue", lwd = 2, main = "ROC Curve for Logistic Regression")
text(0.6, 0.4, labels = paste("AUC:", round(auc_logistic, 2)), col = "blue")

# Plot ROC Curve for Neural Network
plot(roc_nn, col = "red", lwd = 2, main = "ROC Curve for Neural Network")
text(0.6, 0.4, labels = paste("AUC:", round(auc_nn, 2)), col = "red")

# Overlay both ROC Curves
plot(roc_logistic, col = "blue", lwd = 2, main = "ROC Curves for Logistic Regression and Neural Network-Mental Health Status")
plot(roc_nn, col = "red", lwd = 2, add = TRUE)
legend("bottomright", legend = c("Logistic Regression-Mental Health Status", "Neural Network-Mental Health Status"), col = c("blue", "red"), lwd = 2)

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

# Ensure the Mental column is a factor for classification
Training$Mental <- as.factor(Training$Mental)

# Set up 10-fold cross-validation
set.seed(123)
train_control <- trainControl(method = "cv", number = 10)

# Perform 10-fold cross-validation for MLR using caret
mlr_cv_model <- train(Mental ~ Gender + Age + MS + EL + PS + SES + Religion + Residence + Income + History1 + History2 + Duration, 
                      data = Training, method = "glm", family = "binomial", 
                      trControl = train_control)

# Perform 10-fold cross-validation for MLP using caret
mlp_cv_model <- train(Mental ~ Gender + Age + MS + EL + PS + SES + Religion + Residence + Income + History1 + History2 + Duration, 
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
