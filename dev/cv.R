#!/usr/bin/env Rscript

# --- Step 0: Command-line arguments ---
args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 6L) {
  stop("Expected 6 arguments: WRKDIR FILEDIR NUMFILES KFOLDS EPSILON INDEX")
}

WRKDIR   <- args[1]                 # Output directory
FILEDIR  <- args[2]                 # Directory containing scripts / ridge.rds
NUMFILES <- as.integer(args[3])     # Total number of split RDS files (not used here, but passed)
k_folds  <- as.integer(args[4])     # Number of folds (for bookkeeping)
epsilon  <- as.numeric(args[5])     # Noise parameter (currently not used)
index    <- as.integer(args[6])     # Index of file to process (1-based)

cat("========== cv.R START ==========\n")
cat("WRKDIR  :", WRKDIR,  "\n")
cat("FILEDIR :", FILEDIR, "\n")
cat("NUMFILES:", NUMFILES, "\n")
cat("k_folds :", k_folds, "\n")
cat("epsilon :", epsilon, "\n")
cat("index   :", index,   "\n\n")

# --- Load required libraries ---
suppressPackageStartupMessages({
  library(tidyverse)
  library(groupdata2)  # kept for compatibility, even if not used directly
  library(glmnet)
})

# --- Step 1: Locate matching RDS file(s) ---
file_pattern <- "split.rds"
rds_dir      <- file.path(WRKDIR, "pwr_data")
rds_files    <- sort(list.files(path = rds_dir, pattern = file_pattern, full.names = TRUE))
filenames    <- tools::file_path_sans_ext(basename(rds_files))

cat("[INFO] Found", length(rds_files), "split .rds files in", rds_dir, "\n")

if (length(rds_files) == 0) {
  stop("No matching .rds files found in ", rds_dir, ".")
}
if (index > length(rds_files) || index < 1L) {
  stop("Index (", index, ") is out of range [1, ", length(rds_files), "].")
}

# --- Step 2: Load selected RDS file ---
rds_file <- rds_files[index]
cat("[INFO] Reading file:", rds_file, "\n")
data <- readRDS(rds_file)
cat("[INFO] Base name:", filenames[index], "\n")

dataset     <- as.integer(sub(".*full_([0-9]+)_.*", "\\1", filenames[index]))
fold_number <- as.integer(sub(".*fold_([0-9]+)_split.*", "\\1", filenames[index]))

cat("[INFO] Parsed dataset   =", dataset, "\n")
cat("[INFO] Parsed fold_num  =", fold_number, "\n\n")

# --- Step 3: Separate data into training and testing ---
if (!all(c("train", "test") %in% names(data))) {
  stop("The RDS file does not contain $train and $test components.")
}

train_data <- data$train
test_data  <- data$test

# Drop any .folds column if present
train_data <- train_data[, names(train_data) != ".folds", drop = FALSE]
test_data  <- test_data[,  names(test_data)  != ".folds", drop = FALSE]

cat("[INFO] train_data dim:", paste(dim(train_data), collapse = " x "), "\n")
cat("[INFO] test_data  dim:", paste(dim(test_data),  collapse = " x "), "\n\n")

# --- Step 4: Load Ridge Model ---
ridge_path  <- file.path(WRKDIR, "pwr_data", "ridge.rds")
cat("[INFO] Loading ridge model from:", ridge_path, "\n")

ridge_model <- readRDS(ridge_path)
cat("[INFO] class(ridge_model) =", paste(class(ridge_model), collapse = ", "), "\n")

# --- Step 5: Assign y values via ridge_model ---
# We support two cases:
#  1) ridge_model is a glmnet / cv.glmnet object -> use predict()
#  2) ridge_model is a numeric vector of coefficients -> use X %*% beta

X_train <- as.matrix(train_data)
X_test  <- as.matrix(test_data)

if (is.numeric(ridge_model) && is.null(dim(ridge_model))) {
  # Case 2: numeric coefficient vector
  cat("[INFO] ridge_model is numeric; treating as coefficient vector.\n")
  if (length(ridge_model) != ncol(X_train)) {
    stop(
      "Length of ridge_model (", length(ridge_model),
      ") does not match number of columns of train_data (", ncol(X_train), ")."
    )
  }
  y      <- as.vector(X_train %*% ridge_model)
  y_test <- as.vector(X_test  %*% ridge_model)
  
} else if (inherits(ridge_model, c("cv.glmnet", "glmnet"))) {
  # Case 1: glmnet-style model object
  cat("[INFO] ridge_model is glmnet / cv.glmnet; using predict().\n")
  y      <- as.vector(predict(ridge_model, X_train))
  y_test <- as.vector(predict(ridge_model, X_test))
  
} else {
  stop(
    "ridge_model has unsupported class: ",
    paste(class(ridge_model), collapse = ", "),
    ". Expected numeric, glmnet, or cv.glmnet."
  )
}

cat("[INFO] Length y      =", length(y),      "\n")
cat("[INFO] Length y_test =", length(y_test), "\n\n")

# --- Step 6: Train model from Train Data (inner CV Ridge) ---
cat("[INFO] Fitting cv.glmnet on train_data to predict y.\n")
train_ridge_model <- cv.glmnet(X_train, y, alpha = 0)

# --- Step 7: Predict on test_data &
#     Calculate cross-validated R² ---
cat("[INFO] Predicting on test_data.\n")
yhat <- as.vector(predict(train_ridge_model, X_test))

SS_res <- sum((y_test - yhat)^2)
SS_tot <- sum((y_test - mean(y_test))^2)

cv_r2 <- 1 - SS_res / SS_tot
cat("Cross-validated R²:", round(cv_r2, 4), "\n")

# --- Step 8: Save result ---
out_path <- file.path(
  WRKDIR, "pwr_data",
  paste0("data_", dataset, "_fold_", fold_number, "_cvr2.rds")
)
saveRDS(cv_r2, file = out_path)
cat("[INFO] Saved cv_r2 to:", out_path, "\n")
cat("=========== cv.R END ===========\n")
