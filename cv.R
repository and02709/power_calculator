#!/usr/bin/env Rscript

# --- Step 0: Command-line arguments ---
args <- commandArgs(trailingOnly = TRUE)

WRKDIR <- args[1]        # Output directory
FILEDIR <- args[2]       # Directory containing RDS files
NUMFILES <- as.integer(args[3])  # Total number of RDS files (not used here, just passed)
k_folds <- as.integer(args[4])   # Number of folds
index <- as.integer(args[5])     # Index of file to process (1-based)

# --- Load required libraries ---
library(tidyverse)
library(groupdata2)

# --- Step 1: Locate matching RDS file(s) ---
file_pattern <- "split.rds"
rds_files <- sort(list.files(path = paste0(WRKDIR, "/pwr_data"), pattern = file_pattern, full.names = TRUE))
filenames <- tools::file_path_sans_ext(basename(rds_files))


if (length(rds_files) == 0) {
  stop("No matching .rds files found in FILEDIR.")
}
if (index > length(rds_files)) {
  stop("Index exceeds number of available files.")
}

# --- Step 2: Load selected RDS file ---
rds_file <- rds_files[index]
cat("Reading file:", rds_file, "\n")
data <- readRDS(rds_file)
cat("File Name:", filenames[index], "\n")
dataset <- as.integer(sub(".*full_([0-9]+)_.*", "\\1", filenames[index]))
fold_number <- as.integer(sub(".*fold_([0-9]+)_split.*", "\\1", filenames[index]))

# --- Step 3: Separate data into training and testing ---
train_data <- data$train
test_data <- data$test
train_data <- train_data[, names(train_data) != ".folds"]
test_data  <- test_data[, names(test_data) != ".folds"]

# --- Step 4: Load Ridge Model ---
library(glmnet)
ridge_model <- readRDS(paste0(WRKDIR, "/pwr_data/ridge.rds"))

# --- Step 5: Assign y values ---
y <- predict(ridge_model, as.matrix(train_data))
y_var <- var(y)
eps <- eps*y_var
y <- y + rnorm(dim(train_data)[[1]], 0, sqrt(eps))
y_test <- predict(ridge_model, as.matrix(test_data))

# --- Step 6: Train model from Train Data ---
train_ridge_model <- cv.glmnet(as.matrix(train_data), as.matrix(y), alpha=0)
yhat <- predict(train_ridge_model, as.matrix(test_data))

# --- Step 7: Calculate cross-validated R² ---
SS_res <- sum((y_test - yhat)^2)
SS_tot <- sum((y_test - mean(y_test))^2)
cv_r2 <- 1 - SS_res / SS_tot
# cv_r2 <- ifelse(cv_r2 < 0, 0, cv_r2)

cat("Cross-validated R²:", round(cv_r2, 4), "\n")
saveRDS(cv_r2, file=paste0(WRKDIR,"/pwr_data/data_", dataset, "_fold_", fold_number, "_cvr2.rds"))

