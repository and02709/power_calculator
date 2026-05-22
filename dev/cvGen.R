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
file_pattern <- "full_[0-9]+\\.rds"
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

# --- Step 3: Partition into k folds using groupdata2 ---
data <- fold(data, k = k_folds, id_col = NULL)

# --- Step 4: Create a list of data subsets for each fold ---
fold_data <- split(data, data$.folds)

# --- Step 4.5: Create output prefix for saved files ---
output_prefix <- file.path(WRKDIR, "/pwr_data/", filenames[index])

# --- Step 5: Split into k train/test sets and save each ---
for (fold_idx in seq_len(k_folds)) {
  test_set <- fold_data[[fold_idx]]
  train_set <- bind_rows(fold_data[-fold_idx])
  
  split_data <- list(train = train_set, test = test_set)
  
  out_file <- paste0(output_prefix, "_fold_", fold_idx, "_split.rds")
  saveRDS(split_data, out_file)
  
  cat("Saved fold", fold_idx, "-> Train size:", nrow(train_set),
      "Test size:", nrow(test_set), "\n")
}
