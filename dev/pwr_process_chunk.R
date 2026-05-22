args <- commandArgs(trailingOnly = TRUE)

WRKDIR <- args[1]
START <- as.integer(args[2])
END <- as.integer(args[3])
FILEDIR <- args[4]

cat("Processing rows from", START, "to", END, "\n")

# Load the full index file
index_file <- read.table(paste0(WRKDIR,"/pwr_data/pwr_index_file.txt"), header = F)

# Extract the subset for this chunk
chunk_rows <- index_file[START:END,]
n_chunks <- dim(chunk_rows)[[1]]

# Simulation parameters
n_sub <- 10 # number of subjects
n_time <- 100 # number of timepoints per subject

library(doParallel)
library(foreach)
library(dplyr)
library(glmnet)
library(cifti)


cat("Load source code", "\n")
sourceFile <- paste0(FILEDIR,"/brain_simulation.cpp")
# index_file <- read.table(paste0(wrkdir,"/pwr_data/pwr_index_file.txt"), header = F)

# phenos <- read.csv("/home/btervocl/and02709/power_data/ABCDphenotype.BTC.20240509.csv", header = TRUE)
# Y <- phenos$nihtbx_totalcomp_fc

# Load brain simulation function and C++ code once
# source("BrainSource.R")

# Define parameters
cat("Directory path", "\n")
directory_path <- "/home/feczk001/shared/projects/ABCD/gordon_sets/data/group2_10minonly_FD0p1"
files <- list.files(directory_path, pattern = "\\.pconn\\.nii$", full.names = TRUE, recursive = TRUE)

n_files <- length(files)
n_edge <- length(which(upper.tri(cifti::read_cifti(files[1])$data)))

n_draw <- n_chunks
sample_indices <- sample(1:n_files, size = n_draw, replace = TRUE)

cat("Brain Simulation", "\n")

result_list <- lapply(sample_indices, function(i) {
  Rcpp::sourceCpp(sourceFile)
  X <- cifti::read_cifti(files[i])$data
  # realVec <- X[which(upper.tri(X))]
  
  Brainy <- brain_simulation(M = X, n_sub = n_sub, n_time = n_time)
  synthCov <- colMeans(Brainy$Mats$cov_data)
  synthCor <- colMeans(Brainy$Mats$cor_data)
  
  # list(real = realVec, cov = synthCov, cor = synthCor)
  list(cov = synthCov, cor = synthCor)
})

cat("Matrix design", "\n")
# Convert result_list to matrices
synthMatCov <- t(sapply(result_list, function(x) x$cov))
synthMatCor <- t(sapply(result_list, function(x) x$cor))

cat("Save Data", "\n")
purrr::walk(1:nrow(synthMatCov), function(i) {
  name_stem <- paste0(WRKDIR, "/pwr_data/", paste0("dat_size_", chunk_rows[i, 3], "_index_", chunk_rows[i, 2]))
  cat(name_stem, "\n")
  saveRDS(synthMatCov[i, ], paste0(name_stem, "_cov.rds"))
  saveRDS(synthMatCor[i, ], paste0(name_stem, "_cor.rds"))
})

