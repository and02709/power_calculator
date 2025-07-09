args <- commandArgs(trailingOnly = TRUE)

# Extract the arguments
wrkdir <- args[1]
filedir <- args[2]
index <- as.integer(args[3])

# Simulation parameters
n_sub <- 10 # number of subjects
n_time <- 100 # number of timepoints per subject

library(doParallel)
library(foreach)
library(dplyr)
library(glmnet)
library(cifti)
library(rhdf5)

sourceFile <- paste0(filedir,"/brain_simulation.cpp")
index_file <- read.table(paste0(wrkdir,"/pwr_data/pwr_index_file.txt"), header = F)

phenos <- read.csv("/home/btervocl/and02709/power_data/ABCDphenotype.BTC.20240509.csv", header = TRUE)
Y <- phenos$nihtbx_totalcomp_fc

# Load brain simulation function and C++ code once
# source("BrainSource.R")

# Define parameters
directory_path <- "/home/feczk001/shared/projects/ABCD/gordon_sets/data/group2_10minonly_FD0p1"
files <- list.files(directory_path, pattern = "\\.pconn\\.nii$", full.names = TRUE, recursive = TRUE)

n_files <- length(files)
n_edge <- length(which(upper.tri(cifti::read_cifti(files[1])$data)))

n_draw <- index_file[index,2]
sample_indices <- sample(1:n_files, size = n_draw, replace = TRUE)

# Set up parallel backend
n_cores <- 8
cl <- makeCluster(n_cores)
registerDoParallel(cl)

# Run parallel loop using foreach
result_list <- foreach(i = sample_indices, .packages = c("cifti")) %dopar% {
  Rcpp::sourceCpp(sourceFile)
  X <- cifti::read_cifti(files[i])$data
  realVec <- X[which(upper.tri(X))]
  
  Brainy <- brain_simulation(M = X, n_sub = n_sub, n_time = n_time)
  synthCov <- colMeans(Brainy$Mats$cov_data)
  synthCor <- colMeans(Brainy$Mats$cor_data)
  
  list(real = realVec, cov = synthCov, cor = synthCor)
}

# Stop the cluster
stopCluster(cl)

# Convert result_list to matrices
realMat <- t(sapply(result_list, function(x) x$real))
synthMatCov <- t(sapply(result_list, function(x) x$cov))
synthMatCor <- t(sapply(result_list, function(x) x$cor))

real_name <- paste0("real_", index, ".h")
h5createFile(real_name)
h5createDataset(real_name, "matrix", dims = dim(realMat))
h5write(realMat, real_name, "matrix")

cov_name <- paste0("cov_", index, ".h")
h5createFile(cov_name)
h5createDataset(cov_name, "matrix", dims = dim(synthMatCov))
h5write(synthMatCov, cov_name, "matrix")

cor_name <- paste0("cor_", index, ".h")
h5createFile(cor_name)
h5createDataset(cor_name, "matrix", dims = dim(synthMatCor))
h5write(synthMatCor, cor_name, "matrix")