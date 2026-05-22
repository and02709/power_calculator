args <- commandArgs(trailingOnly = TRUE)

WRKDIR <- args[1]
FILEDIR <- args[2]

library(tidyverse)
library(glmnet)
library(cifti)

# directory_path <- "/home/feczk001/shared/projects/ABCD/gordon_sets/data/group2_10minonly_FD0p1"
# files <- list.files(directory_path, pattern = "\\.pconn\\.nii$", full.names = TRUE, recursive = TRUE)
# 
# n_files <- length(files)
# n_edge <- length(which(upper.tri(cifti::read_cifti(files[1])$data)))
# 
# result_list <- lapply(1:n_files, function(i) {
#   X <- cifti::read_cifti(files[i])$data
#   filename <- sub(".*sub-(NDARINV[0-9A-Z]+).*", "\\1", files[i])
#   realVec <- X[which(upper.tri(X))]
#   list(real = realVec, name=filename)
# })
# 
# realMat <- data.frame(t(sapply(result_list, function(x) x$real)))
# realNames <- sapply(result_list, function(x) x$name)
# realMat$ID <- realNames
# 
# phenos <- read.csv("/home/btervocl/and02709/power_data/ABCDphenotype.BTC.20240509.csv", header = TRUE) %>% select(all_of(c("src_subject_id", pheno)))
# colnames(phenos) <- c("ID", "Y")
# phenos$ID <- sub("_", "", phenos$ID)
# 
# df <- dplyr::inner_join(phenos, realMat, by="ID") %>% dplyr::select(-ID)
# X <- df %>% dplyr::select(-Y) %>% as.matrix(.)
# Y <- df %>% dplyr::select(Y) %>% as.matrix(.)
# rm(df)
# complete_cases <- complete.cases(X, Y)
# X_clean <- X[complete_cases, ]
# Y_clean <- Y[complete_cases]
# rm(X,Y)
# 
# ridge_model <- cv.glmnet(x = X_clean, y = Y_clean, alpha = 0)

haufe <- read_csv(paste0(WRKDIR,"/haufe.csv"), col_names = F)
ridge_model <- haufe[upper.tri(haufe)]


saveRDS(ridge_model, file=paste0(WRKDIR,"/pwr_data/ridge.rds"))

