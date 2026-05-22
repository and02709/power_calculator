args <- commandArgs(trailingOnly = TRUE)

WRKDIR <- args[1]
FILEDIR <- args[2]

warning("Running combine_data")
library(tidyverse)

file_list <- list.files(path = paste0(WRKDIR, "/pwr_data"), pattern = "\\.rds$", full.names = TRUE)
n_files <- length(file_list)
cov_list <- list.files(path = paste0(WRKDIR, "/pwr_data"), pattern = "\\_cov.rds$", full.names = TRUE)
n_cov <- length(cov_list)
cor_list <- list.files(path = paste0(WRKDIR, "/pwr_data"), pattern = "\\_cor.rds$", full.names = TRUE)
n_cor <- length(cor_list)

cov_df <- lapply(cov_list, read_rds)
cov_df <- do.call(rbind, cov_df)

size <- as.numeric(sub(".*_size_([0-9]+)_index_.*", "\\1", cov_list))
index <- as.numeric(sub(".*_index_([0-9]+)_cov.*", "\\1", cov_list))

df <- data.frame(size, index, cov_df)

num_sets <- unique(df$size)

for(i in 1:length(num_sets)){
  dataset_name <- paste0("full_",num_sets[i],".rds")
  file_path <- paste0(WRKDIR, "/pwr_data/", dataset_name)
  df_sub <- df %>% filter(size == num_sets[i]) %>% arrange(index) %>% select(-c(size,index))
  saveRDS(df_sub, file_path)
}
