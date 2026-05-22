args <- commandArgs(trailingOnly = TRUE)

WRKDIR <- args[1]
FILEDIR <- args[2]

warning("Running setupCVmetrics")
library(tidyverse)

file_list <- list.files(path = file.path(WRKDIR, "pwr_data"), 
                        pattern = "_fold_[0-9]+_split\\.rds$", 
                        full.names = TRUE)

if (length(file_list) == 0) {
  stop("No matching RDS files found. Please check your WRKDIR and file naming pattern.")
}

filenames <- tools::file_path_sans_ext(basename(file_list))

cv <- data.frame(
  data = as.integer(sub(".*full_([0-9]+)_fold.*", "\\1", filenames)),
  fold = as.integer(sub(".*_fold_([0-9]+)_.*", "\\1", filenames))
)
cv <- data.frame(apply(cv, 2, as.integer))
cv$metric <- 0

saveRDS(cv, file = file.path(WRKDIR, "pwr_data", "cv.rds"))
