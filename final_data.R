args <- commandArgs(trailingOnly = TRUE)

WRKDIR <- args[1]
FILEDIR <- args[2]

warning("Running final_data")
library(tidyverse)

file_list <- list.files(path = paste0(WRKDIR, "/pwr_data"), pattern = "\\_cvr2.rds$", full.names = TRUE)
n_files <- length(file_list)

size <- as.numeric(sub(".*data_([0-9]+)_fold_.*", "\\1", file_list))
fold <- as.numeric(sub(".*_fold_([0-9]+)_cvr2.*", "\\1", file_list))

df <- data.frame(file_list, size, fold, metrics=0)

for(i in 1:n_files){
  met_rds <- readRDS(file_list[i])
  df$metrics[i] <- met_rds
}

saveRDS(df, file=paste0(WRKDIR, "/metrics_data.rds"))

df_summary <- df %>%
  group_by(size) %>%
  summarise(
    mean_metric = mean(metrics, na.rm = TRUE),
    sd_metric = sd(metrics, na.rm = TRUE),
    .groups = "drop"
  )

saveRDS(df_summary, file=paste0(WRKDIR, "/metrics_summary.rds"))

plot_metrics <- ggplot(df_summary, aes(x = as.factor(size), y = mean_metric)) +
  geom_point(size = 3) +
  geom_errorbar(aes(ymin = mean_metric - sd_metric, ymax = mean_metric + sd_metric), width = 0.2) +
  labs(x = "Size", y = "Mean Metric", title = "Mean Metric by Size with Error Bars") +
  theme_minimal()

ggsave(filename = paste0(WRKDIR, "/mean_metric_by_size.png"), plot = plot_metrics, width = 6, height = 4, dpi = 300)
