sample_sizes <- c(50, 100, 200, 500, 1000, 2000, 5000, 10000)
repeated_vector <- rep(sample_sizes, times = sample_sizes)
count_vector <- unlist(lapply(sample_sizes, function(n) 1:n))
n_index <- length(repeated_vector)
index <- 1:n_index

df <- data.frame(index, count_vector, repeated_vector)
colnames(df) <- c("index", "sample_count", "dataset")

write.table(df, file="pwr_index_file.txt", quote=F, col.names = F, row.names = F)