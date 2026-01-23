---
title: "pwr_visualiz"
format: html
editor: visual
---

```{r}
suppressPackageStartupMessages({
  library(RNifti)
  library(RcppCNPy)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(patchwork)
})

WRKDIR  <- "/scratch.global/and02709/second_python"
PWRDATA <- file.path(WRKDIR, "pwr_data")
dir.create(PWRDATA, showWarnings = FALSE, recursive = TRUE)

PCONN_DIR <- "/projects/standard/feczk001/shared/projects/ABCD/gordon_sets/data/group2_10minonly_FD0p1"

# ---- candidate Gordon paths (add more if needed)
gordon_candidates <- c(
  "~/power_calculator/new_work/Gordon_network_order.csv",
  "/scratch.global/and02709/power_calculator/new_work/Gordon_network_order.csv",
  "/users/0/and02709/power_calculator/new_work/Gordon_network_order.csv"
)

# ---- helper: find Gordon CSV robustly
find_gordon_csv <- function(candidates) {
  cand2 <- path.expand(candidates)
  hit <- cand2[file.exists(cand2)]
  if (length(hit) > 0) return(hit[1])

  roots <- unique(path.expand(c("~/power_calculator", "/scratch.global/and02709", "/users/0/and02709")))
  roots <- roots[dir.exists(roots)]
  for (r in roots) {
    f <- list.files(r, pattern="^Gordon_network_order\\.csv$", recursive=TRUE, full.names=TRUE)
    if (length(f) > 0) return(f[1])
  }

  stop(
    "Could not find Gordon_network_order.csv.\nTried:\n  ",
    paste(cand2, collapse="\n  "),
    "\nAlso searched roots:\n  ",
    paste(roots, collapse="\n  ")
  )
}

pick_pconn_file <- function(dir, pattern="\\.pconn\\.nii$", index=1L) {
  files <- list.files(dir, pattern=pattern, full.names=TRUE, recursive=TRUE)
  if (length(files) == 0L) stop("No pconn files found in: ", dir)
  files[index]
}

read_pconn_matrix <- function(path) {
  img  <- RNifti::readNifti(path)
  A <- as.array(img)
  dims <- dim(A)

  if (length(dims) == 2L && dims[1] == dims[2]) {
    M <- A
  } else if (length(dims) == 3L && dims[1] == dims[2] && dims[3] == 1L) {
    M <- drop(A[, , 1])
  } else if (length(dims) == 6L && all(dims[1:4] == 1L) && dims[5] == dims[6]) {
    M <- drop(A[1, 1, 1, 1, , ])
  } else if (length(dims) == 5L && all(dims[1:3] == 1L) && dims[4] == dims[5]) {
    M <- drop(A[1, 1, 1, , ])
  } else {
    stop("Unexpected pconn dims: ", paste(dims, collapse=" x "))
  }

  M <- as.matrix(M)
  storage.mode(M) <- "double"
  stopifnot(nrow(M) == ncol(M))
  M
}

upper_to_mat <- function(vec) {
  L <- length(vec)
  n <- (1 + sqrt(1 + 8 * L)) / 2
  if (abs(n - round(n)) > 1e-8) stop("Vector length not triangular; cannot infer n.")
  n <- as.integer(round(n))
  M <- matrix(0, n, n)
  M[upper.tri(M)] <- vec
  M <- M + t(M)
  diag(M) <- 1
  M
}

plot_mat <- function(M, title, low=-1, high=1, every=2L) {
  n <- nrow(M)
  df <- as.data.frame(M); df$row <- seq_len(n)
  long <- df |>
    pivot_longer(-row, names_to="col", values_to="value") |>
    mutate(col = as.integer(sub("^V","",col))) |>
    filter(row %% every == 0, col %% every == 0)

  ggplot(long, aes(col, row, fill=value)) +
    geom_tile() +
    scale_fill_distiller(palette="RdBu", direction=-1, limits=c(low,high), name="r") +
    coord_fixed() +
    theme_minimal(base_size=12) +
    theme(axis.text=element_blank(), axis.title=element_blank(), panel.grid=element_blank()) +
    ggtitle(title)
}

to_gordon <- function(M_orig, ord0) {
  ord <- as.integer(ord0) + 1L
  M_orig[ord, ord, drop=FALSE]
}

# -----------------------------
# Create empirical_corr.npy + gordon order if missing
# -----------------------------
emp_path <- file.path(PWRDATA, "empirical_corr.npy")
ord_path <- file.path(PWRDATA, "gordon_order_0based.txt")

if (!file.exists(emp_path) || !file.exists(ord_path)) {
  message("[INFO] empirical_corr.npy or gordon_order_0based.txt missing; creating them now...")

  PCONN_PATH <- pick_pconn_file(PCONN_DIR, index=1L)
  message("[EMP] Using pconn: ", PCONN_PATH)

  GORDON_CSV <- find_gordon_csv(gordon_candidates)
  message("[EMP] Using Gordon CSV: ", GORDON_CSV)

  M_emp <- read_pconn_matrix(PCONN_PATH)

  g <- read.csv(GORDON_CSV, stringsAsFactors = FALSE)
  req <- c("ID","net_order","parcel_order")
  miss <- setdiff(req, names(g))
  if (length(miss) > 0) stop("Gordon CSV missing columns: ", paste(miss, collapse=", "))
  if (nrow(g) != nrow(M_emp)) stop("Mismatch: Gordon rows != pconn n")

  ord <- order(g$net_order, g$parcel_order)
  ord0 <- ord - 1L

  M_emp_gordon <- M_emp[ord, ord, drop=FALSE]

  RcppCNPy::npySave(emp_path, M_emp_gordon)
  writeLines(as.character(ord0), ord_path)

  message("[OK] wrote: ", emp_path)
  message("[OK] wrote: ", ord_path)
}

# -----------------------------
# Load empirical + order
# -----------------------------
Emp <- npyLoad(emp_path)
ord0 <- scan(ord_path, quiet=TRUE)

# -----------------------------
# Index *all* simulated dat_size/index files (cov+cor)
# -----------------------------
parse_sim_name <- function(path) {
  bn <- basename(path)
  m <- regexec("^dat_size_([0-9]+)_index_([0-9]+)_(cov|cor)\\.npy$", bn)
  mm <- regmatches(bn, m)[[1]]
  if (length(mm) == 0) return(NULL)
  tibble(
    dat_size = as.integer(mm[2]),
    index    = as.integer(mm[3]),
    kind     = mm[4],
    file     = path
  )
}

sim_all <- list.files(PWRDATA, pattern="^dat_size_\\d+_index_\\d+_(cov|cor)\\.npy$", full.names=TRUE)
if (length(sim_all) == 0L) stop("No dat_size_*_index_*_{cov,cor}.npy files found in: ", PWRDATA)

sim_df <- dplyr::bind_rows(lapply(sim_all, parse_sim_name)) |>
  arrange(dat_size, index, kind)

message("[INFO] Found ", nrow(sim_df), " simulated files.")
print(sim_df |> head(30))

# Summary by dat_size
print(sim_df |>
        count(dat_size, kind, name="n_files") |>
        tidyr::pivot_wider(names_from=kind, values_from=n_files, values_fill=0) |>
        arrange(dat_size))

# -----------------------------
# Choose which simulated file(s) to view
# -----------------------------
# Option A: pick a specific dat_size/index
TARGET_DAT_SIZE <- 2000L
TARGET_INDEX    <- 1004L

# Option B: just pick the first available cor file
# row_pick <- sim_df |> filter(kind=="cor") |> slice(1)

row_pick <- sim_df |>
  filter(dat_size == TARGET_DAT_SIZE, index == TARGET_INDEX, kind == "cor") |>
  slice(1)

if (nrow(row_pick) == 0L) {
  message("[WARN] Requested dat_size/index not found; falling back to first cor file.")
  row_pick <- sim_df |> filter(kind=="cor") |> slice(1)
}

sim_path <- row_pick$file[[1]]
message("[SIM] Using: ", sim_path)

# -----------------------------
# Load sim vec -> matrix -> Gordon reorder
# -----------------------------
vec_sim <- as.numeric(npyLoad(sim_path))
M_sim_orig <- upper_to_mat(vec_sim)

ASSUME_SIM_IS_ORIGINAL_ORDER <- TRUE
Sim <- if (ASSUME_SIM_IS_ORIGINAL_ORDER) to_gordon(M_sim_orig, ord0) else M_sim_orig

if (nrow(Sim) != nrow(Emp)) stop("Dimension mismatch sim vs emp")

# -----------------------------
# Empirical pconn for comparison
# If you used a fixed PCONN1 in the SLURM wrapper, set it here explicitly.
# Otherwise, this uses the same one used to create empirical_corr.npy above.
# -----------------------------
# PCONN_FIXED <- "/projects/.../your_exact_pconn_used_in_simulation.pconn.nii"
# Emp_orig <- read_pconn_matrix(PCONN_FIXED)
# Emp <- to_gordon(Emp_orig, ord0)

# -----------------------------
# Plot side-by-side
# -----------------------------
p1 <- plot_mat(Sim, paste0("Simulated (Gordon space)\n", basename(sim_path)), low=-0.1, high=0.1, every=2L)
p2 <- plot_mat(Emp, "Empirical (Gordon space)", low=-1, high=1, every=2L)

print(p1 / p2)

```
