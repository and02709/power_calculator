EigSplit <- setRefClass(
  "EigSplit",
  fields = list(
    eigenvectors = "matrix",
    sqrt_eigenvalues_pos = "numeric",
    sqrt_eigenvalues_neg = "numeric"
  ),
  
  methods = list(
    initialize = function(symmetric_matrix) {
      # Machine epsilon for float precision
      eps <- .Machine$double.eps
      
      # Perform eigendecomposition
      eig <- eigen(symmetric_matrix, symmetric = TRUE)
      
      # Store the eigenvectors
      eigenvectors <<- eig$vectors
      
      # Positive eigenvalues
      eigenvalues_pos <- pmax(eig$values, eps)
      sqrt_eigenvalues_pos <<- sqrt(eigenvalues_pos)
      
      # Negative eigenvalues (inverted)
      eigenvalues_neg <- pmax(-eig$values, eps)
      sqrt_eigenvalues_neg <<- sqrt(eigenvalues_neg)
    }
  )
)

EigPosNeg <- setRefClass(
  "EigPosNeg",
  fields = list(
    matrixPositive = "matrix",
    matrixNegative = "matrix"
  ),
  
  methods = list(
    initialize = function(symmetric_matrix){
      # Perform split eigendecomposition on the symmetric matrix
      eigsplitObject <- EigSplit(as.matrix(symmetric_matrix))
      
      # Reconstruct SPD matrix for positive eigenvalues
      half <- eigsplitObject$eigenvectors %*% diag(eigsplitObject$sqrt_eigenvalues_pos)
      matrixPositive <<- tcrossprod(half, half)
      
      # Reconstruct the SPD matrix for negative eigenvalues
      half <- eigsplitObject$eigenvectors %*% diag(eigsplitObject$sqrt_eigenvalues_neg)
      matrixNegative <<- tcrossprod(half, half)
    }
  )
)

xaugBuild <- function(n_sub){
  # Construct random design matrix
  x <- matrix(rnorm(n_sub), ncol = 1)
  x_aug <- matrix(0, nrow=n_sub, ncol=2)
  x_aug[,1] <- x[,1]
  x_aug[,2] <- -x[,1]
  
  # Shift the columns of x to be positive
  x_min <- min(x_aug)
  x_aug <- x_aug + abs(x_min)
  
  return(list(x=x, x_aug=x_aug))
}

arrayBuild <- function(M, n_sub, n_time, x_aug, epsilon=0, offset=0){
  
  if(epsilon < 0) stop("epison must be non-negative")
  
  n_node <- dim(M)[[1]]
  
  M_eig <- EigSplit(M)
  
  tdata <- apply(x_aug, 1, function(row,  
                                    n_sub,
                                    n_time,
                                    n_node,
                                    epsilon,
                                    M_eig) {
    trand <- matrix(rnorm(n_time * n_node), nrow = n_time, ncol = n_node)
    temp_t_pos <- sqrt(row[1]) * t(t(trand) * M_eig$sqrt_eigenvalues_pos) %*% t(M_eig$eigenvectors) 
    temp_t_neg <- sqrt(row[2]) * t(t(trand) * M_eig$sqrt_eigenvalues_neg) %*% t(M_eig$eigenvectors)
    
    if(epsilon > 0){
      eps_matrix <- matrix(rnorm(n_time * n_node, offset, epsilon), nrow = n_time, ncol = n_node)
    } else{
      eps_matrix <- matrix(0, nrow = n_time, ncol = n_node)
    }
    
    return(temp_t_pos + temp_t_neg + eps_matrix)
  }, 
  n_sub=n_sub,
  n_time=n_time,
  n_node=n_node,
  epsilon=epsilon,
  M_eig=M_eig,
  simplify = FALSE)
  
  tdata <- abind::abind(tdata, along=3)
  
  return(tdata)
}

vectorize_uplo <- function(mat) {
  mat[upper.tri(mat)]
}

constructMatrices <- function(n_sub, n_time, n_node, tdata){
  n_edge <- (n_node * n_node - n_node) / 2
  cov_data <- apply(tdata, 3, function(x){
    mat <- cov(x, use = "pairwise.complete.obs")
    return(mat[upper.tri(mat)])
  })
  cor_data <- apply(tdata, 3, function(x){
    mat <- cor(x, use = "pairwise.complete.obs")
    return(mat[upper.tri(mat)])
  })
  
  return(list(cov_data=t(cov_data), cor_data=t(cor_data)))
}

# Helper function to unvectorize upper triangular part into a symmetric matrix
unvectorize_uplo <- function(vec, n_node) {
  mat <- matrix(0, nrow = n_node, ncol = n_node)
  mat[upper.tri(mat)] <- vec
  mat <- mat + t(mat)  # Symmetrize
  return(mat)
}

functionalConstruct <- function(x_aug, data_matrices, n_node){
  X <- cbind(1, x_aug$x)
  b_cov <- solve(crossprod(X, X)) %*% crossprod(X, data_matrices$cov_data)
  b_cor <- solve(crossprod(X, X)) %*% crossprod(X, data_matrices$cor_data)
  
  cov_matrix <- unvectorize_uplo(b_cov[2,], n_node)
  cor_matrix <- unvectorize_uplo(b_cor[2,], n_node)
  
  return(list(b_cov=b_cov, cov_matrix=cov_matrix, b_cor=b_cor, cor_matrix=cor_matrix))
}

BrainSimulation <- function(M, n_sub, n_time){
  
  n_node <- dim(M)[[1]]
  x_aug <- xaugBuild(n_sub)
  
  arrayData <- arrayBuild(M = M, n_sub = n_sub, n_time = n_time, x_aug = x_aug$x_aug)
  tempMats <- constructMatrices(n_sub = n_sub, n_time = n_time, n_node = n_node, tdata = arrayData)
  
  functionalData <- functionalConstruct(x_aug = x_aug, data_matrices = tempMats, n_node = n_node)
  # converts
  return(list(functionalData=functionalData, Mats = tempMats))
}