// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
using namespace Rcpp;
using namespace arma;

// [[Rcpp::export]]
List eig_split(const mat& symmetric_matrix) {
  double eps = std::numeric_limits<double>::epsilon();
  
  vec eigval;
  mat eigvec;
  eig_sym(eigval, eigvec, symmetric_matrix);
  
  vec sqrt_eigval_pos = sqrt(clamp(eigval, eps, datum::inf));
  vec sqrt_eigval_neg = sqrt(clamp(-eigval, eps, datum::inf));
  
  return List::create(
    Named("eigenvectors") = eigvec,
    Named("sqrt_eigenvalues_pos") = sqrt_eigval_pos,
    Named("sqrt_eigenvalues_neg") = sqrt_eigval_neg
  );
}

// [[Rcpp::export]]
List eig_pos_neg(const mat& symmetric_matrix) {
  List split = eig_split(symmetric_matrix);
  mat eigvec = as<mat>(split["eigenvectors"]);
  vec sqrt_pos = as<vec>(split["sqrt_eigenvalues_pos"]);
  vec sqrt_neg = as<vec>(split["sqrt_eigenvalues_neg"]);
  
  mat half_pos = eigvec * diagmat(sqrt_pos);
  mat half_neg = eigvec * diagmat(sqrt_neg);
  
  mat matrixPositive = half_pos * half_pos.t();
  mat matrixNegative = half_neg * half_neg.t();
  
  return List::create(
    Named("matrixPositive") = matrixPositive,
    Named("matrixNegative") = matrixNegative
  );
}

// [[Rcpp::export]]
List xaug_build(int n_sub) {
  mat x = randn(n_sub, 1);
  mat x_aug(n_sub, 2);
  x_aug.col(0) = x.col(0);
  x_aug.col(1) = -x.col(0);
  
  double x_min = x_aug.min();
  x_aug = x_aug + std::abs(x_min);
  
  return List::create(Named("x") = x, Named("x_aug") = x_aug);
}

// [[Rcpp::export]]
cube array_build(const mat& M, int n_sub, int n_time, const mat& x_aug, double epsilon = 0.0, double offset = 0.0) {
  if (epsilon < 0.0) stop("epsilon must be non-negative");
  
  int n_node = M.n_rows;
  List M_eig = eig_split(M);
  mat eigvec = as<mat>(M_eig["eigenvectors"]);
  vec sqrt_pos = as<vec>(M_eig["sqrt_eigenvalues_pos"]);
  vec sqrt_neg = as<vec>(M_eig["sqrt_eigenvalues_neg"]);
  
  cube tdata(n_time, n_node, n_sub);
  
  for (int i = 0; i < n_sub; ++i) {
    mat trand = randn(n_time, n_node);
    
    mat temp_t_pos = sqrt(x_aug(i,0)) * (trand.each_row() % sqrt_pos.t()) * eigvec.t();
    mat temp_t_neg = sqrt(x_aug(i,1)) * (trand.each_row() % sqrt_neg.t()) * eigvec.t();
    
    mat eps_matrix;
    if (epsilon > 0.0) {
      eps_matrix = randn(n_time, n_node);
      eps_matrix = eps_matrix * epsilon;
      eps_matrix = eps_matrix + offset;
    } else {
      eps_matrix = zeros<mat>(n_time, n_node);
    }
    
    tdata.slice(i) = temp_t_pos + temp_t_neg + eps_matrix;
  }
  
  return tdata;
}

// [[Rcpp::export]]
vec vectorize_uplo(const mat& mat_in) {
  int n = mat_in.n_rows;
  vec out((n * (n - 1)) / 2);
  int idx = 0;
  for (int i = 0; i < n; ++i) {
    for (int j = i+1; j < n; ++j) {
      out(idx++) = mat_in(i,j);
    }
  }
  return out;
}

// [[Rcpp::export]]
List construct_matrices(int n_node, const cube& tdata) {
  int n_sub = tdata.n_slices;
  int n_edge = (n_node * (n_node - 1)) / 2;
  mat cov_data(n_sub, n_edge);
  mat cor_data(n_sub, n_edge);
  
  for (int i = 0; i < n_sub; ++i) {
    mat temp = tdata.slice(i);
    mat cov_mat = cov(temp);
    mat cor_mat = cor(temp);
    cov_data.row(i) = trans(vectorize_uplo(cov_mat));
    cor_data.row(i) = trans(vectorize_uplo(cor_mat));
  }
  
  return List::create(
    Named("cov_data") = cov_data,
    Named("cor_data") = cor_data
  );
}

// [[Rcpp::export]]
mat unvectorize_uplo(const vec& vec_in, int n_node) {
  mat mat_out = zeros<mat>(n_node, n_node);
  int idx = 0;
  for (int i = 0; i < n_node; ++i) {
    for (int j = i+1; j < n_node; ++j) {
      mat_out(i, j) = vec_in(idx);
      mat_out(j, i) = vec_in(idx);
      idx++;
    }
  }
  return mat_out;
}

// [[Rcpp::export]]
List functional_construct(const mat& x, const mat& cov_data, const mat& cor_data, int n_node) {
  mat X = join_horiz(ones<vec>(x.n_rows), x);
  mat XtX_inv = inv_sympd(X.t() * X);
  mat b_cov = XtX_inv * X.t() * cov_data;
  mat b_cor = XtX_inv * X.t() * cor_data;
  
  mat cov_matrix = unvectorize_uplo(b_cov.row(1).t(), n_node);
  mat cor_matrix = unvectorize_uplo(b_cor.row(1).t(), n_node);
  
  return List::create(
    Named("b_cov") = b_cov,
    Named("cov_matrix") = cov_matrix,
    Named("b_cor") = b_cor,
    Named("cor_matrix") = cor_matrix
  );
}

// [[Rcpp::export]]
List brain_simulation(const mat& M, int n_sub, int n_time) {
  int n_node = M.n_rows;
  List x_aug_list = xaug_build(n_sub);
  mat x = as<mat>(x_aug_list["x"]);
  mat x_aug = as<mat>(x_aug_list["x_aug"]);
  
  cube arrayData = array_build(M, n_sub, n_time, x_aug);
  List tempMats = construct_matrices(n_node, arrayData);
  
  List functionalData = functional_construct(x, tempMats["cov_data"], tempMats["cor_data"], n_node);
  
  return List::create(
    Named("functionalData") = functionalData,
    Named("Mats") = tempMats
  );
}

