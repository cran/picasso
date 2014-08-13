#ifndef MYMATH_H
#define MYMATH_H

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <R.h>
#include "R_ext/BLAS.h"
#include "R_ext/Lapack.h"

double sign(double x);

double max(double x,double y);

int max_idx(double * x, int n);

double max_abs_vec(double * x, int n);

double max_vec(double * x, int n);

int max_norm2_gr(double *x, int *gr, int *gr_size, int gr_n);

int max_abs_idx(double * x, int n);

void max_selc(double *x, double vmax, double *x_s, int n, int *n_s, double z);

double min(double x,double y);

double mean(double *x, int n);

void mean_mvr(double *intcpt, double *y, int n, int p);

void shuffle(int *array, int n);

double vec_inprod(double *x, double *y, int n);

// x = y^T z, y is 1 by m, z is m by n, x is 1 by n
void vec_mat_prod(double *x, double *y, double *z, int m, int n);

// x = y^T z, y is m by n, z is n by d, x is m by d
void vec_mat_prod_mvr(double *x, double *y, double *z, int m, int n, int d);

// || x^T y[,gr] ||
double vec_inprod_gr_2norm(double *x, double *y, int gr, int gr_size, int n);

// || x(n by 1)^T y(n by p) ||
double vec_mat_inprod_2norm(double *y, double *x, int n, int p);

// e-S[act]^T beta[act]
double res(double e, double *S, double *beta, int * set_act, int size_a, int idx);

int is_match(int idx, int * vec, int n);

double dif_2norm(double *x, double *y, int *xa_idx, int n);

// ||x-y||_F
double dif_Fnorm_mvr(double *x, double *y, int *gr_act, int gr_size_a, int d, int p);

double dif_2norm_gr(double *x, double *y, int *gr, int *gr_size, int *gr_act, int gr_act_size);

void vec_copy(double *x, double *y, int *xa_idx, int n);

void vec_copy_gr(double *x, double *y, int *gr, int *gr_size, int *gr_act, int gr_act_size);

// copy x to y
void mat_copy_mvr(double *x, double *y, int *gr_act, int gr_size_a, int d, int p);

// x=y-z
void dif_vec(double *x, double *y, double *z, int n);

void dif_vec_const(double *x, double y, int n);

// x=x-y
void dif_vec_const_mvr(double *x, double *y, double dif, int n, int p);

void dif_vec_vec(double *x, double *y, double z, int n);

// x = x-y[gr]*z[gr]
void dif_vec_gr(double *x, double *y, int gr, int gr_size, double * z, double dif, int n);

// x = x-y[gr]*z[gr]-z1
void dif_vec_const_gr(double *x, double *y, int gr, int gr_size, double * z, double dif, double z1, int n);

void dif_vec_vec_const(double *x, double *y, double z, double z1, int n);

// y = y-dif*x(n by 1)*beta(1 by p)
void dif_mat_mvr(double *y, double *x, double *beta, double dif, int n, int d, int p);

void identfy_actset(double *beta, int *set_act, int *size_a, int d);

void identfy_actgr(double *beta, int *gr_act, int *gr_size_a, int *gr, int *gr_size, int gr_n);

// beta1 = soft(beta1-grad/L, ilambda)
void prox_beta_est(double *beta_tild, double *beta, double *grad, double L, double lamb, int d);

// beta_tild = gr_soft(beta1-grad/L, ilambda)
void prox_beta_est_mvr(double *beta_tild, double *beta, double *grad, double *S, double L, double lamb, int p, int d);

double soft_thresh_l1(double y, double z);

double soft_thresh_scad(double y, double z, double gamma);

double soft_thresh_mcp(double y, double z, double gamma);

double sum_vec_dif(double *x, double *y, int n);

void X_beta_update(double *Xb, double *X, double beta, int n);

void p_update(double *p, double *Xb, double intcpt, int n);

// grad = S * x - e, r is n by 1, A is n by d, x d by 1 with m non-zeros
void grad_scio(double *grad, double *e, double *S, double *x, int *xa_idx, int size_a, int d);

// grad = beta1 - <p-Y, X>/n/w
void get_grad_logit_lin(double *grad, double *beta1, double *p_Y, double *X, int n, int d, double w);

double get_grad_logit_l1(double *p, double *Y, double *X, int n);

// g = <p-Y, X>/n
void get_grad_logit_l1_vec(double *grad, double *p_Y, double *X, int n, int d);

double get_grad_logit_scad(double *p, double *Y, double *X, double beta, double lambda, double gamma, int n);

// grad = <p-Y, X>/n + h_grad(scad)
void get_grad_logit_scad_vec(double *grad, double *p_Y, double *X, double *beta, double lambda, double gamma, int n, int d);

// e[j] - S[act,j]^T beta[act] + h_grad(scad)
double get_grad_scio_scad(double e, double *S, double *beta, int * set_act, int size_a, double lambda, double gamma, int idx);

// grad = e - S[act,]^T beta[act] + h_grad(scad)
void get_grad_scio_scad_vec(double *grad, double *e, double *S, double *beta, int * set_act, int size_a, double lambda, double gamma, int d);

double get_grad_logit_mcp(double *p, double *Y, double *X, double beta, double lambda, double gamma, int n);

// grad = <p-Y, X>/n + h_grad(mcp)
void get_grad_logit_mcp_vec(double *grad, double *p_Y, double *X, double *beta, double lambda, double gamma, int n, int d);

// e[j] - S[act,j]^T beta[act] + h_grad(mcp)
double get_grad_scio_mcp(double e, double *S, double *beta, int * set_act, int size_a, double lambda, double gamma, int idx);

// grad = e - S[act,]^T beta[act] + h_grad(mcp)
void get_grad_scio_mcp_vec(double *grad, double *e, double *S, double *beta, int * set_act, int size_a, double lambda, double gamma, int d);

void get_grad_logit_gr(double *g, double *p, double *Y, double *X, int gr, int gr_size, int n);

// || beta[gr] ||
double norm2_gr_vec(double *beta, int gr,int gr_size);

// || x[gr]-y[gr]/w ||
double norm2_gr_vec_dif(double *x, double *y, double w, int gr,int gr_size);

// || x[c_row,] ||
double norm2_gr_mvr(double *x, int d, int p);

// y[i] = || x[,i] ||^2
void norm2_col_mat(double *y, double *x, int d, int p);

// y[i] = || x[i,] ||^2
void norm2_row_mat(double *y, double *x, int d, int p);

void rtfind(double rt_l, double rt_r, double *x, int c_idx, int start_idx, int n_idx, double beta_hat, double ilambda, double S);

// || beta[c_row,] ||
double norm2_gr_mat(double *beta, int c_row, int d, int p);

// find root of S*rt - beta_hat + ilambda*rt/||x[c_row,]|| = 0
void rtfind_mvr(double rt_l, double rt_r, double *x, int c_col, int c_row, int d,  int p, double beta_hat, double ilambda, double S);

double l1norm(double * x, int n);

void euc_proj(double * v, double z, int n);

double fun1(double lambda, double * v, double z, int n);

double mod_bisec(double * v, double z, int n);

void fabs_vc(double *v_in, double *v_out, int n);

void max_fabs_vc(double *v_in, double *v_out, double *vmax, int *n1, int n, double z);

void sort_up_bubble(double *v, int n);

void get_residual(double *r, double *y, double *A, double *x, int *xa_idx, int *nn, int *mm);

void get_residual_scr(double *r, double *y, double *A, double *x, int *xa_idx, int *nn, int *mm, int *n_scr);

void get_dual(double *u, double *r, double *mmu, int *nn);

void get_dual1(double *u, double *r, double *mmu, int *nn);

void get_dual2(double *u, double *r, double *mmu, int *nn);

void get_grad(double *g, double *A, double *u, int *dd, int *nn);

void get_grad_scr(double *g, double *A, double *u, int *dd, int *nn, int *nn0);

void get_base(double *base, double *u, double *r, double *mmu, int *nn);

// r = y - A* x
void get_residual_mat(double *r, double *y, double *A, double *x, int *idx, int *size, int *nn, int *mm, int *dd);

// u = proj(r)
void get_dual_mat(double *u, double *r, double *mmu, int *nn, int *mm);

// g = -A * u
void get_grad_mat(double *g, double *A, double *u, int *dd, int *nn, int *mm);

// base = u * r - mu * ||u||_F^2/2
void get_base_mat(double *base, double *fro, double *u, double *r, double *mmu, int *nn, int *mm);

void dif_mat(double *x0, double *x1, double *x2, int *nn, int *mm);

void dif_mat2(double *x0, double *x1, double *x2, double *c2, int *nn, int *mm);

// tr(x1'*x2)
double tr_norm(double *x1, double *x2, int *nn, int *mm);

// ||x||_F^2
double fro_norm(double *x, int *nn, int *mm);

// ||x||_12;
double lnorm_12(double *x, int *nn, int *mm);

void trunc_svd(double *U, double *Vt, double *S, double *x, double *eps, int *nn, int *mm, int *min_nnmm);

void equ_mat(double *x0, double * x1, int *nn, int *mm);

void proj_mat_sparse(double *u, int *idx, int *size_u, double *lambda, int *nn, int *mm);


void dantzig_mfista_scr(double *b0, double *A0, double *b, double *A, int *idx_scr, int num_scr, int dim, double *beta, double mu, double *L, int *ite, int *ite2, int *ite3, double lambda, int max_ite, double prec, int intercept, int flag, int nlamb);


void dantzig_mfista_scr2(double *b0, double *A0, double *b, double *A, int *idx_scr, int num_scr, int dim, double *beta, double mu, double *L, int *ite, int *ite2, int *ite3, double lambda, int max_ite, double prec, int intercept, int flag, int nlamb);

#endif
