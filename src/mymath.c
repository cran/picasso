#include "mymath.h"

double sign(double x){
    return (x > 0) ? 1 : ((x < 0) ? -1 : 0);
}

double max(double x,double y){
    return (x > y) ? x : y;
}

int max_idx(double * x, int n){
    int i, idx;
    double tmp = fabs(x[0]);
    
    idx = 0;
    for(i=1; i<n ; i++){
        if(x[i]>tmp){
            tmp = x[i];
            idx = i;
        }
    }
    return idx;
}

double max_abs_vec(double * x, int n){
    int i;
    double tmp = fabs(x[0]);
    
    for(i=1; i<n ; i++){
        tmp = max(tmp, fabs(x[i]));
    }
    return tmp;
}

int max_abs_idx(double * x, int n){
    int i, idx;
    double tmp = fabs(x[0]);
    
    idx = 0;
    for(i=1; i<n ; i++){
        if(fabs(x[i])>tmp){
            tmp = fabs(x[i]);
            idx = i;
        }
    }
    return idx;
}

double max_vec(double * x, int n){
    int i;
    double tmp = x[0];
    
    for(i=1; i<n ; i++){
        tmp = max(tmp, x[i]);
    }
    return tmp;
}

int max_norm2_gr(double *x, int *gr, int *gr_size, int gr_n){
    int i, j, idx;
    double tmp, c_va, max_norm2;
    
    max_norm2 = 0;
    idx = 0;
    for(i=0; i<gr_n; i++){
        tmp = 0;
        for (j=0; j<gr_size[i]; j++) {
            c_va = x[gr[i]+j];
            tmp += c_va*c_va;
        }
        if(tmp>max_norm2){
            max_norm2 = tmp;
            idx = i;
        }
    }
    return idx;
}

void max_selc(double *x, double vmax, double *x_s, int n, int *n_s, double z){
    int i,tmp;
    double thresh = vmax-z;

    tmp = 0;
    for(i=0; i<n ; i++){
        if(x[i]>thresh){
            x_s[tmp] = x[i];
            tmp++;
        }
    }
    *n_s = tmp;
}

double min(double x, double y){
    return (x < y) ? x : y;
}

double mean(double *x, int n){
    int i;
    double tmp;
    
    tmp = 0;
    for(i=0; i<n ;i++){
        tmp += x[i];
    }
    return tmp/(double)n;
}

void mean_mvr(double *intcpt, double *y, int n, int p){
    int i;
    
    for(i=0; i<p; i++){
        intcpt[i] = mean(y+i*n, n);
    }
}

void shuffle(int *array, int n){
    int i, j, t;
    
    if (n > 1)
    {
        for (i = 0; i < n - 1; i++)
        {
            j = i + rand() / (RAND_MAX / (n - i) + 1);
            t = array[j];
            array[j] = array[i];
            array[i] = t;
        }
    }
}

// <x,y>
double vec_inprod(double *x, double *y, int n){
    int i;
    double tmp=0;
    
    for(i=0; i<n; i++){
        tmp += x[i]*y[i];
    }
    return tmp;
}

// x = y^T z, y is 1 by m, z is m by n, x is 1 by n
void vec_mat_prod(double *x, double *y, double *z, int m, int n){
    int i,j;
    
    for(i=0; i<n; i++){
        x[i] = 0;
        for(j=0; j<m; j++){
            x[i] -= z[i*m+j]*y[j];
        }
    }
}

// x = -z^T y, y is n by m, z is n by d, x is m by d
void vec_mat_prod_mvr(double *x, double *y, double *z, int m, int n, int d){
    int i,j,k;
    
    for(i=0; i<d; i++){
        for(j=0; j<m; j++){
            x[j*d+i] = 0;
            for(k=0; k<n; k++){
                x[j*d+i] += y[j*n+k]*z[i*n+k];
            }
        }
    }
}

// || x^T y[,gr] ||
double vec_inprod_gr_2norm(double *x, double *y, int gr, int gr_size, int n){
    int i,j;
    double tmp, tmp1;
    
    tmp = 0;
    for (j=gr; j<gr+gr_size; j++) {
        tmp1 = 0;
        for(i=0; i<n; i++){
            tmp1 += x[i]*y[j*n+i];
        }
        tmp += tmp1*tmp1;
    }
    return sqrt(tmp);
}

// || x(n by 1)^T y(n by p) ||
double vec_mat_inprod_2norm(double *y, double *x, int n, int p){
    int i,j;
    double tmp, tmp1;
    
    tmp = 0;
    for (j=0; j<p; j++) {
        tmp1 = 0;
        for(i=0; i<n; i++){
            tmp1 += x[i]*y[j*n+i];
        }
        tmp += tmp1*tmp1;
    }
    return sqrt(tmp);
}

// e-S[act]^T beta[act]
double res(double e, double *S, double *beta, int * set_act, int size_a, int idx){
    int i, c_idx;
    double tmp;
    
    tmp = e;
    for (i=0; i<size_a; i++) {
        c_idx = set_act[i];
        tmp -= S[c_idx]*beta[c_idx];
    }
    tmp += S[idx]*beta[idx];
    return tmp;
}

// check weather idx is in vec
int is_match(int idx, int * vec, int n){
    int i;
    
    for (i=0; i<n; i++) {
        if (idx == vec[i]) {
            return 1;
        }
    }
    return 0;
}

// ||x-y||_2
double dif_2norm(double *x, double *y, int *xa_idx, int n){
    int i, idx;
    double tmp, norm2;
    
    norm2 = 0;
    for (i=0; i<n; i++) {
        idx = xa_idx[i];
        tmp = x[idx]-y[idx];
        norm2 += tmp*tmp;
    }
    return sqrt(norm2);
}

// ||x-y||_F
double dif_Fnorm_mvr(double *x, double *y, int *gr_act, int gr_size_a, int d, int p){
    int i, j, idx;
    double tmp, Fnorm;
    
    Fnorm = 0;
    for (i=0; i<p; i++) {
        for (j=0; j<gr_size_a; j++) {
            idx = gr_act[j];
            tmp = x[i*d+idx]-y[i*d+idx];
            Fnorm += tmp*tmp;
        }
    }
    return sqrt(Fnorm);
}

// ||x[gr]-y[gr]||_2
double dif_2norm_gr(double *x, double *y, int *gr, int *gr_size, int *gr_act, int gr_act_size){
    int i, j, idx;
    double tmp, norm2;
    
    norm2 = 0;
    for(i=0; i<gr_act_size; i++){
        idx = gr_act[i];
        for (j=0; j<gr_size[idx]; j++) {
            tmp = y[gr[idx]+j] - x[gr[idx]+j];
            norm2 += tmp*tmp;
        }
    }
    return sqrt(norm2);
}

// copy x to y
void vec_copy(double *x, double *y, int *xa_idx, int n){
    int i, idx;
    
    for(i=0; i<n; i++){
        idx = xa_idx[i];
        y[idx] = x[idx];
    }
}

// copy x[gr] to y[gr]
void vec_copy_gr(double *x, double *y, int *gr, int *gr_size, int *gr_act, int gr_act_size){
    int i, j, idx;
    
    for(i=0; i<gr_act_size; i++){
        idx = gr_act[i];
        for (j=0; j<gr_size[idx]; j++) {
            y[gr[idx]+j] = x[gr[idx]+j];
        }
    }
}

// copy x to y
void mat_copy_mvr(double *x, double *y, int *gr_act, int gr_size_a, int d, int p){
    int i, j, idx;
    
    for (i=0; i<p; i++) {
        for (j=0; j<gr_size_a; j++) {
            idx = gr_act[j];
            y[i*d+idx] = x[i*d+idx];
        }
    }
}

// x=x-y
void dif_vec_const(double *x, double y, int n){
    int i;
    
    for (i=0; i<n; i++) {
        x[i] = x[i]-y;
    }
}

// x=x-y
void dif_vec_const_mvr(double *x, double *y, double dif,int n, int p){
    int i, j;
    
    for (i=0; i<p; i++) {
        for (j=0; j<n; j++) {
            x[i*n+j] = x[i*n+j]-dif*y[i];
        }
    }
}

// x=y-z
void dif_vec(double *x, double *y, double *z, int n){
    int i;
    
    for (i=0; i<n; i++) {
        x[i] = y[i]-z[i];
    }
}

// x=x-z*y
void dif_vec_vec(double *x, double *y, double z, int n){
    int i;
    
    for (i=0; i<n; i++) {
        x[i] = x[i]-z*y[i];
    }
}

// x = x-dif*y[gr]*z[gr]
void dif_vec_gr(double *x, double *y, int gr, int gr_size, double * z, double dif, int n){
    int i,j;
    
    for (i=0; i<n; i++) {
        for (j=gr; j<gr+gr_size; j++) {
            x[i] = x[i]-dif*y[j*n+i]*z[j];
        }
    }
}

// x=x-z*y-c
void dif_vec_vec_const(double *x, double *y, double z, double z1, int n){
    int i;
    
    for (i=0; i<n; i++) {
        x[i] = x[i]-z*y[i]-z1;
    }
}

// x = x-y[gr]*z[gr]-z1
void dif_vec_const_gr(double *x, double *y, int gr, int gr_size, double * z, double dif, double z1, int n){
    int i,j;
    
    for (i=0; i<n; i++) {
        for (j=gr; j<gr+gr_size; j++) {
            x[i] = x[i]-dif*y[j*n+i]*z[j];
        }
        x[i] -= z1;
    }
}

// y = y-dif*x(n by 1)*beta(1 by p)
void dif_mat_mvr(double *y, double *x, double *beta, double dif, int n, int d, int p){
    int i,j;
    
    for (i=0; i<p; i++) {
        for (j=0; j<n; j++) {
            y[i*n+j] = y[i*n+j]-dif*x[j]*beta[i*d];
        }
    }
}

void identfy_actset(double *beta, int *set_act, int *size_a, int d){
    int i;
    
    *size_a = 0;
    for (i=0; i<d; i++) {
        if(beta[i]!=0){
            set_act[*size_a] = i;
            (*size_a)++;
        }
    }
}

void identfy_actgr(double *beta, int *gr_act, int *gr_size_a, int *gr, int *gr_size, int gr_n){
    int i;
    
    *gr_size_a = 0;
    for (i=0; i<gr_n; i++) {
        if(norm2_gr_vec(beta, gr[i], gr_size[i])!=0){
            gr_act[*gr_size_a] = i;
            (*gr_size_a)++;
        }
    }
}

// beta_tild = soft(beta1-grad/L, ilambda)
void prox_beta_est(double *beta_tild, double *beta, double *grad, double L, double lamb, int d){
    int i;
    
    for (i=0; i<d; i++) {
        beta_tild[i] = beta[i] - grad[i]/L;
        beta_tild[i] = soft_thresh_l1(beta_tild[i], lamb); //fabs(beta_tild[i])>0 ? sign(beta_tild[i])*(fabs(beta_tild[i])-lamb) : 0;
    }
}

// beta_tild = gr_soft(beta1-grad/L, ilambda)
void prox_beta_est_mvr(double *beta_tild, double *beta, double *grad, double *S, double L, double lamb, int p, int d){
    int i,j;
    double tmp;
    
    for (i=0; i<p; i++) {
        for (j=0; j<d; j++) {
            beta_tild[i*d+j] = beta[i*d+j] - grad[i*d+j]/L;
            //beta_tild[j*p+i] = beta[i*d+j] - grad[i*d+j]/L;
        }
    }
    
    for (i=0; i<d; i++) {
        for (j=0; j<p; j++) {
            tmp = beta_tild[j*d+i];
            if(tmp>lamb){
                rtfind_mvr(0,(tmp-lamb)/S[i], beta_tild, j, i, d, p, tmp, lamb, S[i]);
            }else{
                if(tmp<(-lamb)){
                    rtfind_mvr((tmp+lamb)/S[i], 0, beta_tild, j, i, d, p, tmp, lamb, S[i]);
                }else{
                    beta_tild[j*d+i] = 0;
                }
            }
        }
    }
    
}

// x[i] = soft_l1(y,z);
double soft_thresh_l1(double y, double z){
    
    if(y>z) return y-z;
    else if(y<(-z)) return y+z;
    else return 0;
}

// x[i] = soft_scad(y,z);
double soft_thresh_scad(double y, double z, double gamma){
    
    if(fabs(y)>fabs(gamma*z)) {
        return y;
    }else{
        if(fabs(y)>fabs(2*z)) {
            return soft_thresh_l1(y, gamma*z/(gamma-1))/(1-1/(gamma-1));
        }else{
            return soft_thresh_l1(y, z);
        }
    }
}

// x[i] = soft_mcp(y,z);
double soft_thresh_mcp(double y, double z, double gamma){
    
    if(fabs(y)>fabs(gamma*z)) {
        return y;
    }else{
        return soft_thresh_l1(y, z)/(1-1/gamma);
    }
}


// sum(x-y)
double sum_vec_dif(double *x, double *y, int n){
    int i;
    double tmp = 0;
    
    for (i=0; i<n; i++) {
        tmp += x[i] - y[i];
    }
    return tmp;
}

// Xb = Xb+X*beta
void X_beta_update(double *Xb, double *X, double beta, int n){
    int i;
    
    for (i=0; i<n; i++) {
        Xb[i] = Xb[i] + beta*X[i];
    }
}

// p[i] = 1/(1+exp(-intcpt-Xb[i]))
void p_update(double *p, double *Xb, double intcpt, int n){
    int i;
    double neg_intcpt = -intcpt;
    
    for (i=0; i<n; i++) {
        p[i] = 1/(1+exp(neg_intcpt-Xb[i]));
        if(p[i]>0.999) p[i] = 1;
        if(p[i]<0.001) p[i] = 0;
    }
}

// grad = beta1 - <p-Y, X>/n/w
void get_grad_logit_lin(double *grad, double *beta1, double *p_Y, double *X, int n, int d, double w){
    int i,j;
    double tmp, wn;
    
    wn = (double)n*w;
    for (i=0; i<d; i++) {
        tmp = 0;
        for (j=0; j<n; j++) {
            tmp += p_Y[j]*X[i*n+j];
        }
        grad[i] = beta1[i] - tmp/wn;
    }
}

// g = <p-Y, X>/n
double get_grad_logit_l1(double *p, double *Y, double *X, int n){
    int i;
    double tmp = 0;
    
    for (i=0; i<n; i++) {
        tmp += (p[i]-Y[i])*X[i];
    }
    return tmp/(double)n;
}

// grad = <p-Y, X>/n
void get_grad_logit_l1_vec(double *grad, double *p_Y, double *X, int n, int d){
    int i,j;
    double dbn = (double)n;
    
    for (i=0; i<d; i++) {
        grad[i] = 0;
        for (j=0; j<n; j++) {
            grad[i] += p_Y[j]*X[i*n+j];
        }
        grad[i] = grad[i]/dbn;
    }
}

// g = <p-Y, X>/n + h_grad(scad)
double get_grad_logit_scad(double *p, double *Y, double *X, double beta, double lambda, double gamma, int n){
    int i;
    double tmp, h_grad, beta_abs;
    
    tmp = 0;
    for (i=0; i<n; i++) {
        tmp += (p[i]-Y[i])*X[i];
    }
    
    beta_abs = fabs(beta);
    if(beta_abs<=lambda){
        h_grad = 0;
    }else{
        if(beta_abs<=gamma*lambda){
            h_grad = (lambda*sign(beta)-beta)/(gamma-1);
        }else{
            h_grad = -lambda*sign(beta);
        }
    }
    return tmp/(double)n + h_grad/1.5;
}

// grad = <p-Y, X>/n + h_grad(scad)
void get_grad_logit_scad_vec(double *grad, double *p_Y, double *X, double *beta, double lambda, double gamma, int n, int d){
    int i,j;
    double tmp, h_grad, doublen, gamlamb, beta_abs;
    
    doublen = (double)n;
    gamlamb = gamma*lambda;
    
    for (i=0; i<d; i++) {
        tmp = 0;
        for (j=0; j<n; j++) {
            tmp += (p_Y[j])*X[i*n+j];
        }
        
        beta_abs = fabs(beta[i]);
        if(beta_abs<=lambda){
            h_grad = 0;
        }else{
            if(beta_abs<=gamlamb){
                h_grad = (lambda*sign(beta[i])-beta[i])/(gamma-1);
            }else{
                h_grad = -lambda*sign(beta[i]);
            }
        }
        grad[i] = tmp/doublen + h_grad/1.5;
    }
}

// e[j] - S[act,j]^T beta[act] + h_grad(scad)
double get_grad_scio_scad(double e, double *S, double *beta, int * set_act, int size_a, double lambda, double gamma, int idx){
    int i, c_idx;
    double tmp, beta_abs, h_grad;
    
    tmp = e;
    for (i=0; i<size_a; i++) {
        c_idx = set_act[i];
        tmp -= S[c_idx]*beta[c_idx];
    }
    tmp += S[idx]*beta[idx];
    
    beta_abs = fabs(beta[idx]);
    if(beta_abs<=lambda){
        h_grad = 0;
    }else{
        if(beta_abs<=gamma*lambda){
            h_grad = (lambda*sign(beta[idx])-beta[idx])/(gamma-1);
        }else{
            h_grad = -lambda*sign(beta[idx]);
        }
    }
    return tmp + h_grad/1;
}


// grad = e - S[act,]^T beta[act] + h_grad(scad)
void get_grad_scio_scad_vec(double *grad, double *e, double *S, double *beta, int * set_act, int size_a, double lambda, double gamma, int d){
    int i,j,c_idx;
    double tmp, h_grad, gamlamb, beta_abs;
    
    gamlamb = gamma*lambda;
    
    for (i=0; i<d; i++) {
        tmp = e[i];
        for (j=0; j<size_a; j++) {
            c_idx = set_act[j];
            tmp -= S[i*d+c_idx]*beta[c_idx];
        }
        tmp += S[i*d+i]*beta[i];
        beta_abs = fabs(beta[i]);
        if(beta_abs<=lambda){
            h_grad = 0;
        }else{
            if(beta_abs<=gamlamb){
                h_grad = (lambda*sign(beta[i])-beta[i])/(gamma-1);
            }else{
                h_grad = -lambda*sign(beta[i]);
            }
        }
        grad[i] = tmp + h_grad;
    }
}

// g = <p-Y, X>/n + h_grad(mcp)
double get_grad_logit_mcp(double *p, double *Y, double *X, double beta, double lambda, double gamma, int n){
    int i;
    double tmp, h_grad, beta_abs;
    
    tmp = 0;
    for (i=0; i<n; i++) {
        tmp += (p[i]-Y[i])*X[i];
    }
    
    beta_abs = fabs(beta);
    if(beta_abs<=gamma*lambda){
        h_grad = -beta/gamma;
    }else{
        h_grad = -lambda*sign(beta);
    }
    return tmp/(double)n + h_grad/1.5;
}

// grad = <p-Y, X>/n + h_grad(mcp)
void get_grad_logit_mcp_vec(double *grad, double *p_Y, double *X, double *beta, double lambda, double gamma, int n, int d){
    int i,j;
    double tmp, h_grad, doublen, gamlamb;
    
    doublen = (double)n;
    gamlamb = gamma*lambda;
    for (i=0; i<d; i++) {
        tmp = 0;
        for (j=0; j<n; j++) {
            tmp += (p_Y[j])*X[i*n+j];
        }
        
        if(fabs(beta[i])<=gamlamb){
            h_grad = -beta[i]/gamma;
        }else{
            h_grad = -lambda*sign(beta[i]);
        }
        grad[i] = tmp/doublen + h_grad/1.5;
    }
}

// e[j] - S[act,j]^T beta[act] + h_grad(mcp)
double get_grad_scio_mcp(double e, double *S, double *beta, int * set_act, int size_a, double lambda, double gamma, int idx){
    int i, c_idx;
    double tmp, beta_abs, h_grad;
    
    tmp = e;
    for (i=0; i<size_a; i++) {
        c_idx = set_act[i];
        tmp -= S[c_idx]*beta[c_idx];
    }
    tmp += S[idx]*beta[idx];
    
    beta_abs = fabs(beta[idx]);
    if(beta_abs<=gamma*lambda){
        h_grad = -beta[idx]/gamma;
    }else{
        h_grad = -lambda*sign(beta[idx]);
    }
    return tmp + h_grad/1;
}

// grad = e - S[act,]^T beta[act] + h_grad(mcp)
void get_grad_scio_mcp_vec(double *grad, double *e, double *S, double *beta, int * set_act, int size_a, double lambda, double gamma, int d){
    int i, j, c_idx;
    double tmp, beta_abs, h_grad, gamlamb;
    
    gamlamb = gamma*lambda;
    for (i=0; i<d; i++) {
        tmp = e[i];
        for (j=0; j<size_a; j++) {
            c_idx = set_act[j];
            tmp -= S[i*d+c_idx]*beta[c_idx];
        }
        tmp += S[i*d+i]*beta[i];
        beta_abs = fabs(beta[i]);
        if(beta_abs<=gamlamb){
            h_grad = -beta[i]/gamma;
        }else{
            h_grad = -lambda*sign(beta[i]);
        }
        grad[i] = tmp + h_grad;
    }
}

// g = X[gr]^T (p-Y)/n
void get_grad_logit_gr(double *g, double *p, double *Y, double *X, int gr, int gr_size, int n){
    int i,j;
    
    for (j=gr; j<gr+gr_size; j++) {
        g[j] = 0;
        for (i=0; i<n; i++) {
            g[j] += (p[i]-Y[i])*X[j*n+i];
        }
        g[j] = g[j]/(double)n;
    }
}

// || beta[gr] ||
double norm2_gr_vec(double *beta, int gr,int gr_size){
    int i;
    double tmp=0;
    
    for (i=gr; i<gr+gr_size; i++) {
        tmp += beta[i]*beta[i];
    }
    return sqrt(tmp);
}

// || x[gr]-y[gr]/w ||
double norm2_gr_vec_dif(double *x, double *y, double w, int gr,int gr_size){
    int i;
    double tmp,tmp1;
    
    tmp = 0;
    for (i=gr; i<gr+gr_size; i++) {
        tmp1 = x[i]-y[i]/w;
        tmp += tmp1*tmp1;
    }
    return sqrt(tmp);
}

// || x[c_row,] ||
double norm2_gr_mvr(double *x, int d, int p){
    int i;
    double tmp, norm2;
    
    norm2 = 0;
    for (i=0; i<p; i++) {
        tmp = x[i*d];
        norm2 += tmp*tmp;
    }
    return sqrt(norm2);
}

// y[i] = || x[,i] ||^2, x is p by d
void norm2_col_mat(double *y, double *x, int d, int p){
    int i,j;
    double tmp;
    
    for (i=0; i<d; i++) {
        y[i] = 0;
        for (j=0; j<p; j++) {
            tmp = x[i*p+j];
            y[i] += tmp*tmp;
        }
    }
}

// y[i] = || x[i,] ||^2, x is d by p
void norm2_row_mat(double *y, double *x, int d, int p){
    int i,j;
    double tmp;
    
    for (i=0; i<d; i++) {
        y[i] = 0;
        for (j=0; j<p; j++) {
            tmp = x[j*d+i];
            y[i] += tmp*tmp;
        }
    }
}

// find root of S*rt - beta_hat + ilambda*rt/||x[gr]|| = 0
void rtfind(double rt_l, double rt_r, double *x, int c_idx, int start_idx, int n_idx, double beta_hat, double ilambda, double S){
    int ite = 0;
    double rt_m, func_v, dif;
    
    rt_m = (rt_l+rt_r)/2;
    x[c_idx] = rt_m;
    func_v = S*rt_m - beta_hat + ilambda*rt_m/norm2_gr_vec(x, start_idx, n_idx);
    dif = rt_r - rt_l;
    while (dif > 1e-4 && ite < 100) {
        if (func_v>0) {
            rt_r = rt_m;
        }else{
            rt_l = rt_m;
        }
        rt_m = (rt_l+rt_r)/2;
        x[c_idx] = rt_m;
        func_v = S*rt_m - beta_hat + ilambda*rt_m/norm2_gr_vec(x, start_idx, n_idx);
        dif = rt_r - rt_l;
        ite++;
    }
    rt_m = (rt_l+rt_r)/2;
    x[c_idx] = rt_m;
}

// || beta[c_row,] ||
double norm2_gr_mat(double *beta, int c_row, int d, int p){
    int i;
    double tmp=0;
    
    for (i=0; i<p; i++) {
        tmp += beta[i*d+c_row]*beta[i*d+c_row];
    }
    return sqrt(tmp);
}

// find root of S*rt - beta_hat + ilambda*rt/||x[c_row,]|| = 0
void rtfind_mvr(double rt_l, double rt_r, double *x, int c_col, int c_row, int d,  int p, double beta_hat, double ilambda, double S){
    int ite = 0;
    double rt_m, func_v, dif;
    
    rt_m= (rt_l+rt_r)/2;
    x[c_col*d+c_row] = rt_m;
    func_v = S*rt_m - beta_hat + ilambda*rt_m/norm2_gr_mat(x, c_row, d, p);
    dif = rt_r - rt_l;
    while (dif > 1e-4 && ite < 100) {
        if (func_v>0) {
            rt_r = rt_m;
        }else{
            rt_l = rt_m;
        }
        rt_m = (rt_l+rt_r)/2;
        x[c_col*d+c_row] = rt_m;
        func_v = S*rt_m - beta_hat + ilambda*rt_m/norm2_gr_mat(x, c_row, d, p);
        dif = rt_r - rt_l;
        ite++;
    }
    rt_m = (rt_l+rt_r)/2;
    x[c_col*d+c_row] = rt_m;
}

double l1norm(double * x, int n){
    int i;
    double tmp=0;

    for(i=0; i<n; i++) 
        tmp += fabs(x[i]);
    return tmp;
}

void fabs_vc(double *v_in, double *v_out, int n){
    int i;

    for(i=0; i<n; i++)
        v_out[i] = fabs(v_in[i]);
}

void max_fabs_vc(double *v_in, double *v_out, double *vmax, int *n1, int n, double z){
    int i;
    double tmp, v_abs;

    tmp = 0;
    for(i=0; i<n; i++){
        v_abs = fabs(v_in[i]);
        v_out[i] = v_abs;
        tmp = max(tmp, v_abs);
    }
    *vmax = tmp;
    *n1 = n;
}

void sort_up_bubble(double *v, int n){
    int i,j;
    double tmp;
    int ischanged;

    for(i=n-1; i>=0; i--){ 
        ischanged = 0;
        for(j=0; j<i; j++){
            if(v[j]>v[j+1]){
                tmp = v[j];
                v[j] = v[j+1];
                v[j+1] = tmp;
                ischanged = 1;
            }
        }
        if(ischanged==0) 
            break;
    }
}

// r = y - A * x, r is n by 1, A is n by d, x d by 1 with m non-zeros
void get_residual(double *r, double *y, double *A, double *x, int *xa_idx, int *nn, int *mm)
{
    int i,j,b_idx;
    int n,m;
    double tmp;
    n = *nn;
    m = *mm;
    
    for(i=0;i<n;i++){
        tmp=0;
        for(j=0;j<m;j++){
            b_idx = xa_idx[j];
            tmp+=A[b_idx*n+i]*x[b_idx];
        }
        r[i] = y[i]-tmp;
    }
}

// grad = S * x - e, r is n by 1, A is n by d, x d by 1 with m non-zeros
void grad_scio(double *grad, double *e, double *S, double *x, int *xa_idx, int size_a, int d)
{
    int i,j,b_idx;
    double tmp;
    
    for(i=0;i<d;i++){
        tmp=0;
        for(j=0;j<size_a;j++){
            b_idx = xa_idx[j];
            tmp+=S[i*d+b_idx]*x[b_idx];
        }
        grad[i] = tmp - e[i];
    }
}


void get_residual_scr(double *r, double *y, double *A, double *x, int *xa_idx, int *nn, int *mm, int *n_scr)
{
    int i,j,b_idx;
    int n,m,n_sc;
    double tmp;
    n = *nn;
    m = *mm;
    n_sc = *n_scr;
    
    for(i=0;i<n_sc;i++){
        tmp=0;
        for(j=0;j<m;j++){
            b_idx = xa_idx[j];
            tmp+=A[b_idx*n+i]*x[b_idx];
        }
        r[i] = y[i]-tmp;
    }
}

void get_dual(double *u, double *r, double *mmu, int *nn)
{
    int i,n;
    double mu, zv;
    mu = *mmu;
    n = *nn;
    zv = 1;
    for(i=0;i<n;i++){
        u[i] = r[i]/mu;
    }
    euc_proj(u, zv, n); //euclidean projection
}

void get_dual1(double *u, double *r, double *mmu, int *nn)
{
    int i,n;
    double mu, zv;
    mu = *mmu;
    n = *nn;
    zv = 1;
    for(i=0;i<n;i++){
        u[i] = r[i]/mu;
        if(u[i]>zv)
            u[i] = zv;
        if(u[i]<-zv)
            u[i] = -zv;
    }
}

void get_dual2(double *u, double *r, double *mmu, int *nn)
{
    int i,n;
    double mu, zv, tmp_sum;
    mu = *mmu;
    n = *nn;
    zv = 1;
    tmp_sum = 0;
    for(i=0;i<n;i++){
        u[i] = r[i]/mu;
        tmp_sum += u[i]*u[i];
    }
    tmp_sum = sqrt(tmp_sum);
    if(tmp_sum>=zv){
        for(i=0;i<n;i++){
            u[i] = u[i]/tmp_sum;
        }
    }
}

void get_grad(double *g, double *A, double *u, int *dd, int *nn)
{
    int i,j;
    int d,n;
    
    d = *dd;
    n = *nn;
    
    for(i=0;i<d;i++){
        g[i]=0;
        for(j=0;j<n;j++){
            g[i] -= A[i*n+j]*u[j];
        }
    }
}

void get_grad_scr(double *g, double *A, double *u, int *dd, int *nn, int *nn0)
{
    int i,j;
    int d,n,n0;
    
    d = *dd;
    n = *nn;
    n0 = *nn0;
    
    for(i=0;i<d;i++){
        g[i]=0;
        for(j=0;j<n;j++){
            g[i] -= A[i*n0+j]*u[j];
        }
    }
}

void get_base(double *base, double *u, double *r, double *mmu, int *nn)
{
    int i,n;
    double mu,tmp;
    mu = *mmu;
    n = *nn; 
    tmp = 0;
    for(i=0;i<n;i++){
        tmp += u[i]*u[i];
    }

    *base = 0;
    for(i=0;i<n;i++){
        *base += u[i]*r[i];
    }
    *base -= mu*tmp/2;
}
// r = y - A * x, r is n by m, A is n by d, x d by m
void get_residual_mat(double *r, double *y, double *A, double *x, int *idx_x, int *size_x, int *nn, int *mm, int *dd)
{
    int i,j,k,id;
    int n,m,d,size;
    double tmp;
    n = *nn;
    m = *mm;
    d = *dd;
    size = *size_x;
    
    for(i=0;i<m;i++){
        for(j=0;j<n;j++){
            tmp = 0;
            for(k=0;k<size;k++){
                id = idx_x[k];
                tmp += A[id*n+j]*x[i*d+id];
            }
            r[i*n+j] = y[i*n+j]-tmp;
        }
    }
}

// u = proj(r)
void get_dual_mat(double *u, double *r, double *mmu, int *nn, int *mm)
{
    int i,j,n,m;
    double mu, zv, tmp_sum;
    mu = *mmu;
    n = *nn;
    m = *mm;
    zv = 1;

    for(i=0;i<m;i++){
        tmp_sum = 0;
        for(j=0;j<n;j++){
            u[i*n+j] = r[i*n+j]/mu;
            tmp_sum += u[i*n+j]*u[i*n+j];
        }
        tmp_sum = sqrt(tmp_sum);
        if (tmp_sum >= zv) {
            for(j=0;j<n;j++){
                u[i*n+j] = u[i*n+j]/tmp_sum;
            }
        }
    }
}

void proj_mat_sparse(double *u, int *idx, int *size_u, double *lambda, int *nn, int *mm)
{
    int i,j,n,m,size,flag;
    double zero, tmp_sum;
    n = *nn;
    m = *mm;
    zero = 0;
    size = 0;

    for(i=0;i<n;i++){
        tmp_sum = 0;
        for(j=0;j<m;j++){
            tmp_sum += u[j*n+i]*u[j*n+i];
        }
        tmp_sum = sqrt(tmp_sum);
        flag = 0;
        for(j=0;j<m;j++){
            u[j*n+i] = u[j*n+i]*max(1-*lambda/tmp_sum, zero);
            if(flag == 0){
                if(u[j*n+i] != 0){
                    flag = 1;
                }
            }
        }
        if(flag == 1){
            idx[size] = i;
            size++;
        }
    }
    *size_u = size;
}

// g = -A' * u, g is d by m, A is n by d, u is n by m
void get_grad_mat(double *g, double *A, double *u, int *dd, int *nn, int *mm)
{
    int i,j,k;
    int d,n,m;
    double tmp;

    d = *dd;
    n = *nn;
    m = *mm;
    
    for(i=0;i<m;i++){
        for(j=0;j<d;j++){
            tmp = 0;
            for(k=0;k<n;k++){
                tmp += A[j*n+k]*u[i*n+k];
            }
            g[i*d+j] = -tmp;
        }
    }
}

// base = trace(u' * r) + mu * ||u||_F^2/2, u is n by m, r is n by m
void get_base_mat(double *base, double *fro, double *u, double *r, double *mmu, int *nn, int *mm)
{
    int i,j,n,m;
    double mu,tmp1, tmp2;
    mu = *mmu;
    n = *nn; 
    m = *mm;
    tmp1 = 0;
    for(i=0;i<m;i++){
        for(j=0;j<n;j++){
            tmp1 += u[i*n+j]*r[i*n+j];
        }
    }
    tmp2 = 0;
    for(i=0;i<m;i++){
        for(j=0;j<n;j++){
            tmp2 += u[i*n+j]*u[i*n+j];
        }
    }
    *fro = tmp2;
    *base = tmp1 + mu*tmp2/2;
}

void dif_mat(double *x0, double *x1, double *x2, int *nn, int *mm)
{
    int i,j,n,m;
    
    n = *nn;
    m = *mm;
    for(i=0; i<m; i++){
        for(j=0; j<n; j++){
            x0[i*n+j] = x1[i*n+j] - x2[i*n+j];
        }
    }
}

void dif_mat2(double *x0, double *x1, double *x2, double *cc2, int *nn, int *mm)
{
    int i,j,n,m;
    double c2;
    
    n = *nn;
    m = *mm;
    c2 = *cc2;
    for(i=0; i<m; i++){
        for(j=0; j<n; j++){
            x0[i*n+j] = x1[i*n+j] - c2*x2[i*n+j];
        }
    }
}

// tr(x1'*x2), x1 is n by m, x2, is n by m
double tr_norm(double *x1, double *x2, int *nn, int *mm)
{
    int i,j,k,n,m;
    double trace;
    
    n = *nn;
    m = *mm;
    trace = 0;
    for(i=0; i<m; i++){
        for(j=0; j<m; j++){
            for(k=0; k<n; k++){
                trace += x1[i*n+k]*x2[j*n+k];
            }
        }
    }
    return trace;
}

// ||x||_F^2, x is n by m
double fro_norm(double *x, int *nn, int *mm)
{
    int i,j,n,m;
    double fro;
    
    n = *nn;
    m = *mm;
    fro = 0;
    for(i=0; i<m; i++){
        for(j=0; j<n; j++){
            fro += x[i*n+j]*x[i*n+j];
        }
    }
    return fro;
}

double lnorm_12(double *x, int *nn, int *mm)
{
    int i,j,n,m;
    double lnorm,tmp;
    
    n = *nn;
    m = *mm;
    lnorm = 0;
    for(i=0; i<n; i++){
        tmp = 0;
        for(j=0; j<m; j++){
            tmp += x[j*n+i]*x[j*n+i];
        }
        lnorm += sqrt(tmp);
    }
    return lnorm;
}

void trunc_svd(double *U, double *Vt, double *S, double *x, double *eps, int *nn, int *mm, int *min_nnmm)
{
    int i,j,k,n,m,min_nm;
    double zero, tmp;
    
    n = *nn;
    m = *mm;
    zero = 0;
    min_nm = *min_nnmm;
    for(i=0;i<min_nm;i++){
        S[i] = max(S[i]-*eps, zero);
    }
    for(i=0;i<m;i++){ // z1 = U*S*Vt, U is n by min_dm, S is min_dm, Vt is min_dm by m
        for(j=0;j<n;j++){
            tmp = 0;
            for(k=0;k<min_nm;k++){
                tmp += U[k*n+j]*S[k]*Vt[i*min_nm+k];
            }
            x[i*n+j] = tmp;
        }
    }
}

// x0 <- x1
void equ_mat(double *x0, double *x1, int *nn, int *mm)
{
    int i,j,n,m;
    
    n = *nn;
    m = *mm;
    
    for(i=0; i<m; i++){
        for(j=0; j<n; j++){
            x0[i*n+j] = x1[i*n+j];
        }
    }
}
