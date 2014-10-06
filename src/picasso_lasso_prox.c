#include "mymath.h"

void picasso_lasso_prox(double *Y, double * X, double * S, double * beta, double * intcpt, int * nn, int * dd, int * ite_lamb, int * ite_cyc, int * size_act, double *obj, double *runt, double *lambda, int *nnlambda, double * ggamma, int *mmax_ite, double *pprec, double *LL, int *fflag){
    
    int i, k, m, n, d, max_ite1, max_ite2, nlambda, size_a, size_a1, ite1, ite2, c_idx, flag;
    double gamma, prec1, prec2, ilambda, tmp, dif1, dif2, L, dbn;
    clock_t start, stop;
    
    n = *nn;
    d = *dd;
    max_ite1 = *mmax_ite;
    max_ite2 = *mmax_ite;
    prec1 = *pprec;
    prec2 = *pprec;
    nlambda = *nnlambda;
    gamma = *ggamma;
    flag = *fflag;
    L = *LL;
    dbn = (double)n;
    
    double *beta2 = (double *) Calloc(d, double);
    double *beta1 = (double *) Calloc(d, double);
    double *beta0 = (double *) Calloc(d, double);
    double *beta_tild = (double *) Calloc(d, double);
    int *set_act = (int *) Calloc(d, int);
    double *y_hat = (double *) Calloc(n, double);
    double *grad = (double *) Calloc(d, double);
    for(i=0;i<n;i++){
        y_hat[i] = Y[i];
    }
    start = clock();
    size_a = 0;
    
    for (i=0; i<nlambda; i++) {
        ilambda = lambda[i]*dbn;
        prec1 = (1+prec2*10)*ilambda;
        ite1 = 0;
        dif1 = prec1*2;
        while (dif1>prec1 && ite1<max_ite1) {
            intcpt[i] = mean(y_hat, n);
            dif_vec_const(y_hat, intcpt[i], n); //y_hat = y_hat - intcpt[i]
            vec_mat_prod(grad, y_hat, X, n, d); // grad = X^T y_hat
            if(flag==1){
                prox_beta_est(beta_tild, beta1, grad, L, ilambda/L, d); // beta_tild = soft(beta1-grad/L, ilambda)
            }
            if(flag==2){
                prox_beta_est_mcp(beta_tild, beta1, grad, L, ilambda/L, gamma, d); // beta_tild = soft(beta1-grad/L, ilambda)
            }
            if(flag==3){
                prox_beta_est_scad(beta_tild, beta1, grad, L, ilambda/L, gamma, d); // beta_tild = soft(beta1-grad/L, ilambda)
            }
            identfy_actset(beta_tild, set_act, &size_a, d);
            
            dif_vec_const(y_hat, -intcpt[i], n); //y_hat = y_hat + intcpt[i]
            ite2 = 0;
            dif2 = 1;
            while (dif2>prec2 && ite2<max_ite2) {
                intcpt[i] = mean(y_hat, n);
                dif_vec_const(y_hat, intcpt[i], n); //y_hat = y_hat - intcpt[i]
                
                for (m=0; m<size_a; m++) {
                    c_idx = set_act[m];
                    dif_vec_vec(y_hat, X+c_idx*n, -beta1[c_idx], n); //y_hat = y_hat+beta1[c_idx]*X[,c_idx]
                    tmp = vec_inprod(y_hat, X+c_idx*n, n);
                    if(flag==1){
                        beta1[c_idx] = soft_thresh_l1(tmp/S[c_idx], ilambda/S[c_idx]);
                    }
                    if(flag==2){
                        beta1[c_idx] = soft_thresh_mcp(tmp/S[c_idx], ilambda/S[c_idx], gamma);
                    }
                    if(flag==3){
                        beta1[c_idx] = soft_thresh_scad(tmp/S[c_idx], ilambda/S[c_idx], gamma);
                    }
                    dif_vec_vec(y_hat, X+c_idx*n, beta1[c_idx], n); //y_hat = y_hat-beta1[c_idx]*X[,c_idx]
                }
                dif_vec_const(y_hat, -intcpt[i], n); //y_hat = y_hat + intcpt[i]
                ite2++;
                dif2 = dif_2norm(beta1, beta0, set_act, size_a);
                vec_copy(beta1, beta0, set_act, size_a);
            }
            ite_cyc[i] += ite2;
            vec_mat_prod(grad, y_hat, X, n, d); // grad = X^T grad
            dif1 = max_abs_vec(grad, d);
            //dif1 = dif_2norm(beta1, beta2, set_act, size_a);
            //vec_copy(beta1, beta2, set_act, size_a);
            size_a1 = 0;
            for (k=0; k<size_a; k++) {
                c_idx = set_act[k];
                if(beta1[c_idx]!=0){
                    set_act[size_a1] = c_idx;
                    size_a1++;
                }
            }
            size_a = size_a1;
            ite1++;
        }
        ite_lamb[i] = ite1;
        stop = clock();
        runt[i] = (double)(stop - start)/CLOCKS_PER_SEC;
        vec_copy(beta1, beta+i*d, set_act, size_a);
        size_act[i] = size_a;
    }
    
    Free(beta2);
    Free(beta1);
    Free(beta0);
    Free(beta_tild);
    Free(set_act);
    Free(y_hat);
    Free(grad);
}
