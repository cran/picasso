#include "mymath.h"

void picasso_lasso_scad_greedy(double *Y, double * X, double * S, double * beta, double * intcpt, int * nn, int * dd, int * ite_lamb, int * ite_cyc, int * size_act, double *lambda, int *nnlambda, double * ggamma, int *mmax_ite, double *pprec){
    
    int i, k, m, n, d, max_ite1, max_ite2, nlambda, size_a, size_a1, comb_flag, match, ite1, ite2, c_idx, idx;
    double gamma, prec1, prec2, ilambda, tmp, dif1, dif2, dbn;
    
    n = *nn;
    d = *dd;
    max_ite1 = *mmax_ite;
    max_ite2 = *mmax_ite;
    prec1 = *pprec;
    prec2 = *pprec;
    nlambda = *nnlambda;
    gamma = *ggamma;
    dbn = (double)n;
    
    double *beta2 = (double *) malloc(d*sizeof(double));
    double *beta1 = (double *) malloc(d*sizeof(double));
    double *beta0 = (double *) malloc(d*sizeof(double));
    int *set_act = (int *) malloc(d*sizeof(int));
    double *y_hat = (double *) malloc(n*sizeof(double));
    double *Xy_hat = (double *) malloc(d*sizeof(double));
    for(i=0;i<n;i++){
        y_hat[i] = Y[i];
    }
    for(i=0;i<d;i++){
        beta2[i] = 0;
        beta1[i] = 0;
        beta0[i] = 0;
    }
    size_a = 0;
    
    for (i=0; i<nlambda; i++) {
        ilambda = lambda[i]*dbn;
        ite1 = 0;
        dif1 = 1;
        while (dif1>prec1 && ite1<max_ite1) {
            intcpt[i] = mean(y_hat, n);
            dif_vec_const(y_hat, intcpt[i], n); //y_hat = y_hat - intcpt[i]
            vec_mat_prod(Xy_hat, y_hat, X, n, d); // Xy_hat = -X^T Xy_hat
            idx = max_abs_idx(Xy_hat, d);
            
            comb_flag = 1;
            if(size_a>0){
                match = is_match(idx,set_act,size_a);
                if(match == 1) {
                    comb_flag = 0;
                }
            }
            if(comb_flag==1){
                set_act[size_a] = idx;
                size_a++;
            }
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
                    beta1[c_idx] = soft_thresh_scad(tmp/S[c_idx], ilambda/S[c_idx], gamma);
                    dif_vec_vec(y_hat, X+c_idx*n, beta1[c_idx], n); //y_hat = y_hat-beta1[c_idx]*X[,c_idx]
                }
                dif_vec_const(y_hat, -intcpt[i], n); //y_hat = y_hat + intcpt[i]
                ite2++;
                dif2 = dif_2norm(beta1, beta0, set_act, size_a);
                vec_copy(beta1, beta0, set_act, size_a);
            }
            ite_cyc[i] += ite2;
            dif1 = dif_2norm(beta1, beta2, set_act, size_a);
            vec_copy(beta1, beta2, set_act, size_a);
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
        vec_copy(beta1, beta+i*d, set_act, size_a);
        size_act[i] = size_a;
    }
    
    free(beta2);
    free(beta1);
    free(beta0);
    free(set_act);
    free(y_hat);
    free(Xy_hat);
}
