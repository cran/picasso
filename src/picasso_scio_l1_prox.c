#include "mymath.h"

void picasso_scio_l1_prox(double * S, double * beta, int * dd, int * ite_lamb, int * ite_cyc, double *lambda, int *nnlambda, int *mmax_ite, double *pprec, double * x, int *col_cnz, int *row_idx, double *LL){
    
    int i, j, k, m, d, d_sq, col, max_ite1, max_ite2, nlambda, size_a, size_a1, ite1, ite2, c_idx, cnz;
    double L, prec1, prec2, ilambda, tmp, dif1, dif2;
    
    d = *dd;
    d_sq = d*d;
    max_ite1 = *mmax_ite;
    max_ite2 = *mmax_ite;
    prec1 = *pprec;
    prec2 = *pprec;
    nlambda = *nnlambda;
    L = *LL;
    cnz = 0;
    
    double *beta2 = (double *) malloc(d*sizeof(double));
    double *beta1 = (double *) malloc(d*sizeof(double));
    double *beta0 = (double *) malloc(d*sizeof(double));
    double *beta_tild = (double *) malloc(d*sizeof(double));
    int *set_act = (int *) malloc(d*sizeof(int));
    double *e = (double *) malloc(d*sizeof(double));
    double *grad = (double *) malloc(d*sizeof(double));
    
    for (col=0; col<d; col++) {
        
        for(i=0;i<d;i++){
            beta2[i] = 0;
            beta1[i] = 0;
            beta0[i] = 0;
            e[i] = 0;
        }
        e[col] = 1;
        beta1[col] = 1;
        beta0[col] = 1;
        size_a = 0;
        
        for (i=0; i<nlambda; i++) {
            ilambda = lambda[i];
            ite1 = 0;
            dif1 = 1;
            while (dif1>prec1 && ite1<max_ite1) {
                grad_scio(grad, e, S, beta1, set_act, size_a, d);
                prox_beta_est(beta_tild, beta1, grad, L, ilambda/L, d); // beta_tild = soft(beta1-grad/L, ilambda)
                identfy_actset(beta_tild, set_act, &size_a, d);
                
                ite2 = 0;
                dif2 = 1;
                while (dif2>prec2 && ite2<max_ite2) {
                    for (m=0; m<size_a; m++) {
                        c_idx = set_act[m];
                        tmp = res(e[c_idx], S+c_idx*d, beta1, set_act, size_a, c_idx);
                        beta1[c_idx] = soft_thresh_l1(tmp/S[c_idx*d+c_idx], ilambda/S[c_idx*d+c_idx]);
                    }
                    ite2++;
                    dif2 = dif_2norm(beta1, beta0, set_act, size_a);
                    vec_copy(beta1, beta0, set_act, size_a);
                }
                ite_cyc[i*d+col] += ite2;
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
            
            ite_lamb[i*d+col] += ite1;
            for(j=0; j<size_a; j++){
                c_idx=set_act[j];
                beta[i*d_sq+col*d+c_idx] = beta1[c_idx];
                if(c_idx != col) {
                    x[cnz] = beta1[c_idx];
                    row_idx[cnz] = i*d+c_idx;
                    cnz++;
                }
            }
        }
        col_cnz[col+1]=cnz;
    }
    
    free(beta2);
    free(beta1);
    free(beta0);
    free(beta_tild);
    free(set_act);
    free(e);
    free(grad);
}
