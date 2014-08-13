#include "mymath.h"

void picasso_glasso_prox(double *Y, double * X, double * S, double * beta, double * intcpt, int * nn, int * dd, int * ite_lamb, int * ite_cyc, int * gr_size_act, int * gr, int * gr_nn, int * gr_size, double *lambda, int *nnlambda, int *mmax_ite, double *pprec, double *LL){
    
    int i, k, m, n, d, gr_n, max_ite1, max_ite2, nlambda, gr_size_a, gr_size_a1, ite1, ite2, c_idx, gr_idx;
    double L, prec1, prec2, ilambda, tmp, dif1, dif2, dbn;
    
    n = *nn;
    d = *dd;
    max_ite1 = *mmax_ite;
    max_ite2 = *mmax_ite;
    prec1 = *pprec;
    prec2 = *pprec;
    nlambda = *nnlambda;
    gr_n = *gr_nn;
    L = *LL;
    dbn = (double)n;
    
    double *beta2 = (double *) malloc(d*sizeof(double));
    double *beta1 = (double *) malloc(d*sizeof(double));
    double *beta0 = (double *) malloc(d*sizeof(double));
    double *beta_tild = (double *) malloc(d*sizeof(double));
    int *gr_act = (int *) malloc(gr_n*sizeof(int));
    double *y_hat = (double *) malloc(n*sizeof(double));
    double *grad = (double *) malloc(d*sizeof(double));
    for(i=0;i<n;i++){
        y_hat[i] = Y[i];
    }
    for(i=0;i<d;i++){
        beta2[i] = 0;
        beta1[i] = 0;
        beta0[i] = 0;
    }
    
    gr_size_a = 0;
    for (i=0; i<nlambda; i++) {
        ilambda = lambda[i]*dbn;
        ite1 = 0;
        dif1 = 1;
        while (dif1>prec1 && ite1<max_ite1) {
            intcpt[i] = mean(y_hat, n);
            dif_vec_const(y_hat, intcpt[i], n); //y_hat = y_hat - intcpt[i]
            vec_mat_prod(grad, y_hat, X, n, d); // grad = -X^T y_hat
            prox_beta_est(beta_tild, beta1, grad, L, ilambda/L, d); // beta_tild = soft(beta1-grad/L, ilambda)
            identfy_actgr(beta_tild, gr_act, &gr_size_a, gr, gr_size, gr_n);
            
            dif_vec_const(y_hat, -intcpt[i], n); //y_hat = y_hat + intcpt[i]
            ite2 = 0;
            dif2 = 1;
            while (dif2>prec2 && ite2<max_ite2) {
                intcpt[i] = mean(y_hat, n);
                dif_vec_const(y_hat, intcpt[i], n); //y_hat = y_hat - intcpt[i]
                
                for (m=0; m<gr_size_a; m++) {
                    c_idx = gr_act[m];
                    gr_idx = gr[c_idx];
                    for (k=gr_idx; k<gr_idx+gr_size[c_idx]; k++) {
                        dif_vec_vec(y_hat, X+k*n, -beta1[k], n); //y_hat = y_hat+beta1[k]*X[,k]
                        tmp = vec_inprod(y_hat, X+k*n, n);
                        if(tmp>ilambda){
                            rtfind(0,(tmp-ilambda)/S[k], beta1, k, gr_idx, gr_size[c_idx], tmp, ilambda, S[k]);
                        }else{
                            if(tmp<(-ilambda)){
                                rtfind((tmp+ilambda)/S[k], 0, beta1, k, gr_idx, gr_size[c_idx], tmp, ilambda, S[k]);
                            }else{
                                beta1[k] = 0;
                            }
                        }
                        dif_vec_vec(y_hat, X+k*n, beta1[k], n); //y_hat = y_hat-beta1[k]*X[,k]
                    }
                }
                dif_vec_const(y_hat, -intcpt[i], n); //y_hat = y_hat + intcpt[i]
                ite2++;
                dif2 = dif_2norm_gr(beta1, beta0, gr, gr_size, gr_act, gr_size_a);
                vec_copy_gr(beta1, beta0, gr, gr_size, gr_act, gr_size_a);
            }
            ite_cyc[i] += ite2;
            dif1 = dif_2norm_gr(beta1, beta2, gr, gr_size, gr_act, gr_size_a);
            vec_copy_gr(beta1, beta2, gr, gr_size, gr_act, gr_size_a);
            gr_size_a1 = 0;
            for (k=0; k<gr_size_a; k++) {
                c_idx = gr_act[k];
                if(norm2_gr_vec(beta1,gr[c_idx],gr_size[c_idx]) > 0){
                    gr_act[gr_size_a1] = c_idx;
                    gr_size_a1++;
                }
            }
            gr_size_a = gr_size_a1;
            ite1++;
        }
        ite_lamb[i] = ite1;
        vec_copy_gr(beta1, beta+i*d, gr, gr_size, gr_act, gr_size_a);
        gr_size_act[i] = gr_size_a;
    }
    
    free(beta2);
    free(beta1);
    free(beta0);
    free(beta_tild);
    free(gr_act);
    free(y_hat);
    free(grad);
}
