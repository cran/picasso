#include "mymath.h"

// Y n by p
// X n by d
// beta d by p
void picasso_mvr_greedy(double *Y, double * X, double * S, double * beta, double * intcpt, int * nn, int * dd, int * pp, int * ite_lamb, int * ite_cyc, int * gr_size_act, double *lambda, int *nnlambda, int *mmax_ite, double *pprec){
    
    int i, j, k, m, p, n, d, max_ite1, max_ite2, nlambda, gr_size_a, gr_size_a1, comb_flag, match, ite1, ite2, c_idx, idx;
    double prec1, prec2, ilambda, tmp, dif1, dif2, neg1, pos1;
    
    n = *nn;
    d = *dd;
    p = *pp;
    max_ite1 = *mmax_ite;
    max_ite2 = *mmax_ite;
    prec1 = *pprec;
    prec2 = *pprec;
    nlambda = *nnlambda;
    neg1 = -1;
    pos1 = 1;
    
    double *beta2 = (double *) malloc(d*p*sizeof(double));
    double *beta1 = (double *) malloc(d*p*sizeof(double));
    double *beta0 = (double *) malloc(d*p*sizeof(double));
    int *gr_act = (int *) malloc(d*sizeof(int));
    double *y_hat = (double *) malloc(n*p*sizeof(double));
    double *grad = (double *) malloc(d*p*sizeof(double));
    double *grad_row2 = (double *) malloc(d*sizeof(double));
    for(i=0;i<p;i++){
        for (j=0; j<n; j++) {
            y_hat[i*n+j] = Y[i*n+j];
        }
    }
    for(i=0;i<p;i++){
        for (j=0; j<d; j++) {
            beta2[i*d+j] = 0;
            beta1[i*d+j] = 0;
            beta0[i*d+j] = 0;
        }
    }
    
    gr_size_a = 0;
    for (i=0; i<nlambda; i++) {
        ilambda = lambda[i]*n*p;
        ite1 = 0;
        dif1 = 1;
        while (dif1>prec1 && ite1<max_ite1) {
            mean_mvr(intcpt+i*p, y_hat, n, p);
            dif_vec_const_mvr(y_hat, intcpt+i*p, pos1, n, p); //y_hat = y_hat - intcpt[i]
            vec_mat_prod_mvr(grad, y_hat, X, p, n, d); // grad = -X^T y_hat
            norm2_row_mat(grad_row2, grad, d, p);
            idx = max_idx(grad_row2, d);
            
            comb_flag = 1;
            if(gr_size_a>0){
                match = is_match(idx,gr_act,gr_size_a);
                if(match == 1) {
                    comb_flag = 0;
                }
            }
            if(comb_flag==1){
                gr_act[gr_size_a] = idx;
                gr_size_a++;
            }
            dif_vec_const_mvr(y_hat, intcpt+i*p, neg1, n, p); //y_hat = y_hat + intcpt[i]
            ite2 = 0;
            dif2 = 1;
            while (dif2>prec2 && ite2<max_ite2) {
                mean_mvr(intcpt+i*p, y_hat, n, p);
                dif_vec_const_mvr(y_hat, intcpt+i*p, pos1, n, p); //y_hat = y_hat - intcpt[i]
                
                for (m=0; m<gr_size_a; m++) {
                    c_idx = gr_act[m];
                    dif_mat_mvr(y_hat, X+c_idx*n, beta1+c_idx, neg1, n, d, p); //y_hat = y_hat+X[,c_idx]*beta1[c_idx,]
                    for (k=0; k<p; k++) {
                        tmp = vec_inprod(y_hat+k*n, X+c_idx*n, n);
                        if(tmp>ilambda){
                            rtfind_mvr(0,(tmp-ilambda)/S[k], beta1, k, c_idx, d, p, tmp, ilambda, S[k]);
                        }else{
                            if(tmp<(-ilambda)){
                                rtfind_mvr((tmp+ilambda)/S[k], 0, beta1, k, c_idx, d, p, tmp, ilambda, S[k]);
                            }else{
                                beta1[k*d+c_idx] = 0;
                            }
                        }
                    }
                    dif_mat_mvr(y_hat, X+c_idx*n, beta1+c_idx, pos1, n, d, p); //y_hat = y_hat-X[,c_idx]*beta1[c_idx,]
                }
                dif_vec_const_mvr(y_hat, intcpt+i*p, neg1, n, p); //y_hat = y_hat + intcpt[i]
                ite2++;
                dif2 = dif_Fnorm_mvr(beta1, beta0, gr_act, gr_size_a, d, p);
                mat_copy_mvr(beta1, beta0, gr_act, gr_size_a, d, p);
            }
            ite_cyc[i] += ite2;
            dif1 = dif_Fnorm_mvr(beta1, beta2, gr_act, gr_size_a, d, p);
            mat_copy_mvr(beta1, beta2, gr_act, gr_size_a, d, p);
            gr_size_a1 = 0;
            for (k=0; k<gr_size_a; k++) {
                c_idx = gr_act[k];
                if(norm2_gr_mvr(beta1+c_idx, d, p) > 0){
                    gr_act[gr_size_a1] = c_idx;
                    gr_size_a1++;
                }
            }
            gr_size_a = gr_size_a1;
            ite1++;
        }
        ite_lamb[i] = ite1;
        mat_copy_mvr(beta1, beta+i*d*p, gr_act, gr_size_a, d, p);
        gr_size_act[i] = gr_size_a;
    }
    
    free(beta2);
    free(beta1);
    free(beta0);
    free(gr_act);
    free(y_hat);
    free(grad);
    free(grad_row2);
}
