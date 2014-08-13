#include "mymath.h"

// Y n by p
// X n by d
// beta d by p
void picasso_mvr_cyclic(double *Y, double * X, double * S, double * beta, double * intcpt, int * nn, int * dd, int * pp, int * ite1, int * gr_size_act, double *lambda, int *nnlambda, int *mmax_ite, double *pprec){
    
    int i, j, k, m, p, n, d, max_ite, nlambda, gr_size_a, gr_size_a1, comb_flag, match, ite, c_idx;
    double prec, ilambda, tmp, dif, neg1, pos1;
    
    n = *nn;
    d = *dd;
    p = *pp;
    max_ite = *mmax_ite;
    prec = *pprec;
    nlambda = *nnlambda;
    neg1 = -1;
    pos1 = 1;
    
    double *beta1 = (double *) malloc(d*p*sizeof(double));
    double *beta0 = (double *) malloc(d*p*sizeof(double));
    int *gr_act = (int *) malloc(d*sizeof(int));
    double *y_hat = (double *) malloc(n*p*sizeof(double));
    for(i=0;i<p;i++){
        for (j=0; j<n; j++) {
            y_hat[i*n+j] = Y[i*n+j];
        }
    }
    for(i=0;i<p;i++){
        for (j=0; j<d; j++) {
            beta1[i*d+j] = 0;
            beta0[i*d+j] = 0;
        }
    }
    
    gr_size_a = 0;
    for (i=0; i<nlambda; i++) {
        ilambda = lambda[i]*n*p;
        mean_mvr(intcpt+i*p, y_hat, n, p);
        dif_vec_const_mvr(y_hat, intcpt+i*p, pos1, n, p); //y_hat = y_hat - intcpt[i]
        for(j=0; j<d; j++){
            match = is_match(j,gr_act,gr_size_a);
            if(match == 1){ // if j in set_act
                dif_mat_mvr(y_hat, X+j*n, beta1+j, neg1, n, d, p); //y_hat = y_hat+X[,j]*beta1[j,]
            }
            tmp = vec_mat_inprod_2norm(y_hat, X+j*n, n, p); // || X[,j]^T y_hat ||
            
            if(tmp>ilambda){
                comb_flag = 1;
                if(gr_size_a>0){
                    if(match == 1) {
                        comb_flag = 0;
                    }
                }
                if(comb_flag==1){
                    gr_act[gr_size_a] = j;
                    gr_size_a++;
                }
                if(match == 1){
                    dif_mat_mvr(y_hat, X+j*n, beta1+j, pos1, n, d, p); //y_hat = y_hat-X[,j]*beta1[j,]
                    dif_vec_const_mvr(y_hat, intcpt+i*p, neg1, n, p); //y_hat = y_hat + intcpt[i]
                }else{
                    dif_vec_const_mvr(y_hat, intcpt+i*p, neg1, n, p); //y_hat = y_hat + intcpt[i]
                }
                ite = 0;
                dif = 1;
                while (dif>prec && ite<max_ite) {
                    mean_mvr(intcpt+i*p, y_hat, n, p);
                    dif_vec_const_mvr(y_hat, intcpt+i*p, pos1, n, p); //y_hat = y_hat - intcpt[i]

                    for (m=0; m<gr_size_a; m++) {
                        c_idx = gr_act[m];
                        dif_mat_mvr(y_hat, X+c_idx*n, beta1+c_idx, neg1, n, d, p); //y_hat = y_hat+X[,c_idx]*beta1[c_idx,]
                        for (k=0; k<p; k++) {
                            tmp = vec_inprod(y_hat+k*n, X+c_idx*n, n);
                            if(tmp>ilambda){
                                rtfind_mvr(0,(tmp-ilambda)/S[c_idx], beta1, k, c_idx, d, p, tmp, ilambda, S[c_idx]);
                            }else{
                                if(tmp<(-ilambda)){
                                    rtfind_mvr((tmp+ilambda)/S[c_idx], 0, beta1, k, c_idx, d, p, tmp, ilambda, S[c_idx]);
                                }else{
                                    beta1[k*d+c_idx] = 0;
                                }
                            }
                        }
                        dif_mat_mvr(y_hat, X+c_idx*n, beta1+c_idx, pos1, n, d, p); //y_hat = y_hat-X[,c_idx]*beta1[c_idx,]
                    }
                    dif_vec_const_mvr(y_hat, intcpt+i*p, neg1, n, p); //y_hat = y_hat + intcpt[i]
                    ite++;
                    dif = dif_Fnorm_mvr(beta1, beta0, gr_act, gr_size_a, d, p);
                    mat_copy_mvr(beta1, beta0, gr_act, gr_size_a, d, p);
                }
                dif_vec_const_mvr(y_hat, intcpt+i*p, pos1, n, p); //y_hat = y_hat - intcpt[i]
                ite1[i] += ite;
                gr_size_a1 = 0;
                for (k=0; k<gr_size_a; k++) {
                    c_idx = gr_act[k];
                    if(norm2_gr_mvr(beta1+c_idx, d, p) > 0){
                        gr_act[gr_size_a1] = c_idx;
                        gr_size_a1++;
                    }
                }
                gr_size_a = gr_size_a1;
            }else{
                if(match == 1){ // if j in set_act
                    dif_mat_mvr(y_hat, X+j*n, beta1+j, pos1, n, d, p); //y_hat = y_hat-X[,j]*beta1[j,]
                }
            }
        }
        dif_vec_const_mvr(y_hat, intcpt+i*p, neg1, n, p); //y_hat = y_hat + intcpt[i]
        
        mat_copy_mvr(beta1, beta+i*d*p, gr_act, gr_size_a, d, p);
        gr_size_act[i] = gr_size_a;
    }
    
    free(beta1);
    free(beta0);
    free(gr_act);
    free(y_hat);
}
