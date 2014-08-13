#include "mymath.h"

void picasso_glasso_cyclic(double *Y, double * X, double * S, double * beta, double * intcpt, int * nn, int * dd, int * ite1, int * gr_size_act, int * gr, int * gr_nn, int * gr_size, double *lambda, int *nnlambda, int *mmax_ite, double *pprec){
    
    int i, j, k, m, n, d, gr_n, max_ite, nlambda, gr_size_a, gr_size_a1, comb_flag, match, ite, c_idx, gr_idx;
    double prec, ilambda, tmp, dif, neg1, pos1, dbn;
    
    n = *nn;
    d = *dd;
    max_ite = *mmax_ite;
    prec = *pprec;
    nlambda = *nnlambda;
    gr_n = *gr_nn;
    neg1 = -1;
    pos1 = 1;
    dbn = (double)n;
    
    double *beta1 = (double *) malloc(d*sizeof(double));
    double *beta0 = (double *) malloc(d*sizeof(double));
    int *gr_act = (int *) malloc(gr_n*sizeof(int));
    double *y_hat = (double *) malloc(n*sizeof(double));
    for(i=0;i<n;i++){
        y_hat[i] = Y[i];
    }
    for(i=0;i<d;i++){
        beta1[i] = 0;
        beta0[i] = 0;
    }
    
    gr_size_a = 0;
    for (i=0; i<nlambda; i++) {
        ilambda = lambda[i]*dbn;
        intcpt[i] = mean(y_hat, n);
        dif_vec_const(y_hat, intcpt[i], n); //y_hat = y_hat - intcpt[i]
        for(j=0; j<gr_n; j++){
            match = is_match(j,gr_act,gr_size_a);
            if(match == 1){ // if j in set_act
                dif_vec_gr(y_hat, X, gr[j], gr_size[j], beta1, neg1, n); //y_hat = y_hat+X[,gr]*beta1[gr]
            }
            tmp = vec_inprod_gr_2norm(y_hat, X, gr[j], gr_size[j], n); // || y_hat^T*X[,gr] ||
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
                    dif_vec_const_gr(y_hat, X, gr[j], gr_size[j], beta1, pos1, -intcpt[i], n); //y_hat = y_hat-X[,gr]*beta1[gr] + intcpt[i]
                }else{
                    dif_vec_const(y_hat, -intcpt[i], n); //y_hat = y_hat + intcpt[i]
                }
                ite = 0;
                dif = 1;
                while (dif>prec && ite<max_ite) {
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
                    ite++;
                    dif = dif_2norm_gr(beta1, beta0, gr, gr_size, gr_act, gr_size_a);
                    vec_copy_gr(beta1, beta0, gr, gr_size, gr_act, gr_size_a);
                }
                dif_vec_const(y_hat, intcpt[i], n); //y_hat = y_hat - intcpt[i]
                ite1[i] += ite;
                gr_size_a1 = 0;
                for (k=0; k<gr_size_a; k++) {
                    c_idx = gr_act[k];
                    if(norm2_gr_vec(beta1,gr[c_idx],gr_size[c_idx]) > 0){
                        gr_act[gr_size_a1] = c_idx;
                        gr_size_a1++;
                    }
                }
                gr_size_a = gr_size_a1;
            }else{
                if(match == 1){ // if j in set_act
                    dif_vec_gr(y_hat, X, gr[j], gr_size[j], beta1, pos1, n); //y_hat = y_hat-X[,gr]*beta1[gr]
                }
            }
        }
        dif_vec_const(y_hat, -intcpt[i], n); //y_hat = y_hat + intcpt[i]
        vec_copy_gr(beta1, beta+i*d, gr, gr_size, gr_act, gr_size_a);
        gr_size_act[i] = gr_size_a;
    }
    
    free(beta1);
    free(beta0);
    free(gr_act);
    free(y_hat);
}
