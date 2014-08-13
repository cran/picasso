#include "mymath.h"

void picasso_lasso_mcp_stoc(double *Y, double * X, double * S, double * beta, double * intcpt, int * nn, int * dd, int * ite1, int * size_act, double *lambda, int *nnlambda, double * ggamma, int *mmax_ite, double *pprec){
    
    int i, j, j1, k, m, n, d, max_ite, nlambda, size_a, size_a1, comb_flag, match, ite, c_idx;
    double gamma, prec, ilambda, tmp, dif, dbn;
    
    n = *nn;
    d = *dd;
    max_ite = *mmax_ite;
    prec = *pprec;
    nlambda = *nnlambda;
    gamma = *ggamma;
    dbn = (double)n;
    
    double *beta1 = (double *) malloc(d*sizeof(double));
    double *beta0 = (double *) malloc(d*sizeof(double));
    int *set_act = (int *) malloc(d*sizeof(int));
    int *set_idx = (int *) malloc(d*sizeof(int));
    double *y_hat = (double *) malloc(n*sizeof(double));
    for(i=0;i<n;i++){
        y_hat[i] = Y[i];
    }
    for(i=0;i<d;i++){
        beta1[i] = 0;
        beta0[i] = 0;
        set_idx[i] = i;
    }
    size_a = 0;
    
    for (i=0; i<nlambda; i++) {
        ilambda = lambda[i]*dbn;
        intcpt[i] = mean(y_hat, n);
        dif_vec_const(y_hat, intcpt[i], n); //y_hat = y_hat - intcpt[i]
        shuffle(set_idx, d);
        for(j1=0; j1<d; j1++){
            j = set_idx[j1];
            match = is_match(j,set_act,size_a);
            if(match == 1){ // if j in set_act
                dif_vec_vec(y_hat, X+j*n, -beta1[j], n); //y_hat = y_hat+beta1[j]*X[,j]
            }
            tmp = vec_inprod(y_hat, X+j*n, n);

            if(fabs(tmp)>ilambda){
                comb_flag = 1;
                if(size_a>0){
                    if(match == 1) {
                        comb_flag = 0;
                    }
                }
                if(comb_flag==1){
                    set_act[size_a] = j;
                    size_a++;
                }
                if(match == 1){
                    dif_vec_vec_const(y_hat, X+j*n, beta1[j], -intcpt[i], n); //y_hat = y_hat-beta1[j]*X[,j]+intcpt[i]
                }else{
                    dif_vec_const(y_hat, -intcpt[i], n); //y_hat = y_hat + intcpt[i]
                }
                ite = 0;
                dif = 1;
                while (dif>prec && ite<max_ite) {
                    intcpt[i] = mean(y_hat, n);
                    dif_vec_const(y_hat, intcpt[i], n); //y_hat = y_hat - intcpt[i]

                    for (m=0; m<size_a; m++) {
                        c_idx = set_act[m];
                        dif_vec_vec(y_hat, X+c_idx*n, -beta1[c_idx], n); //y_hat = y_hat+beta1[c_idx]*X[,c_idx]
                        tmp = vec_inprod(y_hat, X+c_idx*n, n);
                        beta1[c_idx] = soft_thresh_mcp(tmp/S[c_idx], ilambda/S[c_idx], gamma);
                        dif_vec_vec(y_hat, X+c_idx*n, beta1[c_idx], n); //y_hat = y_hat-beta1[c_idx]*X[,c_idx]
                    }
                    dif_vec_const(y_hat, -intcpt[i], n); //y_hat = y_hat + intcpt[i]
                    ite++;
                    dif = dif_2norm(beta1, beta0, set_act, size_a);
                    vec_copy(beta1, beta0, set_act, size_a);
                }
                dif_vec_const(y_hat, intcpt[i], n); //y_hat = y_hat - intcpt[i]
                ite1[i] += ite;
                size_a1 = 0;
                for (k=0; k<size_a; k++) {
                    c_idx = set_act[k];
                    if(beta1[c_idx]!=0){
                        set_act[size_a1] = c_idx;
                        size_a1++;
                    }
                }
                size_a = size_a1;
            }else{
                if(match == 1){ // if j in set_act
                    dif_vec_vec(y_hat, X+j*n, beta1[j], n); //y_hat = y_hat-beta1[j]*X[,j]
                }
            }
        }
        dif_vec_const(y_hat, -intcpt[i], n); //y_hat = y_hat + intcpt[i]
        vec_copy(beta1, beta+i*d, set_act, size_a);
        size_act[i] = size_a;
    }
    
    free(beta1);
    free(beta0);
    free(set_act);
    free(set_idx);
    free(y_hat);
}
