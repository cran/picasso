#include "mymath.h"

void picasso_lasso_stoc(double *Y, double * X, double * S, double * beta, double * intcpt, int * nn, int * dd, int * ite_lamb, int * ite_cyc, int * size_act, double *obj, double *runt, double *lambda, int *nnlambda, double * ggamma, int *mmax_ite, double *pprec, int *fflag, int *mmax_act_in, double *ttrunc){
    
    int i, j, j1, k, m, n, d, max_ite1, max_ite2, nlambda, size_a, size_a1, match, ite2, ite1, c_idx, flag, max_act_in, act_in;
    double gamma, prec1, prec2, ilambda, tmp, dif1, dif2, dbn, ilambda1, trunc;
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
    dbn = (double)n;
    max_act_in = *mmax_act_in;
    trunc = *ttrunc;
    
    double *beta2 = (double *) Calloc(d, double);
    double *beta1 = (double *) Calloc(d, double);
    double *beta0 = (double *) Calloc(d, double);
    int *set_act = (int *) Calloc(d, int);
    int *set_idx = (int *) Calloc(d, int);
    double *y_hat = (double *) Calloc(n, double);
    double *grad = (double *) Calloc(d, double);
    for(i=0;i<n;i++){
        y_hat[i] = Y[i];
    }
    for(i=0;i<d;i++){
        set_idx[i] = i;
    }
    start = clock();
    size_a = 0;
    
    for (i=0; i<nlambda; i++) {
        ilambda = lambda[i]*dbn;
        ilambda1 = ilambda*(1+trunc);
        prec1 = (1+trunc)*ilambda;
        ite1 = 0;
        dif1 = prec1*2;
        while (dif1>prec1 && ite1<max_ite1) {
            intcpt[i] = mean(y_hat, n);
            dif_vec_const(y_hat, intcpt[i], n); //y_hat = y_hat - intcpt[i]
            act_in = 0;
            shuffle(set_idx, d);
            for(j1=0; j1<d; j1++){
                j = set_idx[j1];
                match = is_match(j,set_act,size_a);
                if(match == 0){ // if j in set_act
                    tmp = vec_inprod(y_hat, X+j*n, n);
                    if(fabs(tmp)>ilambda1){
                        set_act[size_a] = j;
                        size_a++;
                        act_in++;
                    }
                    if(act_in == max_act_in){
                        break;
                    }
                }
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
            vec_mat_prod(grad, y_hat, X, n, d); // grad = X^T y_hat
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
    Free(set_act);
    Free(set_idx);
    Free(y_hat);
}
