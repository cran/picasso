#include "mymath.h"

void picasso_group_greedy_orth(double *Y, double * X, double * beta, double * intcpt, int * nn, int * dd, int * ite_lamb, int * ite_cyc, int * gr_size_act, double *obj, double *runt, int * gr, int * gr_nn, int * gr_size, double *lambda, int *nnlambda, double * ggamma, int *mmax_ite, double *pprec, int *fflag){
    
    int i, k, m, n, d, gr_n, max_ite1, max_ite2, nlambda, gr_size_a, gr_size_a1, comb_flag, match, ite1, ite2, c_idx, gr_idx, idx, flag;
    double gamma, prec1, prec2, ilambda, tmp, neg1, pos1, dif1, dif2, dbn, dbn1, max_norm2;
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
    gr_n = *gr_nn;
    neg1 = -1;
    pos1 = 1;
    dbn = (double)n;
    dbn1 = 1/dbn;
    
    double *beta2 = (double *) Calloc(d, double);
    double *beta1 = (double *) Calloc(d, double);
    double *beta0 = (double *) Calloc(d, double);
    double *beta_tild = (double *) Calloc(d, double);
    int *gr_act = (int *) Calloc(gr_n, int);
    double *y_hat = (double *) Calloc(n, double);
    double *grad = (double *) Calloc(d, double);
    for(i=0;i<n;i++){
        y_hat[i] = Y[i];
    }
    
    start = clock();
    gr_size_a = 0;
    for (i=0; i<nlambda; i++) {
        ilambda = lambda[i]*dbn;
        prec1 = (1+prec2*10)*ilambda;
        ite1 = 0;
        dif1 = prec1*2;
        while (dif1>prec1 && ite1<max_ite1) {
            intcpt[i] = mean(y_hat, n);
            dif_vec_const(y_hat, intcpt[i], n); //y_hat = y_hat - intcpt[i]
            vec_mat_prod(grad, y_hat, X, n, d); // grad = X^T grad
            max_norm2_gr(grad, gr, gr_size, gr_n, &max_norm2, &idx);
            
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
            dif_vec_const(y_hat, -intcpt[i], n); //y_hat = y_hat + intcpt[i]
            ite2 = 0;
            dif2 = 1;
            while (dif2>prec2 && ite2<max_ite2) {
                intcpt[i] = mean(y_hat, n);
                dif_vec_const(y_hat, intcpt[i], n); //y_hat = y_hat - intcpt[i]
                
                for (m=0; m<gr_size_a; m++) {
                    c_idx = gr_act[m];
                    gr_idx = gr[c_idx];
                    dif_vec_gr(y_hat, X, gr[c_idx], gr_size[c_idx], beta1, neg1, n); //y_hat = y_hat+X[,gr]*beta1[gr]
                    vec_inprod_gr(y_hat, X, beta_tild, gr[c_idx], gr_size[c_idx], n); //  beta_tild[gr] = X[,gr]^T*y_hat
                    tmp = norm2_gr_vec(beta_tild, gr[c_idx], gr_size[c_idx]);
                    for (k=gr_idx; k<gr_idx+gr_size[c_idx]; k++) {
                        if(flag==1){
                            beta1[k] = soft_thresh_gr_l1(tmp,ilambda,beta_tild[k],dbn1);
                        }
                        if(flag==2){
                            beta1[k] = soft_thresh_gr_mcp(tmp,ilambda,beta_tild[k],gamma,dbn1);
                        }
                        if(flag==3){
                            beta1[k] = soft_thresh_gr_scad(tmp,ilambda,beta_tild[k],gamma,dbn1);
                        }
                    }
                    dif_vec_gr(y_hat, X, gr[c_idx], gr_size[c_idx], beta1, pos1, n); //y_hat = y_hat-X[,gr]*beta1[gr]
                }
                dif_vec_const(y_hat, -intcpt[i], n); //y_hat = y_hat + intcpt[i]
                ite2++;
                dif2 = dif_2norm_gr(beta1, beta0, gr, gr_size, gr_act, gr_size_a);
                vec_copy_gr(beta1, beta0, gr, gr_size, gr_act, gr_size_a);
            }
            ite_cyc[i] += ite2;
            vec_mat_prod(grad, y_hat, X, n, d); // grad = X^T grad
            max_norm2_gr(grad, gr, gr_size, gr_n, &dif1, &idx);
            //dif1 = dif_2norm_gr(beta1, beta2, gr, gr_size, gr_act, gr_size_a);
            //vec_copy_gr(beta1, beta2, gr, gr_size, gr_act, gr_size_a);
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
        stop = clock();
        runt[i] = (double)(stop - start)/CLOCKS_PER_SEC;
        vec_copy_gr(beta1, beta+i*d, gr, gr_size, gr_act, gr_size_a);
        gr_size_act[i] = gr_size_a;
    }
    
    Free(beta2);
    Free(beta1);
    Free(beta0);
    Free(beta_tild);
    Free(gr_act);
    Free(y_hat);
    Free(grad);
}
