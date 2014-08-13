#include "mymath.h"

void picasso_logit_gr_greedy(double *Y, double * X, double * beta, double * intcpt, int * nn, int * dd, int * ite_lamb, int * ite_cyc, int * gr_size_act, int * gr, int * gr_nn, int * gr_size, double *lambda, int *nnlambda, int *mmax_ite, double *pprec){
    
    int i, k, m, n, d, gr_n, max_ite1, max_ite2, nlambda, gr_size_a, gr_size_a1, comb_flag, match, ite1, ite2, c_idx, gr_idx, idx;
    double w, wn, prec1, prec2, ilambda, tmp, dif1, dif2;
    
    n = *nn;
    d = *dd;
    max_ite1 = *mmax_ite;
    max_ite2 = *mmax_ite;
    prec1 = *pprec;
    prec2 = *pprec;
    nlambda = *nnlambda;
    w = 0.25;
    wn = w*(double)n;
    gr_n = *gr_nn;
    
    double *beta2 = (double *) malloc(d*sizeof(double));
    double *beta1 = (double *) malloc(d*sizeof(double));
    double *beta0 = (double *) malloc(d*sizeof(double));
    double *grad = (double *) malloc(d*sizeof(double));
    double *p = (double *) malloc(n*sizeof(double));
    double *p_Y = (double *) malloc(n*sizeof(double));
    int *gr_act = (int *) malloc(gr_n*sizeof(int));
    double *Xb = (double *) malloc(n*sizeof(double));
    for(i=0;i<n;i++){
        Xb[i] = 0;
    }
    for(i=0;i<d;i++){
        beta2[i] = 0;
        beta1[i] = 0;
        beta0[i] = 0;
    }
    
    gr_size_a = 0;
    for (i=0; i<nlambda; i++) {
        ilambda = lambda[i]/w;
        if(i>0) {
            intcpt[i] = intcpt[i-1] - sum_vec_dif(p,Y,n)/wn;
        }
        ite1 = 0;
        dif1 = 1;
        while (dif1>prec1 && ite1<max_ite1) {
            p_update(p,Xb,intcpt[i],n); // p[i] = 1/(1+exp(-intcpt-Xb[i]))
            dif_vec(p_Y, p, Y, n); // p_Y = p - Y
            vec_mat_prod(grad, p_Y, X, n, d); // grad = -p_Y^T X
            idx = max_norm2_gr(grad, gr, gr_size, gr_n);
            
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
            ite2 = 0;
            dif2 = 1;
            while (dif2>prec2 && ite2<max_ite2) {
                intcpt[i] = intcpt[i] - sum_vec_dif(p,Y,n)/wn;
                for (m=0; m<gr_size_a; m++) {
                    c_idx = gr_act[m];
                    p_update(p,Xb,intcpt[i],n); // p[i] = 1/(1+exp(-intcpt-Xb[i]))
                    get_grad_logit_gr(grad, p, Y, X, gr[c_idx], gr_size[c_idx], n); // grad = X[gr]^T (p-Y)/n
                    gr_idx = gr[c_idx];
                    for (k=gr_idx; k<gr_idx+gr_size[c_idx]; k++) {
                        X_beta_update(Xb, X+k*n, -beta1[k], n); // X*beta = X*beta-X[,k]*beta1[k]
                        tmp = beta1[k] - grad[k]/w;
                        if(tmp>ilambda){
                            rtfind(0,tmp-ilambda, beta1, k, gr_idx, gr_size[c_idx], tmp, ilambda, 1);
                        }else{
                            if(tmp<(-ilambda)){
                                rtfind(tmp+ilambda, 0, beta1, k, gr_idx, gr_size[c_idx], tmp, ilambda, 1);
                            }else{
                                beta1[k] = 0;
                            }
                        }
                        X_beta_update(Xb, X+k*n, beta1[k], n); // X*beta = X*beta+X[,k]*beta1[k]
                    }
                }
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
    free(p);
    free(p_Y);
    free(grad);
    free(gr_act);
    free(Xb);
}
