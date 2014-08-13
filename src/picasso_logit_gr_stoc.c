#include "mymath.h"

void picasso_logit_gr_stoc(double *Y, double * X, double * beta, double * intcpt, int * nn, int * dd, int * ite1, int * gr_size_act, int * gr, int * gr_nn, int * gr_size, double *lambda, int *nnlambda, int *mmax_ite, double *pprec){
    
    int i, j, j1, k, m, n, d, gr_n, max_ite, nlambda, gr_size_a, gr_size_a1, comb_flag, match, ite, c_idx, gr_idx;
    double w, wn, prec, ilambda, tmp, dif;
    
    n = *nn;
    d = *dd;
    max_ite = *mmax_ite;
    prec = *pprec;
    nlambda = *nnlambda;
    w = 0.25;
    wn = w*(double)n;
    gr_n = *gr_nn;
    
    double *beta1 = (double *) malloc(d*sizeof(double));
    double *beta0 = (double *) malloc(d*sizeof(double));
    double *g = (double *) malloc(d*sizeof(double));
    int *gr_act = (int *) malloc(gr_n*sizeof(int));
    int *set_idx = (int *) malloc(gr_n*sizeof(int));
    double *p = (double *) malloc(n*sizeof(double));
    double *Xb = (double *) malloc(n*sizeof(double));
    for(i=0;i<n;i++){
        Xb[i] = 0;
    }
    for(i=0;i<d;i++){
        beta1[i] = 0;
        beta0[i] = 0;
    }
    for(i=0;i<gr_n;i++){
        set_idx[i] = i;
    }
    gr_size_a = 0;
    
    for (i=0; i<nlambda; i++) {
        ilambda = lambda[i]/w;
        if(i>0) {
            intcpt[i] = intcpt[i-1] - sum_vec_dif(p,Y,n)/wn;
        }
        shuffle(set_idx, gr_n);
        for(j1=0; j1<gr_n; j1++){
            j = set_idx[j1];
            p_update(p,Xb,intcpt[i],n); // p[i] = 1/(1+exp(-intcpt-Xb[i]))
            get_grad_logit_gr(g, p, Y, X, gr[j], gr_size[j], n); // g[gr] = X[gr]^T (p-Y)/n
            tmp = norm2_gr_vec_dif(beta1, g, w, gr[j], gr_size[j]); // || beta1[gr]-g[gr]/w ||
            if(tmp>ilambda){
                comb_flag = 1;
                match = is_match(j,gr_act,gr_size_a);
                if(gr_size_a>0){
                    if(match == 1) {
                        comb_flag = 0;
                    }
                }
                if(comb_flag==1){
                    gr_act[gr_size_a] = j;
                    gr_size_a++;
                }
                ite = 0;
                dif = 1;
                while (dif>prec && ite<max_ite) {
                    intcpt[i] = intcpt[i] - sum_vec_dif(p,Y,n)/wn;
                    for (m=0; m<gr_size_a; m++) {
                        c_idx = gr_act[m];
                        p_update(p,Xb,intcpt[i],n); // p[i] = 1/(1+exp(-intcpt-Xb[i]))
                        get_grad_logit_gr(g, p, Y, X, gr[c_idx], gr_size[c_idx], n); // g = X[gr]^T (p-Y)/n
                        gr_idx = gr[c_idx];
                        for (k=gr_idx; k<gr_idx+gr_size[c_idx]; k++) {
                            X_beta_update(Xb, X+k*n, -beta1[k], n); // X*beta = X*beta-X[,k]*beta1[k]
                            tmp = beta1[k] - g[k]/w;
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
                    ite++;
                    dif = dif_2norm_gr(beta1, beta0, gr, gr_size, gr_act, gr_size_a);
                    vec_copy_gr(beta1, beta0, gr, gr_size, gr_act, gr_size_a);
                }
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
            }
        }
        vec_copy_gr(beta1, beta+i*d, gr, gr_size, gr_act, gr_size_a);
        gr_size_act[i] = gr_size_a;
    }
    
    free(beta1);
    free(beta0);
    free(g);
    free(gr_act);
    free(set_idx);
    free(p);
    free(Xb);
}
