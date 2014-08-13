#include "mymath.h"

void picasso_logit_scad_greedy(double *Y, double * X, double * beta, double * intcpt, int * nn, int * dd, int * ite_lamb, int * ite_cyc, int * size_act, double *lambda, int *nnlambda, double *ggamma, int *mmax_ite, double *pprec){
    
    int i, k, m, n, d, max_ite1, max_ite2, nlambda, size_a, size_a1, comb_flag, match, ite1, ite2, c_idx, idx;
    double gamma, w, wn, g, prec1, prec2, ilambda, ilambda0, tmp, dif1, dif2;
    
    n = *nn;
    d = *dd;
    max_ite1 = *mmax_ite;
    max_ite2 = *mmax_ite;
    prec1 = *pprec;
    prec2 = *pprec;
    nlambda = *nnlambda;
    w = 0.25;
    wn = w*(double)n;
    gamma = *ggamma;
    
    double *beta2 = (double *) malloc(d*sizeof(double));
    double *beta1 = (double *) malloc(d*sizeof(double));
    double *beta0 = (double *) malloc(d*sizeof(double));
    int *set_act = (int *) malloc(d*sizeof(int));
    double *grad = (double *) malloc(d*sizeof(double));
    double *p = (double *) malloc(n*sizeof(double));
    double *p_Y = (double *) malloc(n*sizeof(double));
    double *Xb = (double *) malloc(n*sizeof(double));
    for(i=0;i<n;i++){
        Xb[i] = 0;
    }
    for(i=0;i<d;i++){
        beta2[i] = 0;
        beta1[i] = 0;
        beta0[i] = 0;
    }
    size_a = 0;
    
    for (i=0; i<nlambda; i++) {
        ilambda0 = lambda[i];
        ilambda = lambda[i]/w;
        if(i>0) {
            intcpt[i] = intcpt[i-1] - sum_vec_dif(p,Y,n)/wn;
        }
        ite1 = 0;
        dif1 = 1;
        while (dif1>prec1 && ite1<max_ite1) {
            p_update(p,Xb,intcpt[i],n); // p[i] = 1/(1+exp(-intcpt-Xb[i]))
            
            dif_vec(p_Y, p, Y, n); // p_Y = p - Y
            get_grad_logit_scad_vec(grad, p_Y, X, beta1, ilambda0, gamma, n, d); // grad = <p-Y, X>/n + h_grad(scad)
            idx = max_abs_idx(grad, d);
            
            comb_flag = 1;
            if(size_a>0){
                match = is_match(idx,set_act,size_a);
                if(match == 1) {
                    comb_flag = 0;
                }
            }
            if(comb_flag==1){
                set_act[size_a] = idx;
                size_a++;
            }
            ite2 = 0;
            dif2 = 1;
            while (dif2>prec2 && ite2<max_ite2) {
                intcpt[i] = intcpt[i] - sum_vec_dif(p,Y,n)/wn;
                for (m=0; m<size_a; m++) {
                    c_idx = set_act[m];
                    p_update(p,Xb,intcpt[i],n); // p[i] = 1/(1+exp(-intcpt-Xb[i]))
                    g = get_grad_logit_scad(p, Y, X+c_idx*n, beta1[c_idx], ilambda0, gamma, n); // g = <p-Y, X>/n + h_grad(scad)
                    tmp = beta1[c_idx] - g/w;
                    X_beta_update(Xb, X+c_idx*n, -beta1[c_idx], n); // X*beta = X*beta-X[,c_idx]*beta1[c_idx]
                    beta1[c_idx] = soft_thresh_l1(tmp, ilambda);
                    X_beta_update(Xb, X+c_idx*n, beta1[c_idx], n); // X*beta = X*beta+X[,c_idx]*beta1[c_idx]
                }
                ite2++;
                dif2 = dif_2norm(beta1, beta0, set_act, size_a);
                vec_copy(beta1, beta0, set_act, size_a);
            }
            ite_cyc[i] += ite2;
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
        ite_lamb[i] = ite1;
        vec_copy(beta1, beta+i*d, set_act, size_a);
        size_act[i] = size_a;
    }
    
    free(beta2);
    free(beta1);
    free(beta0);
    free(grad);
    free(set_act);
    free(p);
    free(p_Y);
    free(Xb);
}