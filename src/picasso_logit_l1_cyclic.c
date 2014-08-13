#include "mymath.h"

void picasso_logit_l1_cyclic(double *Y, double * X, double * beta, double * intcpt, int * nn, int * dd, int * ite1, int * size_act, double *lambda, int *nnlambda, int *mmax_ite, double *pprec){
    
    int i, j, k, m, n, d, max_ite, nlambda, size_a, size_a1, comb_flag, match, ite, c_idx;
    double w, wn ,g, prec, ilambda, tmp, dif;
    
    n = *nn;
    d = *dd;
    max_ite = *mmax_ite;
    prec = *pprec;
    nlambda = *nnlambda;
    w = 0.25;
    wn = w*(double)n;
    
    double *beta1 = (double *) malloc(d*sizeof(double));
    double *beta0 = (double *) malloc(d*sizeof(double));
    int *set_act = (int *) malloc(d*sizeof(int));
    double *p = (double *) malloc(n*sizeof(double));
    double *Xb = (double *) malloc(n*sizeof(double));
    for(i=0;i<n;i++){
        Xb[i] = 0;
    }
    for(i=0;i<d;i++){
        beta1[i] = 0;
        beta0[i] = 0;
    }
    size_a = 0;
    
    for (i=0; i<nlambda; i++) {
        ilambda = lambda[i]/w;
        if(i>0) {
            intcpt[i] = intcpt[i-1] - sum_vec_dif(p,Y,n)/wn;
        }
        for(j=0; j<d; j++){
            p_update(p,Xb,intcpt[i],n); // p[i] = 1/(1+exp(-intcpt-Xb[i]))
            g = get_grad_logit_l1(p, Y, X+j*n, n); // g = <p-Y, X>/n
            tmp = beta1[j] - g/w;
            if(fabs(tmp)>ilambda){
                comb_flag = 1;
                match = is_match(j,set_act,size_a);
                if(size_a>0){
                    if(match == 1) {
                        comb_flag = 0;
                    }
                }
                if(comb_flag==1){
                    set_act[size_a] = j;
                    size_a++;
                }
                ite = 0;
                dif = 1;
                while (dif>prec && ite<max_ite) {
                    intcpt[i] = intcpt[i] - sum_vec_dif(p,Y,n)/wn;
                    for (m=0; m<size_a; m++) {
                        c_idx = set_act[m];
                        p_update(p,Xb,intcpt[i],n); // p[i] = 1/(1+exp(-intcpt-Xb[i]))
                        g = get_grad_logit_l1(p, Y, X+c_idx*n, n); // g = <p-Y, X>
                        tmp = beta1[c_idx] - g/w;
                        X_beta_update(Xb, X+c_idx*n, -beta1[c_idx], n); // X*beta = X*beta-X[,c_idx]*beta1[c_idx]
                        beta1[c_idx] = soft_thresh_l1(tmp, ilambda);
                        X_beta_update(Xb, X+c_idx*n, beta1[c_idx], n); // X*beta = X*beta+X[,c_idx]*beta1[c_idx]
                    }
                    ite++;
                    dif = dif_2norm(beta1, beta0, set_act, size_a);
                    vec_copy(beta1, beta0, set_act, size_a);
                }
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
            }
        }
        vec_copy(beta1, beta+i*d, set_act, size_a);
        size_act[i] = size_a;
    }
    
    free(beta1);
    free(beta0);
    free(set_act);
    free(p);
    free(Xb);
}
