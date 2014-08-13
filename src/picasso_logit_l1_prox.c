#include "mymath.h"

void picasso_logit_l1_prox(double *Y, double * X, double * beta, double * intcpt, int * nn, int * dd, int * ite_lamb, int * ite_cyc, int * size_act, double *lambda, int *nnlambda, int *mmax_ite, double *pprec, double *LL){
    
    int i, k, m, n, d, max_ite1, max_ite2, nlambda, size_a, size_a1, ite1, ite2, c_idx;
    double w, wn, g, L, prec1, prec2, ilambda, ilambda0, tmp, dif1, dif2;
    
    n = *nn;
    d = *dd;
    max_ite1 = *mmax_ite;
    max_ite2 = *mmax_ite;
    prec1 = *pprec;
    prec2 = *pprec;
    nlambda = *nnlambda;
    w = 0.25;
    L = (*LL)*w;
    wn = w*(double)n;
    
    double *beta2 = (double *) malloc(d*sizeof(double));
    double *beta1 = (double *) malloc(d*sizeof(double));
    double *beta0 = (double *) malloc(d*sizeof(double));
    double *beta_tild = (double *) malloc(d*sizeof(double));
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
            get_grad_logit_l1_vec(grad, p_Y, X, n, d); // grad = <p-Y, X>
            prox_beta_est(beta_tild, beta1, grad, L, ilambda0/L, d); // beta_tild = soft(beta1-grad/L, ilambda)
            identfy_actset(beta_tild, set_act, &size_a, d);
            
            ite2 = 0;
            dif2 = 1;
            while (dif2>prec2 && ite2<max_ite2) {
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
    free(beta_tild);
    free(grad);
    free(set_act);
    free(p);
    free(p_Y);
    free(Xb);
}
