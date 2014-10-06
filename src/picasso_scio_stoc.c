#include "mymath.h"

void picasso_scio_stoc(double * S, double * beta, int * dd, int * ite_lamb, int * ite_cyc, double *lambda, int *nnlambda, int *mmax_ite, double *pprec, double * x, int *col_cnz, int *row_idx, double *obj, double *runt, double * ggamma, int *fflag){
    
    int i, j, j1, k, m, d, d_sq, col, max_ite, nlambda, size_a, size_a1, comb_flag, match, ite, ite1, c_idx, cnz, flag;
    double prec, ilambda, tmp, dif, gamma;
    clock_t start, stop;
    
    d = *dd;
    d_sq = d*d;
    max_ite = *mmax_ite;
    prec = *pprec;
    nlambda = *nnlambda;
    cnz = 0;
    gamma = *ggamma;
    flag = *fflag;
    
    double *beta1 = (double *) Calloc(d, double);
    double *beta0 = (double *) Calloc(d, double);
    int *set_act = (int *) Calloc(d, int);
    int *set_idx = (int *) Calloc(d, int);
    double *e = (double *) Calloc(d, double);
    
    for (col=0; col<d; col++) {
        
        start = clock();
        for(i=0;i<d;i++){
            beta1[i] = 0;
            beta0[i] = 0;
            e[i] = 0;
            set_idx[i] = i;
        }
        e[col] = 1;
        beta1[col] = 1;
        beta0[col] = 1;
        size_a = 0;
        
        for (i=0; i<nlambda; i++) {
            ilambda = lambda[i];
            shuffle(set_idx, d);
            ite1 = 0;
            for(j1=0; j1<d; j1++){
                j = set_idx[j1];
                tmp = res(e[j], S+j*d, beta1, set_act, size_a, j);
                
                if(fabs(tmp)>ilambda){
                    match = is_match(j,set_act,size_a);
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
                    ite = 0;
                    dif = 1;
                    while (dif>prec && ite<max_ite) {
                        for (m=0; m<size_a; m++) {
                            c_idx = set_act[m];
                            tmp = res(e[c_idx], S+c_idx*d, beta1, set_act, size_a, c_idx);
                            if(flag==1){
                                beta1[c_idx] = soft_thresh_l1(tmp/S[c_idx*d+c_idx], ilambda/S[c_idx*d+c_idx]);
                            }
                            if(flag==2){
                                beta1[c_idx] = soft_thresh_mcp(tmp/S[c_idx*d+c_idx], ilambda/S[c_idx*d+c_idx], gamma);
                            }
                            if(flag==3){
                                beta1[c_idx] = soft_thresh_scad(tmp/S[c_idx*d+c_idx], ilambda/S[c_idx*d+c_idx], gamma);
                            }
                        }
                        ite++;
                        dif = dif_2norm(beta1, beta0, set_act, size_a);
                        vec_copy(beta1, beta0, set_act, size_a);
                    }
                    ite_cyc[i*d+col] += ite;
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
            }
            ite_lamb[i*d+col] = ite1;
            stop = clock();
            runt[i*d+col] = (double)(stop - start)/CLOCKS_PER_SEC;
            
            for(j=0; j<size_a; j++){
                c_idx=set_act[j];
                beta[i*d_sq+col*d+c_idx] = beta1[c_idx];
                if(c_idx != col) {
                    x[cnz] = beta1[c_idx];
                    row_idx[cnz] = i*d+c_idx;
                    cnz++;
                }
            }
        }
        col_cnz[col+1]=cnz;
    }
    
    Free(beta1);
    Free(beta0);
    Free(set_act);
    Free(set_idx);
    Free(e);
}
