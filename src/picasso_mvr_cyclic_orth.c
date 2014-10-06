#include "mymath.h"

// Y n by p
// X n by d
// beta d by p
void picasso_mvr_cyclic_orth(double *Y, double * X, double * beta, double * intcpt, int * nn, int * dd, int * pp, int * ite_lamb, int * ite_cyc, int * gr_size_act, double *obj, double *runt, double *xinvc, double *lambda, int *nnlambda, double * ggamma, int *mmax_ite, double *pprec, double *uinv, int *fflag, int *mmax_act_gr_in, double *ttrunc){
    
    int i, j, k, m, p, n, d, max_ite1, max_ite2, nlambda, gr_size_a, gr_size_a1, match, ite1, ite2, c_idx, flag, max_act_gr_in, act_gr_in;
    double gamma, prec1, prec2, ilambda, tmp, dif1, dif2, neg1, pos1, np, np1, n1, ilambda1, trunc;
    clock_t start, stop;
    
    n = *nn;
    d = *dd;
    p = *pp;
    max_ite1 = *mmax_ite;
    max_ite2 = *mmax_ite;
    prec1 = *pprec;
    prec2 = *pprec;
    nlambda = *nnlambda;
    gamma = *ggamma;
    flag = *fflag;
    neg1 = -1;
    pos1 = 1;
    np = (double)n*(double)p;
    np1 = 1/np;
    n1 = 1/(double)n;
    max_act_gr_in = *mmax_act_gr_in;
    trunc = *ttrunc;
    
    double *beta2 = (double *) Calloc(d*p, double);
    double *beta1 = (double *) Calloc(d*p, double);
    double *beta0 = (double *) Calloc(d*p, double);
    double *beta_tild = (double *) Calloc(d*p, double);
    int *gr_act = (int *) Calloc(d, int);
    double *y_hat = (double *) Calloc(n*p, double);
    double *grad = (double *) Calloc(d*p, double);
    double *grad_row2 = (double *) Calloc(d, double);
    for(i=0;i<p;i++){
        for (j=0; j<n; j++) {
            y_hat[i*n+j] = Y[i*n+j];
        }
    }
    
    start = clock();
    gr_size_a = 0;
    for (i=0; i<nlambda; i++) {
        ilambda = lambda[i]*n;
        ilambda1 = (1+trunc)*ilambda;
        prec1 = (1+trunc)*ilambda;
        ite1 = 0;
        dif1 = prec1*2;
        while (dif1>prec1 && ite1<max_ite1) {
            mean_mvr(intcpt+i*p, y_hat, n, p);
            dif_vec_const_mvr(y_hat, intcpt+i*p, pos1, n, p); //y_hat = y_hat - intcpt[i]
            act_gr_in = 0;
            for(j=0; j<d; j++){
                match = is_match(j,gr_act,gr_size_a);
                if(match == 0){ // if j in set_act
                    tmp = vec_mat_inprod_2norm(y_hat, X+j*n, n, p); // || X[,j]^T y_hat ||
                    if(tmp>ilambda1){
                        gr_act[gr_size_a] = j;
                        gr_size_a++;
                        act_gr_in++;
                    }
                    if(act_gr_in == max_act_gr_in){
                        break;
                    }
                }
            }
            dif_vec_const_mvr(y_hat, intcpt+i*p, neg1, n, p); //y_hat = y_hat + intcpt[i]
            ite2 = 0;
            dif2 = 1;
            while (dif2>prec2 && ite2<max_ite2) {
                mean_mvr(intcpt+i*p, y_hat, n, p);
                dif_vec_const_mvr(y_hat, intcpt+i*p, pos1, n, p); //y_hat = y_hat - intcpt[i]
                
                for (m=0; m<gr_size_a; m++) {
                    c_idx = gr_act[m];
                    dif_mat_mvr(y_hat, X+c_idx*n, beta1+c_idx, neg1, n, d, p); //y_hat = y_hat+X[,c_idx]*beta1[c_idx,]
                    vec_inprod_mvr(y_hat, X, beta_tild, c_idx, n, d, p); //  beta_tild[c_idx,] = X[,c_idx]^T*y_hat
                    tmp = norm2_gr_mvr(beta_tild+c_idx, d, p);
                    for (k=0; k<p; k++) {
                        if(flag==1){
                            beta1[k*d+c_idx] = soft_thresh_gr_l1(tmp,ilambda,beta_tild[k*d+c_idx],n1);
                        }
                        if(flag==2){
                            beta1[k*d+c_idx] = soft_thresh_gr_mcp(tmp,ilambda,beta_tild[k*d+c_idx],gamma,n1);
                        }
                        if(flag==3){
                            beta1[k*d+c_idx] = soft_thresh_gr_scad(tmp,ilambda,beta_tild[k*d+c_idx],gamma,n1);
                        }
                    }
                    dif_mat_mvr(y_hat, X+c_idx*n, beta1+c_idx, pos1, n, d, p); //y_hat = y_hat-X[,c_idx]*beta1[c_idx,]
                }
                dif_vec_const_mvr(y_hat, intcpt+i*p, neg1, n, p); //y_hat = y_hat + intcpt[i]
                ite2++;
                dif2 = dif_Fnorm_mvr(beta1, beta0, gr_act, gr_size_a, d, p);
                mat_copy_mvr(beta1, beta0, gr_act, gr_size_a, d, p);
            }
            ite_cyc[i] += ite2;
            vec_mat_prod_mvr(grad, y_hat, X, p, n, d, pos1); // grad = X^T y_hat
            norm2_row_mat(grad_row2, grad, d, p);
            dif1 = max_vec(grad_row2, d);
            //dif1 = dif_Fnorm_mvr(beta1, beta2, gr_act, gr_size_a, d, p);
            //mat_copy_mvr(beta1, beta2, gr_act, gr_size_a, d, p);
            gr_size_a1 = 0;
            for (k=0; k<gr_size_a; k++) {
                c_idx = gr_act[k];
                if(norm2_gr_mvr(beta1+c_idx, d, p) > 0){
                    gr_act[gr_size_a1] = c_idx;
                    gr_size_a1++;
                }
            }
            gr_size_a = gr_size_a1;
            ite1++;
        }
        ite_lamb[i] = ite1;
        obj[i] = get_obj_mvr(y_hat, beta1, xinvc, uinv, gr_act, gr_size_a, n, d, p,ilambda)/np;
        stop = clock();
        runt[i] = (double)(stop - start)/CLOCKS_PER_SEC;
        
        mat_copy_mvr(beta1, beta+i*d*p, gr_act, gr_size_a, d, p);
        gr_size_act[i] = gr_size_a;
    }
    
    Free(beta2);
    Free(beta1);
    Free(beta0);
    Free(beta_tild);
    Free(gr_act);
    Free(y_hat);
    Free(grad);
    Free(grad_row2);
}
