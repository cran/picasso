#include "mymath.h"

void picasso_logit_gr_greedy_orth(double *Y, double * X, double * beta, double * intcpt, int * nn, int * dd, int * ite_lamb, int * ite_cyc, int * gr_size_act, double *obj, double *runt, int * gr, int * gr_nn, int * gr_size, double *lambda, int *nnlambda, double * ggamma, int *mmax_ite, double *pprec, int *fflag){
    
    int i, k, m, n, d, gr_n, max_ite1, max_ite2, nlambda, gr_size_a, gr_size_a1, comb_flag, match, ite1, ite2, c_idx, gr_idx, idx, flag;
    double gamma, w, wn, prec1, prec2, ilambda, ilambda0, tmp, dif1, dif2, neg1, pos1, max_norm2;
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
    w = 0.25;
    wn = w*(double)n;
    gr_n = *gr_nn;
    neg1 = -1;
    pos1 = 1;
    
    double *beta2 = (double *) Calloc(d, double);
    double *beta1 = (double *) Calloc(d, double);
    double *beta0 = (double *) Calloc(d, double);
    double *beta_tild = (double *) Calloc(d, double);
    double *grad = (double *) Calloc(d, double);
    double *p = (double *) Calloc(n, double);
    double *p_Y = (double *) Calloc(n, double);
    int *gr_act = (int *) Calloc(gr_n, int);
    double *Xb = (double *) Calloc(n, double);
    
    start = clock();
    gr_size_a = 0;
    for (i=0; i<nlambda; i++) {
        ilambda0 = lambda[i];
        ilambda = lambda[i]/w;
        if(i>0) {
            intcpt[i] = intcpt[i-1] - sum_vec_dif(p,Y,n)/wn;
        }
        prec1 = (1+prec2*10)*ilambda;
        ite1 = 0;
        dif1 = prec1*2;
        while (dif1>prec1 && ite1<max_ite1) {
            p_update(p,Xb,intcpt[i],n); // p[i] = 1/(1+exp(-intcpt-Xb[i]))
            dif_vec(p_Y, p, Y, n); // p_Y = p - Y
            //vec_mat_prod(grad, p_Y, X, n, d); // grad = p_Y^T X
            if(flag==1){
                get_grad_logit_gr_l1_all(grad, p_Y, X, gr, gr_size, gr_n, n); // g[gr] = X[gr]^T (p-Y)/n
            }
            if(flag==2){
                get_grad_logit_gr_mcp_all(grad, p_Y, X, beta1, gr, gr_size, gr_n, ilambda0, gamma, n); // g[gr] = <p-Y, X[,gr]>/n + h_grad(mcp)
            }
            if(flag==3){
                get_grad_logit_gr_scad_all(grad, p_Y, X, beta1, gr, gr_size, gr_n, ilambda0, gamma, n); // g[gr] = <p-Y, X[,gr]>/n + h_grad(scad)
            }
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
            ite2 = 0;
            dif2 = 1;
            while (dif2>prec2 && ite2<max_ite2) {
                intcpt[i] = intcpt[i] - sum_vec_dif(p,Y,n)/wn;
                for (m=0; m<gr_size_a; m++) {
                    c_idx = gr_act[m];
                    gr_idx = gr[c_idx];
                    p_update(p,Xb,intcpt[i],n); // p[i] = 1/(1+exp(-intcpt-Xb[i]))
                    dif_vec(p_Y, p, Y, n); // p_Y = p - Y
                    if(flag==1){
                        get_grad_logit_gr_l1(grad, p_Y, X, gr[c_idx], gr_size[c_idx], n); // g[gr] = <p-Y, X[,gr]>/n
                    }
                    if(flag==2){
                        get_grad_logit_gr_mcp(grad, p_Y, X, beta1, gr[c_idx], gr_size[c_idx], ilambda0, gamma, n); // g[gr] = <p-Y, X[,gr]>/n + h_grad(mcp)
                    }
                    if(flag==3){
                        get_grad_logit_gr_scad(grad, p_Y, X, beta1, gr[c_idx], gr_size[c_idx], ilambda0, gamma, n); // g[gr] = <p-Y, X[,gr]>/n + h_grad(scad)
                    }
                    X_beta_update_gr(Xb, X, beta1, gr_idx, gr_size[c_idx],n,neg1);
                    logit_gr_vec_dif(beta_tild, beta1, grad, w, gr[c_idx], gr_size[c_idx]); //beta_tild[gr] = beta1[gr] - g/w
                    tmp = norm2_gr_vec(beta_tild, gr[c_idx], gr_size[c_idx]);
                    for (k=gr_idx; k<gr_idx+gr_size[c_idx]; k++) {
                        beta1[k] = soft_thresh_gr_l1(tmp,ilambda,beta_tild[k],1);
                    }
                    X_beta_update_gr(Xb, X, beta1, gr_idx, gr_size[c_idx],n,pos1);
                }
                ite2++;
                dif2 = dif_2norm_gr(beta1, beta0, gr, gr_size, gr_act, gr_size_a);
                vec_copy_gr(beta1, beta0, gr, gr_size, gr_act, gr_size_a);
            }
            ite_cyc[i] += ite2;
            p_update(p,Xb,intcpt[i],n); // p[i] = 1/(1+exp(-intcpt-Xb[i]))
            dif_vec(p_Y, p, Y, n); // p_Y = p - Y
            if(flag==1){
                get_grad_logit_gr_l1_all(grad, p_Y, X, gr, gr_size, gr_n, n); // g[gr] = X[gr]^T (p-Y)/n
            }
            if(flag==2){
                get_grad_logit_gr_mcp_all(grad, p_Y, X, beta1, gr, gr_size, gr_n, ilambda0, gamma, n); // g[gr] = <p-Y, X[,gr]>/n + h_grad(mcp)
            }
            if(flag==3){
                get_grad_logit_gr_scad_all(grad, p_Y, X, beta1, gr, gr_size, gr_n, ilambda0, gamma, n); // g[gr] = <p-Y, X[,gr]>/n + h_grad(scad)
            }
            max_norm2_gr(grad, gr, gr_size, gr_n, &dif1, &idx);
            //dif1 = dif_2norm(beta1, beta2, set_act, size_a);
            //vec_copy(beta1, beta2, set_act, size_a);
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
    Free(p);
    Free(p_Y);
    Free(grad);
    Free(gr_act);
    Free(Xb);
}
