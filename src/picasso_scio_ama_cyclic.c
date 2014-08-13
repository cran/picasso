#include "mymath.h"

//extern "C" {
    // LU decomoposition of a general matrix
    //void dgetrf_(int* M, int *N, double* A, int* lda, int* IPIV, int* INFO);
    
    // generate inverse of a matrix given its LU decomposition
    //void dgetri_(int* N, double* A, int* lda, int* IPIV, double* WORK, int* lwork, int* INFO);
//}

void inverse(double* A, int N)
{
    int *IPIV = (int *) malloc((N+1)*sizeof(int));
    int LWORK = N*N;
    double *WORK = (double *) malloc(LWORK*sizeof(double));
    int INFO;
    
    dgetrf_(&N,&N,A,&N,IPIV,&INFO);
    dgetri_(&N,A,&N,IPIV,WORK,&LWORK,&INFO);
    
    free(IPIV);
    free(WORK);
}

void picasso_scio_ama_cyclic(double * S, double * beta, int * dd, int * ite1, double *lambda, int *nnlambda, int *mmax_ite, double *pprec, double * x, int *col_cnz, int *row_idx){
    
    int i, j, k, m, d, d_sq, col, max_ite, nlambda, size_a, size_a1, comb_flag, match, ite, c_idx, cnz;
    double prec, ilambda, tmp, dif;
    
    d = *dd;
    d_sq = d*d;
    max_ite = *mmax_ite;
    prec = *pprec;
    nlambda = *nnlambda;
    cnz = 0;
    
    double *beta1 = (double *) malloc(d*sizeof(double));
    double *beta0 = (double *) malloc(d*sizeof(double));
    int *set_act = (int *) malloc(d*sizeof(int));
    double *e = (double *) malloc(d*sizeof(double));
    
    for (col=0; col<d; col++) {
        
        for(i=0;i<d;i++){
            beta1[i] = 0;
            beta0[i] = 0;
            e[i] = 0;
        }
        e[col] = 1;
        size_a = 0;
        
        for (i=0; i<nlambda; i++) {
            ilambda = lambda[i];
            for(j=0; j<d; j++){
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
                            beta1[c_idx] = soft_thresh_l1(tmp/S[c_idx*d+c_idx], ilambda/S[c_idx*d+c_idx]);
                        }
                        ite++;
                        dif = dif_2norm(beta1, beta0, set_act, size_a);
                        vec_copy(beta1, beta0, set_act, size_a);
                    }
                    ite1[i*d+col] += ite;
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
    
    free(beta1);
    free(beta0);
    free(set_act);
    free(e);
}
