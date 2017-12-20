#include "mathutils.h"

#define BIG_EXPONENT (690) 

double sqrt_mse_obj_change(const double *r, 
    const double *X, const double L, int k, int n, double beta_new, double beta_old){
    double delta = beta_new - beta_old;

    double tmp = 0.0;
    double a = 0.0;
    for (int i = 0; i < n; i++){
        tmp = r[i] / L;
        tmp = tmp*tmp/n;
        a += X[k*n+i]*X[k*n+i]*(1-tmp);
    }

    a = a/(n*L);

    double obj_change = a*delta*delta*0.5;
    return obj_change; 
}


void coordinate_update(double * beta, double gr, double S, 
                        int standardized, double lambda){
    double tmp = 0;
    if (standardized)
        tmp = gr + *beta;
    else
        tmp = gr + (*beta) * S;

    *beta = soft_thresh_l1(tmp, lambda);


    if (!standardized)
        *beta = (*beta) / S;
}

void coordinate_update_nonlinear(double * beta, double gr, double S, 
                        int standardized, double lambda, double gamma, int method_flag){
    double tmp = 0;
    if (standardized)
        tmp = gr + *beta;
    else
        tmp = gr + (*beta) * S;

    if (method_flag == 1)
        *beta = soft_thresh_l1(tmp, lambda);
    
    if (method_flag == 2)
        *beta = soft_thresh_mcp(tmp, lambda, gamma);

    if (method_flag == 3)
        *beta = soft_thresh_scad(tmp, lambda, gamma);

    if (!standardized)
        *beta = (*beta) / S;
}


double truncate_exponent(double x, double a){
    double t = fabs(x);

    if (t > a){
        if (x > 0)
            return(a);
        else
            return(-a);
    } else {
        return(x);
    }
}

int min_int(int a, int b){
    if (a < b)
        return(a);
    else
        return(b);
}


double penalty_derivative(int method_flag, double x, double lambda, double gamma){
    // mcp
    if (method_flag == 2){
        if (fabs(x) > lambda * gamma){
            return(0);
        } else  {
            return(lambda - fabs(x)/gamma);
        }
    }

    // scad
    if (method_flag == 3){
        if (fabs(x) > lambda * gamma){
            return(0);
        } else if (fabs(x) > lambda){
            return((lambda*gamma-fabs(x))/(gamma-1));
        } else {
            return(lambda);
        }
    }

    return(lambda);
}


double get_penalty_value(int method_flag, double x, double lambda, double gamma){
    // lasso
    if (method_flag == 1){
        x = fabs(x);
        return(lambda * x);
    }

    // mcp
    if (method_flag == 2){
        x = fabs(x);
        if (x > gamma * lambda){
            return(lambda*lambda*gamma/2);
        } else {
            return(lambda*(x - x*x/(2*lambda*gamma)));
        }
    }

    // scad
    if (method_flag == 3){
        x = fabs(x);
        if (x > gamma * lambda){
            return((gamma+1)*lambda*lambda/2);
        } else if (x > lambda){
            return(-(x*x - 2*lambda*gamma*x +lambda*lambda)/(2*(gamma-1)));
        } else {
            return(lambda*x);
        }
    }
    return(0);
}

double get_penalized_logistic_loss(int method_flag, double *p, double * Y, double * Xb, double * beta, 
                                double intcpt, int n, int d, double lambda, double gamma){
    int i;
    double v = 0.0;
    for (i = 0; i<n; i++){
        v -= Y[i]*(intcpt+Xb[i]); 
    }
    for (i = 0; i<n; i++)
    if (p[i] > 1e-8) {
        v -= (log(p[i]) - intcpt - Xb[i]);
    }

    v = v/n;
    
    for (i = 0; i<d; i++){
        v += get_penalty_value(method_flag, fabs(beta[i]), lambda, gamma);
    }

    return(v); 
}

double get_penalized_sqrt_mse_loss(int method_flag, const double * Y, 
            const double * Xb, double * beta, int n, int d, 
            const double* lambda, double gamma){
    int i;
    double v = 0.0;
    for (i = 0; i<n; i++){
        v += (Y[i]-Xb[i])*(Y[i]-Xb[i]); 
    }

    v = sqrt(v/n);
    
    for (i = 0; i<d; i++){
        v += get_penalty_value(method_flag, fabs(beta[i]), lambda[i], gamma);
    }

    return(v); 
}

double get_sqrt_mse_loss(const double * Y, 
            double * Xb, double intcpt, int n, int d){
    int i;
    double v = 0.0;
    for (i = 0; i<n; i++){
        v += (Y[i]-Xb[i]-intcpt)*(Y[i]-Xb[i]-intcpt); 
    }

    v = sqrt(v/n);
    
    return(v); 
}



double get_penalized_poisson_loss(int method_flag, double *p, double * Y, double * Xb, double * beta, 
                                double intcpt, int n, int d, double lambda, double gamma){
    int i;
    double v = 0.0;
    for (i = 0; i<n; i++){
        v -= Y[i]*(intcpt+Xb[i]); 
    }
    for (i = 0; i<n; i++)
        v += p[i];
    

    v = v/n;

    for (i = 0; i<d; i++){
        v += get_penalty_value(method_flag, fabs(beta[i]), lambda, gamma);
    }

    return(v); 
}


double mean(double *x, int n){
    int i;
    double tmp;
    
    tmp = 0;
    for(i=0; i<n ;i++){
        tmp += x[i];
    }
    return tmp/(double)n;
}

// <x,y>
double vec_inprod(double *x, double *y, int n){
    int i;
    double tmp=0;
    
    for(i=0; i<n; i++){
        tmp += x[i]*y[i];
    }
    return tmp;
}



// copy x to y
void vec_copy(double *x, double *y, int *xa_idx, int n){
    int i, idx;
    
    for(i=0; i<n; i++){
        idx = xa_idx[i];
        y[idx] = x[idx];
    }
}


// x[i] = soft_l1(y,lamb);
double soft_thresh_l1(double y, double lamb){
    
    if(y>lamb) return y-lamb;
    else if(y<(-lamb)) return y+lamb;
    else return 0;
}

// x[i] = soft_scad(y,lamb);
double soft_thresh_scad(double y, double lamb, double gamma){
    
    if(fabs(y)>fabs(gamma*lamb)) {
        return y;
    }else{
        if(fabs(y)>fabs(2*lamb)) {
            return soft_thresh_l1(y, gamma*lamb/(gamma-1))/(1-1/(gamma-1));
        }else{
            return soft_thresh_l1(y, lamb);
        }
    }
}

// x[i] = soft_mcp(y,lamb);
double soft_thresh_mcp(double y, double lamb, double gamma){
    
    if(fabs(y)>fabs(gamma*lamb)) {
        return y;
    }else{
        return soft_thresh_l1(y, lamb)/(1-1/gamma);
    }
}

double soft_thresh_gr_l1(double y, double lamb, double beta, double dbn1){
    if(y>lamb) return (1-lamb/y)*beta*dbn1;
    else return 0;
}

double soft_thresh_gr_mcp(double y, double lamb, double beta, double gamma, double dbn1){
    if(y>lamb*gamma){
        return beta*dbn1;
    }else{
        //return soft_thresh_gr_l1(y, lamb, beta, dbn1)*gamma/(gamma-1);
        if(y>lamb) return (1-lamb/y)*beta*dbn1*gamma/(gamma-1);
        else return 0;
    }
}

double soft_thresh_gr_scad(double y, double lamb, double beta, double gamma, double dbn1){
    if(y>lamb*gamma){
        return beta*dbn1;
    }else{
        if(y>2*lamb){
            //return soft_thresh_gr_l1(y, lamb*gamma/(gamma-1), beta, dbn1)*(gamma-1)/(gamma-2);
            if(y>lamb*gamma/(gamma-1)) return (1-lamb*gamma/(gamma-1)/y)*beta*dbn1*(gamma-1)/(gamma-2);
            else return 0;
        }else{
            //return soft_thresh_gr_l1(y, lamb, beta, dbn1);
            if(y>lamb) return (1-lamb/y)*beta*dbn1;
            else return 0;
        }
    }
}



// Xb = Xb+X*beta
void X_beta_update(double *Xb, const double *X, double beta, int n){
    int i;
    
    for (i = 0; i < n; i++) {
        Xb[i] = Xb[i] + beta*X[i];
    }
}


// p[i] = 1/(1+exp(-intcpt-Xb[i]))
void p_update(double *p, double *Xb, double intcpt, int n){
    int i;    
    for (i = 0; i < n; i++) {
        p[i] = 1/(1+exp(truncate_exponent(-intcpt-Xb[i], BIG_EXPONENT)));
        if (p[i] > 0.999) p[i] = 1;
        if (p[i] < 0.001) p[i] = 0;
    }
}

void standardize_design(double * X, double * xx, double * xm, 
                            double * xinvc, int * nn, int * dd) {
    int i, j, jn, n, d;
    
    n = *nn;
    d = *dd;
    
    for (j = 0; j < d; j++) {
        // Center
        xm[j] = 0;
        jn = j*n;
        for (i = 0; i < n; i++) 
            xm[j] += X[jn+i];
        
        xm[j] = xm[j] / n;
        for (i = 0; i < n; i++) 
            xx[jn+i] = X[jn+i] - xm[j];
        
        // Scale
        xinvc[j] = 0;
        for (i = 0; i < n; i++) {
            xinvc[j] += xx[jn+i]*xx[jn+i];
        }

        if (xinvc[j] > 0){
           xinvc[j] = 1/sqrt(xinvc[j]/(n-1));
            for (i = 0; i < n; i++) {
                xx[jn+i] = xx[jn+i]*xinvc[j];
            } 
        }
    }
}

